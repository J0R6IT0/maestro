use crate::audio::{MEL_BANDS, MelPatches, PATCH_FRAMES, PatchExtractor};
use anyhow::{Context, Result, anyhow, bail};
use ort::{
    environment::Environment,
    ep::ExecutionProviderDispatch,
    session::{OutputSelector, RunOptions, Session},
    value::{Outlet, TensorRef},
};
use serde_json::Value;
use std::path::Path;
use tracing::info;

#[cfg(all(feature = "cuda", feature = "openvino"))]
compile_error!("features `cuda` and `openvino` are mutually exclusive");

#[cfg(feature = "cuda")]
const EXECUTION_PROVIDER: &str = "CUDA";
#[cfg(feature = "openvino")]
const EXECUTION_PROVIDER: &str = "OpenVINO";
#[cfg(not(any(feature = "cuda", feature = "openvino")))]
const EXECUTION_PROVIDER: &str = "CPU";

const INPUT: &str = "melspectrogram";
const EMBEDDINGS: &str = "layer_06_embeddings";
const PREDICTIONS: &str = "activations";

pub struct Model {
    session: Session,
    labels: Vec<String>,
    name: String,
}

pub struct Analysis {
    /// concat(CLS, DIST, mean of the other tokens) from layer 7, averaged over patches (2304-d).
    pub embedding: Vec<f32>,
    pub styles: Vec<(String, f32)>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let labels = read_labels(&path.with_extension("json"))?;
        let name = path
            .file_stem()
            .context("model path has no file name")?
            .to_string_lossy()
            .into_owned();
        let env = ort::init().build()?;
        let session = create_session(&env, path)?;
        check_graph(&session)?;

        let mut model = Self {
            session,
            labels,
            name,
        };

        model
            .analyze(PatchExtractor::new().patches_from_samples(&[0.0]))
            .with_context(|| format!("test inference on {EXECUTION_PROVIDER} failed"))?;
        info!(execution_provider = EXECUTION_PROVIDER, "model loaded");
        Ok(model)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn analyze(&mut self, patches: MelPatches) -> Result<Analysis> {
        let count = patches.patch_count as f32;
        let mut embedding = Vec::new();
        let mut scores = vec![0.0f32; self.labels.len()];
        let options = RunOptions::new()?.with_outputs(
            OutputSelector::no_default()
                .with(EMBEDDINGS)
                .with(PREDICTIONS),
        );

        for patch in patches.values.as_chunks::<{ PATCH_FRAMES * MEL_BANDS }>().0 {
            let input = TensorRef::from_array_view(([1, PATCH_FRAMES, MEL_BANDS], &patch[..]))?;
            let outputs = self
                .session
                .run_with_options(ort::inputs![INPUT => input], &options)?;

            let (shape, tokens) = outputs[EMBEDDINGS].try_extract_tensor::<f32>()?;
            let dim = *shape
                .last()
                .context("embeddings tensor without dimensions")? as usize;
            add_patch_embedding(&mut embedding, tokens, dim, count);

            let (_, predictions) = outputs[PREDICTIONS].try_extract_tensor::<f32>()?;
            for (score, p) in scores.iter_mut().zip(predictions) {
                *score += p / count;
            }
        }

        let mut styles: Vec<(String, f32)> = self.labels.iter().cloned().zip(scores).collect();
        styles.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(Analysis { embedding, styles })
    }
}

fn read_labels(meta_path: &Path) -> Result<Vec<String>> {
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(meta_path)
            .with_context(|| format!("reading {}", meta_path.display()))?,
    )?;
    serde_json::from_value(meta["classes"].clone()).context("model metadata: no `classes`")
}

fn check_graph(session: &Session) -> Result<()> {
    let has = |outlets: &[Outlet], name: &str| outlets.iter().any(|o| o.name() == name);
    if has(session.inputs(), INPUT)
        && has(session.outputs(), EMBEDDINGS)
        && has(session.outputs(), PREDICTIONS)
    {
        return Ok(());
    }
    let found: Vec<&str> = session
        .inputs()
        .iter()
        .chain(session.outputs())
        .map(|o| o.name())
        .collect();
    bail!(
        "unexpected ONNX graph: expected input '{INPUT}' and outputs '{EMBEDDINGS}', \
         '{PREDICTIONS}'; found {found:?}"
    );
}

/// Adds one patch's concat(CLS, DIST, mean of the other tokens), divided by the number of patches,
/// to `embedding`. `tokens` is [1, tokens, dim], with CLS first and DIST second.
fn add_patch_embedding(embedding: &mut Vec<f32>, tokens: &[f32], dim: usize, patches: f32) {
    let mut rows = tokens.chunks_exact(dim);
    let (cls, dist) = (rows.next().unwrap(), rows.next().unwrap());
    let mut others_sum = vec![0.0; dim];
    let mut others = 0;
    for row in rows {
        for (sum, x) in others_sum.iter_mut().zip(row) {
            *sum += x;
        }
        others += 1;
    }
    embedding.resize(dim * 3, 0.0);
    for d in 0..dim {
        let others_mean = others_sum[d] / others as f32;
        embedding[d] += cls[d] / patches;
        embedding[dim + d] += dist[d] / patches;
        embedding[2 * dim + d] += others_mean / patches;
    }
}

#[cfg(not(feature = "openvino"))]
fn create_session(env: &Environment, path: &Path) -> Result<Session> {
    #[cfg(feature = "cuda")]
    let providers = [ort::ep::CUDA::default().build().error_on_failure()];
    #[cfg(not(feature = "cuda"))]
    let providers: [ExecutionProviderDispatch; 0] = [];
    load_session(env, path, &providers)
}

#[cfg(feature = "openvino")]
fn create_session(env: &Environment, path: &Path) -> Result<Session> {
    let openvino = |device: &str| {
        [ort::ep::OpenVINO::default()
            .with_device_type(device)
            .build()
            .error_on_failure()]
    };
    load_session(env, path, &openvino("GPU")).or_else(|e| {
        tracing::warn!("no Intel GPU found, OpenVINO will run on the CPU: {e:#}");
        load_session(env, path, &openvino("CPU"))
    })
}

fn load_session(
    env: &Environment,
    path: &Path,
    providers: &[ExecutionProviderDispatch],
) -> Result<Session> {
    Session::builder(env)?
        .with_execution_providers(providers)
        .map_err(|e| anyhow!("registering {EXECUTION_PROVIDER}: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}
