//! Preparation of audio files. ffmpeg decodes them to mono 16 kHz, then a port of Essentia's
//! MAEST preprocessing (TensorflowPredictMAEST + TensorflowInputMusiCNN) converts them into mel
//! patches: 512-sample frames / 256 hop, Hann window, magnitude spectrum, 96 Slaney mel bands
//! (power, linear weighting, unit_tri), log10(1 + 10000·x), then MAEST's mean/std normalization.

use anyhow::{Context, Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{
    env, fs,
    io::{self, BufReader, ErrorKind, Read},
    ops::Range,
    path::Path,
    process::{Command, Stdio},
    sync::Arc,
    thread,
};
use tempfile::NamedTempFile;
use tracing::debug;

pub const MEL_BANDS: usize = 96;
pub const PATCH_FRAMES: usize = 1876;
const SAMPLE_RATE_HZ: usize = 16000;
const FRAME_SAMPLES: usize = 512;
const HOP_SAMPLES: usize = 256;
const PATCH_STRIDE_FRAMES: usize = 1875;
/// Long tracks are analyzed in this many evenly spaced 30s patches.
const MAX_PATCHES: usize = 8;
const LOG_MEL_MEAN: f32 = 2.067_556_9;
const LOG_MEL_STD: f32 = 1.268_292_8;
const SILENT_BAND: f32 = normalize_log_mel(0.0);

const fn normalize_log_mel(log_mel: f32) -> f32 {
    (log_mel - LOG_MEL_MEAN) / (2.0 * LOG_MEL_STD)
}

pub fn check_ffmpeg() -> Result<()> {
    let status = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("ffmpeg not found on PATH")?;
    if !status.success() {
        bail!("ffmpeg -version failed: {status}");
    }
    Ok(())
}

const TEMP_FILE_PREFIX: &str = "maestro-audio-";

pub fn create_temp_file() -> io::Result<NamedTempFile> {
    tempfile::Builder::new().prefix(TEMP_FILE_PREFIX).tempfile()
}

/// Deletes the temp files of analyzers that were killed. Only safe while no other one runs.
pub fn remove_stale_temp_files() {
    let Ok(entries) = fs::read_dir(env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(TEMP_FILE_PREFIX)
        {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Ready mel patches: `patch_count * PATCH_FRAMES * MEL_BANDS` values, row-major.
pub struct MelPatches {
    pub values: Vec<f32>,
    pub patch_count: usize,
}

pub struct PatchExtractor {
    hann_window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    filterbank: Vec<MelFilter>,
}

struct MelFilter {
    first_bin: usize,
    /// Triangle weights of the FFT bins starting from `first_bin`.
    weights: Vec<f32>,
}

/// Working memory of the FFT, reused between frames.
struct FftBuffers {
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl PatchExtractor {
    pub fn new() -> Self {
        Self {
            hann_window: hann_window(),
            fft: FftPlanner::new().plan_fft_forward(FRAME_SAMPLES),
            filterbank: slaney_filterbank(),
        }
    }

    pub fn extract_patches(&self, path: &Path, expected_secs: usize) -> Result<MelPatches> {
        let expected_starts = patch_start_frames(expected_secs * SAMPLE_RATE_HZ / HOP_SAMPLES + 1);
        let (sample_count, mut patch_audio) = decode_patch_audio(path, &expected_starts)?;
        let start_frames = patch_start_frames(sample_count / HOP_SAMPLES + 1);
        if start_frames != expected_starts {
            debug!(
                expected_secs,
                sample_count, "the patches moved, decoding again"
            );
            patch_audio = decode_patch_audio(path, &start_frames)?.1;
        }

        let patches = self.build_patches(sample_count, &start_frames, |i| {
            (patch_audio[i].sample_range.start, &patch_audio[i].samples)
        });
        if patches.values.iter().any(|v| !v.is_finite()) {
            bail!("the audio has NaN, infinite or out of range samples");
        }
        Ok(patches)
    }

    pub fn patches_from_samples(&self, samples: &[f32]) -> MelPatches {
        let start_frames = patch_start_frames(samples.len() / HOP_SAMPLES + 1);
        self.build_patches(samples.len(), &start_frames, |_| (0, samples))
    }

    /// Builds the patches starting at `start_frames` of a track with `sample_count` samples.
    /// `patch_audio(i)` returns where the samples under patch `i` begin, and those samples.
    fn build_patches<'a>(
        &self,
        sample_count: usize,
        start_frames: &[usize],
        patch_audio: impl Fn(usize) -> (usize, &'a [f32]),
    ) -> MelPatches {
        let mut fft = self.fft_buffers();
        if start_frames.is_empty() {
            let (audio_start, samples) = patch_audio(0);
            let mut values = vec![SILENT_BAND; PATCH_FRAMES * MEL_BANDS];
            for (frame_idx, frame_bands) in values
                .as_chunks_mut::<MEL_BANDS>()
                .0
                .iter_mut()
                .take(sample_count / HOP_SAMPLES + 1)
                .enumerate()
            {
                self.write_frame_bands(samples, audio_start, frame_idx, frame_bands, &mut fft);
            }
            return MelPatches {
                values,
                patch_count: 1,
            };
        }

        let mut values = vec![0.0; start_frames.len() * PATCH_FRAMES * MEL_BANDS];
        for (patch_idx, (patch, &start_frame)) in values
            .as_chunks_mut::<{ PATCH_FRAMES * MEL_BANDS }>()
            .0
            .iter_mut()
            .zip(start_frames)
            .enumerate()
        {
            let (audio_start, samples) = patch_audio(patch_idx);
            for (frame_idx, frame_bands) in
                patch.as_chunks_mut::<MEL_BANDS>().0.iter_mut().enumerate()
            {
                self.write_frame_bands(
                    samples,
                    audio_start,
                    start_frame + frame_idx,
                    frame_bands,
                    &mut fft,
                );
            }
        }
        MelPatches {
            values,
            patch_count: start_frames.len(),
        }
    }

    fn fft_buffers(&self) -> FftBuffers {
        FftBuffers {
            spectrum: vec![Complex::default(); FRAME_SAMPLES],
            scratch: vec![Complex::default(); self.fft.get_inplace_scratch_len()],
        }
    }

    /// Writes the normalized log-mel bands of frame `frame_idx` to `out`. The frame is centered at
    /// sample `frame_idx * HOP_SAMPLES` and zero padded where `samples`, which begin at sample
    /// `audio_start`, don't cover it.
    fn write_frame_bands(
        &self,
        samples: &[f32],
        audio_start: usize,
        frame_idx: usize,
        out: &mut [f32],
        fft: &mut FftBuffers,
    ) {
        let frame_start =
            (frame_idx * HOP_SAMPLES) as isize - (FRAME_SAMPLES / 2 + audio_start) as isize;
        for (i, (bin, weight)) in fft.spectrum.iter_mut().zip(&self.hann_window).enumerate() {
            let sample = usize::try_from(frame_start + i as isize)
                .ok()
                .and_then(|s| samples.get(s))
                .copied()
                .unwrap_or(0.0);
            *bin = Complex::new(sample * weight, 0.0);
        }
        self.fft
            .process_with_scratch(&mut fft.spectrum, &mut fft.scratch);

        for (out, filter) in out.iter_mut().zip(&self.filterbank) {
            let energy: f32 = fft.spectrum[filter.first_bin..]
                .iter()
                .zip(&filter.weights)
                .map(|(bin, w)| bin.norm_sqr() * w)
                .sum();
            *out = normalize_log_mel((1.0 + 10000.0 * energy).log10());
        }
    }
}

/// The samples of a patch, the first one being at `sample_range.start`.
struct PatchAudio {
    sample_range: Range<usize>,
    samples: Vec<f32>,
}

/// Decodes the whole file and keeps the samples under the patches starting at `start_frames`,
/// or under the first patch if there are none. Also returns the file's total sample count.
fn decode_patch_audio(path: &Path, start_frames: &[usize]) -> Result<(usize, Vec<PatchAudio>)> {
    let mut patch_audio: Vec<PatchAudio> = if start_frames.is_empty() {
        &[0][..]
    } else {
        start_frames
    }
    .iter()
    .map(|&start_frame| {
        let sample_range = (start_frame * HOP_SAMPLES).saturating_sub(FRAME_SAMPLES / 2)
            ..(start_frame + PATCH_FRAMES - 1) * HOP_SAMPLES + FRAME_SAMPLES / 2;
        PatchAudio {
            samples: Vec::with_capacity(sample_range.len()),
            sample_range,
        }
    })
    .collect();
    let (mut samples_seen, mut first_open) = (0, 0);
    decode_with_ffmpeg(path, |sample| {
        while patch_audio
            .get(first_open)
            .is_some_and(|patch| patch.sample_range.end <= samples_seen)
        {
            first_open += 1;
        }
        for patch in &mut patch_audio[first_open..] {
            if patch.sample_range.start > samples_seen {
                break;
            }
            patch.samples.push(sample);
        }
        samples_seen += 1;
    })?;
    if samples_seen == 0 {
        bail!("ffmpeg produced no audio");
    }
    Ok((samples_seen, patch_audio))
}

/// Decodes to f32 16 kHz mono.
fn decode_with_ffmpeg(path: &Path, on_sample: impl FnMut(f32)) -> Result<()> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vn", "-rematrix_maxval", "1.0", "-ac", "1", "-ar"])
        .arg(SAMPLE_RATE_HZ.to_string())
        .args(["-f", "f32le", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting ffmpeg")?;

    let mut stderr = child.stderr.take().unwrap();
    let stderr_reader = thread::spawn(move || {
        let mut stderr_bytes = Vec::new();
        let _ = stderr.read_to_end(&mut stderr_bytes);
        stderr_bytes
    });
    let read_result = read_samples(child.stdout.take().unwrap(), on_sample);
    let status = child.wait()?;
    let stderr_bytes = stderr_reader.join().unwrap_or_default();

    if !status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&stderr_bytes).trim()
        );
    }
    read_result.context("reading ffmpeg output")
}

fn read_samples(reader: impl Read, mut on_sample: impl FnMut(f32)) -> io::Result<()> {
    let mut reader = BufReader::new(reader);
    let mut buf = [0; 4];
    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => on_sample(f32::from_le_bytes(buf)),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

fn patch_start_frames(frame_count: usize) -> Vec<usize> {
    let starts: Vec<usize> = (0..)
        .map(|patch_idx| patch_idx * PATCH_STRIDE_FRAMES)
        .take_while(|start| start + PATCH_FRAMES <= frame_count)
        .collect();
    if starts.len() <= MAX_PATCHES {
        return starts;
    }
    let total = starts.len();
    (0..MAX_PATCHES)
        .map(|i| starts[i * (total - 1) / (MAX_PATCHES - 1)])
        .collect()
}

/// Essentia's default Hann window.
fn hann_window() -> Vec<f32> {
    (0..FRAME_SAMPLES)
        .map(|i| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SAMPLES - 1) as f32).cos()
        })
        .collect()
}

/// Slaney's mel scale.
fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz * 3.0 / 200.0
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// Essentia TriangularBands with Slaney mel edges.
fn slaney_filterbank() -> Vec<MelFilter> {
    let max_mel = hz_to_mel(SAMPLE_RATE_HZ as f64 / 2.0);
    let edges_hz: Vec<f64> = (0..MEL_BANDS + 2)
        .map(|i| mel_to_hz(max_mel * i as f64 / (MEL_BANDS + 1) as f64))
        .collect();
    let bin_hz = (SAMPLE_RATE_HZ as f64 / 2.0) / (FRAME_SAMPLES / 2) as f64;

    edges_hz
        .windows(3)
        .map(|edges| {
            let (low, peak, high) = (edges[0], edges[1], edges[2]);
            let first_bin = (low / bin_hz).ceil() as usize;
            let last_bin = ((high / bin_hz).floor() as usize).min(FRAME_SAMPLES / 2);
            let area = (high - low) / 2.0;
            let weights = (first_bin..=last_bin)
                .map(|bin| {
                    let hz = bin as f64 * bin_hz;
                    let weight = if hz < peak {
                        (hz - low) / (peak - low)
                    } else {
                        (high - hz) / (high - peak)
                    };
                    (weight / area) as f32
                })
                .collect();
            MelFilter { first_bin, weights }
        })
        .collect()
}
