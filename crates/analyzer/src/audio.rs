//! Preparation of audio files. ffmpeg decodes them to mono 16 kHz, then a port of Essentia's
//! MAEST preprocessing (TensorflowPredictMAEST + TensorflowInputMusiCNN) converts them into mel
//! patches: 512-sample frames / 256 hop, Hann window, magnitude spectrum, 96 Slaney mel bands
//! (power, linear weighting, unit_tri), log10(1 + 10000·x), then MAEST's mean/std normalization.

use anyhow::{Context, Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::Arc,
    thread,
};

pub const BANDS: usize = 96;
pub const PATCH_FRAMES: usize = 1876;
const SAMPLE_RATE: usize = 16000;
const FRAME: usize = 512;
const HOP: usize = 256;
const PATCH_HOP: usize = 1875;
/// Long tracks are analyzed in this many evenly spaced 30s patches.
const MAX_PATCHES: usize = 8;
const MEAN: f32 = 2.067_556_9;
const STD: f32 = 1.268_292_8;
const SILENCE: f32 = normalize(0.0);

const fn normalize(log_mel: f32) -> f32 {
    (log_mel - MEAN) / (2.0 * STD)
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

/// Ready mel patches: `count * PATCH_FRAMES * BANDS` values, row-major.
pub struct Patches {
    pub data: Vec<f32>,
    pub count: usize,
}

pub struct Decoder {
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    bands: Vec<MelBand>,
}

struct MelBand {
    first_bin: usize,
    /// Triangle weights of the FFT bins starting from `first_bin`.
    weights: Vec<f32>,
}

/// Working memory of the FFT, reused between frames.
struct FftBuffers {
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            window: hann_window(),
            fft: FftPlanner::new().plan_fft_forward(FRAME),
            bands: slaney_filterbank(),
        }
    }

    pub fn decode(&self, file: Vec<u8>) -> Result<Patches> {
        let patches = self.patches(&decode_samples(file)?);
        if patches.data.iter().any(|x| !x.is_finite()) {
            bail!("the audio has NaN, infinite or out of range samples");
        }
        Ok(patches)
    }

    pub fn patches(&self, audio: &[f32]) -> Patches {
        let frames = audio.len() / HOP + 1;
        let starts = patch_starts(frames);
        let mut fft = self.fft_buffers();

        if starts.is_empty() {
            let mut data = vec![SILENCE; PATCH_FRAMES * BANDS];
            for (frame, bands) in data
                .as_chunks_mut::<BANDS>()
                .0
                .iter_mut()
                .take(frames)
                .enumerate()
            {
                self.frame_into(audio, frame, bands, &mut fft);
            }
            return Patches { data, count: 1 };
        }

        let mut data = vec![0.0; starts.len() * PATCH_FRAMES * BANDS];
        for (patch, &start) in data
            .as_chunks_mut::<{ PATCH_FRAMES * BANDS }>()
            .0
            .iter_mut()
            .zip(&starts)
        {
            for (offset, bands) in patch.as_chunks_mut::<BANDS>().0.iter_mut().enumerate() {
                self.frame_into(audio, start + offset, bands, &mut fft);
            }
        }
        Patches {
            data,
            count: starts.len(),
        }
    }

    fn fft_buffers(&self) -> FftBuffers {
        FftBuffers {
            spectrum: vec![Complex::default(); FRAME],
            scratch: vec![Complex::default(); self.fft.get_inplace_scratch_len()],
        }
    }

    /// Normalized log-mel bands of `frame` (centered at `frame * HOP`, zero padded).
    fn frame_into(&self, audio: &[f32], frame: usize, bands: &mut [f32], fft: &mut FftBuffers) {
        let first_sample = (frame * HOP) as isize - (FRAME / 2) as isize;
        for (i, (bin, w)) in fft.spectrum.iter_mut().zip(&self.window).enumerate() {
            let sample = usize::try_from(first_sample + i as isize)
                .ok()
                .and_then(|s| audio.get(s))
                .copied()
                .unwrap_or(0.0);
            *bin = Complex::new(sample * w, 0.0);
        }
        self.fft
            .process_with_scratch(&mut fft.spectrum, &mut fft.scratch);

        for (value, band) in bands.iter_mut().zip(&self.bands) {
            let energy: f32 = fft.spectrum[band.first_bin..]
                .iter()
                .zip(&band.weights)
                .map(|(bin, w)| bin.norm_sqr() * w)
                .sum();
            *value = normalize((1.0 + 10000.0 * energy).log10());
        }
    }
}

/// Decodes to f32 16 kHz mono.
fn decode_samples(file: Vec<u8>) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i", "pipe:0", "-vn"])
        .args(["-rematrix_maxval", "1.0", "-ac", "1", "-ar"])
        .arg(SAMPLE_RATE.to_string())
        .args(["-f", "f32le", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting ffmpeg")?;

    let mut stdin = child.stdin.take().unwrap();
    let writer = thread::spawn(move || stdin.write_all(&file));
    let out = child.wait_with_output()?;
    let _ = writer.join();

    if !out.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let samples: Vec<f32> = out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    if samples.is_empty() {
        bail!("ffmpeg produced no audio");
    }
    Ok(samples)
}

fn patch_starts(frames: usize) -> Vec<usize> {
    let starts: Vec<usize> = (0..)
        .map(|p| p * PATCH_HOP)
        .take_while(|start| start + PATCH_FRAMES <= frames)
        .collect();
    if starts.len() <= MAX_PATCHES {
        return starts;
    }
    let n = starts.len();
    (0..MAX_PATCHES)
        .map(|i| starts[i * (n - 1) / (MAX_PATCHES - 1)])
        .collect()
}

/// Essentia's default Hann window.
fn hann_window() -> Vec<f32> {
    (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME - 1) as f32).cos())
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
fn slaney_filterbank() -> Vec<MelBand> {
    let high = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let edges: Vec<f64> = (0..BANDS + 2)
        .map(|i| mel_to_hz(high * i as f64 / (BANDS + 1) as f64))
        .collect();
    let bin_hz = (SAMPLE_RATE as f64 / 2.0) / (FRAME / 2) as f64;

    edges
        .windows(3)
        .map(|edges| {
            let (low, peak, high) = (edges[0], edges[1], edges[2]);
            let first_bin = (low / bin_hz).ceil() as usize;
            let last_bin = ((high / bin_hz).floor() as usize).min(FRAME / 2);
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
            MelBand { first_bin, weights }
        })
        .collect()
}
