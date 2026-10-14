/// The distance between the most different tracks. Distances can reach 1 (opposite vectors), but
/// embeddings centered on their library's mean are never that far apart.
const MOST_DIFFERENT: f32 = 0.75;

/// Distance of two normalized vectors from their cosine.
pub fn distance(cos: f32) -> f32 {
    (1.0 - cos) / 2.0
}

/// Similarity of two normalized vectors from their distance.
pub fn similarity(distance: f32) -> f32 {
    (1.0 - distance / MOST_DIFFERENT).clamp(0.0, 1.0)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scales `v` to unit length.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
}

/// Distance between two raw embeddings (as the model outputs them), relative to their length.
/// Unlike normalized vectors, it doesn't depend on the library.
pub fn raw_distance(a: &[f32], b: &[f32]) -> f32 {
    let apart: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    apart.sqrt() / (dot(a, a).sqrt() * dot(b, b).sqrt()).sqrt()
}
