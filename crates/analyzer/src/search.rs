//! Writes the search index (see `common::search_index`) from the vector index.

use crate::index::rows_per_thread;
use anyhow::Result;
use common::{
    db::{
        IDS_PER_QUERY,
        schema::{tracks, vector_data, vectors},
        types::{Embedding, Neighbors},
    },
    search_index::{self, NodeRecord},
};
use diesel::{prelude::*, sqlite::SqliteConnection};
use ndarray::{Array1, Array2, ArrayView2, Axis, linalg::general_mat_mul};
use std::{collections::HashMap, thread, time::Instant};
use tracing::info;

/// Components kept per vector.
const DIMS: usize = 128;
/// Neighbors kept per track.
const FANOUT: usize = 16;
/// Vectors the principal components are estimated from.
const PCA_SAMPLE: usize = 10_000;
const PCA_ITERATIONS: usize = 30;
/// Vectors the main direction of a half is estimated from.
const DIRECTION_SAMPLE: usize = 8192;
const DIRECTION_ITERATIONS: usize = 6;
/// Tracks below which halving stops (their order doesn't matter).
const LEAF: usize = 32;
/// Rows per query when streaming the vectors.
const PAGE: i64 = 1024;

pub fn write(conn: &mut SqliteConnection) -> Result<()> {
    let started = Instant::now();
    let basis = Basis::estimate(sample(conn)?.view());
    let tracks = encode_tracks(conn, &basis)?;
    let order = locality_order(tracks.decoded(&basis.scales).view());
    let durations = durations(conn)?;

    // Nodes are numbered in that order, and links refer to their numbers.
    let node_of: HashMap<i32, u32> = (0..)
        .zip(&order)
        .map(|(n, &t)| (tracks.idx[t], n))
        .collect();
    let links: Vec<Vec<u32>> = order
        .iter()
        .map(|&t| {
            tracks.links[t]
                .iter()
                .filter_map(|i| node_of.get(i).copied())
                .collect()
        })
        .collect();
    let nodes: Vec<NodeRecord> = order
        .iter()
        .zip(&links)
        .map(|(&t, links)| NodeRecord {
            idx: tracks.idx[t],
            duration: durations.get(&tracks.idx[t]).copied().unwrap_or(0),
            links,
            code: tracks.code(t),
        })
        .collect();
    search_index::write(&search_index::path(), &basis.scales, FANOUT, &nodes)?;
    info!(
        tracks = nodes.len(),
        secs = started.elapsed().as_secs_f32(),
        "wrote the search index"
    );
    Ok(())
}

fn durations(conn: &mut SqliteConnection) -> Result<HashMap<i32, i32>> {
    Ok(tracks::table
        .inner_join(vectors::table.on(vectors::track_id.eq(tracks::id)))
        .select((vectors::idx, tracks::duration))
        .load::<(i32, i32)>(conn)?
        .into_iter()
        .collect())
}

fn sample(conn: &mut SqliteConnection) -> Result<Array2<f32>> {
    let indexed: Vec<i32> = vectors::table
        .filter(vectors::neighbors.is_not_null())
        .order(vectors::idx)
        .select(vectors::idx)
        .load(conn)?;
    let picked: Vec<i32> = indexed
        .iter()
        .copied()
        .step_by(indexed.len().div_ceil(PCA_SAMPLE).max(1))
        .collect();
    let mut rows = Vec::new();
    for ids in picked.chunks(IDS_PER_QUERY) {
        let vectors: Vec<Embedding> = vector_data::table
            .filter(vector_data::idx.eq_any(ids))
            .select(vector_data::vec)
            .load(conn)?;
        rows.extend(vectors.into_iter().flat_map(|Embedding(v)| v));
    }
    let dim = rows.len().checked_div(picked.len()).unwrap_or(0);
    Ok(Array2::from_shape_vec((picked.len(), dim), rows)?)
}

/// The top DIMS principal components.
struct Basis {
    /// Orthonormal, as columns.
    components: Array2<f32>,
    ///  Per component, the size of one int8 step.
    scales: Vec<f32>,
}

impl Basis {
    /// By subspace iteration on the sample's covariance, which is just sampleᵀ·sample since the
    /// vectors are already centered.
    fn estimate(sample: ArrayView2<f32>) -> Self {
        let dim = sample.ncols();
        let covariance = product(sample.t(), sample);
        let mut components = Array2::zeros((dim, DIMS.min(dim)));

        components.diag_mut().fill(1.0);
        for _ in 0..PCA_ITERATIONS {
            components = orthonormalized(product(covariance.view(), components.view()));
        }

        let projections = sample.dot(&components);
        let scales = projections
            .columns()
            .into_iter()
            .map(|c| (c.fold(0.0_f32, |max, &x| max.max(x.abs())) / 127.0).max(f32::MIN_POSITIVE))
            .collect();

        Self { components, scales }
    }

    /// The codes of `vectors` (rows), row-major.
    fn encode(&self, vectors: ArrayView2<f32>) -> Vec<i8> {
        let projections = vectors.dot(&self.components);
        projections
            .rows()
            .into_iter()
            .flat_map(|row| row.into_iter().zip(&self.scales))
            .map(|(&projection, &scale)| search_index::encode(projection, scale))
            .collect()
    }
}

/// Makes the columns orthonormal without changing the space they span (modified Gram-Schmidt).
/// A column that adds no new direction becomes zero.
fn orthonormalized(m: Array2<f32>) -> Array2<f32> {
    // Works on the transpose, so each column is a contiguous row.
    let mut rows = m.t().as_standard_layout().into_owned();
    for k in 0..rows.nrows() {
        let (previous, mut rest) = rows.view_mut().split_at(Axis(0), k);
        let mut row = rest.row_mut(0);
        for previous in previous.rows() {
            let overlap = row.dot(&previous);
            row.scaled_add(-overlap, &previous);
        }
        let norm = row.dot(&row).sqrt();
        let scale = if norm > 1e-6 { 1.0 / norm } else { 0.0 };
        row.mapv_inplace(|x| x * scale);
    }
    rows.t().as_standard_layout().into_owned()
}

fn product(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let mut out = Array2::zeros((a.nrows(), b.ncols()));
    let rows = rows_per_thread(a.nrows());
    thread::scope(|s| {
        for (a, mut out) in a
            .axis_chunks_iter(Axis(0), rows)
            .zip(out.axis_chunks_iter_mut(Axis(0), rows))
        {
            s.spawn(move || general_mat_mul(1.0, &a, &b, 0.0, &mut out));
        }
    });
    out
}

#[derive(Default)]
struct Encoded {
    idx: Vec<i32>,
    links: Vec<Vec<i32>>,
    /// Row-major.
    codes: Vec<i8>,
}

impl Encoded {
    fn code(&self, row: usize) -> &[i8] {
        &self.codes[row * DIMS..(row + 1) * DIMS]
    }

    fn decoded(&self, scales: &[f32]) -> Array2<f32> {
        let mut vectors = Vec::with_capacity(self.codes.len());
        for code in self.codes.as_chunks::<DIMS>().0 {
            vectors.extend(search_index::decode(*code, scales));
        }
        Array2::from_shape_vec((self.idx.len(), DIMS), vectors).unwrap()
    }
}

fn encode_tracks(conn: &mut SqliteConnection, basis: &Basis) -> Result<Encoded> {
    let mut encoded = Encoded::default();
    let mut after = 0;
    loop {
        let page: Vec<(i32, Neighbors, Embedding)> = vectors::table
            .inner_join(vector_data::table)
            .filter(vectors::idx.gt(after))
            .filter(vectors::neighbors.is_not_null())
            .order(vectors::idx)
            .limit(PAGE)
            .select((
                vectors::idx,
                vectors::neighbors.assume_not_null(),
                vector_data::vec,
            ))
            .load(conn)?;
        let Some(&(last, ..)) = page.last() else {
            return Ok(encoded);
        };
        after = last;
        let rows: Vec<f32> = page
            .iter()
            .flat_map(|(.., v)| v.0.iter().copied())
            .collect();
        let vectors = ArrayView2::from_shape((page.len(), rows.len() / page.len()), &rows)?;
        encoded.codes.extend(basis.encode(vectors));
        for (idx, Neighbors(list), _) in page {
            encoded.idx.push(idx);
            encoded
                .links
                .push(list.iter().take(FANOUT).map(|n| n.idx).collect());
        }
    }
}

/// Row indices of `vectors`, ordered so that similar rows end up close together.
fn locality_order(vectors: ArrayView2<f32>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..vectors.nrows()).collect();
    halve(vectors, &mut order);
    order
}

/// Splits `rows` in half along the direction in which they vary most, then splits each half the
/// same way, until groups are down to LEAF rows.
fn halve(vectors: ArrayView2<f32>, rows: &mut [usize]) {
    if rows.len() <= LEAF {
        return;
    }
    let direction = main_direction(vectors, rows);
    let mut keyed: Vec<(f32, usize)> = rows
        .iter()
        .map(|&row| (vectors.row(row).dot(&direction), row))
        .collect();
    let middle = keyed.len() / 2;
    keyed.select_nth_unstable_by(middle, |a, b| a.0.total_cmp(&b.0));
    for (slot, (_, row)) in rows.iter_mut().zip(keyed) {
        *slot = row;
    }
    let (low, high) = rows.split_at_mut(middle);
    halve(vectors, low);
    halve(vectors, high);
}

/// The direction in which the `rows` of `vectors` vary most, estimated by power iteration on at
/// most DIRECTION_SAMPLE of them.
fn main_direction(vectors: ArrayView2<f32>, rows: &[usize]) -> Array1<f32> {
    let picked: Vec<usize> = rows
        .iter()
        .copied()
        .step_by(rows.len().div_ceil(DIRECTION_SAMPLE))
        .collect();
    let sample = vectors.select(Axis(0), &picked);
    let centered = &sample - &sample.mean_axis(Axis(0)).unwrap();
    let mut direction = centered.row(0).to_owned();
    for _ in 0..DIRECTION_ITERATIONS {
        direction = centered.t().dot(&centered.dot(&direction));
        let norm = direction.dot(&direction).sqrt();
        if norm == 0.0 {
            break;
        }
        direction /= norm;
    }
    direction
}
