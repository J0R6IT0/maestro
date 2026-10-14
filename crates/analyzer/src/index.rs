//! Exact K-nearest-neighbor lists, precomputed so that similarity queries are just lookups.
//!
//! Embeddings are stored in `vector_data`, centered on a frozen library mean and L2-normalized.
//! Neighbor lists live separately in `vectors` so that updating the list doesn't have to rewrite
//! the whole row.
//!
//! New vectors are indexed CHUNK at a time. Each chunk is compared against every indexed vector,
//! BLOCK rows at a time. That gives the chunk its own top K and updates any existing list that a
//! new vector now belongs in. Memory should stay at a few hundred MB regardless of library size.
//! O(N² · d) spread over the passes.
//!
//! When a vector is deleted, a trigger records it in `removed_vectors`. Every list that pointed
//! to it is recomputed.
//!
//! Each new list is then checked for neighbors sitting almost exactly on its track.
//! Their raw distance (see `common::vector::raw_distance`) goes into `near_identical_pairs`,
//! which the server uses to tell which tracks have the same audio.
//!
//! Any update that changes the index also rebuilds the search index (see `search.rs`).

use crate::search;
use anyhow::Result;
use common::{
    db::{
        IDS_PER_QUERY, NEIGHBORS_PER_TRACK,
        models::{IndexMeta, NewVector, NewVectorData},
        raw_embeddings,
        schema::{
            analyses, embeddings_changed, index_meta, index_queue, near_identical_pairs,
            near_identical_pending, removed_vectors, stale_vectors, vector_data, vectors,
        },
        types::{Embedding, Neighbor, Neighbors},
    },
    search_index,
    vector::{distance, raw_distance},
};
use diesel::{dsl::exists, prelude::*, sqlite::SqliteConnection};
use ndarray::{ArrayView2, Axis};
use std::{collections::HashSet, thread, time::Instant};
use tracing::{info, info_span};

const K: usize = NEIGHBORS_PER_TRACK;
const CHUNK: usize = 4096;
const BLOCK: usize = 4096;
/// Rows per query when streaming embeddings or lists.
const PAGE: i64 = 1024;
/// How far the library mean may drift from the frozen one before the index is rebuilt around the
/// new mean, as a fraction of how spread out the embeddings are (their RMS distance from the mean).
const MAX_DRIFT: f64 = 0.05;
/// Neighbors closer than this in the index are stored as near-identical pairs.
const NEAR_IDENTICAL: f32 = 0.005;

pub fn update(conn: &mut SqliteConnection) -> Result<()> {
    update_with(conn, false)
}

pub fn update_end_of_pass(conn: &mut SqliteConnection) -> Result<()> {
    update_with(conn, true)
}

fn update_with(conn: &mut SqliteConnection, always_check_drift: bool) -> Result<()> {
    let _span = info_span!("index").entered();
    let tracks = tracks_with_embedding(conn)?;
    if tracks == 0 {
        return Ok(());
    }
    let started = Instant::now();
    let meta = prepare(conn, tracks, always_check_drift)?;
    copy_embeddings(conn, meta)?;
    let removed = mark_stale(conn)?;
    let refreshed = refresh_stale(conn, started)?;
    let indexed = index_pending(conn, started)?;

    let near_identical_pairs = find_near_identical_pairs(conn)?;
    if near_identical_pairs > 0 {
        info!(
            pairs = near_identical_pairs,
            "found nearly identical tracks to check for duplicates"
        );
    }

    if removed + refreshed + indexed > 0 || !search_index::is_current(&search_index::path()) {
        search::write(conn)?;
    }
    Ok(())
}

fn tracks_with_embedding(conn: &mut SqliteConnection) -> Result<usize> {
    let count: i64 = analyses::table
        .filter(analyses::embedding.is_not_null())
        .count()
        .get_result(conn)?;
    Ok(count as usize)
}

/// Loads the index's metadata, first rebuilding the index if it doesn't exist yet, was built for a
/// different K, or the library mean has drifted too far.
fn prepare(
    conn: &mut SqliteConnection,
    tracks: usize,
    always_check_drift: bool,
) -> Result<IndexMeta> {
    let stored: Option<IndexMeta> = index_meta::table
        .select(IndexMeta::as_select())
        .first(conn)
        .optional()?;
    let Some(mut meta) = stored.filter(|meta| meta.k as usize == K) else {
        info!(tracks, "building the index");
        let (mean, _) = library_stats(conn)?;
        return rebuild(conn, mean, tracks);
    };

    let rebuild_interrupted = meta.rebuild_after.is_some();
    let check_due = always_check_drift || tracks >= 2 * meta.track_count as usize;

    if rebuild_interrupted || !check_due || !embeddings_changed(conn)? {
        return Ok(meta);
    }

    let (mean, spread) = library_stats(conn)?;
    let drift = drift(&mean.0, &meta.mean.0, spread);
    if drift > MAX_DRIFT {
        info!(drift, tracks, "library mean drifted, rebuilding the index");
        return rebuild(conn, mean, tracks);
    }
    info!(drift, tracks, "library mean checked, no rebuild needed");
    meta.track_count = tracks as i32;
    save_meta(conn, &meta)?;
    Ok(meta)
}

/// Clears the index and freezes `mean` as the new center, so every track gets re-indexed around it.
fn rebuild(conn: &mut SqliteConnection, mean: Embedding, tracks: usize) -> Result<IndexMeta> {
    let meta = IndexMeta {
        id: IndexMeta::ID,
        mean,
        track_count: tracks as i32,
        k: K as i32,
        rebuild_after: Some(String::new()),
    };
    conn.immediate_transaction(|conn| {
        diesel::delete(vectors::table).execute(conn)?;
        diesel::delete(index_queue::table).execute(conn)?;
        diesel::delete(removed_vectors::table).execute(conn)?;
        diesel::delete(stale_vectors::table).execute(conn)?;
        save_meta(conn, &meta)
    })?;
    Ok(meta)
}

fn save_meta(conn: &mut SqliteConnection, meta: &IndexMeta) -> Result<()> {
    diesel::insert_into(index_meta::table)
        .values(meta)
        .on_conflict(index_meta::id)
        .do_update()
        .set(meta)
        .execute(conn)?;
    Ok(())
}

fn embeddings_after(conn: &mut SqliteConnection, after: &str) -> Result<Vec<(String, Embedding)>> {
    Ok(analyses::table
        .filter(analyses::track_id.gt(after))
        .filter(analyses::embedding.is_not_null())
        .order(analyses::track_id)
        .limit(PAGE)
        .select((analyses::track_id, analyses::embedding.assume_not_null()))
        .load(conn)?)
}

fn embeddings_changed(conn: &mut SqliteConnection) -> Result<bool> {
    Ok(diesel::select(exists(
        embeddings_changed::table.select(embeddings_changed::id),
    ))
    .get_result(conn)?)
}

/// The library mean and how spread out the embeddings are around it (their RMS distance from it).
fn library_stats(conn: &mut SqliteConnection) -> Result<(Embedding, f64)> {
    diesel::delete(embeddings_changed::table).execute(conn)?;
    let mut sum: Vec<f64> = Vec::new();
    let (mut squares, mut n) = (0f64, 0usize);
    let mut after = String::new();
    loop {
        let page = embeddings_after(conn, &after)?;
        let Some((last, _)) = page.last() else { break };
        after = last.clone();
        for (_, Embedding(embedding)) in &page {
            sum.resize(embedding.len(), 0.0);
            for (s, &v) in sum.iter_mut().zip(embedding) {
                *s += v as f64;
                squares += (v as f64).powi(2);
            }
            n += 1;
        }
    }
    let mean: Vec<f64> = sum.iter().map(|s| s / n.max(1) as f64).collect();
    let spread = (squares / n.max(1) as f64 - mean.iter().map(|m| m * m).sum::<f64>())
        .max(0.0)
        .sqrt();
    Ok((Embedding(mean.iter().map(|&m| m as f32).collect()), spread))
}

/// How far the mean moved, relative to the spread.
fn drift(mean: &[f32], frozen: &[f32], spread: f64) -> f64 {
    let moved = mean
        .iter()
        .zip(frozen)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if moved == 0.0 {
        0.0
    } else {
        moved / spread.max(1e-12)
    }
}

/// Adds normalized embeddings to `vectors`.
fn copy_embeddings(conn: &mut SqliteConnection, mut meta: IndexMeta) -> Result<()> {
    while let Some(after) = &meta.rebuild_after {
        let page = embeddings_after(conn, after)?;
        meta.rebuild_after = page.last().map(|(id, _)| id.clone());
        conn.immediate_transaction(|conn| {
            for (id, mut embedding) in page {
                normalize(&mut embedding.0, &meta.mean.0);
                insert_vector(conn, &id, &embedding)?;
            }
            save_meta(conn, &meta)
        })?;
    }
    loop {
        let embedding = analyses::table
            .filter(analyses::track_id.eq(index_queue::track_id))
            .select(analyses::embedding)
            .single_value();
        let page: Vec<(String, Option<Embedding>)> = index_queue::table
            .limit(PAGE)
            .select((index_queue::track_id, embedding))
            .load(conn)?;
        if page.is_empty() {
            return Ok(());
        }
        conn.immediate_transaction(|conn| -> Result<()> {
            for (id, embedding) in page {
                if let Some(mut embedding) = embedding {
                    normalize(&mut embedding.0, &meta.mean.0);
                    insert_vector(conn, &id, &embedding)?;
                }
                diesel::delete(index_queue::table.find(&id)).execute(conn)?;
            }
            Ok(())
        })?;
    }
}

fn insert_vector(conn: &mut SqliteConnection, track_id: &str, vec: &Embedding) -> Result<()> {
    let idx: Option<i32> = diesel::insert_into(vectors::table)
        .values(NewVector { track_id })
        .on_conflict_do_nothing()
        .returning(vectors::idx)
        .get_result(conn)
        .optional()?;
    if let Some(idx) = idx {
        diesel::insert_into(vector_data::table)
            .values(NewVectorData { idx, vec })
            .execute(conn)?;
    }
    Ok(())
}

/// Centers on the library mean and L2-normalizes. Raw transformer embeddings all point roughly the
/// same way (any two score around 0.9 by cosine), so without centering every distance would look alike.
fn normalize(v: &mut [f32], mean: &[f32]) {
    v.iter_mut().zip(mean).for_each(|(v, m)| *v -= m);
    common::vector::normalize(v);
}

fn mark_stale(conn: &mut SqliteConnection) -> Result<usize> {
    let removed: Vec<i32> = removed_vectors::table
        .select(removed_vectors::idx)
        .load(conn)?;
    if removed.is_empty() {
        return Ok(0);
    }
    let removed_set: HashSet<i32> = removed.iter().copied().collect();
    let mut after = 0;
    loop {
        let page: Vec<(i32, Neighbors)> = vectors::table
            .filter(vectors::idx.gt(after))
            .filter(vectors::neighbors.is_not_null())
            .order(vectors::idx)
            .limit(PAGE)
            .select((vectors::idx, vectors::neighbors.assume_not_null()))
            .load(conn)?;
        let Some(&(last, _)) = page.last() else { break };
        after = last;
        let stale: Vec<i32> = page
            .iter()
            .filter(|(_, list)| list.0.iter().any(|n| removed_set.contains(&n.idx)))
            .map(|&(idx, _)| idx)
            .collect();
        if !stale.is_empty() {
            conn.immediate_transaction(|conn| -> Result<()> {
                for idx in stale {
                    diesel::insert_into(stale_vectors::table)
                        .values(stale_vectors::idx.eq(idx))
                        .on_conflict_do_nothing()
                        .execute(conn)?;
                }
                Ok(())
            })?;
        }
    }

    conn.immediate_transaction(|conn| -> Result<()> {
        for ids in removed.chunks(IDS_PER_QUERY) {
            diesel::delete(removed_vectors::table.filter(removed_vectors::idx.eq_any(ids)))
                .execute(conn)?;
        }
        Ok(())
    })?;

    Ok(removed.len())
}

fn refresh_stale(conn: &mut SqliteConnection, started: Instant) -> Result<usize> {
    let mut refreshed = 0;
    loop {
        let chunk = stale_chunk(conn)?;
        if chunk.is_empty() {
            return Ok(refreshed);
        }
        index_chunk(conn, &chunk, Mode::Refresh)?;
        refreshed += chunk.len();
        info!(
            refreshed,
            secs = started.elapsed().as_secs_f32(),
            "refreshed neighbor lists that pointed to removed tracks"
        );
    }
}

fn index_pending(conn: &mut SqliteConnection, started: Instant) -> Result<usize> {
    let (mut indexed, mut updated) = (0, 0);
    loop {
        let chunk = pending_chunk(conn)?;
        if chunk.is_empty() {
            return Ok(indexed);
        }
        updated += index_chunk(conn, &chunk, Mode::Index)?;
        indexed += chunk.len();
        info!(
            indexed,
            updated,
            secs = started.elapsed().as_secs_f32(),
            "indexed new tracks and updated their neighbors' lists"
        );
    }
}

struct VectorRows<Key> {
    keys: Vec<Key>,
    data: Vec<f32>,
}

/// Vectors to index, by idx.
type Chunk = VectorRows<i32>;
/// Vectors to scan, by idx with their list if indexed.
type Block = VectorRows<(i32, Option<Neighbors>)>;

impl<Key> VectorRows<Key> {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn matrix(&self, dim: usize) -> Result<ArrayView2<'_, f32>> {
        Ok(ArrayView2::from_shape((self.len(), dim), &self.data)?)
    }
}

impl<Key> FromIterator<(Key, Embedding)> for VectorRows<Key> {
    fn from_iter<I: IntoIterator<Item = (Key, Embedding)>>(rows: I) -> Self {
        let rows = rows.into_iter();
        let mut out = Self {
            keys: Vec::with_capacity(rows.size_hint().0),
            data: Vec::new(),
        };
        for (key, Embedding(vector)) in rows {
            out.keys.push(key);
            out.data.extend(vector);
        }
        out
    }
}

fn pending_chunk(conn: &mut SqliteConnection) -> Result<Chunk> {
    Ok(vectors::table
        .inner_join(vector_data::table)
        .filter(vectors::neighbors.is_null())
        .order(vectors::idx)
        .limit(CHUNK as i64)
        .select((vectors::idx, vector_data::vec))
        .load::<(i32, Embedding)>(conn)?
        .into_iter()
        .collect())
}

fn stale_chunk(conn: &mut SqliteConnection) -> Result<Chunk> {
    Ok(stale_vectors::table
        .inner_join(vector_data::table.on(vector_data::idx.eq(stale_vectors::idx)))
        .order(stale_vectors::idx)
        .limit(CHUNK as i64)
        .select((stale_vectors::idx, vector_data::vec))
        .load::<(i32, Embedding)>(conn)?
        .into_iter()
        .collect())
}

fn block_between(conn: &mut SqliteConnection, after: i32, until: i32) -> Result<Block> {
    Ok(vectors::table
        .inner_join(vector_data::table)
        .filter(vectors::idx.gt(after))
        .filter(vectors::idx.le(until))
        .order(vectors::idx)
        .limit(BLOCK as i64)
        .select((vectors::idx, vectors::neighbors, vector_data::vec))
        .load::<(i32, Option<Neighbors>, Embedding)>(conn)?
        .into_iter()
        .map(|(idx, list, vector)| ((idx, list), vector))
        .collect())
}

fn set_neighbors(conn: &mut SqliteConnection, idx: i32, list: &Neighbors) -> Result<()> {
    diesel::update(vectors::table.find(idx))
        .set(vectors::neighbors.eq(list))
        .execute(conn)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Index,
    Refresh,
}

/// Computes the neighbor lists of the chunk's vectors. Returns how many existing lists changed.
///
/// Index: also adds each chunk vector to any existing list it now belongs in.
/// Refresh: scans the whole library but leaves other lists alone.
fn index_chunk(conn: &mut SqliteConnection, chunk: &Chunk, mode: Mode) -> Result<usize> {
    let mut lists = ChunkLists::new(chunk, mode)?;
    let until = match mode {
        Mode::Index => *chunk.keys.last().unwrap(),
        Mode::Refresh => i32::MAX,
    };
    let (mut after, mut updated) = (0, 0);
    loop {
        let block = block_between(conn, after, until)?;
        let Some(&(last, _)) = block.keys.last() else {
            break;
        };
        after = last;
        let newcomers = lists.add_block(&block)?;
        let changed = merge_newcomers(block, newcomers);
        if changed.is_empty() {
            continue;
        }
        updated += changed.len();
        conn.immediate_transaction(|conn| -> Result<()> {
            for (idx, list) in &changed {
                set_neighbors(conn, *idx, list)?;
            }
            Ok(())
        })?;
    }
    conn.immediate_transaction(|conn| -> Result<()> {
        for (idx, list) in lists.into_sorted() {
            set_neighbors(conn, idx, &list)?;
            match mode {
                Mode::Index => {
                    diesel::insert_into(near_identical_pending::table)
                        .values(near_identical_pending::idx.eq(idx))
                        .on_conflict_do_nothing()
                        .execute(conn)?;
                }
                Mode::Refresh => {
                    diesel::delete(stale_vectors::table.find(idx)).execute(conn)?;
                }
            }
        }
        Ok(())
    })?;
    Ok(updated)
}

struct ChunkLists<'a> {
    mode: Mode,
    ids: &'a [i32],
    vectors: ArrayView2<'a, f32>,
    lists: Vec<TopK>,
    rows_per_thread: usize,
}

impl<'a> ChunkLists<'a> {
    fn new(chunk: &'a Chunk, mode: Mode) -> Result<Self> {
        Ok(Self {
            mode,
            ids: &chunk.keys,
            vectors: chunk.matrix(chunk.data.len() / chunk.len())?,
            lists: chunk.keys.iter().map(|_| TopK::new()).collect(),
            rows_per_thread: rows_per_thread(chunk.len()),
        })
    }

    /// Adds the block's vectors to the chunk's lists. Returns, per block row, the chunk vectors
    /// that get into its list.
    fn add_block(&mut self, block: &Block) -> Result<Vec<Vec<Neighbor>>> {
        let scan = BlockScan {
            mode: self.mode,
            vectors: block.matrix(self.vectors.ncols())?,
            keys: &block.keys,
            thresholds: block
                .keys
                .iter()
                .map(|(_, list)| entry_threshold(list.as_ref(), self.mode))
                .collect(),
        };

        let rows = self.rows_per_thread;
        let candidates: Vec<(usize, Neighbor)> = thread::scope(|s| {
            let scan = &scan;
            let workers: Vec<_> = self
                .vectors
                .axis_chunks_iter(Axis(0), rows)
                .zip(self.ids.chunks(rows))
                .zip(self.lists.chunks_mut(rows))
                .map(|((queries, ids), lists)| s.spawn(move || scan.run(queries, ids, lists)))
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });

        let mut newcomers = vec![Vec::new(); block.len()];
        for (row, neighbor) in candidates {
            newcomers[row].push(neighbor);
        }
        Ok(newcomers)
    }

    fn into_sorted(self) -> impl Iterator<Item = (i32, Neighbors)> + 'a {
        self.ids
            .iter()
            .copied()
            .zip(self.lists.into_iter().map(TopK::into_sorted))
    }
}

/// A block of the library, shared by the threads. Each compares it against its slice of the chunk.
struct BlockScan<'a> {
    mode: Mode,
    vectors: ArrayView2<'a, f32>,
    keys: &'a [(i32, Option<Neighbors>)],
    /// Per block row (see `entry_threshold`).
    thresholds: Vec<f32>,
}

impl BlockScan<'_> {
    /// Adds the block's vectors to the lists of the chunk vectors `queries` (ids `ids`). Returns
    /// the queries that get into a block vector's list, with that vector's block row.
    fn run(
        &self,
        queries: ArrayView2<f32>,
        ids: &[i32],
        lists: &mut [TopK],
    ) -> Vec<(usize, Neighbor)> {
        let similarities = queries.dot(&self.vectors.t());
        let mut newcomers = Vec::new();
        for ((similarities, &query), list) in similarities.outer_iter().zip(ids).zip(lists) {
            for (row, (&similarity, &(idx, ref block_list))) in
                similarities.iter().zip(self.keys).enumerate()
            {
                let is_pending = block_list.is_none();
                if idx == query || (self.mode == Mode::Refresh && is_pending) {
                    continue;
                }
                let distance = distance(similarity);
                list.push(idx, distance);
                if distance < self.thresholds[row] {
                    newcomers.push((
                        row,
                        Neighbor {
                            idx: query,
                            distance,
                        },
                    ));
                }
            }
        }
        newcomers
    }
}

pub fn rows_per_thread(rows: usize) -> usize {
    let threads = thread::available_parallelism().map_or(4, |n| n.get());
    rows.div_ceil(threads).max(1)
}

/// How close a chunk vector must be to join a block vector's list, closer than its current K-th
/// neighbor.
fn entry_threshold(list: Option<&Neighbors>, mode: Mode) -> f32 {
    match (mode, list) {
        (Mode::Refresh, _) | (_, None) => f32::NEG_INFINITY,
        (Mode::Index, Some(list)) => list.0.get(K - 1).map_or(f32::INFINITY, |n| n.distance),
    }
}

fn merge_newcomers(block: Block, newcomers: Vec<Vec<Neighbor>>) -> Vec<(i32, Neighbors)> {
    block
        .keys
        .into_iter()
        .zip(newcomers)
        .filter_map(|((idx, list), newcomers)| {
            let Neighbors(mut list) = list?;
            let new: Vec<Neighbor> = newcomers
                .into_iter()
                .filter(|n| !list.iter().any(|l| l.idx == n.idx))
                .collect();
            if new.is_empty() {
                return None;
            }
            list.extend(new);
            list.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            list.truncate(K);
            Some((idx, Neighbors(list)))
        })
        .collect()
}

/// Collects the K smallest distances.
struct TopK {
    items: Vec<Neighbor>,
    cutoff: f32,
}

impl TopK {
    fn new() -> Self {
        Self {
            items: Vec::with_capacity(2 * K),
            cutoff: f32::INFINITY,
        }
    }

    fn push(&mut self, idx: i32, distance: f32) {
        if distance >= self.cutoff {
            return;
        }
        self.items.push(Neighbor { idx, distance });
        if self.items.len() == 2 * K {
            self.items
                .select_nth_unstable_by(K - 1, |a, b| a.distance.total_cmp(&b.distance));
            self.items.truncate(K);
            self.cutoff = self.items[K - 1].distance;
        }
    }

    fn into_sorted(mut self) -> Neighbors {
        self.items.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        self.items.truncate(K);
        Neighbors(self.items)
    }
}

/// Stores the near-identical pairs of the pending vectors' lists, with their raw distance. Returns
/// how many new pairs it stored.
fn find_near_identical_pairs(conn: &mut SqliteConnection) -> Result<usize> {
    let mut found = 0;
    loop {
        let page: Vec<(i32, Option<Neighbors>)> = near_identical_pending::table
            .inner_join(vectors::table)
            .order(near_identical_pending::idx)
            .limit(PAGE)
            .select((near_identical_pending::idx, vectors::neighbors))
            .load(conn)?;
        if page.is_empty() {
            return Ok(found);
        }
        let pairs: Vec<(i32, i32)> = page
            .iter()
            .filter_map(|(idx, list)| Some((*idx, list.as_ref()?)))
            .flat_map(|(idx, Neighbors(list))| {
                list.iter()
                    .take_while(|n| n.distance < NEAR_IDENTICAL)
                    .map(move |n| (idx.min(n.idx), idx.max(n.idx)))
            })
            .collect();
        let involved: Vec<i32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let embeddings = raw_embeddings(conn, &involved)?;
        conn.immediate_transaction(|conn| -> Result<()> {
            for (a, b) in pairs {
                let (Some(x), Some(y)) = (embeddings.get(&a), embeddings.get(&b)) else {
                    continue;
                };
                found += diesel::insert_into(near_identical_pairs::table)
                    .values((
                        near_identical_pairs::a.eq(a),
                        near_identical_pairs::b.eq(b),
                        near_identical_pairs::raw_distance.eq(raw_distance(x, y)),
                    ))
                    .on_conflict_do_nothing()
                    .execute(conn)?;
            }
            let done: Vec<i32> = page.iter().map(|&(idx, _)| idx).collect();
            diesel::delete(
                near_identical_pending::table.filter(near_identical_pending::idx.eq_any(done)),
            )
            .execute(conn)?;
            Ok(())
        })?;
    }
}
