use anyhow::Result;
use common::db::{
    IDS_PER_QUERY,
    schema::{albums, analyses, libraries, near_identical_pairs, track_artists, tracks, vectors},
    types::Neighbors,
};
use diesel::{dsl, prelude::*, sqlite::SqliteConnection};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Serialize)]
pub struct Track {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub album: String,
    #[serde(skip)]
    pub duration: i32,
}

pub enum Missing {
    Unknown,
    Excluded,
    NotAnalyzed,
    Failed(String),
    NotIndexed,
}

pub fn missing(conn: &mut SqliteConnection, id: &str) -> Result<Missing> {
    let excluded: Option<bool> = tracks::table
        .inner_join(libraries::table)
        .filter(tracks::id.eq(id))
        .select(libraries::excluded)
        .first(conn)
        .optional()?;
    match excluded {
        None => return Ok(Missing::Unknown),
        Some(true) => return Ok(Missing::Excluded),
        Some(false) => {}
    }
    let error: Option<Option<String>> = analyses::table
        .find(id)
        .select(analyses::error)
        .first(conn)
        .optional()?;
    Ok(match error {
        None => Missing::NotAnalyzed,
        Some(Some(error)) => Missing::Failed(error),
        Some(None) => Missing::NotIndexed,
    })
}

/// Vectors joined with their track, skipping those in excluded libraries.
#[dsl::auto_type(no_type_alias)]
fn included_vectors() -> _ {
    vectors::table
        .inner_join(tracks::table.on(tracks::id.eq(vectors::track_id)))
        .inner_join(libraries::table.on(libraries::id.eq(tracks::library_id)))
        .filter(libraries::excluded.eq(false))
}

/// The idx and neighbor list of track `id`, or None if it isn't indexed (or its library is
/// excluded). The list can reference vectors removed since, or in excluded libraries.
pub fn neighbors(conn: &mut SqliteConnection, id: &str) -> Result<Option<(i32, Neighbors)>> {
    Ok(included_vectors()
        .filter(vectors::track_id.eq(id))
        .filter(vectors::neighbors.is_not_null())
        .select((vectors::idx, vectors::neighbors.assume_not_null()))
        .first(conn)
        .optional()?)
}

/// The vector idx of track `id`, if it has a vector and its library isn't excluded.
pub fn idx(conn: &mut SqliteConnection, id: &str) -> Result<Option<i32>> {
    Ok(included_vectors()
        .filter(vectors::track_id.eq(id))
        .select(vectors::idx)
        .first(conn)
        .optional()?)
}

/// The tracks of the vectors `idx`. Missing idx (removed since) and tracks of excluded libraries
/// are left out.
pub fn tracks_by_idx(conn: &mut SqliteConnection, idx: &[i32]) -> Result<HashMap<i32, Track>> {
    let mut out = HashMap::with_capacity(idx.len());
    for ids in idx.chunks(IDS_PER_QUERY) {
        let rows: Vec<(i32, String, String, String, String, i32)> = included_vectors()
            .inner_join(albums::table.on(albums::id.eq(tracks::album_id)))
            .filter(vectors::idx.eq_any(ids))
            .select((
                vectors::idx,
                tracks::id,
                tracks::title,
                tracks::artist,
                albums::name,
                tracks::duration,
            ))
            .load(conn)?;
        for (idx, id, title, artist, album, duration) in rows {
            out.insert(
                idx,
                Track {
                    id,
                    title,
                    artist,
                    album,
                    duration,
                },
            );
        }
    }
    Ok(out)
}

/// The raw distances of the near-identical pairs among the vectors `idx`, by (lower idx, higher
/// idx).
pub fn near_identical_pairs(
    conn: &mut SqliteConnection,
    idx: &[i32],
) -> Result<HashMap<(i32, i32), f32>> {
    let wanted: HashSet<i32> = idx.iter().copied().collect();
    let mut pairs = HashMap::new();
    for ids in idx.chunks(IDS_PER_QUERY) {
        let rows: Vec<(i32, i32, f32)> = near_identical_pairs::table
            .filter(near_identical_pairs::a.eq_any(ids))
            .select((
                near_identical_pairs::a,
                near_identical_pairs::b,
                near_identical_pairs::raw_distance,
            ))
            .load(conn)?;
        pairs.extend(
            rows.into_iter()
                .filter(|(_, b, _)| wanted.contains(b))
                .map(|(a, b, distance)| ((a, b), distance)),
        );
    }
    Ok(pairs)
}

/// The artist ids of the tracks of the vectors `idx` that have any.
pub fn artist_ids(conn: &mut SqliteConnection, idx: &[i32]) -> Result<HashMap<i32, Vec<String>>> {
    let mut artists: HashMap<i32, Vec<String>> = HashMap::new();
    for ids in idx.chunks(IDS_PER_QUERY) {
        let rows: Vec<(i32, String)> = vectors::table
            .inner_join(track_artists::table.on(track_artists::track_id.eq(vectors::track_id)))
            .filter(vectors::idx.eq_any(ids))
            .select((vectors::idx, track_artists::artist_id))
            .load(conn)?;
        for (idx, artist) in rows {
            artists.entry(idx).or_default().push(artist);
        }
    }
    Ok(artists)
}
