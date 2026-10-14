use crate::provider;
use anyhow::Result;
use common::db::{
    IDS_PER_QUERY,
    models::{Album, Artist, Library, Track, TrackArtist},
    schema::{albums, analyses, artists, libraries, track_artists, tracks},
};
use diesel::{
    dsl::{exists, not},
    prelude::*,
    sqlite::SqliteConnection,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use tracing::{info, warn};

pub fn libraries(
    conn: &mut SqliteConnection,
    provided: &[provider::Library],
    excluded: &[String],
) -> Result<HashSet<String>> {
    for unknown in excluded
        .iter()
        .filter(|&name| !provided.iter().any(|l| is_named(l, name)))
    {
        warn!(library = %unknown, "EXCLUDED_LIBRARIES names a library the provider doesn't have");
    }
    let excluded_ids: HashSet<String> = provided
        .iter()
        .filter(|library| excluded.iter().any(|name| is_named(library, name)))
        .map(|library| library.id.clone())
        .collect();

    conn.immediate_transaction(|conn| -> Result<()> {
        for library in provided {
            let row = Library {
                id: library.id.clone(),
                name: library.name.clone(),
                excluded: excluded_ids.contains(&library.id),
            };
            diesel::insert_into(libraries::table)
                .values(&row)
                .on_conflict(libraries::id)
                .do_update()
                .set(&row)
                .execute(conn)?;
        }
        let ids: Vec<&str> = provided.iter().map(|l| l.id.as_str()).collect();
        let removed =
            diesel::delete(libraries::table.filter(libraries::id.ne_all(ids))).execute(conn)?;
        if removed > 0 {
            warn!(removed, "deleted libraries the provider no longer has");
        }
        Ok(())
    })?;

    if excluded_ids.is_empty() {
        info!(libraries = provided.len(), "libraries synced");
    } else {
        let names: Vec<&str> = provided
            .iter()
            .filter(|library| excluded_ids.contains(&library.id))
            .map(|library| library.name.as_str())
            .collect();
        info!(libraries = provided.len(), excluded = %names.join(", "), "libraries synced");
    }
    Ok(excluded_ids)
}

fn is_named(library: &provider::Library, name_or_id: &str) -> bool {
    library.id == name_or_id || library.name.to_lowercase() == name_or_id.to_lowercase()
}

pub struct AlbumSync {
    pub changed: bool,
    pub tracks_added: usize,
    pub tracks_updated: usize,
    pub pending: Vec<Track>,
}

pub fn album(
    conn: &mut SqliteConnection,
    library_id: &str,
    provided: &provider::Album,
) -> Result<AlbumSync> {
    let wanted = AlbumRows::new(library_id, provided);
    let stored = StoredRows::load(conn, &wanted)?;
    let changes = Changes::between(&stored, &wanted);
    let changed = !changes.is_empty();
    if changed {
        changes.write(conn)?;
    }

    let tracks_added = changes
        .tracks
        .iter()
        .filter(|t| !stored.tracks.contains_key(&t.id))
        .count();
    Ok(AlbumSync {
        changed,
        tracks_added,
        tracks_updated: changes.tracks.len() - tracks_added,
        pending: pending_tracks(conn, &wanted.track_ids)?,
    })
}

struct AlbumRows<'a> {
    album: Album,
    tracks: Vec<Track>,
    track_ids: Vec<&'a str>,
    artist_names: HashMap<&'a str, &'a str>,
    track_artists: HashMap<&'a str, BTreeSet<&'a str>>,
}

impl<'a> AlbumRows<'a> {
    fn new(library_id: &str, provided: &'a provider::Album) -> Self {
        let album = Album {
            id: provided.id.clone(),
            library_id: library_id.to_string(),
            name: provided.name.clone(),
            artist: provided.artist.clone(),
        };
        let tracks = provided
            .tracks
            .iter()
            .map(|t| Track {
                id: t.id.clone(),
                library_id: library_id.to_string(),
                album_id: provided.id.clone(),
                title: t.title.clone(),
                artist: t.artist.clone(),
                duration: t.duration,
            })
            .collect();
        let track_ids = provided.tracks.iter().map(|t| t.id.as_str()).collect();

        let mut artist_names = HashMap::new();
        let mut track_artists = HashMap::new();
        for track in &provided.tracks {
            let artist_ids: &mut BTreeSet<&str> =
                track_artists.entry(track.id.as_str()).or_default();
            for artist in &track.artists {
                artist_names.insert(artist.id.as_str(), artist.name.as_str());
                artist_ids.insert(artist.id.as_str());
            }
        }

        Self {
            album,
            tracks,
            track_ids,
            artist_names,
            track_artists,
        }
    }
}

struct StoredRows {
    album: Option<Album>,
    tracks: HashMap<String, Track>,
    artist_names: HashMap<String, String>,
    track_artists: HashMap<String, BTreeSet<String>>,
}

impl StoredRows {
    fn load(conn: &mut SqliteConnection, wanted: &AlbumRows) -> Result<Self> {
        let artist_ids: Vec<&str> = wanted.artist_names.keys().copied().collect();

        let album = albums::table
            .find(&wanted.album.id)
            .select(Album::as_select())
            .first(conn)
            .optional()?;
        let tracks = tracks::table
            .filter(tracks::id.eq_any(&wanted.track_ids))
            .select(Track::as_select())
            .load(conn)?
            .into_iter()
            .map(|t| (t.id.clone(), t))
            .collect();
        let artist_names = artists::table
            .filter(artists::id.eq_any(&artist_ids))
            .select((artists::id, artists::name))
            .load(conn)?
            .into_iter()
            .collect();
        let mut track_artists: HashMap<String, BTreeSet<String>> = HashMap::new();
        for link in track_artists::table
            .filter(track_artists::track_id.eq_any(&wanted.track_ids))
            .select(TrackArtist::as_select())
            .load(conn)?
        {
            track_artists
                .entry(link.track_id)
                .or_default()
                .insert(link.artist_id);
        }

        Ok(Self {
            album,
            tracks,
            artist_names,
            track_artists,
        })
    }
}

struct Changes<'a> {
    album: Option<&'a Album>,
    artists: Vec<Artist>,
    tracks: Vec<&'a Track>,
    track_artists: Vec<(&'a str, &'a BTreeSet<&'a str>)>,
}

impl<'a> Changes<'a> {
    fn between(stored: &StoredRows, wanted: &'a AlbumRows<'a>) -> Self {
        let album = (stored.album.as_ref() != Some(&wanted.album)).then_some(&wanted.album);
        let artists = wanted
            .artist_names
            .iter()
            .filter(|&(id, name)| {
                stored
                    .artist_names
                    .get(*id)
                    .is_none_or(|stored| stored.to_lowercase() != name.to_lowercase())
            })
            .map(|(id, name)| Artist {
                id: id.to_string(),
                name: name.to_string(),
            })
            .collect();
        let tracks = wanted
            .tracks
            .iter()
            .filter(|t| stored.tracks.get(&t.id) != Some(*t))
            .collect();
        let track_artists = wanted
            .track_artists
            .iter()
            .filter(|&(track, artists)| !same_artists(stored.track_artists.get(*track), artists))
            .map(|(track, artists)| (*track, artists))
            .collect();

        Self {
            album,
            artists,
            tracks,
            track_artists,
        }
    }

    fn is_empty(&self) -> bool {
        self.album.is_none()
            && self.artists.is_empty()
            && self.tracks.is_empty()
            && self.track_artists.is_empty()
    }

    fn write(&self, conn: &mut SqliteConnection) -> Result<()> {
        conn.immediate_transaction(|conn| {
            if let Some(album) = self.album {
                diesel::insert_into(albums::table)
                    .values(album)
                    .on_conflict(albums::id)
                    .do_update()
                    .set(album)
                    .execute(conn)?;
            }
            for artist in &self.artists {
                diesel::insert_into(artists::table)
                    .values(artist)
                    .on_conflict(artists::id)
                    .do_update()
                    .set(artist)
                    .execute(conn)?;
            }
            for &track in &self.tracks {
                diesel::insert_into(tracks::table)
                    .values(track)
                    .on_conflict(tracks::id)
                    .do_update()
                    .set(track)
                    .execute(conn)?;
            }
            for &(track, artists) in &self.track_artists {
                diesel::delete(track_artists::table.filter(track_artists::track_id.eq(track)))
                    .execute(conn)?;
                let rows: Vec<TrackArtist> = artists
                    .iter()
                    .map(|artist| TrackArtist {
                        track_id: track.to_string(),
                        artist_id: artist.to_string(),
                    })
                    .collect();
                if !rows.is_empty() {
                    diesel::insert_into(track_artists::table)
                        .values(&rows)
                        .execute(conn)?;
                }
            }
            Ok(())
        })
    }
}

fn same_artists(stored: Option<&BTreeSet<String>>, wanted: &BTreeSet<&str>) -> bool {
    stored
        .into_iter()
        .flatten()
        .map(String::as_str)
        .eq(wanted.iter().copied())
}

fn pending_tracks(conn: &mut SqliteConnection, track_ids: &[&str]) -> Result<Vec<Track>> {
    let mut pending: Vec<Track> = tracks::table
        .filter(tracks::id.eq_any(track_ids))
        .filter(not(exists(
            analyses::table
                .filter(analyses::track_id.eq(tracks::id))
                .filter(analyses::error.is_null()),
        )))
        .select(Track::as_select())
        .load(conn)?;
    let position: HashMap<&str, usize> = track_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    pending.sort_by_key(|t| position[t.id.as_str()]);
    Ok(pending)
}

pub fn stored_albums(
    conn: &mut SqliteConnection,
    libraries: &[String],
) -> Result<Vec<(String, String)>> {
    Ok(albums::table
        .filter(albums::library_id.eq_any(libraries))
        .select((albums::id, albums::library_id))
        .load(conn)?)
}

pub fn remove_missing(
    conn: &mut SqliteConnection,
    libraries: &[String],
    seen_albums: &HashSet<String>,
    seen_tracks: &HashSet<String>,
) -> Result<()> {
    let gone_tracks: Vec<String> = tracks::table
        .filter(tracks::library_id.eq_any(libraries))
        .select(tracks::id)
        .load::<String>(conn)?
        .into_iter()
        .filter(|id| !seen_tracks.contains(id))
        .collect();
    let gone_albums: Vec<String> = stored_albums(conn, libraries)?
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| !seen_albums.contains(id))
        .collect();

    conn.immediate_transaction(|conn| {
        for ids in gone_tracks.chunks(IDS_PER_QUERY) {
            diesel::delete(tracks::table.filter(tracks::id.eq_any(ids))).execute(conn)?;
        }
        for ids in gone_albums.chunks(IDS_PER_QUERY) {
            diesel::delete(albums::table.filter(albums::id.eq_any(ids))).execute(conn)?;
        }
        let artists = diesel::delete(artists::table.filter(not(exists(
            track_artists::table.filter(track_artists::artist_id.eq(artists::id)),
        ))))
        .execute(conn)?;
        if !gone_tracks.is_empty() || !gone_albums.is_empty() || artists > 0 {
            info!(
                tracks = gone_tracks.len(),
                albums = gone_albums.len(),
                artists,
                "deleted what the provider no longer has"
            );
        }
        Ok(())
    })
}
