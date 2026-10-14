use extism_pdk::{config, info};
use nd_pdk::host::http::{self, HTTPRequest};
use nd_pdk::metadata::{
    self, SimilarSongsByTrackProvider, SimilarSongsByTrackRequest, SimilarSongsResponse,
};
use nd_pdk::sonicsimilarity::{
    self, FindSonicPathRequest, GetSonicSimilarTracksRequest, SonicMatch, SonicSimilarity,
    SonicSimilarityResponse,
};
use nd_pdk::types::SongRef;
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde::Deserialize;

const DEFAULT_URL: &str = "http://localhost:3000";
const MAX_SIMILAR: i32 = 100;
const MIN_PATH: i32 = 2;
const MAX_PATH: i32 = 200;
const TIMEOUT_MS: i32 = 10_000;

nd_pdk::register_sonicsimilarity!(Maestro);
nd_pdk::register_metadata_similar_songs_by_track!(Maestro);

#[derive(Default)]
struct Maestro;

impl SonicSimilarity for Maestro {
    fn get_sonic_similar_tracks(
        &self,
        req: GetSonicSimilarTracksRequest,
    ) -> Result<SonicSimilarityResponse, sonicsimilarity::Error> {
        let tracks = similar(req.song.id, req.count).map_err(sonicsimilarity::Error::new)?;
        Ok(SonicSimilarityResponse {
            matches: tracks.into_iter().map(SonicMatch::from).collect(),
        })
    }

    fn find_sonic_path(
        &self,
        req: FindSonicPathRequest,
    ) -> Result<SonicSimilarityResponse, sonicsimilarity::Error> {
        let mut query = vec![("from", req.start_song.id), ("to", req.end_song.id)];
        if req.count > 0 {
            query.push(("count", req.count.clamp(MIN_PATH, MAX_PATH).to_string()));
        }

        let tracks = get("path", &query)
            .and_then(|tracks| tracks.map_err(|message| format!("no path: {message}")))
            .map_err(sonicsimilarity::Error::new)?;
        Ok(SonicSimilarityResponse {
            matches: tracks.into_iter().map(SonicMatch::from).collect(),
        })
    }
}

impl SimilarSongsByTrackProvider for Maestro {
    fn get_similar_songs_by_track(
        &self,
        req: SimilarSongsByTrackRequest,
    ) -> Result<SimilarSongsResponse, metadata::Error> {
        let tracks = similar(req.id, req.count).map_err(metadata::Error::new)?;
        Ok(SimilarSongsResponse {
            songs: tracks.into_iter().map(SongRef::from).collect(),
        })
    }
}

#[derive(Deserialize)]
struct Tracks {
    tracks: Vec<Track>,
}

#[derive(Deserialize)]
struct Track {
    id: String,
    title: String,
    artist: String,
    album: String,
    similarity: f64,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

fn similar(id: String, count: i32) -> Result<Vec<Track>, String> {
    let mut query = vec![("id", id)];
    if count > 0 {
        query.push(("count", count.min(MAX_SIMILAR).to_string()));
    }
    Ok(get("similar", &query)?.unwrap_or_else(|message| {
        info!("no similar tracks: {message}");
        Vec::new()
    }))
}

fn get(endpoint: &str, query: &[(&str, String)]) -> Result<Result<Vec<Track>, String>, String> {
    let query: Vec<String> = query
        .iter()
        .map(|(key, value)| format!("{key}={}", utf8_percent_encode(value, NON_ALPHANUMERIC)))
        .collect();
    let url = format!("{}/{endpoint}?{}", server_url()?, query.join("&"));

    let response = http::send(HTTPRequest {
        method: "GET".into(),
        url: url.clone(),
        timeout_ms: TIMEOUT_MS,
        ..Default::default()
    })
    .map_err(|e| format!("GET {url}: {e}"))?
    .ok_or_else(|| format!("GET {url}: no response"))?;

    match response.status_code {
        200 => {}
        404 => return Ok(Err(message(&response.body))),
        status => return Err(format!("GET {url}: {status}: {}", message(&response.body))),
    }
    let Tracks { tracks } = serde_json::from_slice(&response.body)
        .map_err(|e| format!("GET {url}: bad response: {e}"))?;
    Ok(Ok(tracks))
}

fn server_url() -> Result<String, String> {
    let url = config::get("url")
        .map_err(|e| format!("reading config: {e}"))?
        .filter(|url| !url.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_URL.into());
    Ok(url.trim().trim_end_matches('/').to_string())
}

impl From<Track> for SongRef {
    fn from(track: Track) -> Self {
        Self {
            id: track.id,
            name: track.title,
            artist: track.artist,
            album: track.album,
            ..Default::default()
        }
    }
}

impl From<Track> for SonicMatch {
    fn from(track: Track) -> Self {
        let similarity = track.similarity;
        Self {
            song: track.into(),
            similarity,
        }
    }
}

fn message(body: &[u8]) -> String {
    serde_json::from_slice::<ErrorBody>(body)
        .map(|b| b.error)
        .unwrap_or_else(|_| String::from_utf8_lossy(body).into_owned())
}
