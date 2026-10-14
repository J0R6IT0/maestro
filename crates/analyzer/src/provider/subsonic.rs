use super::{Album, ArtistRef, Library, Provider, Track};
use anyhow::{Context, Result, bail};
use common::env;
use reqwest::{
    Url,
    blocking::{Client, Response},
    header::CONTENT_TYPE,
};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};
use serde_json::Value;
use std::time::Duration;

const API_VERSION: &str = "1.16.1";
const CLIENT_NAME: &str = "maestro";
const TIMEOUT: Duration = Duration::from_secs(300);
const ALBUM_PAGE: usize = 500;
const OPUS_KBPS: &str = "128";

#[derive(Clone)]
pub struct Settings {
    pub url: String,
    pub user: String,
    pub password: String,
    pub format: String,
}

impl Settings {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            url: env::required("SUBSONIC_URL")?,
            user: env::required("SUBSONIC_USER")?,
            password: env::required("SUBSONIC_PASSWORD")?,
            format: env::var_or("SUBSONIC_FORMAT", "opus"),
        })
    }
}

pub struct Subsonic {
    settings: Settings,
    base: Url,
    http: Client,
}

impl Subsonic {
    pub fn new(settings: Settings) -> Result<Self> {
        let base = Url::parse(&format!("{}/rest/", settings.url.trim_end_matches('/')))
            .context("invalid SUBSONIC_URL")?;
        let http = Client::builder().timeout(TIMEOUT).build()?;

        Ok(Self {
            settings,
            base,
            http,
        })
    }

    fn url(&self, endpoint: &str, params: &[(&str, &str)]) -> Url {
        let salt = format!("{:016x}", rand::random::<u64>());
        let token = format!(
            "{:x}",
            md5::compute(format!("{}{salt}", self.settings.password))
        );
        let mut url = self
            .base
            .join(endpoint)
            .expect("endpoint is a relative path");

        url.query_pairs_mut()
            .extend_pairs([
                ("u", self.settings.user.as_str()),
                ("t", &token),
                ("s", &salt),
                ("v", API_VERSION),
                ("c", CLIENT_NAME),
                ("f", "json"),
            ])
            .extend_pairs(params);
        url
    }

    fn get(&self, endpoint: &str, params: &[(&str, &str)]) -> reqwest::Result<Response> {
        self.http
            .get(self.url(endpoint, params))
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| e.without_url())
    }

    fn call(&self, endpoint: &'static str, params: &[(&str, &str)]) -> Result<Value> {
        let mut body: Value = self
            .get(endpoint, params)
            .and_then(|r| r.json().map_err(|e| e.without_url()))
            .with_context(|| format!("{endpoint} request failed"))?;

        let response = body
            .get_mut("subsonic-response")
            .map(Value::take)
            .unwrap_or_default();

        match response["status"].as_str() {
            Some("ok") => Ok(response),
            Some(_) => Err(ApiError::new(endpoint, &response).into()),
            None => bail!("{endpoint}: not a Subsonic response"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{endpoint}: {message} (Subsonic error {code})")]
struct ApiError {
    endpoint: &'static str,
    code: i64,
    message: String,
}

impl ApiError {
    const DATA_NOT_FOUND: i64 = 70;

    fn new(endpoint: &'static str, response: &Value) -> Self {
        Self {
            endpoint,
            code: response["error"]["code"].as_i64().unwrap_or(0),
            message: response["error"]["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string(),
        }
    }

    fn is_not_found(error: &anyhow::Error) -> bool {
        error
            .downcast_ref::<Self>()
            .is_some_and(|e| e.code == Self::DATA_NOT_FOUND)
    }
}

impl Provider for Subsonic {
    fn libraries(&self) -> Result<Vec<Library>> {
        let mut response = self.call("getMusicFolders", &[])?;
        let folders: Vec<MusicFolder> = take_list(&mut response, "/musicFolders/musicFolder")?;
        Ok(folders.into_iter().map(Library::from).collect())
    }

    fn album_ids(&self, library_id: &str, offset: usize) -> Result<Vec<String>> {
        let (size, offset) = (ALBUM_PAGE.to_string(), offset.to_string());
        let mut response = self.call(
            "getAlbumList2",
            &[
                ("type", "newest"),
                ("size", &size),
                ("offset", &offset),
                ("musicFolderId", library_id),
            ],
        )?;
        let albums: Vec<AlbumId> = take_list(&mut response, "/albumList2/album")?;
        Ok(albums.into_iter().map(|a| a.id).collect())
    }

    fn album(&self, id: &str) -> Result<Option<Album>> {
        let mut response = match self.call("getAlbum", &[("id", id)]) {
            Err(e) if ApiError::is_not_found(&e) => return Ok(None),
            response => response?,
        };
        let album = response
            .get_mut("album")
            .map(Value::take)
            .context("getAlbum: no album in the response")?;
        let album: SubsonicAlbum = serde_json::from_value(album).context("getAlbum")?;
        Ok(Some(album.into()))
    }

    fn fetch_audio(&self, track_id: &str) -> Result<Vec<u8>> {
        let mut params = vec![
            ("id", track_id),
            ("format", self.settings.format.as_str()),
            ("estimateContentLength", "false"),
        ];
        if self.settings.format == "opus" {
            params.push(("maxBitRate", OPUS_KBPS));
        }
        let response = self
            .get("stream", &params)
            .context("stream request failed")?;

        let is_error = response
            .headers()
            .get(CONTENT_TYPE)
            .is_some_and(|v| v.as_bytes().starts_with(b"application/json"));
        let body = response
            .bytes()
            .map_err(|e| e.without_url())
            .context("stream download failed")?;
        if is_error {
            let error: Value = serde_json::from_slice(&body).unwrap_or_default();
            return Err(ApiError::new("stream", &error["subsonic-response"]).into());
        }
        Ok(body.to_vec())
    }
}

#[derive(Deserialize)]
struct MusicFolder {
    #[serde(deserialize_with = "string_or_number")]
    id: String,
    #[serde(default, deserialize_with = "null_as_default")]
    name: String,
}

#[derive(Deserialize)]
struct AlbumId {
    #[serde(deserialize_with = "string_or_number")]
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubsonicAlbum {
    #[serde(deserialize_with = "string_or_number")]
    id: String,
    #[serde(default, deserialize_with = "null_as_default")]
    name: String,
    #[serde(default, deserialize_with = "null_as_default")]
    artist: String,
    display_artist: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    song: Vec<Song>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Song {
    #[serde(deserialize_with = "string_or_number")]
    id: String,
    #[serde(default, deserialize_with = "null_as_default")]
    title: String,
    #[serde(default, deserialize_with = "null_as_default")]
    artist: String,
    display_artist: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    duration: i32,
    #[serde(default, deserialize_with = "null_as_default")]
    artists: Vec<SongArtist>,
    #[serde(default, deserialize_with = "optional_string_or_number")]
    artist_id: Option<String>,
}

#[derive(Deserialize)]
struct SongArtist {
    #[serde(deserialize_with = "string_or_number")]
    id: String,
    #[serde(default, deserialize_with = "null_as_default")]
    name: String,
}

impl From<MusicFolder> for Library {
    fn from(folder: MusicFolder) -> Self {
        Self {
            id: folder.id,
            name: folder.name,
        }
    }
}

impl From<SubsonicAlbum> for Album {
    fn from(album: SubsonicAlbum) -> Self {
        Self {
            id: album.id,
            artist: album.display_artist.unwrap_or(album.artist),
            name: album.name,
            tracks: album.song.into_iter().map(Track::from).collect(),
        }
    }
}

impl From<Song> for Track {
    fn from(song: Song) -> Self {
        let artists = if song.artists.is_empty() {
            song.artist_id
                .map(|id| ArtistRef {
                    id,
                    name: song.artist.clone(),
                })
                .into_iter()
                .collect()
        } else {
            song.artists
                .into_iter()
                .map(|a| ArtistRef {
                    id: a.id,
                    name: a.name,
                })
                .collect()
        };
        Self {
            id: song.id,
            title: song.title,
            artist: song.display_artist.unwrap_or(song.artist),
            duration: song.duration,
            artists,
        }
    }
}

fn take_list<T: DeserializeOwned>(response: &mut Value, pointer: &str) -> Result<Vec<T>> {
    match response.pointer_mut(pointer).map(Value::take) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(items) => Ok(serde_json::from_value(items)?),
    }
}

fn null_as_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Id {
    Text(String),
    Number(i64),
}

impl From<Id> for String {
    fn from(id: Id) -> Self {
        match id {
            Id::Text(s) => s,
            Id::Number(n) => n.to_string(),
        }
    }
}

fn string_or_number<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Id::deserialize(d).map(String::from)
}

fn optional_string_or_number<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    Ok(Option::<Id>::deserialize(d)?.map(String::from))
}
