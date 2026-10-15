use super::{Album, ArtistRef, CLIENT_NAME, Library, Provider, Track, USER_AGENT};
use anyhow::{Context, Result, bail};
use common::env;
use reqwest::{
    StatusCode, Url,
    blocking::{Client, RequestBuilder, Response},
    header::CONTENT_TYPE,
};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    sync::{LazyLock, OnceLock},
    time::Duration,
};
use tracing::info;

const API_VERSION: &str = "1.16.1";
const TIMEOUT: Duration = Duration::from_secs(300);
const ALBUM_PAGE: usize = 500;
const OPUS_KBPS: u32 = 128;

static CLIENT_INFO: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "name": CLIENT_NAME,
        "platform": std::env::consts::OS,
        "directPlayProfiles": [
            { "containers": [], "audioCodecs": ["mp3", "aac", "opus", "vorbis"], "protocols": ["http"] }
        ],
        "transcodingProfiles": [
            { "container": "flac", "audioCodec": "flac", "protocol": "http", "maxAudioChannels": 1 },
            { "container": "opus", "audioCodec": "opus", "protocol": "http", "maxAudioChannels": 1 }
        ],
        "codecProfiles": [
            {
                "type": "AudioCodec",
                "name": "flac",
                "limitations": [
                    { "name": "audioSamplerate", "comparison": "LessThanEqual", "values": ["16000"], "required": true },
                    { "name": "audioBitdepth", "comparison": "LessThanEqual", "values": ["16"], "required": true }
                ]
            },
            {
                "type": "AudioCodec",
                "name": "opus",
                "limitations": [
                    { "name": "audioBitrate", "comparison": "LessThanEqual", "values": [(OPUS_KBPS * 1000).to_string()], "required": false }
                ]
            }
        ]
    })
});

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
    transcoding: OnceLock<bool>,
}

impl Subsonic {
    pub fn new(settings: Settings) -> Result<Self> {
        let base = Url::parse(&format!("{}/rest/", settings.url.trim_end_matches('/')))
            .context("invalid SUBSONIC_URL")?;
        let http = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(TIMEOUT)
            .build()?;

        Ok(Self {
            settings,
            base,
            http,
            transcoding: OnceLock::new(),
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

    fn send(request: RequestBuilder) -> reqwest::Result<Response> {
        request
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| e.without_url())
    }

    fn call(&self, endpoint: &'static str, params: &[(&str, &str)]) -> Result<Value> {
        Self::execute(endpoint, self.http.get(self.url(endpoint, params)))
    }

    fn execute(endpoint: &'static str, request: RequestBuilder) -> Result<Value> {
        let mut body: Value = Self::send(request)
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

    fn download(&self, endpoint: &'static str, params: &[(&str, &str)]) -> Result<Vec<u8>> {
        let response = Self::send(self.http.get(self.url(endpoint, params)))
            .with_context(|| format!("{endpoint} request failed"))?;

        let is_error = response
            .headers()
            .get(CONTENT_TYPE)
            .is_some_and(|v| v.as_bytes().starts_with(b"application/json"));
        let body = response
            .bytes()
            .map_err(|e| e.without_url())
            .with_context(|| format!("{endpoint} download failed"))?;
        if is_error {
            let error: Value = serde_json::from_slice(&body).unwrap_or_default();
            return Err(ApiError::new(endpoint, &error["subsonic-response"]).into());
        }
        Ok(body.to_vec())
    }

    fn supports_transcoding(&self) -> Result<bool> {
        if let Some(&supported) = self.transcoding.get() {
            return Ok(supported);
        }
        let supported = match self.call("getOpenSubsonicExtensions", &[]) {
            Ok(mut response) => take_list::<Extension>(&mut response, "/openSubsonicExtensions")
                .is_ok_and(|extensions| extensions.iter().any(|e| e.name == "transcoding")),
            Err(e) if is_unsupported(&e) => false,
            Err(e) => return Err(e),
        };
        if self.transcoding.set(supported).is_ok() {
            if supported {
                info!("server supports the transcoding extension");
            } else {
                info!(
                    "server lacks the transcoding extension, streaming as {}",
                    self.settings.format
                );
            }
        }
        Ok(supported)
    }

    fn fetch_transcoded(&self, track_id: &str) -> Result<Vec<u8>> {
        let media = [("mediaId", track_id), ("mediaType", "song")];
        let request = self
            .http
            .post(self.url("getTranscodeDecision", &media))
            .json(&*CLIENT_INFO);
        let mut response = Self::execute("getTranscodeDecision", request)?;
        let decision: TranscodeDecision =
            take(&mut response, "getTranscodeDecision", "transcodeDecision")?;

        let Some(params) = decision
            .transcode_params
            .filter(|_| decision.can_direct_play || decision.can_transcode)
        else {
            bail!(
                "getTranscodeDecision: {}",
                decision
                    .error_reason
                    .filter(|r| !r.is_empty())
                    .as_deref()
                    .unwrap_or("no compatible format")
            );
        };
        self.download(
            "getTranscodeStream",
            &[
                ("mediaId", track_id),
                ("mediaType", "song"),
                ("transcodeParams", &params),
            ],
        )
    }

    fn fetch_stream(&self, track_id: &str) -> Result<Vec<u8>> {
        let kbps = OPUS_KBPS.to_string();
        let mut params = vec![
            ("id", track_id),
            ("format", self.settings.format.as_str()),
            ("estimateContentLength", "false"),
        ];
        if self.settings.format == "opus" {
            params.push(("maxBitRate", &kbps));
        }
        self.download("stream", &params)
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
        let album: SubsonicAlbum = take(&mut response, "getAlbum", "album")?;
        Ok(Some(album.into()))
    }

    fn fetch_audio(&self, track_id: &str) -> Result<Vec<u8>> {
        if self.supports_transcoding()? {
            self.fetch_transcoded(track_id)
        } else {
            self.fetch_stream(track_id)
        }
    }
}

#[derive(Deserialize)]
struct Extension {
    name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TranscodeDecision {
    #[serde(default)]
    can_direct_play: bool,
    #[serde(default)]
    can_transcode: bool,
    transcode_params: Option<String>,
    error_reason: Option<String>,
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

fn is_unsupported(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ApiError>().is_some()
        || error
            .downcast_ref::<reqwest::Error>()
            .and_then(reqwest::Error::status)
            == Some(StatusCode::NOT_FOUND)
}

fn take<T: DeserializeOwned>(response: &mut Value, endpoint: &'static str, key: &str) -> Result<T> {
    let value = response
        .get_mut(key)
        .map(Value::take)
        .with_context(|| format!("{endpoint}: no {key} in the response"))?;
    serde_json::from_value(value).context(endpoint)
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
