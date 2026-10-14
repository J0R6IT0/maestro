use crate::config::ProviderConfig;
use anyhow::Result;

pub mod subsonic;

const USER_AGENT: &str = concat!(
    "maestro/",
    env!("CARGO_PKG_VERSION"),
    " (https://github.com/J0R6IT0/maestro)"
);

pub fn connect(config: &ProviderConfig) -> Result<Box<dyn Provider>> {
    Ok(match config {
        ProviderConfig::Subsonic(settings) => Box::new(subsonic::Subsonic::new(settings.clone())?),
    })
}

pub trait Provider: Send + Sync {
    fn libraries(&self) -> Result<Vec<Library>>;
    /// A page of album ids, newest first and starting at `offset`. Empty past the end.
    fn album_ids(&self, library_id: &str, offset: usize) -> Result<Vec<String>>;
    /// None if the provider doesn't have it anymor.
    fn album(&self, id: &str) -> Result<Option<Album>>;
    fn fetch_audio(&self, track_id: &str) -> Result<Vec<u8>>;
}

pub struct Library {
    pub id: String,
    pub name: String,
}

pub struct Album {
    pub id: String,
    pub name: String,
    pub artist: String,
    pub tracks: Vec<Track>,
}

pub struct Track {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub duration: i32,
    pub artists: Vec<ArtistRef>,
}

pub struct ArtistRef {
    pub id: String,
    pub name: String,
}
