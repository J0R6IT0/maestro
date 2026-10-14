use crate::provider::subsonic;
use anyhow::{Result, bail};
use common::env;

pub const MODEL_PATH: &str = "models/discogs-maest-30s-pw-519l-2.onnx";

pub struct Config {
    pub provider: ProviderConfig,
    pub download_workers: usize,
    pub excluded_libraries: Vec<String>,
}

pub enum ProviderConfig {
    Subsonic(subsonic::Settings),
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let provider = match env::var_or("PROVIDER", "subsonic").as_str() {
            "subsonic" => ProviderConfig::Subsonic(subsonic::Settings::from_env()?),
            other => bail!("unknown PROVIDER {other:?} (supported: subsonic)"),
        };
        Ok(Self {
            provider,
            download_workers: env::parsed_or("DOWNLOAD_WORKERS", 4)?,
            excluded_libraries: env::list("EXCLUDED_LIBRARIES"),
        })
    }
}
