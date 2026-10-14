use crate::env;
use tracing::warn;
use tracing_subscriber::{EnvFilter, fmt::time::ChronoLocal};

const DEFAULT_LEVEL: &str = "info,ort=warn";

pub fn init() {
    let requested = env::var("LOG_LEVEL");
    let (filter, rejected) = match requested.as_deref().map(EnvFilter::try_new) {
        Some(Ok(filter)) => (filter, None),
        Some(Err(_)) => (EnvFilter::new(DEFAULT_LEVEL), requested),
        None => (EnvFilter::new(DEFAULT_LEVEL), None),
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_timer(ChronoLocal::rfc_3339())
        .with_thread_names(true)
        .with_writer(std::io::stderr)
        .init();

    if let Some(level) = rejected {
        warn!("invalid LOG_LEVEL {level:?}, using {DEFAULT_LEVEL}");
    }
}
