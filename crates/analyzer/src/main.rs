use crate::config::Config;
use anyhow::{Context, Result, bail};
use common::{db::DATABASE_PATH, logging};
use std::{
    fs::{self, File, TryLockError},
    process::ExitCode,
};
use tracing::error;

mod audio;
mod config;
mod index;
mod model;
mod pipeline;
mod provider;
mod search;
mod store;
mod sync;

fn main() -> ExitCode {
    dotenvy::dotenv().ok();
    logging::init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let config = Config::from_env()?;
    let pool = common::db::connect(DATABASE_PATH)?;
    let _lock = lock()?;

    pipeline::run(&config, &pool)
}

fn lock() -> Result<File> {
    let path = fs::canonicalize(DATABASE_PATH)?.with_extension("lock");
    let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => bail!("another analyzer is already running"),
        Err(TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("locking {}", path.display()))
        }
    }
}
