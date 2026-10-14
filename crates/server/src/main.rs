use crate::{api::AppState, filter::Filter};
use anyhow::{Context, Result};
use common::{db::DATABASE_PATH, env, logging};
use std::process::ExitCode;
use tokio::{net::TcpListener, signal};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::{Level, error, info};

mod api;
mod filter;
mod path;
mod store;

#[tokio::main]
async fn main() -> ExitCode {
    dotenvy::dotenv().ok();
    logging::init();

    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let addr = env::var_or("LISTEN_ADDR", "0.0.0.0:3000");
    let filter = Filter::from_env()?;
    let pool = common::db::connect(DATABASE_PATH)?;

    let app = api::router(AppState { pool, filter }).layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
            .on_response(DefaultOnResponse::new().level(Level::INFO)),
    );
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("can't listen on {addr}"))?;
    info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let ctrl_c = async {
        signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    info!("shutting down");
}
