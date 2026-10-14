use crate::{
    filter::Filter,
    path,
    store::{self, Missing, Track},
};
use axum::{
    Json, Router,
    extract::{Query, State, rejection::QueryRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use common::{
    db::{NEIGHBORS_PER_TRACK, Pool},
    search_index::{self, Node, SearchIndex},
    vector::similarity,
};
use diesel::{Connection, sqlite::SqliteConnection};
use serde::{Deserialize, Serialize};
use std::{io, ops::RangeInclusive, time::Instant};
use tracing::{debug, error};

const SIMILAR_COUNT: usize = 20;
const SIMILAR_COUNTS: RangeInclusive<usize> = 1..=NEIGHBORS_PER_TRACK;
const PATH_COUNT: usize = 25;
const PATH_COUNTS: RangeInclusive<usize> = 2..=200;

#[derive(Clone)]
pub struct AppState {
    pub pool: Pool,
    pub filter: Filter,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/similar", get(similar))
        .route("/path", get(path))
        .with_state(state)
}

#[derive(Serialize)]
struct Tracks<T> {
    tracks: Vec<T>,
}

#[derive(Deserialize)]
struct SimilarParams {
    id: String,
    count: Option<usize>,
}

#[derive(Serialize)]
struct Match {
    #[serde(flatten)]
    track: Track,
    similarity: f32,
}

impl Match {
    fn new((distance, track): (f32, Track)) -> Self {
        Self {
            track,
            similarity: similarity(distance),
        }
    }
}

async fn similar(
    State(state): State<AppState>,
    params: Result<Query<SimilarParams>, QueryRejection>,
) -> Result<Json<Tracks<Match>>, ApiError> {
    let params = parse(params)?;
    let count = count(params.count, SIMILAR_COUNT, SIMILAR_COUNTS)?;
    let filter = state.filter;
    let tracks = with_conn(&state.pool, move |conn| {
        let Some(ranked) = filter.neighbors(conn, &params.id, count)? else {
            return Err(missing(conn, &params.id));
        };
        Ok(ranked.into_iter().map(Match::new).collect())
    })
    .await?;
    Ok(Json(Tracks { tracks }))
}

#[derive(Deserialize)]
struct PathParams {
    from: String,
    to: String,
    count: Option<usize>,
}

async fn path(
    State(state): State<AppState>,
    params: Result<Query<PathParams>, QueryRejection>,
) -> Result<Json<Tracks<Match>>, ApiError> {
    let params = parse(params)?;
    let count = count(params.count, PATH_COUNT, PATH_COUNTS)?;
    if params.from == params.to {
        return Err(ApiError::BadRequest(
            "from and to must be different tracks".into(),
        ));
    }
    let filter = state.filter;
    let tracks = with_conn(&state.pool, move |conn| {
        let started = Instant::now();
        let index = open_search_index()?;
        let from = seed(conn, &index, &params.from)?;
        let to = seed(conn, &index, &params.to)?;
        let path = path::find(conn, &filter, &index, from, to, count)?;
        debug!(
            ms = started.elapsed().as_millis() as u64,
            seen = path.seen_count,
            route = path.route_len,
            candidates = path.candidate_count,
            "path found"
        );
        Ok(path.tracks.into_iter().map(Match::new).collect())
    })
    .await?;
    Ok(Json(Tracks { tracks }))
}

async fn with_conn<T, F>(pool: &Pool, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&mut SqliteConnection) -> Result<T, ApiError> + Send + 'static,
{
    let pool = pool.clone();
    tokio::task::spawn_blocking(move || {
        let mut conn = pool.get()?;
        SqliteConnection::transaction(&mut conn, f)
    })
    .await?
}

fn parse<T>(query: Result<Query<T>, QueryRejection>) -> Result<T, ApiError> {
    query
        .map(|Query(params)| params)
        .map_err(|e| ApiError::BadRequest(e.body_text()))
}

fn count(
    requested: Option<usize>,
    default: usize,
    allowed: RangeInclusive<usize>,
) -> Result<usize, ApiError> {
    let count = requested.unwrap_or(default);
    if !allowed.contains(&count) {
        return Err(ApiError::BadRequest(format!(
            "count must be between {} and {}",
            allowed.start(),
            allowed.end()
        )));
    }
    Ok(count)
}

fn open_search_index() -> Result<SearchIndex, ApiError> {
    SearchIndex::open(&search_index::path()).map_err(|e| match e.kind() {
        io::ErrorKind::NotFound => ApiError::NotFound("no track is indexed yet".into()),
        _ => e.into(),
    })
}

fn seed(conn: &mut SqliteConnection, index: &SearchIndex, id: &str) -> Result<Node, ApiError> {
    match store::idx(conn, id)?.and_then(|idx| index.node(idx)) {
        Some(node) => Ok(node),
        None => Err(missing(conn, id)),
    }
}

fn missing(conn: &mut SqliteConnection, id: &str) -> ApiError {
    let message = match store::missing(conn, id) {
        Ok(Missing::Unknown) => format!("unknown track {id}"),
        Ok(Missing::Excluded) => format!("track {id} is in an excluded library"),
        Ok(Missing::NotAnalyzed) => format!("track {id} has not been analyzed yet"),
        Ok(Missing::Failed(error)) => format!("track {id} failed analysis: {error}"),
        Ok(Missing::NotIndexed) => format!("track {id} is not indexed yet"),
        Err(e) => return e.into(),
    };
    ApiError::NotFound(message)
}

pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Internal(anyhow::Error),
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self::Internal(e.into())
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message),
            Self::Internal(e) => {
                error!("{e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
            }
        };
        (status, Json(ErrorBody { error })).into_response()
    }
}
