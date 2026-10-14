use diesel::connection::SimpleConnection;
use diesel::r2d2::{ConnectionManager, CustomizeConnection, PooledConnection};
use diesel::sqlite::SqliteConnection;
use diesel::{Connection as _, ExpressionMethods, QueryDsl, QueryResult, RunQueryDsl};
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};
use schema::{analyses, vectors};
use std::{collections::HashMap, time::Duration};
use types::Embedding;

pub mod models;
pub mod schema;
pub mod types;

pub type Pool = diesel::r2d2::Pool<ConnectionManager<SqliteConnection>>;
pub type Connection = PooledConnection<ConnectionManager<SqliteConnection>>;

pub const DATABASE_PATH: &str = "maestro.db";
pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");
pub const IDS_PER_QUERY: usize = 1000;
pub const NEIGHBORS_PER_TRACK: usize = 100;

const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database connection: {0}")]
    Connection(#[from] diesel::ConnectionError),
    #[error("database pool: {0}")]
    Pool(#[from] diesel::r2d2::PoolError),
    #[error("database migration: {0}")]
    Migration(Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Query(#[from] diesel::result::Error),
}

pub fn connect(path: &str) -> Result<Pool, Error> {
    let mut conn = SqliteConnection::establish(path)?;

    set_pragmas(&mut conn)?;

    conn.batch_execute("PRAGMA journal_mode = WAL;")?;
    conn.run_pending_migrations(MIGRATIONS)
        .map_err(Error::Migration)?;

    Ok(Pool::builder()
        .connection_customizer(Box::new(Pragmas))
        .build(ConnectionManager::new(path))?)
}

fn set_pragmas(conn: &mut SqliteConnection) -> diesel::QueryResult<()> {
    conn.batch_execute(&format!(
        "PRAGMA busy_timeout = {};
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;",
        BUSY_TIMEOUT.as_millis()
    ))
}

#[derive(Debug)]
struct Pragmas;

impl CustomizeConnection<SqliteConnection, diesel::r2d2::Error> for Pragmas {
    fn on_acquire(&self, conn: &mut SqliteConnection) -> Result<(), diesel::r2d2::Error> {
        set_pragmas(conn).map_err(diesel::r2d2::Error::QueryError)
    }
}

pub fn raw_embeddings(
    conn: &mut SqliteConnection,
    idx: &[i32],
) -> QueryResult<HashMap<i32, Vec<f32>>> {
    let mut embeddings = HashMap::with_capacity(idx.len());
    for ids in idx.chunks(IDS_PER_QUERY) {
        let rows: Vec<(i32, Option<Embedding>)> = vectors::table
            .inner_join(analyses::table)
            .filter(vectors::idx.eq_any(ids))
            .select((vectors::idx, analyses::embedding))
            .load(conn)?;
        embeddings.extend(rows.into_iter().filter_map(|(i, e)| Some((i, e?.0))));
    }
    Ok(embeddings)
}
