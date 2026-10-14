use crate::model::Analysis;
use anyhow::Result;
use chrono::Utc;
use common::db::{
    models::{self, NewStyle, TrackStyle},
    schema::{analyses, index_queue, styles, track_styles},
    types::Embedding,
};
use diesel::{prelude::*, sqlite::SqliteConnection};
use std::collections::HashMap;

const MAX_STYLES: usize = 10;
const MIN_STYLE_SCORE: f32 = 0.1;

pub fn register_styles(
    conn: &mut SqliteConnection,
    labels: &[String],
) -> Result<HashMap<String, i32>> {
    conn.immediate_transaction(|conn| {
        for name in labels {
            diesel::insert_into(styles::table)
                .values(NewStyle { name })
                .on_conflict_do_nothing()
                .execute(conn)?;
        }
        Ok(styles::table
            .select((styles::name, styles::id))
            .load::<(String, i32)>(conn)?
            .into_iter()
            .collect())
    })
}

pub fn save_analysis(
    conn: &mut SqliteConnection,
    model: &str,
    track_id: &str,
    result: Result<Analysis, String>,
    style_ids: &HashMap<String, i32>,
) -> Result<()> {
    let (embedding, error, styles) = match result {
        Ok(analysis) => (Some(Embedding(analysis.embedding)), None, analysis.styles),
        Err(e) => (None, Some(e), Vec::new()),
    };
    let queue_for_index = embedding.is_some();
    let row = models::Analysis {
        track_id: track_id.to_string(),
        model: model.to_string(),
        embedding,
        error,
        analyzed_at: Utc::now().naive_utc(),
    };
    let styles: Vec<TrackStyle> = styles
        .iter()
        .take(MAX_STYLES)
        .take_while(|(_, score)| *score >= MIN_STYLE_SCORE)
        .map(|(name, score)| TrackStyle {
            track_id: track_id.to_string(),
            style_id: style_ids[name],
            score: *score,
        })
        .collect();

    conn.immediate_transaction(|conn| {
        diesel::delete(analyses::table.find(track_id)).execute(conn)?;
        diesel::insert_into(analyses::table)
            .values(&row)
            .execute(conn)?;
        if !styles.is_empty() {
            diesel::insert_into(track_styles::table)
                .values(&styles)
                .execute(conn)?;
        }
        if queue_for_index {
            diesel::insert_into(index_queue::table)
                .values(index_queue::track_id.eq(track_id))
                .execute(conn)?;
        }
        Ok(())
    })
}
