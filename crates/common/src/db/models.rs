use super::schema::*;
use super::types::Embedding;
use chrono::NaiveDateTime;
use diesel::prelude::*;

#[derive(Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Insertable, AsChangeset)]
#[diesel(table_name = libraries, check_for_backend(diesel::sqlite::Sqlite))]
pub struct Library {
    pub id: String,
    pub name: String,
    pub excluded: bool,
}

#[derive(Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Insertable, AsChangeset)]
#[diesel(table_name = artists, check_for_backend(diesel::sqlite::Sqlite))]
pub struct Artist {
    pub id: String,
    pub name: String,
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Queryable,
    Selectable,
    Identifiable,
    Associations,
    Insertable,
    AsChangeset,
)]
#[diesel(table_name = albums, belongs_to(Library), check_for_backend(diesel::sqlite::Sqlite))]
pub struct Album {
    pub id: String,
    pub library_id: String,
    pub name: String,
    pub artist: String,
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Queryable,
    Selectable,
    Identifiable,
    Associations,
    Insertable,
    AsChangeset,
)]
#[diesel(table_name = tracks, belongs_to(Library), belongs_to(Album), check_for_backend(diesel::sqlite::Sqlite))]
pub struct Track {
    pub id: String,
    pub library_id: String,
    pub album_id: String,
    pub title: String,
    pub artist: String,
    pub duration: i32,
}

#[derive(
    Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Associations, Insertable,
)]
#[diesel(
    table_name = track_artists,
    primary_key(track_id, artist_id),
    belongs_to(Track),
    belongs_to(Artist),
    check_for_backend(diesel::sqlite::Sqlite)
)]
pub struct TrackArtist {
    pub track_id: String,
    pub artist_id: String,
}

#[derive(
    Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Associations, Insertable,
)]
#[diesel(
    table_name = analyses,
    primary_key(track_id),
    belongs_to(Track),
    check_for_backend(diesel::sqlite::Sqlite)
)]
pub struct Analysis {
    pub track_id: String,
    pub model: String,
    pub embedding: Option<Embedding>,
    pub error: Option<String>,
    pub analyzed_at: NaiveDateTime,
}

#[derive(Debug, Clone, PartialEq, Queryable, Selectable, Identifiable)]
#[diesel(table_name = styles, check_for_backend(diesel::sqlite::Sqlite))]
pub struct Style {
    pub id: i32,
    pub name: String,
}

#[derive(Debug, Clone, Insertable)]
#[diesel(table_name = styles)]
pub struct NewStyle<'a> {
    pub name: &'a str,
}

#[derive(
    Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Associations, Insertable,
)]
#[diesel(
    table_name = track_styles,
    primary_key(track_id, style_id),
    belongs_to(Analysis, foreign_key = track_id),
    belongs_to(Style),
    check_for_backend(diesel::sqlite::Sqlite)
)]
pub struct TrackStyle {
    pub track_id: String,
    pub style_id: i32,
    pub score: f32,
}

#[derive(Debug, Clone, Insertable)]
#[diesel(table_name = vectors)]
pub struct NewVector<'a> {
    pub track_id: &'a str,
}

#[derive(Debug, Clone, Insertable)]
#[diesel(table_name = vector_data)]
pub struct NewVectorData<'a> {
    pub idx: i32,
    pub vec: &'a Embedding,
}

#[derive(Debug, Clone, PartialEq, Queryable, Selectable, Identifiable, Insertable, AsChangeset)]
#[diesel(table_name = index_meta, treat_none_as_null = true, check_for_backend(diesel::sqlite::Sqlite))]
pub struct IndexMeta {
    pub id: i32,
    pub mean: Embedding,
    pub track_count: i32,
    pub k: i32,
    pub rebuild_after: Option<String>,
}

impl IndexMeta {
    pub const ID: i32 = 0;
}
