// @generated automatically by Diesel CLI.

diesel::table! {
    albums (id) {
        id -> Text,
        library_id -> Text,
        name -> Text,
        artist -> Text,
    }
}

diesel::table! {
    analyses (track_id) {
        track_id -> Text,
        model -> Text,
        embedding -> Nullable<Binary>,
        error -> Nullable<Text>,
        analyzed_at -> Timestamp,
    }
}

diesel::table! {
    artists (id) {
        id -> Text,
        name -> Text,
    }
}

diesel::table! {
    embeddings_changed (id) {
        id -> Integer,
    }
}

diesel::table! {
    index_meta (id) {
        id -> Integer,
        mean -> Binary,
        track_count -> Integer,
        k -> Integer,
        rebuild_after -> Nullable<Text>,
    }
}

diesel::table! {
    index_queue (track_id) {
        track_id -> Text,
    }
}

diesel::table! {
    libraries (id) {
        id -> Text,
        name -> Text,
        excluded -> Bool,
    }
}

diesel::table! {
    removed_vectors (idx) {
        idx -> Integer,
    }
}

diesel::table! {
    near_identical_pairs (a, b) {
        a -> Integer,
        b -> Integer,
        raw_distance -> Float,
    }
}

diesel::table! {
    near_identical_pending (idx) {
        idx -> Integer,
    }
}

diesel::table! {
    stale_vectors (idx) {
        idx -> Integer,
    }
}

diesel::table! {
    styles (id) {
        id -> Integer,
        name -> Text,
    }
}

diesel::table! {
    track_artists (track_id, artist_id) {
        track_id -> Text,
        artist_id -> Text,
    }
}

diesel::table! {
    track_styles (track_id, style_id) {
        track_id -> Text,
        style_id -> Integer,
        score -> Float,
    }
}

diesel::table! {
    tracks (id) {
        id -> Text,
        library_id -> Text,
        album_id -> Text,
        title -> Text,
        artist -> Text,
        duration -> Integer,
    }
}

diesel::table! {
    vector_data (idx) {
        idx -> Integer,
        vec -> Binary,
    }
}

diesel::table! {
    vectors (idx) {
        idx -> Integer,
        track_id -> Text,
        neighbors -> Nullable<Binary>,
    }
}

diesel::joinable!(albums -> libraries (library_id));
diesel::joinable!(analyses -> tracks (track_id));
diesel::joinable!(index_queue -> analyses (track_id));
diesel::joinable!(near_identical_pending -> vectors (idx));
diesel::joinable!(stale_vectors -> vectors (idx));
diesel::joinable!(track_artists -> artists (artist_id));
diesel::joinable!(track_artists -> tracks (track_id));
diesel::joinable!(track_styles -> analyses (track_id));
diesel::joinable!(track_styles -> styles (style_id));
diesel::joinable!(tracks -> albums (album_id));
diesel::joinable!(tracks -> libraries (library_id));
diesel::joinable!(vector_data -> vectors (idx));
diesel::joinable!(vectors -> analyses (track_id));

diesel::allow_tables_to_appear_in_same_query!(
    albums,
    analyses,
    artists,
    embeddings_changed,
    index_meta,
    index_queue,
    libraries,
    removed_vectors,
    near_identical_pairs,
    near_identical_pending,
    stale_vectors,
    styles,
    track_artists,
    track_styles,
    tracks,
    vector_data,
    vectors,
);
