CREATE TABLE libraries (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    excluded BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE artists (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE albums (
    id TEXT PRIMARY KEY NOT NULL,
    library_id TEXT NOT NULL REFERENCES libraries (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    artist TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX albums_library ON albums (library_id);

CREATE TABLE tracks (
    id TEXT PRIMARY KEY NOT NULL,
    library_id TEXT NOT NULL REFERENCES libraries (id) ON DELETE CASCADE,
    album_id TEXT NOT NULL REFERENCES albums (id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    duration INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX tracks_library ON tracks (library_id);
CREATE INDEX tracks_album ON tracks (album_id);

CREATE TABLE track_artists (
    track_id TEXT NOT NULL REFERENCES tracks (id) ON DELETE CASCADE,
    artist_id TEXT NOT NULL REFERENCES artists (id) ON DELETE CASCADE,
    PRIMARY KEY (track_id, artist_id)
) WITHOUT ROWID;

CREATE INDEX track_artists_artist ON track_artists (artist_id);

CREATE TABLE analyses (
    track_id TEXT PRIMARY KEY NOT NULL REFERENCES tracks (id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding BLOB,
    error TEXT,
    analyzed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((embedding IS NULL) != (error IS NULL))
);

CREATE INDEX analyses_embedded ON analyses (track_id) WHERE embedding IS NOT NULL;

CREATE TABLE styles (
    id INTEGER PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE track_styles (
    track_id TEXT NOT NULL REFERENCES analyses (track_id) ON DELETE CASCADE,
    style_id INTEGER NOT NULL REFERENCES styles (id) ON DELETE CASCADE,
    score REAL NOT NULL,
    PRIMARY KEY (track_id, style_id)
) WITHOUT ROWID;

CREATE INDEX track_styles_style ON track_styles (style_id, score);

CREATE TABLE vectors (
    idx INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    track_id TEXT NOT NULL UNIQUE REFERENCES analyses (track_id) ON DELETE CASCADE,
    neighbors BLOB
);

CREATE INDEX vectors_pending ON vectors (idx) WHERE neighbors IS NULL;

CREATE TABLE vector_data (
    idx INTEGER PRIMARY KEY NOT NULL REFERENCES vectors (idx) ON DELETE CASCADE,
    vec BLOB NOT NULL
);

CREATE TABLE index_queue (
    track_id TEXT PRIMARY KEY NOT NULL REFERENCES analyses (track_id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE removed_vectors (
    idx INTEGER PRIMARY KEY NOT NULL
);

CREATE TABLE stale_vectors (
    idx INTEGER PRIMARY KEY NOT NULL REFERENCES vectors (idx) ON DELETE CASCADE
);

CREATE TABLE near_identical_pairs (
    a INTEGER NOT NULL REFERENCES vectors (idx) ON DELETE CASCADE,
    b INTEGER NOT NULL REFERENCES vectors (idx) ON DELETE CASCADE,
    raw_distance REAL NOT NULL,
    PRIMARY KEY (a, b),
    CHECK (a < b)
) WITHOUT ROWID;

CREATE INDEX near_identical_pairs_b ON near_identical_pairs (b);

CREATE TABLE near_identical_pending (
    idx INTEGER PRIMARY KEY NOT NULL REFERENCES vectors (idx) ON DELETE CASCADE
);

CREATE TRIGGER vectors_removed AFTER DELETE ON vectors
BEGIN
    INSERT OR IGNORE INTO removed_vectors (idx) VALUES (OLD.idx);
END;

CREATE TABLE index_meta (
    id INTEGER PRIMARY KEY NOT NULL CHECK (id = 0),
    mean BLOB NOT NULL,
    track_count INTEGER NOT NULL,
    k INTEGER NOT NULL,
    rebuild_after TEXT
);

CREATE TABLE embeddings_changed (
    id INTEGER PRIMARY KEY NOT NULL CHECK (id = 0)
);

CREATE TRIGGER analyses_embedding_added AFTER INSERT ON analyses
WHEN NEW.embedding IS NOT NULL
BEGIN
    INSERT OR IGNORE INTO embeddings_changed (id) VALUES (0);
END;

CREATE TRIGGER analyses_embedding_updated AFTER UPDATE OF embedding ON analyses
BEGIN
    INSERT OR IGNORE INTO embeddings_changed (id) VALUES (0);
END;

CREATE TRIGGER analyses_embedding_removed AFTER DELETE ON analyses
WHEN OLD.embedding IS NOT NULL
BEGIN
    INSERT OR IGNORE INTO embeddings_changed (id) VALUES (0);
END;
