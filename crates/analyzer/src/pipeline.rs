use crate::{
    audio::{self, MelPatches, PatchExtractor},
    config::{Config, MODEL_PATH},
    index,
    model::{Analysis, Model},
    provider::{self, Provider},
    store, sync,
};
use anyhow::{Context, Result, anyhow, bail};
use common::db::{Pool, models::Track};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::{
    collections::HashSet,
    iter,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Instant,
};
use tempfile::NamedTempFile;
use tracing::{debug, error, info, info_span, warn};

const ALBUM_WORKERS: usize = 4;
const ALBUM_BUFFER: usize = 64;
const TRACK_BUFFER: usize = 64;
const DECODED_BUFFER: usize = 8;
const INDEX_EVERY: usize = 4096;

struct ListedAlbum {
    library_id: String,
    id: String,
}

struct Decoded {
    track: Track,
    patches: Result<MelPatches>,
}

#[derive(Default)]
struct Pass {
    seen_libraries: Mutex<HashSet<String>>,
    seen_albums: Mutex<HashSet<String>>,
    seen_tracks: Mutex<HashSet<String>>,
    unanalyzed: AtomicUsize,
    incomplete: AtomicBool,
}

impl Pass {
    fn found_unanalyzed(&self, tracks: usize) {
        self.unanalyzed.fetch_add(tracks, Ordering::Relaxed);
    }

    fn has_unanalyzed(&self) -> bool {
        self.unanalyzed.load(Ordering::Relaxed) > 0
    }

    fn set_incomplete(&self) {
        self.incomplete.store(true, Ordering::Relaxed);
    }

    fn is_complete(&self) -> bool {
        !self.incomplete.load(Ordering::Relaxed)
    }
}

pub fn run(config: &Config, pool: &Pool) -> Result<()> {
    let started = Instant::now();
    audio::check_ffmpeg()?;
    audio::remove_stale_temp_files();
    let provider = provider::connect(&config.provider)?;
    let pass = Pass::default();

    let (album_tx, album_rx) = bounded(ALBUM_BUFFER);
    let (track_tx, track_rx) = bounded(TRACK_BUFFER);
    let (decoded_tx, decoded_rx) = bounded(DECODED_BUFFER);

    let analyzed = thread::scope(|s| -> Result<usize> {
        let (provider, pass) = (&*provider, &pass);
        let lister = thread::Builder::new()
            .name("lister".into())
            .spawn_scoped(s, move || {
                list_albums(provider, pool, pass, &config.excluded_libraries, album_tx)
            })?;

        for i in 0..ALBUM_WORKERS {
            let (album_rx, track_tx) = (album_rx.clone(), track_tx.clone());
            thread::Builder::new()
                .name(format!("album-{i}"))
                .spawn_scoped(s, move || {
                    sync_albums(provider, pool, pass, album_rx, track_tx)
                })?;
        }

        for i in 0..config.download_workers {
            let (track_rx, decoded_tx) = (track_rx.clone(), decoded_tx.clone());
            thread::Builder::new()
                .name(format!("decode-{i}"))
                .spawn_scoped(s, move || decode_tracks(provider, track_rx, decoded_tx))?;
        }

        drop((album_rx, track_tx, track_rx, decoded_tx));

        let analyzed = analyze(pool, decoded_rx);
        match lister.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                pass.set_incomplete();
                error!("album listing failed: {e:#}");
            }
            Err(_) => return Err(anyhow!("album lister panicked")),
        }
        analyzed
    })?;
    if !pass.has_unanalyzed() {
        info!("nothing new to analyze");
    }

    if pass.is_complete() {
        if let Err(e) = remove_missing(&*provider, pool, &pass) {
            warn!("couldn't delete what the provider no longer has, will retry next pass: {e:#}");
        }
    } else {
        warn!("pass incomplete, removed tracks will be deleted after the next complete one");
    }

    index::update_end_of_pass(&mut *pool.get()?)?;

    let secs = started.elapsed().as_secs_f32();
    info!(analyzed, secs, "pass finished");
    Ok(())
}

fn list_albums(
    provider: &dyn Provider,
    pool: &Pool,
    pass: &Pass,
    excluded_libraries: &[String],
    albums: Sender<ListedAlbum>,
) -> Result<()> {
    let libraries = provider.libraries()?;
    if libraries.is_empty() {
        bail!("the provider returned no libraries");
    }
    let excluded = sync::libraries(&mut *pool.get()?, &libraries, excluded_libraries)?;

    for library in libraries.iter().filter(|l| !excluded.contains(&l.id)) {
        let _span = info_span!("library", id = %library.id, name = %library.name).entered();
        pass.seen_libraries
            .lock()
            .unwrap()
            .insert(library.id.clone());

        let (mut offset, mut listed) = (0, 0);
        loop {
            let page = provider.album_ids(&library.id, offset)?;
            if page.is_empty() {
                break;
            }
            debug!(offset, albums = page.len(), "listed a page of albums");
            offset += page.len();
            for id in page {
                let first_seen = pass.seen_albums.lock().unwrap().insert(id.clone());
                if !first_seen {
                    continue;
                }
                listed += 1;
                let album = ListedAlbum {
                    library_id: library.id.clone(),
                    id,
                };
                if albums.send(album).is_err() {
                    debug!("album workers stopped");
                    pass.set_incomplete();
                    return Ok(());
                }
            }
        }
        info!(albums = listed, "library listed");
    }
    Ok(())
}

fn sync_albums(
    provider: &dyn Provider,
    pool: &Pool,
    pass: &Pass,
    albums: Receiver<ListedAlbum>,
    pending: Sender<Track>,
) {
    for album in albums {
        let _span = info_span!("album", id = %album.id).entered();
        let tracks = match sync_album(provider, pool, pass, &album) {
            Ok(tracks) => tracks,
            Err(e) => {
                pass.set_incomplete();
                warn!("album sync failed, will retry next pass: {e:#}");
                continue;
            }
        };
        pass.found_unanalyzed(tracks.len());
        for track in tracks {
            if pending.send(track).is_err() {
                pass.set_incomplete();
                return;
            }
        }
    }
}

fn sync_album(
    provider: &dyn Provider,
    pool: &Pool,
    pass: &Pass,
    listed: &ListedAlbum,
) -> Result<Vec<Track>> {
    let Some(album) = provider.album(&listed.id)? else {
        debug!("album deleted since it was listed");
        pass.seen_albums.lock().unwrap().remove(&listed.id);
        return Ok(Vec::new());
    };
    pass.seen_tracks
        .lock()
        .unwrap()
        .extend(album.tracks.iter().map(|t| t.id.clone()));
    let synced = sync::album(&mut *pool.get()?, &listed.library_id, &album)?;
    let (added, updated, pending) = (
        synced.tracks_added,
        synced.tracks_updated,
        synced.pending.len(),
    );
    if synced.changed {
        info!(name = %album.name, added, updated, pending, "album synced");
    } else {
        debug!(name = %album.name, pending, "album unchanged");
    }
    Ok(synced.pending)
}

fn decode_tracks(provider: &dyn Provider, pending: Receiver<Track>, decoded: Sender<Decoded>) {
    let extractor = PatchExtractor::new();
    for track in pending {
        let _span = info_span!("track", id = %track.id).entered();
        let started = Instant::now();
        let (file, bytes) = match download(provider, &track.id) {
            Ok(downloaded) => downloaded,
            Err(e) => {
                warn!("download failed, will retry next pass: {e:#}");
                continue;
            }
        };
        let downloaded = started.elapsed();
        let patches =
            extractor.extract_patches(file.path(), track.duration.try_into().unwrap_or(0));
        drop(file);
        debug!(
            bytes,
            download_ms = downloaded.as_millis(),
            decode_ms = (started.elapsed() - downloaded).as_millis(),
            patches = patches.as_ref().map_or(0, |p| p.patch_count),
            "decoded"
        );
        if decoded.send(Decoded { track, patches }).is_err() {
            return;
        }
    }
}

fn download(provider: &dyn Provider, track_id: &str) -> Result<(NamedTempFile, u64)> {
    let mut file = audio::create_temp_file().context("creating a temporary file")?;
    let bytes = provider.fetch_audio(track_id, &mut file)?;
    Ok((file, bytes))
}

fn analyze(pool: &Pool, decoded: Receiver<Decoded>) -> Result<usize> {
    let Ok(first) = decoded.recv() else {
        return Ok(0);
    };
    info!(path = MODEL_PATH, "loading model");
    let mut model = Model::load(Path::new(MODEL_PATH))?;
    let style_ids = store::register_styles(&mut *pool.get()?, model.labels())?;

    let (mut stored, mut analyzed_since_index) = (0, 0);
    for Decoded { track, patches } in iter::once(first).chain(decoded) {
        let _span = info_span!("track", id = %track.id).entered();
        let result = match patches {
            Ok(patches) => Some(model.analyze(patches)?)
                .filter(|analysis| analysis.embedding.iter().all(|x| x.is_finite()))
                .ok_or_else(|| "the model output NaN or infinite values".to_string()),
            Err(e) => Err(format!("{e:#}")),
        };
        match &result {
            Ok(analysis) => {
                info!(artist = %track.artist, title = %track.title, styles = %top_styles(analysis), "analyzed");
                analyzed_since_index += 1;
            }
            Err(e) => warn!(artist = %track.artist, title = %track.title, "couldn't analyze: {e}"),
        }
        store::save_analysis(
            &mut *pool.get()?,
            model.name(),
            &track.id,
            result,
            &style_ids,
        )?;
        stored += 1;
        if analyzed_since_index == INDEX_EVERY {
            index::update(&mut *pool.get()?)?;
            analyzed_since_index = 0;
        }
    }
    Ok(stored)
}

fn top_styles(analysis: &Analysis) -> String {
    let top: Vec<String> = analysis
        .styles
        .iter()
        .take(3)
        .map(|(style, score)| format!("{style} {score:.2}"))
        .collect();
    top.join(", ")
}

fn remove_missing(provider: &dyn Provider, pool: &Pool, pass: &Pass) -> Result<()> {
    let conn = &mut *pool.get()?;
    let libraries: Vec<String> = pass
        .seen_libraries
        .lock()
        .unwrap()
        .iter()
        .cloned()
        .collect();
    let mut seen_albums = pass.seen_albums.lock().unwrap();
    let mut seen_tracks = pass.seen_tracks.lock().unwrap();

    for (id, library_id) in sync::stored_albums(conn, &libraries)? {
        if seen_albums.contains(&id) {
            continue;
        }
        let Some(album) = provider.album(&id)? else {
            continue;
        };
        info!(id, name = %album.name, "found an album the listing missed, it will be analyzed next pass");
        seen_tracks.extend(album.tracks.iter().map(|t| t.id.clone()));
        sync::album(conn, &library_id, &album)?;
        seen_albums.insert(id);
    }
    sync::remove_missing(conn, &libraries, &seen_albums, &seen_tracks)
}
