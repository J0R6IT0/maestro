use crate::store::{self, Track};
use anyhow::Result;
use common::{db::raw_embeddings, env, vector::raw_distance};
use diesel::sqlite::SqliteConnection;
use std::collections::HashMap;

const SAME_AUDIO: f32 = 0.011;

#[derive(Clone, Copy)]
pub struct Filter {
    versions: Option<(Match, Level)>,
    max_duration_increase: Option<f64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Match {
    Title,
    TitleAndArtist,
}

#[derive(Clone, Copy)]
enum Level {
    Recording,
    Song,
}

impl Level {
    fn max_distance(self) -> f32 {
        match self {
            Self::Recording => 0.1,
            Self::Song => 0.3,
        }
    }
}

impl Filter {
    pub fn from_env() -> Result<Self, env::Error> {
        Ok(Self {
            versions: versions()?,
            max_duration_increase: max_duration_increase()?,
        })
    }

    pub fn max_duration(&self, longest_seed: i32) -> Option<i32> {
        let increase = self.max_duration_increase?;
        (longest_seed > 0).then(|| (f64::from(longest_seed) * (1.0 + increase)).floor() as i32)
    }

    pub fn apply(
        &self,
        conn: &mut SqliteConnection,
        seeds: &[i32],
        candidates: &[i32],
        limit: usize,
    ) -> Result<Vec<(i32, Track)>> {
        let all: Vec<i32> = seeds.iter().chain(candidates).copied().collect();
        let mut tracks = store::tracks_by_idx(conn, &all)?;
        let longest_seed = seeds
            .iter()
            .filter_map(|i| tracks.get(i))
            .map(|t| t.duration)
            .max();
        let max_duration = self.max_duration(longest_seed.unwrap_or(0));
        let duplicates = Duplicates::load(conn, &all, &tracks, self.versions)?;

        let mut taken: Vec<i32> = seeds
            .iter()
            .copied()
            .filter(|i| tracks.contains_key(i))
            .collect();
        let mut kept = Vec::new();
        for &i in candidates {
            if kept.len() == limit {
                break;
            }
            let Some(track) = tracks.get(&i) else {
                continue;
            };
            let is_too_long = max_duration.is_some_and(|max| track.duration > max);
            if is_too_long || taken.iter().any(|&j| duplicates.are(i, j)) {
                continue;
            }
            taken.push(i);
            kept.push(i);
        }
        Ok(kept
            .into_iter()
            .filter_map(|i| Some((i, tracks.remove(&i)?)))
            .collect())
    }

    pub fn neighbors(
        &self,
        conn: &mut SqliteConnection,
        id: &str,
        n: usize,
    ) -> Result<Option<Vec<(f32, Track)>>> {
        let Some((seed, list)) = store::neighbors(conn, id)? else {
            return Ok(None);
        };

        let distances: HashMap<i32, f32> = list.0.iter().map(|n| (n.idx, n.distance)).collect();
        let ids: Vec<i32> = list.0.iter().map(|n| n.idx).collect();
        let mut considered = (2 * n).min(ids.len());

        let kept = loop {
            let kept = self.apply(conn, &[seed], &ids[..considered], n)?;
            if kept.len() == n || considered == ids.len() {
                break kept;
            }
            considered = ids.len();
        };

        Ok(Some(
            kept.into_iter()
                .map(|(i, track)| (distances[&i], track))
                .collect(),
        ))
    }
}

struct Duplicates {
    near_identical_pairs: HashMap<(i32, i32), f32>,
    versions: Option<Versions>,
}

impl Duplicates {
    fn load(
        conn: &mut SqliteConnection,
        idx: &[i32],
        tracks: &HashMap<i32, Track>,
        versions: Option<(Match, Level)>,
    ) -> Result<Self> {
        Ok(Self {
            near_identical_pairs: store::near_identical_pairs(conn, idx)?,
            versions: match versions {
                Some((matching, level)) => Some(Versions::load(conn, tracks, matching, level)?),
                None => None,
            },
        })
    }

    fn are(&self, i: i32, j: i32) -> bool {
        let same_audio = self
            .near_identical_pairs
            .get(&(i.min(j), i.max(j)))
            .is_some_and(|&distance| distance < SAME_AUDIO);
        same_audio || self.versions.as_ref().is_some_and(|v| v.are(i, j))
    }
}

struct Versions {
    matching: Match,
    max_distance: f32,
    titles: HashMap<i32, String>,
    embeddings: HashMap<i32, Vec<f32>>,
    artist_ids: HashMap<i32, Vec<String>>,
    artist_names: HashMap<i32, String>,
}

impl Versions {
    fn load(
        conn: &mut SqliteConnection,
        tracks: &HashMap<i32, Track>,
        matching: Match,
        level: Level,
    ) -> Result<Self> {
        let titles: HashMap<i32, String> = tracks
            .iter()
            .map(|(&i, t)| (i, title_key(&t.title)))
            .collect();
        let shared = with_shared_title(&titles);
        let (artist_ids, artist_names) = match matching {
            Match::TitleAndArtist => (
                store::artist_ids(conn, &shared)?,
                shared
                    .iter()
                    .map(|&i| (i, tracks[&i].artist.trim().to_lowercase()))
                    .collect(),
            ),
            Match::Title => (HashMap::new(), HashMap::new()),
        };
        Ok(Self {
            matching,
            max_distance: level.max_distance(),
            embeddings: raw_embeddings(conn, &shared)?,
            artist_ids,
            artist_names,
            titles,
        })
    }

    fn are(&self, i: i32, j: i32) -> bool {
        self.titles[&i] == self.titles[&j]
            && (self.matching == Match::Title || self.share_artist(i, j))
            && self.sound_alike(i, j)
    }

    fn share_artist(&self, i: i32, j: i32) -> bool {
        if let (Some(a), Some(b)) = (self.artist_ids.get(&i), self.artist_ids.get(&j)) {
            return a.iter().any(|id| b.contains(id));
        }
        let (a, b) = (&self.artist_names[&i], &self.artist_names[&j]);
        !a.is_empty() && a == b
    }

    fn sound_alike(&self, i: i32, j: i32) -> bool {
        self.embeddings
            .get(&i)
            .zip(self.embeddings.get(&j))
            .is_some_and(|(a, b)| raw_distance(a, b) < self.max_distance)
    }
}

fn versions() -> Result<Option<(Match, Level)>, env::Error> {
    let strategy = env::var_or("DEDUP_STRATEGY", "title+artist");
    let matching = match strategy.trim().to_lowercase().as_str() {
        "title+artist" => Match::TitleAndArtist,
        "title" => Match::Title,
        "off" => return Ok(None),
        _ => {
            let reason = "expected title+artist, title or off";
            return Err(env::invalid("DEDUP_STRATEGY", strategy, reason));
        }
    };
    let level = env::var_or("DEDUP_LEVEL", "recording");
    let level = match level.trim().to_lowercase().as_str() {
        "recording" => Level::Recording,
        "song" => Level::Song,
        _ => {
            return Err(env::invalid(
                "DEDUP_LEVEL",
                level,
                "expected recording or song",
            ));
        }
    };
    Ok(Some((matching, level)))
}

fn max_duration_increase() -> Result<Option<f64>, env::Error> {
    const NAME: &str = "MAX_DURATION_INCREASE";
    let Some(value) = env::var(NAME) else {
        return Ok(None);
    };
    match value.trim().trim_end_matches('%').trim().parse::<f64>() {
        Ok(percent) if percent >= 0.0 && percent.is_finite() => Ok(Some(percent / 100.0)),
        _ => Err(env::invalid(
            NAME,
            value,
            "expected a percentage, e.g. 50 or 50%",
        )),
    }
}

fn with_shared_title(titles: &HashMap<i32, String>) -> Vec<i32> {
    let mut count: HashMap<&str, usize> = HashMap::new();
    for title in titles.values() {
        *count.entry(title).or_default() += 1;
    }
    titles
        .iter()
        .filter(|(_, title)| count[title.as_str()] > 1)
        .map(|(&i, _)| i)
        .collect()
}

fn title_key(title: &str) -> String {
    let lower = title.to_lowercase();
    let (mut plain, mut depth) = (String::new(), 0u32);
    for c in lower.chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => plain.push(c),
            _ => {}
        }
    }
    let mut cut = plain.as_str();
    for sep in [" - ", " – ", " — ", " feat. ", " ft. ", " featuring "] {
        if let Some(at) = cut.find(sep) {
            cut = &cut[..at];
        }
    }
    let mut key = words(cut);
    if key.is_empty() {
        key = words(&lower);
    }
    if key.is_empty() {
        key = lower.trim().to_string();
    }
    let parts = part_numbers(&lower);
    if parts.is_empty() {
        key
    } else {
        format!("{key} #{}", parts.join(" "))
    }
}

fn part_numbers(lower: &str) -> Vec<&str> {
    const MARKERS: [&str; 10] = [
        "part", "pt", "vol", "volume", "op", "chapter", "movement", "mvt", "act", "book",
    ];
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let mut numbers: Vec<&str> = words
        .windows(2)
        .filter(|w| MARKERS.contains(&w[0]) && is_numeral(w[1]))
        .map(|w| w[1])
        .collect();
    for sep in [" - ", " – ", " — "] {
        if let Some((_, after)) = lower.split_once(sep)
            && let Some((first, _)) = after.split_once(". ")
            && is_numeral(first)
        {
            numbers.push(first);
        }
    }
    numbers
}

fn is_numeral(word: &str) -> bool {
    const WORDS: [&str; 10] = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ];
    !word.is_empty()
        && (word.chars().all(|c| c.is_ascii_digit())
            || word.chars().all(|c| "ivxlc".contains(c))
            || WORDS.contains(&word))
}

fn words(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
