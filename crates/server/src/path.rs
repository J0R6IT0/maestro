//! A path of at most N tracks from song A to song B (both included), where each step changes
//! the sound gradually.
//!
//! A straight line from A to B runs through empty space and the nearest track to any point on it
//! is usually A or B itself. Instead, the path follows the library through the neighbor graph of
//! the search index (see `common::search_index`).
//!
//! 1. Route: search the graph from A to B and from B to A, then take the cheapest route from
//!    A to B through the tracks those searches saw. A step costs its distance squared, so one
//!    big jump costs more than several small ones. If no route is found, retry with a wider search.
//!    If that fails too, the route is just A -> B.
//! 2. Targets: N points evenly spaced along the route. The first is A and the last is B.
//! 3. Corridor: for each target, a beam search finds the BEAM_WIDTH nearest tracks within the
//!    duration limit. Targets are searched alternately from both ends towards the middle,
//!    and each search starts from the nearest tracks seen so far, so every search is local.
//! 4. Path: every track found (route included) is a candidate for the target it's nearest to.
//!    One candidate per target is chosen jointly (Viterbi), so that the sum over the path of
//!    (distance from the previous track)² + (distance to the target)² is lowest. If the filter
//!    drops tracks on that path, they are removed from the candidates and the path is chosen again.
//!    A target that no candidate is nearest to is skipped, so the path can be shorter than N but
//!    never longer.
//!
//! Cost is about N · BEAM_WIDTH² per filter round, independently of library size.

use crate::{
    filter::Filter,
    store::{self, Track},
};
use anyhow::Result;
use common::{
    search_index::{Node, SearchIndex},
    vector::{distance, dot},
};
use diesel::sqlite::SqliteConnection;
use ndarray::{Array2, ArrayView1};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
};

const BEAM_WIDTH: usize = 64;
const ROUTE_RETRY_FACTOR: usize = 4;

pub struct Path {
    /// The tracks from A to B, each with its distance to A.
    pub tracks: Vec<(f32, Track)>,
    pub route_len: usize,
    pub seen_count: usize,
    pub candidate_count: usize,
}

pub fn find(
    conn: &mut SqliteConnection,
    filter: &Filter,
    index: &SearchIndex,
    a: Node,
    b: Node,
    n: usize,
) -> Result<Path> {
    // If A and B are the same audio, there is nothing in between.
    if n <= 2 || index.vector(a) == index.vector(b) {
        return Ok(Path {
            tracks: load_tracks(conn, index, &[a, b])?,
            route_len: 2,
            seen_count: 0,
            candidate_count: 0,
        });
    }

    let (route, route_seen) = find_route(index, a, b);
    let points: Vec<Vec<f32>> = route.iter().map(|&node| index.vector(node)).collect();
    let max_duration = filter.max_duration(index.duration(a).max(index.duration(b)));
    let mut explorer = Explorer::new(index, evenly_spaced(&points, n), BEAM_WIDTH, max_duration);
    for &node in &route {
        explorer.measure(node);
    }
    let mut found: HashSet<Node> = route.iter().copied().collect();
    for target in ends_to_middle(n) {
        found.extend(explorer.search(target, None));
    }

    let candidates = explorer.candidates(&found, a, b, n);
    let candidate_count = candidates.len();
    let chosen = choose_filtered(conn, filter, &explorer, a, b, n, candidates)?;
    Ok(Path {
        tracks: load_tracks(conn, index, &chosen)?,
        route_len: route.len(),
        seen_count: route_seen + explorer.seen.len(),
        candidate_count,
    })
}

/// The cheapest route from `a` to `b` along neighbor links, and how many tracks were seen.
fn find_route(index: &SearchIndex, a: Node, b: Node) -> (Vec<Node>, usize) {
    let flat = [index.vector(a), index.vector(b)].concat();
    let end_vectors = Array2::from_shape_vec((2, flat.len() / 2), flat).unwrap();
    let mut seen = 0;
    for width in [BEAM_WIDTH, ROUTE_RETRY_FACTOR * BEAM_WIDTH] {
        let mut explored = Explorer::new(index, end_vectors.clone(), width, None);
        explored.search(1, Some(a));
        explored.search(0, Some(b));
        seen += explored.seen.len();
        if let Some(route) = cheapest_route(&explored.links(), a, b) {
            return (route, seen);
        }
    }
    (vec![a, b], seen)
}

/// Dijkstra over `links`, where each step costs its distance squared.
fn cheapest_route(links: &HashMap<Node, Vec<(Node, f32)>>, a: Node, b: Node) -> Option<Vec<Node>> {
    let mut best: HashMap<Node, (f32, Node)> = HashMap::from([(a, (0.0, a))]);
    let mut queue = BinaryHeap::from([Reverse(Scored {
        distance: 0.0,
        node: a,
    })]);
    while let Some(Reverse(Scored {
        distance: cost,
        node: u,
    })) = queue.pop()
    {
        if u == b {
            break;
        }
        if cost > best[&u].0 {
            continue;
        }
        for &(v, step) in links.get(&u).into_iter().flatten() {
            let cost_via_u = cost + step * step;
            if best.get(&v).is_none_or(|&(old, _)| cost_via_u < old) {
                best.insert(v, (cost_via_u, u));
                queue.push(Reverse(Scored {
                    distance: cost_via_u,
                    node: v,
                }));
            }
        }
    }

    best.contains_key(&b).then(|| {
        let mut route = vec![b];
        while *route.last().unwrap() != a {
            route.push(best[route.last().unwrap()].1);
        }
        route.reverse();
        route
    })
}

/// `n` unit vectors evenly spaced along the route, following the great circle between
/// consecutive points. The first and last rows are the route's ends.
fn evenly_spaced(points: &[Vec<f32>], n: usize) -> Array2<f32> {
    let angles: Vec<f32> = points
        .windows(2)
        .map(|w| dot(&w[0], &w[1]).clamp(-1.0, 1.0).acos())
        .collect();
    let total: f32 = angles.iter().sum();
    let mut out = Array2::zeros((n, points[0].len()));
    let (mut segment, mut segment_start) = (0, 0.0);
    for i in 0..n {
        let at = total * i as f32 / (n - 1) as f32;
        while segment + 1 < angles.len() && at > segment_start + angles[segment] {
            segment_start += angles[segment];
            segment += 1;
        }
        let (p, q, theta) = (&points[segment], &points[segment + 1], angles[segment]);
        let (wp, wq) = if theta == 0.0 {
            (1.0, 0.0)
        } else {
            let s = ((at - segment_start) / theta).clamp(0.0, 1.0);
            let sin = theta.sin();
            (((1.0 - s) * theta).sin() / sin, (s * theta).sin() / sin)
        };
        for (k, o) in out.row_mut(i).iter_mut().enumerate() {
            *o = wp * p[k] + wq * q[k];
        }
    }
    out
}

fn ends_to_middle(n: usize) -> impl Iterator<Item = usize> {
    let inner = 1..n - 1;
    inner
        .clone()
        .zip(inner.rev())
        .flat_map(|(low, high)| [low, high])
        .take(n - 2)
}

#[derive(Clone, Copy)]
struct Candidate {
    node: Node,
    target: usize,
    target_distance: f32,
}

fn choose_filtered(
    conn: &mut SqliteConnection,
    filter: &Filter,
    explorer: &Explorer,
    a: Node,
    b: Node,
    n: usize,
    mut candidates: Vec<Candidate>,
) -> Result<Vec<Node>> {
    let index = explorer.index;
    let end_idxs = [index.idx(a), index.idx(b)];
    loop {
        let path = choose(explorer, a, b, n, &candidates);
        let inner: HashSet<Node> = path[1..path.len() - 1].iter().copied().collect();
        let preferred: Vec<i32> = candidates
            .iter()
            .filter(|c| inner.contains(&c.node))
            .map(|c| index.idx(c.node))
            .collect();
        let kept: HashSet<i32> = filter
            .apply(conn, &end_idxs, &preferred, usize::MAX)?
            .into_iter()
            .map(|(idx, _)| idx)
            .collect();
        if kept.len() == preferred.len() {
            return Ok(path);
        }
        candidates.retain(|c| !inner.contains(&c.node) || kept.contains(&index.idx(c.node)));
    }
}

/// Picks one candidate per target between A and B (Viterbi).
fn choose(explorer: &Explorer, a: Node, b: Node, n: usize, candidates: &[Candidate]) -> Vec<Node> {
    let mut slots: Vec<Vec<(Node, f32)>> = vec![Vec::new(); n];
    slots[0].push((a, 0.0));
    for candidate in candidates {
        slots[candidate.target].push((candidate.node, candidate.target_distance));
    }
    slots[n - 1].push((b, 0.0));
    slots.retain(|slot| !slot.is_empty());

    let vectors: Vec<Array2<f32>> = slots
        .iter()
        .map(|slot| explorer.vectors(slot.iter().map(|&(node, _)| node)))
        .collect();
    let to_previous: Vec<Array2<f32>> = vectors
        .windows(2)
        .map(|pair| pair[1].dot(&pair[0].t()).mapv(distance))
        .collect();
    viterbi(&slots, &to_previous)
}

/// Picks one candidate per slot, minimizing the sum of squared step distances and squared target
/// distances. Each candidate is (node, distance to the slot's target), and `to_previous[s - 1]`
/// holds the distances from slot s's candidates (rows) to slot s - 1's (columns). Returns the
/// chosen nodes (the first and last slots are A and B).
fn viterbi(slots: &[Vec<(Node, f32)>], to_previous: &[Array2<f32>]) -> Vec<Node> {
    let mut best: Vec<Vec<(f32, usize)>> = vec![vec![(0.0, 0)]];
    for s in 1..slots.len() {
        let previous = &best[s - 1];
        let costs = slots[s]
            .iter()
            .zip(to_previous[s - 1].rows())
            .map(|(&(_, to_target), steps)| {
                let mut cheapest = (f32::INFINITY, 0);
                for (k, (&(previous_cost, _), &step)) in previous.iter().zip(steps).enumerate() {
                    let cost = previous_cost + step * step + to_target * to_target;
                    if cost < cheapest.0 {
                        cheapest = (cost, k);
                    }
                }
                cheapest
            })
            .collect();
        best.push(costs);
    }

    let mut k = 0;
    let mut path = Vec::with_capacity(slots.len());
    for s in (0..slots.len()).rev() {
        path.push(slots[s][k].0);
        k = best[s][k].1;
    }
    path.reverse();
    path
}

fn load_tracks(
    conn: &mut SqliteConnection,
    index: &SearchIndex,
    path: &[Node],
) -> Result<Vec<(f32, Track)>> {
    let idx: Vec<i32> = path.iter().map(|&node| index.idx(node)).collect();
    let mut tracks = store::tracks_by_idx(conn, &idx)?;
    let start = index.vector(path[0]);
    Ok(path
        .iter()
        .zip(&idx)
        .filter_map(|(&node, idx)| {
            let track = tracks.remove(idx)?;
            Some((distance(dot(&start, &index.vector(node))), track))
        })
        .collect())
}

/// A node with its distance, ordered by distance (then by node).
#[derive(Clone, Copy, PartialEq)]
struct Scored {
    distance: f32,
    node: Node,
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.node.cmp(&other.node))
    }
}

struct Nearest {
    width: usize,
    heap: BinaryHeap<Scored>,
}

impl Nearest {
    fn new(width: usize) -> Self {
        Self {
            width,
            heap: BinaryHeap::with_capacity(width + 1),
        }
    }

    fn bound(&self) -> f32 {
        match self.heap.peek() {
            Some(farthest) if self.heap.len() >= self.width => farthest.distance,
            _ => f32::INFINITY,
        }
    }

    fn offer(&mut self, track: Scored) -> bool {
        if track.distance >= self.bound() {
            return false;
        }
        self.heap.push(track);
        if self.heap.len() > self.width {
            self.heap.pop();
        }
        true
    }

    fn into_nodes(self) -> Vec<Node> {
        self.heap.into_iter().map(|s| s.node).collect()
    }
}

/// A track's vector and its distance to each target.
struct Measured {
    vector: Vec<f32>,
    to_targets: Vec<f32>,
}

/// Runs beam searches around a set of target points and keeps every track it measured.
struct Explorer<'a> {
    index: &'a SearchIndex,
    /// One target per row.
    targets: Array2<f32>,
    width: usize,
    max_duration: Option<i32>,
    seen: HashMap<Node, Measured>,
}

impl<'a> Explorer<'a> {
    fn new(
        index: &'a SearchIndex,
        targets: Array2<f32>,
        width: usize,
        max_duration: Option<i32>,
    ) -> Self {
        Self {
            index,
            targets,
            width,
            max_duration,
            seen: HashMap::new(),
        }
    }

    fn within_max_duration(&self, node: Node) -> bool {
        self.max_duration
            .is_none_or(|max| self.index.duration(node) <= max)
    }

    fn measure(&mut self, node: Node) -> &Measured {
        let (index, targets) = (self.index, &self.targets);
        self.seen.entry(node).or_insert_with(|| {
            let vector = index.vector(node);
            let to_targets = targets
                .dot(&ArrayView1::from(&vector))
                .mapv(distance)
                .to_vec();
            Measured { vector, to_targets }
        })
    }

    /// Beam search over the graph for the `width` tracks nearest to `target` that fit the max
    /// duration. It starts from `seed`, or from the nearest tracks seen so far.
    fn search(&mut self, target: usize, seed: Option<Node>) -> Vec<Node> {
        let (index, width) = (self.index, self.width);
        let mut seeds: Vec<Scored> = match seed {
            Some(node) => vec![Scored {
                distance: self.measure(node).to_targets[target],
                node,
            }],
            None => self
                .seen
                .iter()
                .map(|(&node, seen)| Scored {
                    distance: seen.to_targets[target],
                    node,
                })
                .collect(),
        };
        if seeds.len() > width {
            seeds.select_nth_unstable(width - 1);
            seeds.truncate(width);
        }
        let mut visited: HashSet<Node> = seeds.iter().map(|s| s.node).collect();
        let mut to_expand: BinaryHeap<Reverse<Scored>> =
            seeds.iter().copied().map(Reverse).collect();
        let mut beam = Nearest::new(width);
        let mut found = Nearest::new(width);
        for start in seeds {
            beam.offer(start);
            if self.within_max_duration(start.node) {
                found.offer(start);
            }
        }

        while let Some(Reverse(Scored { distance: d, node })) = to_expand.pop() {
            if d > beam.bound() {
                break;
            }
            for link in index.links(node) {
                if !visited.insert(link) {
                    continue;
                }
                let scored = Scored {
                    distance: self.measure(link).to_targets[target],
                    node: link,
                };
                if self.within_max_duration(link) {
                    found.offer(scored);
                }
                if beam.offer(scored) {
                    to_expand.push(Reverse(scored));
                }
            }
        }
        found.into_nodes()
    }

    /// The neighbor links between seen tracks, in both directions, with their distances.
    fn links(&self) -> HashMap<Node, Vec<(Node, f32)>> {
        let mut links: HashMap<Node, Vec<(Node, f32)>> = HashMap::new();
        for (&u, seen) in &self.seen {
            for v in self.index.links(u) {
                let Some(other) = self.seen.get(&v) else {
                    continue;
                };
                let step = distance(dot(&seen.vector, &other.vector));
                links.entry(u).or_default().push((v, step));
                links.entry(v).or_default().push((u, step));
            }
        }
        links
    }

    /// Every track in `found` that fits the max duration, except A and B, becomes a candidate for
    /// the inner target it's nearest to, so no track can be picked twice. Sorted by distance, then
    /// by node, so the same request always gives the same path.
    fn candidates(&self, found: &HashSet<Node>, a: Node, b: Node, n: usize) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = found
            .iter()
            .filter(|&&node| node != a && node != b && self.within_max_duration(node))
            .map(|&node| {
                let to_targets = &self.seen[&node].to_targets;
                let target = (1..n - 1)
                    .min_by(|&x, &y| to_targets[x].total_cmp(&to_targets[y]))
                    .unwrap();
                Candidate {
                    node,
                    target,
                    target_distance: to_targets[target],
                }
            })
            .collect();
        candidates.sort_by(|x, y| {
            x.target_distance
                .total_cmp(&y.target_distance)
                .then(x.node.cmp(&y.node))
        });
        candidates
    }

    /// The vectors of the seen tracks `nodes`, one per row.
    fn vectors(&self, nodes: impl Iterator<Item = Node>) -> Array2<f32> {
        let vectors: Vec<f32> = nodes
            .flat_map(|node| self.seen[&node].vector.iter().copied())
            .collect();
        let dims = self.targets.ncols();
        Array2::from_shape_vec((vectors.len() / dims, dims), vectors).unwrap()
    }
}
