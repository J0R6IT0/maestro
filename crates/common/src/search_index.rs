//! The search index is a compact copy of the vector index for path searches, stored in a file next to
//! the database. The analyzer rewrites it after each index update. The server opens it per request, so
//! a path costs one sequential read (or a few page reads if it is too large) and no memory between requests.
//!
//! A node is the track's vector compressed to a few int8 values along the principal components of the library,
//! its nearest neighbors as links to other nodes, and its duration. Nodes are stored so that similar tracks
//! are next to each other.
//!
//! Layout, little-endian:
//! - header: MAGIC, VERSION, node count, dims, fanout (u32 each), then per dim the value of one
//!   code unit (f32)
//! - nodes: track idx (i32), duration in seconds (i32), `fanout` links (u32 node numbers, NO_LINK
//!   past the last one), `dims` codes (i8)
//! - lookup: per node, its idx (i32) and node number (u32), sorted by idx

use crate::{db::DATABASE_PATH, vector::normalize};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Read, Write},
    ops::Deref,
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 4] = b"MSRC";
const VERSION: u32 = 2;
const HEADER_FIELDS: usize = 5;
const NO_LINK: u32 = u32::MAX;
const IDX_AT: usize = 0;
const DURATION_AT: usize = 4;
const LINKS_AT: usize = 8;
/// Indexes up to this size are read whole, larger ones mapped.
const READ_WHOLE_UP_TO: u64 = 128 << 20;

pub fn path() -> PathBuf {
    fs::canonicalize(DATABASE_PATH)
        .unwrap_or_else(|_| DATABASE_PATH.into())
        .with_extension("search")
}

/// The position of a node in the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Node(u32);

pub struct NodeRecord<'a> {
    pub idx: i32,
    pub duration: i32,
    /// Node numbers in the order nodes are writen.
    pub links: &'a [u32],
    pub code: &'a [i8],
}

pub fn encode(projection: f32, scale: f32) -> i8 {
    (projection / scale).round().clamp(-127.0, 127.0) as i8
}

pub fn decode(codes: impl IntoIterator<Item = i8>, scales: &[f32]) -> Vec<f32> {
    let mut vector: Vec<f32> = codes
        .into_iter()
        .zip(scales)
        .map(|(code, scale)| code as f32 * scale)
        .collect();
    normalize(&mut vector);
    vector
}

pub fn write(path: &Path, scales: &[f32], fanout: usize, nodes: &[NodeRecord]) -> io::Result<()> {
    let tmp = path.with_extension("search.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);
    out.write_all(MAGIC)?;
    for field in [
        VERSION,
        nodes.len() as u32,
        scales.len() as u32,
        fanout as u32,
    ] {
        out.write_all(&field.to_le_bytes())?;
    }
    for scale in scales {
        out.write_all(&scale.to_le_bytes())?;
    }
    for node in nodes {
        out.write_all(&node.idx.to_le_bytes())?;
        out.write_all(&node.duration.to_le_bytes())?;
        let padding = fanout - node.links.len();
        for link in node
            .links
            .iter()
            .chain(std::iter::repeat_n(&NO_LINK, padding))
        {
            out.write_all(&link.to_le_bytes())?;
        }
        out.write_all(&node.code.iter().map(|&c| c as u8).collect::<Vec<_>>())?;
    }
    let mut lookup: Vec<(i32, u32)> = (0..).zip(nodes).map(|(n, node)| (node.idx, n)).collect();
    lookup.sort_unstable();
    for (idx, node) in lookup {
        out.write_all(&idx.to_le_bytes())?;
        out.write_all(&node.to_le_bytes())?;
    }
    out.into_inner()?.sync_all()?;
    fs::rename(tmp, path)
}

pub fn is_current(path: &Path) -> bool {
    let mut header = [0; 8];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .is_ok_and(|()| &header[..4] == MAGIC && u32_at(&header, 4) == VERSION)
}

pub struct SearchIndex {
    bytes: Bytes,
    nodes: usize,
    fanout: usize,
    scales: Vec<f32>,
}

impl SearchIndex {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let bytes = if file.metadata()?.len() <= READ_WHOLE_UP_TO {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            Bytes::Read(bytes)
        } else {
            // SAFETY: the index is replaced by renaming a new file over it, never written in place.
            Bytes::Mapped(unsafe { memmap2::Mmap::map(&file)? })
        };
        Self::parse(bytes)
    }

    fn parse(bytes: Bytes) -> io::Result<Self> {
        let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
        if bytes.len() < 4 * HEADER_FIELDS || &bytes[..4] != MAGIC {
            return Err(invalid("not a search index"));
        }
        let field = |n: usize| u32_at(&bytes, 4 * n) as usize;
        if field(1) != VERSION as usize {
            return Err(invalid("unsupported search index version"));
        }
        let (nodes, dims, fanout) = (field(2), field(3), field(4));
        if bytes.len() < 4 * (HEADER_FIELDS + dims) {
            return Err(invalid("truncated search index"));
        }
        let scales: Vec<f32> = (0..dims)
            .map(|d| f32::from_bits(u32_at(&bytes, 4 * (HEADER_FIELDS + d))))
            .collect();
        let index = Self {
            bytes,
            nodes,
            fanout,
            scales,
        };
        if index.bytes.len() != index.lookup_start() + 8 * nodes {
            return Err(invalid("truncated search index"));
        }
        Ok(index)
    }

    pub fn node(&self, idx: i32) -> Option<Node> {
        let entry = |n: usize| {
            let at = self.lookup_start() + 8 * n;
            (u32_at(&self.bytes, at) as i32, u32_at(&self.bytes, at + 4))
        };
        let (mut low, mut high) = (0, self.nodes);
        while low < high {
            let middle = (low + high) / 2;
            if entry(middle).0 < idx {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (low < self.nodes && entry(low).0 == idx).then(|| Node(entry(low).1))
    }

    pub fn idx(&self, node: Node) -> i32 {
        u32_at(&self.bytes, self.node_start(node) + IDX_AT) as i32
    }

    pub fn duration(&self, node: Node) -> i32 {
        u32_at(&self.bytes, self.node_start(node) + DURATION_AT) as i32
    }

    pub fn links(&self, node: Node) -> impl Iterator<Item = Node> + '_ {
        let start = self.node_start(node) + LINKS_AT;
        (0..self.fanout)
            .map(move |k| u32_at(&self.bytes, start + 4 * k))
            .take_while(|&link| link != NO_LINK)
            .map(Node)
    }

    pub fn vector(&self, node: Node) -> Vec<f32> {
        let start = self.node_start(node) + self.codes_at();
        let codes = &self.bytes[start..start + self.scales.len()];
        decode(codes.iter().map(|&code| code as i8), &self.scales)
    }

    fn codes_at(&self) -> usize {
        LINKS_AT + 4 * self.fanout
    }

    fn node_size(&self) -> usize {
        self.codes_at() + self.scales.len()
    }

    fn node_start(&self, Node(node): Node) -> usize {
        4 * (HEADER_FIELDS + self.scales.len()) + node as usize * self.node_size()
    }

    fn lookup_start(&self) -> usize {
        self.node_start(Node(self.nodes as u32))
    }
}

enum Bytes {
    Read(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Read(bytes) => bytes,
            Self::Mapped(map) => map,
        }
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
