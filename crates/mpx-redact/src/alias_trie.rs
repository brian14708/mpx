use std::ops::Range;

/// A radix trie over issued alias bytes, built from keys in sorted order.
pub(crate) struct AliasTrie {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    labels: Vec<u8>,
}

#[derive(Default)]
struct Node {
    terminal: bool,
    edges: Range<usize>,
}

struct Edge {
    label: Range<usize>,
    child: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum Cursor {
    Node(usize),
    Edge { index: usize, offset: usize },
}

impl Cursor {
    pub(crate) const ROOT: Self = Self::Node(0);
}

impl AliasTrie {
    pub(crate) fn new<'a>(keys: impl Iterator<Item = &'a [u8]>) -> Self {
        let keys: Vec<_> = keys.collect();
        let mut trie = Self {
            nodes: vec![Node::default()],
            edges: Vec::new(),
            labels: Vec::new(),
        };
        let mut pending = vec![(0, 0, keys.as_slice())];
        while let Some((index, depth, mut group)) = pending.pop() {
            let terminal = group.first().is_some_and(|key| key.len() == depth);
            if terminal {
                group = &group[1..];
            }
            let start = trie.edges.len();
            for keys in group.chunk_by(|left, right| left[depth] == right[depth]) {
                let first = &keys[0][depth..];
                let last = &keys[keys.len() - 1][depth..];
                // The first and last keys bound the prefix shared by the whole group.
                let common = first.iter().zip(last).take_while(|(a, b)| a == b).count();
                let label_start = trie.labels.len();
                trie.labels.extend_from_slice(&first[..common]);
                let child = trie.nodes.len();
                trie.nodes.push(Node::default());
                trie.edges.push(Edge {
                    label: label_start..trie.labels.len(),
                    child,
                });
                pending.push((child, depth + common, keys));
            }
            trie.nodes[index] = Node {
                terminal,
                edges: start..trie.edges.len(),
            };
        }
        trie
    }

    pub(crate) fn advance(&self, mut cursor: Cursor, mut bytes: &[u8]) -> Option<Cursor> {
        while let Some(&first) = bytes.first() {
            let (index, offset) = match cursor {
                Cursor::Node(node) => {
                    let edges = &self.nodes[node].edges;
                    let index = self.edges[edges.clone()]
                        .binary_search_by_key(&first, |edge| self.labels[edge.label.start])
                        .ok()?;
                    (edges.start + index, 0)
                }
                Cursor::Edge { index, offset } => (index, offset),
            };
            let edge = &self.edges[index];
            let label = &self.labels[edge.label.start + offset..edge.label.end];
            let shared = label.len().min(bytes.len());
            if label[..shared] != bytes[..shared] {
                return None;
            }
            cursor = if shared == label.len() {
                Cursor::Node(edge.child)
            } else {
                Cursor::Edge {
                    index,
                    offset: offset + shared,
                }
            };
            bytes = &bytes[shared..];
        }
        Some(cursor)
    }

    pub(crate) fn is_terminal(&self, cursor: Cursor) -> bool {
        matches!(cursor, Cursor::Node(node) if self.nodes[node].terminal)
    }
}
