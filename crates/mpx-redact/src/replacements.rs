use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Bound::{Excluded, Unbounded},
    sync::Arc,
};

/// The reversible tags and aliases emitted during an exchange, paired with their originals.
#[derive(Clone, Default)]
pub struct ReplacementMap(Arc<Table>);

#[derive(Clone, Default)]
struct Table {
    entries: BTreeMap<String, String>,
    index: RestorationIndex,
}

#[derive(Clone, Default)]
pub(crate) struct RestorationIndex {
    pub(crate) alias_limit: usize,
    pub(crate) marker_lengths: BTreeSet<usize>,
    pub(crate) marker_prefix: Option<String>,
    pub(crate) marker_starts: BTreeSet<char>,
}

impl fmt::Debug for ReplacementMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplacementMap")
            .field("entry_count", &self.0.entries.len())
            .finish_non_exhaustive()
    }
}

impl ReplacementMap {
    /// Looks up the original text for an exact emitted tag or alias.
    #[must_use]
    pub fn get(&self, replacement: &str) -> Option<&str> {
        self.0.entries.get(replacement).map(String::as_str)
    }

    /// Iterates over the recorded pairs sorted by replacement text.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&str, &str)> {
        self.0
            .entries
            .iter()
            .map(|(replacement, original)| (replacement.as_str(), original.as_str()))
    }

    pub(crate) fn has_longer_with_prefix(&self, prefix: &str) -> bool {
        self.0
            .entries
            .range::<str, _>((Excluded(prefix), Unbounded))
            .next()
            .is_some_and(|(replacement, _)| replacement.starts_with(prefix))
    }

    pub(crate) fn index(&self) -> &RestorationIndex {
        &self.0.index
    }

    pub(crate) fn extend(&mut self, entries: BTreeMap<String, String>, marker_prefix: &str) {
        if entries.is_empty() {
            return;
        }
        let table = Arc::make_mut(&mut self.0);
        let index = &mut table.index;
        let had_markers = !index.marker_lengths.is_empty();
        let mut any_marker = false;
        for replacement in entries.keys() {
            if replacement.starts_with(marker_prefix) {
                any_marker = true;
                index.marker_lengths.insert(replacement.len());
            } else {
                index.alias_limit = index.alias_limit.max(replacement.len());
            }
        }
        if any_marker {
            if !had_markers {
                index.marker_prefix = Some(marker_prefix.to_owned());
            } else if index.marker_prefix.as_deref() != Some(marker_prefix) {
                index.marker_prefix = None;
            }
            index.marker_starts.extend(marker_prefix.chars().take(1));
        }
        table.entries.extend(entries);
    }
}
