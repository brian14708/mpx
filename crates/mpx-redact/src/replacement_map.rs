use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Bound::{Excluded, Unbounded},
    sync::{Arc, OnceLock},
};

use crate::{alias_trie::AliasTrie, replacement_template::is_alias};

/// The replacements emitted during an exchange, paired with their originals.
///
/// Forward lookups include permanent masks. Reverse lookups include reversible
/// tags and aliases.
#[derive(Clone, Default)]
pub struct ReplacementMap(Arc<Table>);

#[derive(Clone, Default)]
struct Table {
    entries: BTreeMap<String, String>,
    permanent: BTreeMap<String, BTreeSet<String>>,
    index: RestorationIndex,
    alias_trie: OnceLock<Arc<AliasTrie>>,
}

#[derive(Clone, Default)]
pub(crate) struct RestorationIndex {
    pub(crate) alias_lengths: Option<(usize, usize)>,
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

    /// Iterates over distinct `(original, replacement)` pairs, including permanent masks.
    pub fn forward_entries(&self) -> impl Iterator<Item = (&str, &str)> {
        let permanent = self.0.permanent.iter().flat_map(|(original, values)| {
            values
                .iter()
                .map(move |replacement| (original.as_str(), replacement.as_str()))
        });
        self.entries()
            .map(|(replacement, original)| (original, replacement))
            .chain(permanent)
    }

    /// Iterates over the distinct masks emitted for an original value.
    pub fn replacements_for<'a>(&'a self, original: &'a str) -> impl Iterator<Item = &'a str> {
        let permanent = self
            .0
            .permanent
            .get(original)
            .into_iter()
            .flatten()
            .map(String::as_str);
        self.entries()
            .filter(move |&(_, source)| source == original)
            .map(|(replacement, _)| replacement)
            .chain(permanent)
    }

    pub(crate) fn permanent_values(&self) -> impl Iterator<Item = &str> {
        self.0.permanent.values().flatten().map(String::as_str)
    }

    pub(crate) fn extend_permanent(&mut self, entries: BTreeMap<String, BTreeSet<String>>) {
        if entries.is_empty() {
            return;
        }
        let table = Arc::make_mut(&mut self.0);
        for (original, values) in entries {
            table.permanent.entry(original).or_default().extend(values);
        }
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

    pub(crate) fn alias_trie(&self) -> Arc<AliasTrie> {
        Arc::clone(self.0.alias_trie.get_or_init(|| {
            let aliases = self
                .0
                .entries
                .keys()
                .filter(|key| is_alias(key))
                .map(String::as_bytes);
            Arc::new(AliasTrie::new(aliases))
        }))
    }

    pub(crate) fn extend_markers(
        &mut self,
        entries: BTreeMap<String, String>,
        marker_prefix: &str,
    ) {
        if entries.is_empty() {
            return;
        }
        let table = Arc::make_mut(&mut self.0);
        let index = &mut table.index;
        let had_markers = !index.marker_lengths.is_empty();
        for replacement in entries.keys() {
            index.marker_lengths.insert(replacement.len());
        }
        if !had_markers {
            index.marker_prefix = Some(marker_prefix.to_owned());
        } else if index.marker_prefix.as_deref() != Some(marker_prefix) {
            index.marker_prefix = None;
        }
        index.marker_starts.extend(marker_prefix.chars().take(1));
        table.entries.extend(entries);
    }

    pub(crate) fn extend_aliases(&mut self, entries: BTreeMap<String, String>) {
        if entries.is_empty() {
            return;
        }
        let table = Arc::make_mut(&mut self.0);
        let _ = table.alias_trie.take();
        for replacement in entries.keys() {
            let length = replacement.len();
            table.index.alias_lengths = Some(
                table
                    .index
                    .alias_lengths
                    .map_or((length, length), |(min, max)| {
                        (min.min(length), max.max(length))
                    }),
            );
        }
        table.entries.extend(entries);
    }
}
