use std::{fmt, mem, sync::Arc};

use crate::{
    ReplacementMap,
    alias_trie::{AliasTrie, Cursor},
    replacement_template::is_value_char,
};

/// Incrementally restores generated replacements across UTF-8 text chunks.
///
/// Pass chunks from one logical field to [`Self::push`], then call [`Self::finish`].
pub struct Restorer {
    replacements: ReplacementMap,
    pending: String,
    aliases: Option<AliasReader>,
}

impl fmt::Debug for Restorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Restorer")
            .field("pending_bytes", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl Restorer {
    pub(crate) fn new(replacements: ReplacementMap) -> Self {
        let aliases = replacements
            .index()
            .alias_lengths
            .map(|(min, max)| AliasReader::new(min, max));
        Self {
            replacements,
            pending: String::new(),
            aliases,
        }
    }

    pub(crate) fn restore(mut self, text: &str) -> String {
        let mut output = String::with_capacity(text.len());
        self.scan_complete::<false>(text, &mut output);
        output
    }

    /// Consumes one chunk and returns text ready for delivery.
    ///
    /// A trailing candidate is buffered while it can still match an issued replacement.
    #[must_use]
    pub fn push(&mut self, chunk: &str) -> String {
        let mut output = String::with_capacity(chunk.len());
        if chunk.is_empty() {
            return output;
        }
        if let Some(aliases) = &mut self.aliases {
            aliases
                .trie
                .get_or_insert_with(|| self.replacements.alias_trie());
        }
        if self.replacements.index().marker_lengths.is_empty() {
            write_plain::<true>(&mut self.aliases, &self.replacements, chunk, &mut output);
            return output;
        }
        if self.pending.is_empty() {
            let consumed = self.scan::<true>(chunk, &mut output, false);
            self.pending.push_str(&chunk[consumed..]);
        } else {
            let mut input = mem::take(&mut self.pending);
            input.push_str(chunk);
            let consumed = self.scan::<true>(&input, &mut output, false);
            if consumed != 0 {
                input.drain(..consumed);
            }
            self.pending = input;
        }
        if !self.pending.is_empty()
            && let Some(aliases) = &mut self.aliases
        {
            aliases.before_pending_marker(&self.pending, &self.replacements, &mut output);
        }
        output
    }

    /// Ends the text field and returns its remaining text.
    #[must_use]
    pub fn finish(mut self) -> String {
        let mut output = String::new();
        let pending = mem::take(&mut self.pending);
        self.scan_complete::<true>(&pending, &mut output);
        output
    }

    fn scan_complete<const STREAMING: bool>(&mut self, input: &str, output: &mut String) {
        self.scan::<STREAMING>(input, output, true);
        if let Some(aliases) = &mut self.aliases {
            aliases.flush::<STREAMING>(&self.replacements, output);
        }
    }

    fn scan<const STREAMING: bool>(
        &mut self,
        input: &str,
        output: &mut String,
        final_chunk: bool,
    ) -> usize {
        let Self {
            aliases,
            replacements,
            ..
        } = self;
        let index = replacements.index();
        let Some(&max_marker_bytes) = index.marker_lengths.last() else {
            write_plain::<STREAMING>(aliases, replacements, input, output);
            return input.len();
        };

        let prefix = index.marker_prefix.as_deref();
        let single_start = if index.marker_starts.len() == 1 {
            index.marker_starts.first().copied()
        } else {
            None
        };
        let mut remaining = input;
        while let Some(start) = single_start.map_or_else(
            || remaining.find(|ch| index.marker_starts.contains(&ch)),
            |start| remaining.find(start),
        ) {
            write_plain::<STREAMING>(aliases, replacements, &remaining[..start], output);
            let candidate = &remaining[start..];
            if !final_chunk
                && candidate.len() < max_marker_bytes
                && replacements.has_longer_with_prefix(candidate)
            {
                return input.len() - candidate.len();
            }
            let lookup = |length| {
                let marker = candidate.get(..length)?;
                replacements.get(marker).map(|original| (length, original))
            };
            let matched = if prefix.is_some_and(|prefix| !candidate.starts_with(prefix)) {
                None
            } else {
                index
                    .marker_lengths
                    .range(..=candidate.len())
                    .rev()
                    .copied()
                    .find_map(lookup)
            };
            if let Some((end, original)) = matched {
                if let Some(aliases) = aliases {
                    aliases.flush::<STREAMING>(replacements, output);
                }
                output.push_str(original);
                remaining = &candidate[end..];
                continue;
            }
            let next = candidate.ceil_char_boundary(1);
            write_plain::<STREAMING>(aliases, replacements, &candidate[..next], output);
            remaining = &candidate[next..];
        }

        write_plain::<STREAMING>(aliases, replacements, remaining, output);
        input.len()
    }
}

fn write_plain<const STREAMING: bool>(
    aliases: &mut Option<AliasReader>,
    replacements: &ReplacementMap,
    text: &str,
    output: &mut String,
) {
    if let Some(aliases) = aliases {
        aliases.push::<STREAMING>(text, replacements, output);
    } else {
        output.push_str(text);
    }
}

struct AliasReader {
    min_bytes: usize,
    limit: usize,
    trie: Option<Arc<AliasTrie>>,
    cursor: Cursor,
    value: String,
    literal: bool,
    dot: TrailingDot,
}

#[derive(Default, PartialEq, Eq)]
enum TrailingDot {
    #[default]
    None,
    Buffered,
    Emitted,
}

impl AliasReader {
    const fn new(min_bytes: usize, limit: usize) -> Self {
        Self {
            min_bytes,
            limit,
            trie: None,
            cursor: Cursor::ROOT,
            value: String::new(),
            literal: false,
            dot: TrailingDot::None,
        }
    }

    fn push<const STREAMING: bool>(
        &mut self,
        text: &str,
        replacements: &ReplacementMap,
        output: &mut String,
    ) {
        if text.is_empty() {
            return;
        }
        match mem::take(&mut self.dot) {
            TrailingDot::None => {}
            TrailingDot::Buffered => {
                if text.starts_with(is_value_char) {
                    self.buffer_value::<STREAMING>(".", output);
                } else {
                    self.write_value::<STREAMING>("", replacements, output);
                    output.push('.');
                }
            }
            TrailingDot::Emitted => self.literal = text.starts_with(is_value_char),
        }

        let mut start = 0;
        let mut in_value = self.literal || !self.value.is_empty();
        let mut dot = false;
        for (offset, ch) in text.char_indices() {
            if is_value_char(ch) {
                in_value = true;
                dot = false;
            } else if ch == '.' && in_value && !dot {
                dot = true;
            } else {
                let end = offset - usize::from(dot);
                self.write_value::<STREAMING>(&text[start..end], replacements, output);
                start = offset + ch.len_utf8();
                output.push_str(&text[end..start]);
                in_value = false;
                dot = false;
            }
        }
        // A dot at the chunk boundary belongs to the value only if the next
        // character can continue it.
        let end = text.len() - usize::from(dot);
        self.buffer_value::<STREAMING>(&text[start..end], output);
        if dot {
            if !STREAMING
                || !self.literal
                    && self.trie.as_deref().is_none_or(|trie| {
                        trie.is_terminal(self.cursor) || trie.advance(self.cursor, b".").is_some()
                    })
            {
                self.dot = TrailingDot::Buffered;
            } else {
                self.emit_unmatched(output);
                output.push('.');
                self.dot = TrailingDot::Emitted;
            }
        }
    }

    fn buffer_value<const STREAMING: bool>(&mut self, text: &str, output: &mut String) {
        if text.is_empty() {
            return;
        }
        if !self.literal && text.len() <= self.limit - self.value.len() {
            if STREAMING && let Some(trie) = &self.trie {
                if let Some(cursor) = trie.advance(self.cursor, text.as_bytes()) {
                    self.cursor = cursor;
                    self.value.push_str(text);
                    return;
                }
            } else {
                self.value.push_str(text);
                return;
            }
        }
        self.emit_unmatched(output);
        output.push_str(text);
    }

    fn emit_unmatched(&mut self, output: &mut String) {
        output.push_str(&self.value);
        self.value.clear();
        self.cursor = Cursor::ROOT;
        self.literal = true;
    }

    fn write_value<const STREAMING: bool>(
        &mut self,
        text: &str,
        replacements: &ReplacementMap,
        output: &mut String,
    ) {
        if self.value.is_empty() {
            let value = if !self.literal && (self.min_bytes..=self.limit).contains(&text.len()) {
                replacements.get(text).unwrap_or(text)
            } else {
                text
            };
            output.push_str(value);
        } else {
            self.buffer_value::<STREAMING>(text, output);
            let value = if self.value.len() >= self.min_bytes
                && (!STREAMING
                    || self
                        .trie
                        .as_deref()
                        .is_none_or(|trie| trie.is_terminal(self.cursor)))
            {
                replacements.get(&self.value).unwrap_or(&self.value)
            } else {
                &self.value
            };
            output.push_str(value);
            self.value.clear();
            self.cursor = Cursor::ROOT;
        }
        self.literal = false;
    }

    fn before_pending_marker(
        &mut self,
        marker: &str,
        replacements: &ReplacementMap,
        output: &mut String,
    ) {
        if let Some(after_dot) = marker.strip_prefix('.')
            && self.dot == TrailingDot::None
            && after_dot.chars().next().is_none_or(is_value_char)
        {
            // If the marker fails, the dot may still continue this alias.
            if !self.value.is_empty()
                && let Some(trie) = &self.trie
                && !trie.is_terminal(self.cursor)
            {
                let end = after_dot
                    .find(|ch| !is_value_char(ch))
                    .map_or(marker.len(), |i| i + 1);
                if trie
                    .advance(self.cursor, &marker.as_bytes()[..end])
                    .is_none()
                {
                    self.emit_unmatched(output);
                }
            }
        } else {
            self.flush::<true>(replacements, output);
        }
    }

    fn flush<const STREAMING: bool>(&mut self, replacements: &ReplacementMap, output: &mut String) {
        self.write_value::<STREAMING>("", replacements, output);
        if mem::take(&mut self.dot) == TrailingDot::Buffered {
            output.push('.');
        }
    }
}
