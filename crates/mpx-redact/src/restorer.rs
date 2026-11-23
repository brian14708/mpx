use std::{fmt, mem};

use crate::{ReplacementMap, template::is_value_char};

/// Incrementally restores generated replacements across UTF-8 text chunks.
///
/// Pass chunks from one logical field in order, then call [`Self::finish`].
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
        let alias_limit = replacements.index().alias_limit;
        let aliases = (alias_limit > 0).then(|| AliasReader::new(alias_limit));
        Self {
            replacements,
            pending: String::new(),
            aliases,
        }
    }

    pub(crate) fn restore(mut self, text: &str) -> String {
        let mut output = self.push(text);
        self.finish_into(&mut output);
        output
    }

    /// Consumes one chunk and returns text ready for delivery.
    #[must_use]
    pub fn push(&mut self, chunk: &str) -> String {
        let mut output = String::with_capacity(chunk.len());
        if chunk.is_empty() {
            return output;
        }
        if self.pending.is_empty() {
            let consumed = self.scan(chunk, &mut output, false);
            self.pending.push_str(&chunk[consumed..]);
        } else {
            let mut input = mem::take(&mut self.pending);
            input.push_str(chunk);
            let consumed = self.scan(&input, &mut output, false);
            if consumed != 0 {
                input.drain(..consumed);
            }
            self.pending = input;
        }
        output
    }

    /// Ends the text field and returns its remaining text.
    #[must_use]
    pub fn finish(self) -> String {
        let mut output = String::new();
        self.finish_into(&mut output);
        output
    }

    fn finish_into(mut self, output: &mut String) {
        let pending = mem::take(&mut self.pending);
        self.scan(&pending, output, true);
        if let Some(aliases) = &mut self.aliases {
            aliases.flush(&self.replacements, output);
        }
    }

    fn scan(&mut self, input: &str, output: &mut String, final_chunk: bool) -> usize {
        let Self {
            aliases,
            replacements,
            ..
        } = self;
        let index = replacements.index();
        let Some(&max_marker_bytes) = index.marker_lengths.last() else {
            write_plain(aliases, replacements, input, output);
            return input.len();
        };

        let prefix = index.marker_prefix.as_deref();
        let mut remaining = input;
        while let Some(start) = prefix.map_or_else(
            || remaining.find(|ch| index.marker_starts.contains(&ch)),
            |prefix| remaining.find(prefix),
        ) {
            write_plain(aliases, replacements, &remaining[..start], output);
            let candidate = &remaining[start..];
            if !final_chunk
                && candidate.len() < max_marker_bytes
                && replacements.has_longer_with_prefix(candidate)
            {
                return input.len() - candidate.len();
            }
            let matched = index
                .marker_lengths
                .range(..=candidate.len())
                .rev()
                .find_map(|&length| {
                    let marker = candidate.get(..length)?;
                    replacements.get(marker).map(|original| (length, original))
                });
            if let Some((end, original)) = matched {
                if let Some(aliases) = aliases {
                    aliases.flush(replacements, output);
                }
                output.push_str(original);
                remaining = &candidate[end..];
                continue;
            }
            let next = candidate.ceil_char_boundary(1);
            write_plain(aliases, replacements, &candidate[..next], output);
            remaining = &candidate[next..];
        }

        let Some(prefix) = prefix.filter(|_| !final_chunk) else {
            write_plain(aliases, replacements, remaining, output);
            return input.len();
        };

        let tail_start =
            remaining.ceil_char_boundary(remaining.len().saturating_sub(prefix.len() - 1));
        let pending = remaining[tail_start..]
            .char_indices()
            .map(|(offset, _)| tail_start + offset)
            .find(|&offset| prefix.starts_with(&remaining[offset..]))
            .unwrap_or(remaining.len());
        write_plain(aliases, replacements, &remaining[..pending], output);
        input.len() - remaining.len() + pending
    }
}

fn write_plain(
    aliases: &mut Option<AliasReader>,
    replacements: &ReplacementMap,
    text: &str,
    output: &mut String,
) {
    if let Some(aliases) = aliases {
        aliases.push(text, replacements, output);
    } else {
        output.push_str(text);
    }
}

struct AliasReader {
    limit: usize,
    value: String,
    literal: bool,
    dot: bool,
}

impl AliasReader {
    const fn new(limit: usize) -> Self {
        Self {
            limit,
            value: String::new(),
            literal: false,
            dot: false,
        }
    }

    fn push(&mut self, text: &str, replacements: &ReplacementMap, output: &mut String) {
        for ch in text.chars() {
            if is_value_char(ch) {
                if mem::take(&mut self.dot) {
                    self.push_value('.', output);
                }
                self.push_value(ch, output);
            } else if ch == '.' && !self.dot && (self.literal || !self.value.is_empty()) {
                self.dot = true;
            } else {
                self.flush(replacements, output);
                output.push(ch);
            }
        }
    }

    fn push_value(&mut self, ch: char, output: &mut String) {
        if !self.literal && self.value.len() + ch.len_utf8() <= self.limit {
            self.value.push(ch);
        } else {
            output.push_str(&self.value);
            self.value.clear();
            self.literal = true;
            output.push(ch);
        }
    }

    fn flush(&mut self, replacements: &ReplacementMap, output: &mut String) {
        let original = replacements.get(&self.value);
        output.push_str(original.unwrap_or(&self.value));
        self.value.clear();
        self.literal = false;
        if mem::take(&mut self.dot) {
            output.push('.');
        }
    }
}
