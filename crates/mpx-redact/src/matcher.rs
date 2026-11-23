use std::{fmt, ops::Range, sync::Arc};

use regex::Regex;

use crate::Error;

/// A detector callback used by [`Matcher::custom`].
pub type Detector = dyn Fn(&str) -> Result<Vec<Range<usize>>, Error> + Send + Sync;

/// Determines which spans a redaction rule matches.
#[derive(Clone)]
#[non_exhaustive]
pub enum Matcher {
    /// An exact, case-sensitive string. An empty literal matches no spans.
    Literal(String),
    /// A compiled Rust regular expression. If it has a capture group, group 1
    /// is selected. Otherwise the full match is selected. Zero-width matches
    /// are ignored.
    Regex(Regex),
    /// A caller-supplied detector that returns UTF-8 byte ranges.
    Custom(Arc<Detector>),
}

impl fmt::Debug for Matcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(_) => f.debug_tuple("Literal").finish_non_exhaustive(),
            Self::Regex(_) => f.debug_tuple("Regex").finish_non_exhaustive(),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

impl Matcher {
    /// Creates an exact, case-sensitive literal matcher.
    #[must_use]
    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal(value.into())
    }

    /// Compiles a Rust regular expression.
    ///
    /// # Errors
    /// Returns an error if the expression cannot compile.
    pub fn regex(pattern: &str) -> Result<Self, Error> {
        Regex::new(pattern)
            .map(Self::Regex)
            .map_err(|source| Error::InvalidRegex { source })
    }

    /// Creates a custom detector over one field's UTF-8 text.
    pub fn custom(
        detector: impl Fn(&str) -> Result<Vec<Range<usize>>, Error> + Send + Sync + 'static,
    ) -> Self {
        Self::Custom(Arc::new(detector))
    }

    pub(crate) fn find(&self, text: &str) -> Result<Vec<Range<usize>>, Error> {
        let mut matches = Vec::new();
        match self {
            Self::Literal(literal) => {
                let Some(first) = literal.chars().next() else {
                    return Ok(matches);
                };
                let mut cursor = 0;
                while let Some(offset) = text[cursor..].find(literal) {
                    let start = cursor + offset;
                    matches.push(start..start + literal.len());
                    cursor = start + first.len_utf8();
                }
            }
            Self::Regex(regex) => {
                if regex.captures_len() == 1 {
                    for matched in regex.find_iter(text).filter(|matched| !matched.is_empty()) {
                        matches.push(matched.range());
                    }
                } else {
                    for captures in regex.captures_iter(text) {
                        if let Some(selected) =
                            captures.get(1).filter(|selected| !selected.is_empty())
                        {
                            matches.push(selected.range());
                        }
                    }
                }
            }
            Self::Custom(detector) => return detector(text),
        }
        Ok(matches)
    }
}
