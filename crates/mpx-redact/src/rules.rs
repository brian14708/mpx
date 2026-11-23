use std::{cmp::Reverse, fmt, ops::Range};

use crate::{Error, Matcher, Replacement};

/// Reusable redaction rules.
///
/// A rule is supplied as a `(Matcher, Replacement)` tuple.
///
/// ~~~
/// use mpx_redact::{Matcher, Replacement, Rules, Slot};
///
/// let mask = Replacement::template("{}-{}@{}", &[
///     Slot::wordlist(["ada", "mira"])?,
///     Slot::wordlist(["chen", "patel"])?,
///     Slot::wordlist(["example.com", "example.net"])?,
/// ])?;
/// let email = Matcher::regex(
///     r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}",
/// )?;
/// let rules = Rules::new([(email, mask)]);
/// assert_eq!(rules.try_find("Email person@example.net")?, [6..24]);
/// # Ok::<(), mpx_redact::Error>(())
/// ~~~
#[derive(Clone)]
pub struct Rules {
    rules: Vec<(Matcher, Replacement)>,
    merge_overlaps: bool,
}

pub(crate) struct RuleMatch<'a> {
    pub(crate) span: Range<usize>,
    pub(crate) replacement: &'a Replacement,
}

impl fmt::Debug for Rules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rules")
            .field("rule_count", &self.rules.len())
            .field("merge_overlaps", &self.merge_overlaps)
            .finish()
    }
}

impl Rules {
    /// Collects rules in the order supplied.
    #[must_use]
    pub fn new(rules: impl IntoIterator<Item = (Matcher, Replacement)>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
            merge_overlaps: false,
        }
    }

    /// Creates the built-in detector set with kind-specific redaction markers.
    ///
    /// Overlapping built-in matches are combined. The leftmost match supplies
    /// the marker kind, with longer matches taking precedence at the same start.
    ///
    /// # Panics
    /// The built-in marker template is a compile-time constant and cannot fail
    /// to compile unless the template implementation changes.
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            rules: crate::builtin_rules::rules(),
            merge_overlaps: true,
        }
    }

    pub(crate) fn literal_replacements(&self) -> impl Iterator<Item = &str> {
        self.rules
            .iter()
            .filter_map(|(_, replacement)| replacement.literal_replacement())
    }

    /// Finds UTF-8 byte ranges, ordered by start and then end.
    ///
    /// Literal matches can overlap. Each regular expression uses nonoverlapping matches.
    ///
    /// This best-effort search skips failed detectors and discards invalid spans.
    /// Its result may omit matches. Use [`Self::try_find`] when detection must
    /// succeed for every rule, such as when selecting spans to redact.
    #[must_use]
    pub fn find(&self, text: &str) -> Vec<Range<usize>> {
        self.try_find_matches(text, false)
            .map(spans_of)
            .unwrap_or_default()
    }

    /// Finds UTF-8 byte ranges, ordered by start and then end.
    ///
    /// # Errors
    /// Returns an error if a detector fails or returns an invalid span.
    pub fn try_find(&self, text: &str) -> Result<Vec<Range<usize>>, Error> {
        self.try_find_matches(text, true).map(spans_of)
    }

    pub(crate) fn try_find_matches(
        &self,
        text: &str,
        strict: bool,
    ) -> Result<Vec<RuleMatch<'_>>, Error> {
        let mut matches = Vec::new();
        for rule in &self.rules {
            let spans = match rule.0.find(text) {
                Ok(spans) => spans,
                Err(_) if !strict => continue,
                Err(error) => return Err(error),
            };
            for span in spans {
                if !valid_span(text, &span) {
                    if strict {
                        return Err(Error::DetectorFailed {
                            message: "detector returned an invalid UTF-8 span".to_owned(),
                        });
                    }
                    continue;
                }
                matches.push(RuleMatch {
                    span,
                    replacement: &rule.1,
                });
            }
        }
        if self.merge_overlaps {
            matches.sort_by_key(|matched| (matched.span.start, Reverse(matched.span.end)));
            // The leftmost, longest match supplies the replacement.
            merge_overlapping(matches, |current, _| Ok(current))
        } else {
            matches.sort_by_key(|matched| (matched.span.start, matched.span.end));
            Ok(matches)
        }
    }
}

fn spans_of(matches: Vec<RuleMatch<'_>>) -> Vec<Range<usize>> {
    matches.into_iter().map(|matched| matched.span).collect()
}

/// Masks supplied UTF-8 byte ranges with a literal replacement.
///
/// Overlaps merge into one replacement. Adjacent ranges get separate replacements.
///
/// # Errors
///
/// Returns an error if any span is empty or invalid for the input text.
pub fn mask(text: &str, spans: &[Range<usize>], replacement: &str) -> Result<String, Error> {
    let mask = Replacement::mask(replacement);
    let spans = merge_matches(masked_spans(text, spans, &mask)?)?;
    replace_matches(text, spans, |output, _, _| {
        output.push_str(replacement);
        Ok(())
    })
}

pub(crate) fn masked_spans<'a>(
    text: &str,
    spans: &[Range<usize>],
    replacement: &'a Replacement,
) -> Result<Vec<RuleMatch<'a>>, Error> {
    let mut matches = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().enumerate() {
        if !valid_span(text, span) {
            return Err(Error::InvalidSpan { index });
        }
        matches.push(RuleMatch {
            span: span.clone(),
            replacement,
        });
    }
    matches.sort_unstable_by_key(|matched| (matched.span.start, matched.span.end));
    Ok(matches)
}

fn valid_span(text: &str, span: &Range<usize>) -> bool {
    !span.is_empty() && text.get(span.clone()).is_some()
}

pub(crate) fn replace_matches<'a>(
    text: &str,
    merged: Vec<RuleMatch<'a>>,
    mut rewrite: impl FnMut(&mut String, RuleMatch<'a>, Option<usize>) -> Result<(), Error>,
) -> Result<String, Error> {
    let mut spans = merged.into_iter().peekable();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(matched) = spans.next() {
        output.push_str(&text[cursor..matched.span.start]);
        cursor = matched.span.end;
        rewrite(
            &mut output,
            matched,
            spans.peek().map(|next| next.span.start),
        )?;
    }
    output.push_str(&text[cursor..]);
    Ok(output)
}

pub(crate) fn merge_matches(spans: Vec<RuleMatch<'_>>) -> Result<Vec<RuleMatch<'_>>, Error> {
    merge_overlapping(spans, Replacement::merge)
}

/// Merges overlapping sorted spans, combining their replacements with `combine`.
fn merge_overlapping<'a>(
    mut spans: Vec<RuleMatch<'a>>,
    mut combine: impl FnMut(&'a Replacement, &'a Replacement) -> Result<&'a Replacement, Error>,
) -> Result<Vec<RuleMatch<'a>>, Error> {
    let mut result = Ok(());
    spans.dedup_by(|next, current| {
        if result.is_err() || next.span.start >= current.span.end {
            return false;
        }
        match combine(current.replacement, next.replacement) {
            Ok(replacement) => {
                current.span.end = current.span.end.max(next.span.end);
                current.replacement = replacement;
                true
            }
            Err(error) => {
                result = Err(error);
                false
            }
        }
    });
    result.map(|()| spans)
}
