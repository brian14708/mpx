//! Shared `Arbitrary` generators and oracles for the redaction fuzz targets.

use std::{collections::HashSet, ops::Range, sync::LazyLock};

use arbitrary::Arbitrary;
use mpx_redact::{
    Error, Matcher, RedactionSession, Replacement, ReplacementMap, Restorer, Rules, Slot,
    SlotGenerator,
};

/// Upper bounds that keep generated cases small and fast.
pub const MAX_RULES: usize = 6;
pub const MAX_FIELDS: usize = 4;
pub const MAX_TOKENS: usize = 16;
pub const MAX_CHARS: usize = 24;
/// Most spans, words, or values a single generated detector or slot uses.
pub const MAX_ITEMS: usize = 8;

/// Regexes paired with a sample that each one matches, so fields can contain hits.
pub const REGEXES: [(&str, &str); 5] = [
    (r"\d+", "2026"),
    (r"[a-z]+@[a-z]+", "ada@host"),
    (r"key-([0-9a-f]{4,})", "key-beef01"),
    (r"\b\w{3}\b", "abc"),
    (r"末+", "末末"),
];

// Compiling Unicode regexes dominates a run, so compile them once.
static COMPILED: LazyLock<Vec<Matcher>> = LazyLock::new(|| {
    REGEXES
        .iter()
        .map(|(pattern, _)| Matcher::regex(pattern).unwrap())
        .collect()
});

/// Text placed next to matches: value characters, a dot (which can join
/// aliases), marker delimiters, whitespace, and a multibyte character.
pub const GLUE: [&str; 20] = [
    "", " ", "\n", "a", "Z", "0", "@", "_", "%", "+", "-", ".", "{{", "}}", "<", "末", "|", "#",
    "::", "|b",
];

/// Delimiters that open and close a framed template, so marker text can
/// overlap adjacent delimiters.
pub const SYMMETRIC: [&str; 4] = ["|", "#", "::", "~~"];

fn glue(index: u8) -> &'static str {
    GLUE[usize::from(index) % GLUE.len()]
}

fn short(text: &str) -> &str {
    let end = text
        .char_indices()
        .nth(MAX_CHARS)
        .map_or(text.len(), |(index, _)| index);
    &text[..end]
}

pub fn pick<T>(items: &[T], index: u8) -> Option<&T> {
    (!items.is_empty()).then(|| &items[usize::from(index) % items.len()])
}

/// A byte range resolved against the text a detector or caller receives.
#[derive(Debug, Clone, Arbitrary)]
pub struct SpanSpec {
    pub start: u16,
    pub len: u8,
    /// Use the raw offsets, which are often empty, out of range, or inside a code point.
    pub raw: bool,
}

impl SpanSpec {
    /// Resolves the span. Unless `raw`, the result is nonempty and on char boundaries.
    #[must_use]
    pub fn resolve(&self, text: &str) -> Option<Range<usize>> {
        let start = usize::from(self.start);
        let end = start + usize::from(self.len);
        if self.raw {
            return Some(start..end);
        }
        if text.is_empty() {
            return None;
        }
        let start = text.floor_char_boundary(start % text.len());
        let end = text.ceil_char_boundary(end.clamp(start + 1, text.len()));
        Some(start..end)
    }
}

/// Emits one of a fixed set of alphanumeric values, as a well-behaved custom slot.
struct Pick(Vec<String>);

impl SlotGenerator for Pick {
    fn generate(&self, seed: [u8; 32], _: usize, output: &mut String) {
        output.push_str(&self.0[usize::from(seed[0]) % self.0.len()]);
    }
}

#[derive(Debug, Arbitrary)]
pub enum SlotSpec {
    Literal(String),
    Digits(u8),
    Base16(u8),
    Base32(u8),
    Number(u64, u64),
    Words(Vec<String>),
    Custom(Vec<String>),
}

impl SlotSpec {
    #[must_use]
    pub fn build(&self) -> Option<Slot> {
        let width = |width: &u8| usize::from(*width % 24);
        Some(match self {
            Self::Literal(value) => Slot::literal(short(value)),
            Self::Digits(w) => Slot::digits(width(w)),
            Self::Base16(w) => Slot::base16(width(w)),
            Self::Base32(w) => Slot::base32(width(w)),
            Self::Number(low, high) => Slot::number(*low..=*high),
            Self::Words(words) => {
                Slot::wordlist(words.iter().take(MAX_ITEMS).map(|w| short(w))).ok()?
            }
            Self::Custom(values) => {
                let values: Vec<String> = values
                    .iter()
                    .take(MAX_ITEMS)
                    .map(|value| short(value).to_owned())
                    .filter(|value| !value.is_empty() && value.chars().all(char::is_alphanumeric))
                    .collect();
                if values.is_empty() {
                    return None;
                }
                Slot::custom(Pick(values))
            }
        })
    }
}

/// A replacement whose permanence is chosen by the caller.
#[derive(Debug, Arbitrary)]
pub enum ReplacementSpec {
    /// A `{{KIND_abcdefgh}}` marker; always reversible.
    Kind(bool),
    /// `<{}>`: a framed replacement.
    Framed(SlotSpec),
    /// `{}`: an unframed alias.
    Alias(SlotSpec),
    /// `|{}|` and similar: framed by the same delimiter on both sides.
    Symmetric(u8, SlotSpec),
    Template(String, Vec<SlotSpec>),
    Block,
}

impl ReplacementSpec {
    /// Builds the replacement, or `None` when the generated template is invalid.
    ///
    /// # Panics
    /// Panics if template validity depends on reversibility.
    #[must_use]
    pub fn build(&self, permanent: bool) -> Option<Replacement> {
        let template = |pattern: &str, slots: &[Slot]| {
            let reversible = Replacement::template(pattern, slots);
            let fixed = Replacement::permanent_template(pattern, slots);
            assert_eq!(
                reversible.is_ok(),
                fixed.is_ok(),
                "template validity must not depend on reversibility"
            );
            if permanent { fixed } else { reversible }.ok()
        };
        match self {
            Self::Kind(api) => Replacement::kind(if *api { "API_KEY" } else { "SECRET" }).ok(),
            Self::Framed(slot) => template("<{}>", &[slot.build()?]),
            Self::Alias(slot) => template("{}", &[slot.build()?]),
            Self::Symmetric(delimiter, slot) => {
                let delimiter = SYMMETRIC[usize::from(*delimiter) % SYMMETRIC.len()];
                template(&format!("{delimiter}{{}}{delimiter}"), &[slot.build()?])
            }
            Self::Template(pattern, slots) => {
                let slots = slots
                    .iter()
                    .take(MAX_ITEMS)
                    .map(SlotSpec::build)
                    .collect::<Option<Vec<_>>>()?;
                template(short(pattern), &slots)
            }
            Self::Block => Some(Replacement::block()),
        }
    }

    #[must_use]
    pub const fn is_block(&self) -> bool {
        matches!(self, Self::Block)
    }
}

/// A custom detector that returns generated spans or fails.
#[derive(Debug, Clone, Arbitrary)]
pub struct DetectorSpec {
    pub spans: Vec<SpanSpec>,
    pub fail: bool,
}

impl DetectorSpec {
    /// Returns the spans the detector reports for `text`, or `None` if it fails.
    #[must_use]
    pub fn spans(&self, text: &str) -> Option<Vec<Range<usize>>> {
        (!self.fail).then(|| {
            self.spans
                .iter()
                .take(MAX_ITEMS)
                .filter_map(|span| span.resolve(text))
                .collect()
        })
    }

    #[must_use]
    pub fn matcher(&self) -> Matcher {
        let spec = self.clone();
        Matcher::custom(move |text| {
            spec.spans(text).ok_or_else(|| Error::DetectorFailed {
                message: "generated failure".to_owned(),
            })
        })
    }

    /// Whether the detector can fail or report a span that is not valid UTF-8.
    #[must_use]
    pub fn can_fail(&self) -> bool {
        self.fail || self.spans.iter().take(MAX_ITEMS).any(|span| span.raw)
    }
}

#[derive(Debug, Arbitrary)]
pub enum MatcherSpec {
    Literal(String),
    Regex(u8),
    Custom(DetectorSpec),
}

impl MatcherSpec {
    /// Builds the matcher and a sample it matches, or `None` for an empty literal.
    #[must_use]
    pub fn build(&self) -> Option<(Matcher, String)> {
        Some(match self {
            Self::Literal(value) if value.is_empty() => return None,
            Self::Literal(value) => {
                let value = short(value);
                (Matcher::literal(value), value.to_owned())
            }
            Self::Regex(index) => {
                let index = usize::from(*index) % REGEXES.len();
                (COMPILED[index].clone(), REGEXES[index].1.to_owned())
            }
            // Custom spans are positional, so any text exercises them.
            Self::Custom(detector) => (detector.matcher(), String::new()),
        })
    }

    #[must_use]
    pub fn can_fail(&self) -> bool {
        matches!(self, Self::Custom(detector) if detector.can_fail())
    }
}

#[derive(Debug, Arbitrary)]
pub struct RuleSpec {
    pub matcher: MatcherSpec,
    pub replacement: ReplacementSpec,
}

/// Rules built from specs, with a sample of text each rule matches.
pub struct BuiltRules {
    pub rules: Rules,
    pub samples: Vec<String>,
    pub can_fail: bool,
}

impl BuiltRules {
    /// Builds at most [`MAX_RULES`] rules, skipping invalid ones.
    #[must_use]
    pub fn new(specs: &[RuleSpec], permanent: bool) -> Self {
        let mut rules = Vec::new();
        let mut samples = Vec::new();
        let mut can_fail = false;
        for spec in specs.iter().take(MAX_RULES) {
            let (Some((matcher, sample)), Some(replacement)) =
                (spec.matcher.build(), spec.replacement.build(permanent))
            else {
                continue;
            };
            can_fail |= spec.matcher.can_fail();
            rules.push((matcher, replacement));
            samples.push(sample);
        }
        Self {
            rules: Rules::new(rules),
            samples,
            can_fail,
        }
    }
}

/// A piece of an original field.
#[derive(Debug, Arbitrary)]
pub enum Token {
    /// Text that the selected rule matches.
    Rule(u8),
    Glue(u8),
    Text(String),
}

/// Renders an original field from tokens and rule samples.
#[must_use]
pub fn render_field(tokens: &[Token], samples: &[String]) -> String {
    let mut field = String::new();
    for token in tokens.iter().take(MAX_TOKENS) {
        match token {
            Token::Rule(index) => field.push_str(pick(samples, *index).map_or("", String::as_str)),
            Token::Glue(index) => field.push_str(glue(*index)),
            Token::Text(text) => field.push_str(short(text)),
        }
    }
    field
}

/// A piece of model output that echoes, damages, or surrounds issued replacements.
#[derive(Debug, Arbitrary)]
pub enum NoiseToken {
    Issued(u8),
    Prefix(u8, u8),
    Suffix(u8, u8),
    /// Replaces one character of an issued replacement.
    Mutate(u8, u8, char),
    Glue(u8),
    Text(String),
}

/// Renders model output from tokens and every replacement the session issued.
#[must_use]
pub fn render_noise(tokens: &[NoiseToken], session: &RedactionSession) -> String {
    let map = session.replacement_map();
    let issued: Vec<&str> = map.forward_entries().map(|(_, issued)| issued).collect();
    let floor = |text: &str, at: u8| text.floor_char_boundary(usize::from(at) % (text.len() + 1));
    let mut output = String::new();
    for token in tokens.iter().take(MAX_TOKENS * 2) {
        match token {
            NoiseToken::Issued(index) => {
                output.push_str(pick(&issued, *index).copied().unwrap_or(""))
            }
            NoiseToken::Prefix(index, at) => {
                let text = pick(&issued, *index).copied().unwrap_or("");
                output.push_str(&text[..floor(text, *at)]);
            }
            NoiseToken::Suffix(index, at) => {
                let text = pick(&issued, *index).copied().unwrap_or("");
                output.push_str(&text[floor(text, *at)..]);
            }
            NoiseToken::Mutate(index, at, ch) => {
                let text = pick(&issued, *index).copied().unwrap_or("");
                let mut chars: Vec<char> = text.chars().collect();
                if !chars.is_empty() {
                    let len = chars.len();
                    chars[usize::from(*at) % len] = *ch;
                }
                output.extend(chars);
            }
            NoiseToken::Glue(index) => output.push_str(glue(*index)),
            NoiseToken::Text(text) => output.push_str(short(text)),
        }
    }
    output
}

/// Optional session settings.
#[derive(Debug, Arbitrary)]
pub struct SessionSpec {
    pub key: Option<Vec<u8>>,
    pub scope: Vec<u8>,
    pub fallback: Option<ReplacementSpec>,
}

impl SessionSpec {
    /// Builds a fresh session. A fallback that is invalid is omitted.
    #[must_use]
    pub fn build(&self, permanent: bool) -> RedactionSession {
        let mut session = RedactionSession::new().with_scope(&self.scope);
        if let Some(key) = &self.key {
            session = session.with_hmac_key(key);
        }
        if let Some(fallback) = self
            .fallback
            .as_ref()
            .and_then(|spec| spec.build(permanent))
        {
            session = session.with_fallback(fallback);
        }
        session
    }
}

/// Restores `text` in chunks split at `cuts`.
#[must_use]
pub fn restore_chunked(session: &RedactionSession, text: &str, cuts: &[u16]) -> String {
    restore_chunks(session.restorer(), crate::split_str(text, cuts))
}

/// Feeds `chunks` through `restorer` and finishes it.
#[must_use]
pub fn restore_chunks<'a>(
    mut restorer: Restorer,
    chunks: impl IntoIterator<Item = &'a str>,
) -> String {
    let mut output = String::new();
    for chunk in chunks {
        output.push_str(&restorer.push(chunk));
    }
    output.push_str(&restorer.finish());
    output
}

/// Checks restoration against an independent expected value and chunk partitions.
pub fn assert_restores(session: &RedactionSession, masked: &str, expected: &str, cuts: &[u16]) {
    assert_eq!(
        session.restore(masked),
        expected,
        "whole restore of {masked:?} with {:?}",
        mappings(session)
    );
    assert_eq!(
        restore_chunked(session, masked, cuts),
        expected,
        "chunked restore of {masked:?} at {cuts:?}"
    );
}

/// Checks that chunked restoration agrees with whole restoration.
pub fn assert_restore_consistent(session: &RedactionSession, text: &str, cuts: &[u16]) {
    assert_eq!(
        restore_chunked(session, text, cuts),
        session.restore(text),
        "chunked restore of {text:?} at {cuts:?}"
    );
}

/// Takes an ordered snapshot of both reversible and permanent mappings.
#[must_use]
pub fn mappings(session: &RedactionSession) -> Vec<(String, String)> {
    map_mappings(&session.replacement_map())
}

/// Takes an ordered snapshot of a replacement map's forward mappings.
#[must_use]
pub fn map_mappings(map: &ReplacementMap) -> Vec<(String, String)> {
    let mut entries: Vec<_> = map
        .forward_entries()
        .map(|(original, replacement)| (original.to_owned(), replacement.to_owned()))
        .collect();
    entries.sort();
    entries
}

/// Checks that the replacement map's views agree and that each issued
/// reversible replacement restores alone.
pub fn assert_session_consistent(session: &RedactionSession) {
    let map = session.replacement_map();
    let entries: Vec<_> = map.entries().collect();
    assert!(
        entries.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "entries are not sorted and unique by replacement"
    );
    let forward: HashSet<_> = map.forward_entries().collect();
    for &(replacement, original) in &entries {
        assert_eq!(map.get(replacement), Some(original));
        assert!(
            map.replacements_for(original)
                .any(|issued| issued == replacement)
        );
        assert!(forward.contains(&(original, replacement)));
        assert_eq!(
            session.restore(replacement),
            original,
            "lone {replacement:?}"
        );
    }
    for &(original, replacement) in &forward {
        assert!(
            map.replacements_for(original)
                .any(|issued| issued == replacement)
        );
    }
}

/// Accepts only the failures generated rules can legitimately cause.
///
/// # Panics
/// Panics on any other error.
pub fn assert_mask_error(error: &Error, detectors_can_fail: bool) {
    assert!(
        matches!(
            error,
            Error::ReplacementUnavailable { .. } | Error::OverlappingReplacements | Error::Blocked
        ) || detectors_can_fail && matches!(error, Error::DetectorFailed { .. }),
        "unexpected masking failure: {error:?}"
    );
}

/// Runs `mask`, checking that any failure is legitimate and leaves the mappings unchanged.
///
/// # Errors
/// Returns the masking error after checking it.
pub fn mask_atomically(
    session: &mut RedactionSession,
    detectors_can_fail: bool,
    mask: impl FnOnce(&mut RedactionSession) -> Result<String, Error>,
) -> Result<String, Error> {
    let before = mappings(session);
    mask(session).inspect_err(|error| {
        assert_mask_error(error, detectors_can_fail);
        assert_eq!(mappings(session), before, "failed masking changed mappings");
    })
}

/// Unions overlapping sorted spans, as masking does; adjacent spans stay separate.
#[must_use]
pub fn merge_spans(spans: Vec<Range<usize>>) -> Vec<Range<usize>> {
    merge_tagged(spans.into_iter().map(|span| (span, ())).collect())
        .into_iter()
        .map(|(span, ())| span)
        .collect()
}

/// [`merge_spans`] carrying a tag; each union keeps the earliest-sorted span's tag.
#[must_use]
pub fn merge_tagged<T>(mut spans: Vec<(Range<usize>, T)>) -> Vec<(Range<usize>, T)> {
    spans.sort_by_key(|(span, _)| (span.start, span.end));
    let mut merged: Vec<(Range<usize>, T)> = Vec::with_capacity(spans.len());
    for (span, tag) in spans {
        match merged.last_mut() {
            Some((last, _)) if span.start < last.end => last.end = last.end.max(span.end),
            _ => merged.push((span, tag)),
        }
    }
    merged
}

/// Checks that every merged match in `original` received a recorded replacement.
pub fn assert_all_masked(session: &RedactionSession, original: &str, rules: &Rules) {
    let map = session.replacement_map();
    let spans = rules
        .try_find(original)
        .expect("masking succeeded, so detection did");
    for span in merge_spans(spans) {
        let value = &original[span];
        assert!(
            map.replacements_for(value).next().is_some(),
            "match {value:?} in {original:?} has no replacement"
        );
    }
}
