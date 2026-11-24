//! Checks span-level masking against a reference model: literal-mask rules,
//! the free `mask` function, `RedactionSession::mask_spans`, and custom
//! detectors that report invalid spans or fail.
#![no_main]

use std::ops::Range;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::redact::{
    DetectorSpec, MAX_ITEMS, MAX_RULES, MAX_TOKENS, ReplacementSpec, SpanSpec, assert_mask_error,
    assert_restores, assert_session_consistent, mappings, merge_tagged, pick,
};
use mpx_redact::{Error, Matcher, RedactionSession, Replacement, Rules};

#[derive(Debug, Arbitrary)]
enum Piece {
    Needle(u8),
    Text(String),
}

#[derive(Debug, Arbitrary)]
struct Input {
    /// Literal needles and their permanent masks.
    needles: Vec<(String, String)>,
    text: Vec<Piece>,
    spans: Vec<SpanSpec>,
    mask: String,
    replacement: ReplacementSpec,
    permanent: bool,
    detectors: Vec<DetectorSpec>,
    cuts: Vec<u16>,
}

/// Replaces sorted matches, merging overlaps so the earliest-sorted match's
/// mask covers the union. Adjacent matches stay separate.
fn reference(text: &str, matches: Vec<(Range<usize>, &str)>) -> (String, Vec<(String, String)>) {
    let mut output = String::new();
    let mut cursor = 0;
    let mut recorded = Vec::new();
    for (span, mask) in merge_tagged(matches) {
        output.push_str(&text[cursor..span.start]);
        output.push_str(mask);
        recorded.push((text[span.clone()].to_owned(), mask.to_owned()));
        cursor = span.end;
    }
    output.push_str(&text[cursor..]);
    recorded.sort();
    recorded.dedup();
    (output, recorded)
}

/// Finds every occurrence, including overlapping ones, by checking each char boundary.
fn occurrences(text: &str, needle: &str) -> Vec<Range<usize>> {
    text.char_indices()
        .map(|(start, _)| start)
        .filter(|&start| text[start..].starts_with(needle))
        .map(|start| start..start + needle.len())
        .collect()
}

fn valid(text: &str, span: &Range<usize>) -> bool {
    !span.is_empty() && text.get(span.clone()).is_some()
}

fuzz_target!(|input: Input| {
    let needles: Vec<(&str, &str)> = input
        .needles
        .iter()
        .take(MAX_RULES)
        .filter(|(needle, _)| !needle.is_empty())
        .map(|(needle, mask)| (needle.as_str(), mask.as_str()))
        .collect();
    let mut text = String::new();
    for piece in input.text.iter().take(MAX_TOKENS) {
        match piece {
            Piece::Needle(index) => {
                if let Some((needle, _)) = pick(&needles, *index) {
                    text.push_str(needle);
                }
            }
            Piece::Text(value) => text.push_str(value),
        }
    }

    // Permanent literal masks never collide in a fresh session, so the output is exact.
    let rules = Rules::new(
        needles
            .iter()
            .map(|(needle, mask)| (Matcher::literal(*needle), Replacement::mask(*mask))),
    );
    let matches: Vec<_> = needles
        .iter()
        .flat_map(|(needle, mask)| {
            occurrences(&text, needle)
                .into_iter()
                .map(|span| (span, *mask))
        })
        .collect();
    let (expected, recorded) = reference(&text, matches);
    let mut session = RedactionSession::new();
    let masked = session
        .mask(&text, &rules)
        .expect("literal masks in a fresh session succeed");
    assert_eq!(masked, expected, "literal masks of {text:?}");
    assert_eq!(mappings(&session), recorded);
    assert_eq!(
        session.restore(&masked),
        masked,
        "permanent masks are not restored"
    );

    // Caller-supplied spans: the first invalid span is reported by input index.
    let spans: Vec<Range<usize>> = input
        .spans
        .iter()
        .take(MAX_ITEMS)
        .filter_map(|span| span.resolve(&text))
        .collect();
    let invalid = spans.iter().position(|span| !valid(&text, span));
    let free = mpx_redact::mask(&text, &spans, &input.mask);
    let mut session = RedactionSession::new();
    let mask = Replacement::mask(&input.mask);
    let via_session = session.mask_spans(&text, &spans, &mask);
    match invalid {
        Some(index) => {
            for result in [&free, &via_session] {
                assert!(
                    matches!(result, Err(Error::InvalidSpan { index: reported }) if *reported == index),
                    "expected invalid span {index}: {result:?}"
                );
            }
        }
        None => {
            let matches = spans
                .iter()
                .map(|span| (span.clone(), input.mask.as_str()))
                .collect();
            let (expected, _) = reference(&text, matches);
            assert_eq!(free.expect("valid spans mask"), expected);
            assert_eq!(via_session.expect("valid spans mask"), expected);
        }
    }

    // Spans with a generated replacement.
    if let Some(replacement) = input.replacement.build(input.permanent) {
        let mut session = RedactionSession::new();
        match session.mask_spans(&text, &spans, &replacement) {
            Ok(masked) => {
                assert!(invalid.is_none() && !(input.replacement.is_block() && !spans.is_empty()));
                if !input.permanent {
                    assert_restores(&session, &masked, &text, &input.cuts);
                }
                assert_session_consistent(&session);
            }
            Err(Error::InvalidSpan { index }) => assert_eq!(Some(index), invalid),
            Err(error) => {
                assert!(invalid.is_none());
                assert_eq!(
                    matches!(error, Error::Blocked),
                    input.replacement.is_block() && !spans.is_empty(),
                );
                assert_mask_error(&error, false);
                assert!(
                    mappings(&session).is_empty(),
                    "failed masking recorded mappings"
                );
            }
        }
    }

    // Custom detectors: best-effort `find` skips failures and invalid spans;
    // `try_find` and masking report them.
    let detectors: Vec<_> = input.detectors.iter().take(4).collect();
    let rules = Rules::new(
        detectors
            .iter()
            .map(|detector| (detector.matcher(), Replacement::kind("SECRET").unwrap())),
    );
    let mut found = Vec::new();
    let mut fails = false;
    for detector in &detectors {
        match detector.spans(&text) {
            Some(spans) => {
                fails |= spans.iter().any(|span| !valid(&text, span));
                found.extend(spans.into_iter().filter(|span| valid(&text, span)));
            }
            None => fails = true,
        }
    }
    found.sort_by_key(|span| (span.start, span.end));
    assert_eq!(rules.find(&text), found);
    match rules.try_find(&text) {
        Ok(spans) => {
            assert!(!fails);
            assert_eq!(spans, found);
        }
        Err(error) => {
            assert!(fails);
            assert!(matches!(error, Error::DetectorFailed { .. }), "{error:?}");
        }
    }
    let mut session = RedactionSession::new();
    match session.mask(&text, &rules) {
        Ok(masked) => {
            assert!(!fails);
            assert_restores(&session, &masked, &text, &input.cuts);
        }
        Err(error) => {
            assert!(
                fails || !matches!(error, Error::DetectorFailed { .. }),
                "{error:?}"
            );
            assert_mask_error(&error, fails);
        }
    }
});
