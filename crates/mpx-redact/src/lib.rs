//! Literal, regex, and custom masking for UTF-8 text.
//!
//! Use [`Rules`] to detect spans, [`mask`] for permanent masking,
//! [`Rules::builtin`] for the inventory-backed detector set, and
//! [`RedactionSession`] for explicitly configured templates.
//!
//! ```
//! use mpx_redact::{Replacement, Matcher, RedactionSession, Rules};
//!
//! let rule = (Matcher::regex(r"\bID-[0-9]+\b")?, Replacement::mask("[REDACTED]"));
//! let rules = Rules::new([rule]);
//! let message = "Project Nightjar has ID-1234";
//!
//! let mut session = RedactionSession::new();
//! let masked = session.mask(message, &rules)?;
//! let mut restorer = session.restorer();
//! let mut output = restorer.push(&masked[..10]);
//! output.push_str(&restorer.push(&masked[10..]));
//! output.push_str(&restorer.finish());
//! assert_eq!(output, "Project Nightjar has [REDACTED]");
//! # Ok::<(), mpx_redact::Error>(())
//! ```

mod alias_trie;
mod builtin_rules;
mod matcher;
mod prepared_aliases;
mod replacement;
mod replacement_map;
mod replacement_template;
mod restorer;
mod rules;
mod session;
mod slot;
mod slot_generator;

pub use matcher::{Detector, Matcher};
pub use replacement::Replacement;
pub use replacement_map::ReplacementMap;
pub use restorer::Restorer;
pub use rules::{Rules, mask};
pub use session::RedactionSession;
pub use slot::Slot;
pub use slot_generator::{Base16, Base32, Digits, Number, SlotGenerator, WordList};

/// An error compiling rules or masking text.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A regular expression could not be compiled.
    #[error("redaction regex could not be compiled")]
    InvalidRegex {
        /// The error returned by the regular-expression engine.
        #[source]
        source: regex::Error,
    },
    /// A span was empty, out of range, reversed, or split a UTF-8 code point.
    #[error("span {index} is not a nonempty UTF-8 byte range in the input")]
    InvalidSpan {
        /// Zero-based index of the span.
        index: usize,
    },
    /// A positional template could not be compiled.
    #[error("template must contain valid slots and render one value")]
    InvalidTemplate,
    /// A rule configured to block masking matched the input.
    #[error("request blocked by a redaction rule")]
    Blocked,
    /// A word-list entry could not be parsed as one value.
    #[error("word list entry {index} must be a single value")]
    InvalidWordListValue {
        /// Zero-based position in the table.
        index: usize,
    },
    /// Overlapping rules used incompatible replacements.
    #[error("overlapping redaction rules require the same replacement")]
    OverlappingReplacements,
    /// A custom detector could not complete safely.
    #[error("custom redaction detector failed: {message}")]
    DetectorFailed {
        /// Detector failure detail.
        message: String,
    },
    /// A template could not produce a usable replacement.
    #[error("could not use the redaction replacement from template {template:?}")]
    ReplacementUnavailable {
        /// The configured template pattern.
        template: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replacement() -> Replacement {
        Replacement::kind("SECRET").unwrap()
    }

    #[test]
    fn overlapping_rules_merge_equal_templates_from_separate_calls() {
        let rules = Rules::new([
            (Matcher::literal("secret-1"), replacement()),
            (Matcher::regex(r"secret-\d").unwrap(), replacement()),
        ]);
        let mut session = RedactionSession::new();
        let masked = session.mask("x secret-1 y", &rules).unwrap();
        assert!(masked.starts_with("x {{SECRET_"), "{masked:?}");
        assert_eq!(session.restore(&masked), "x secret-1 y");

        let digits = |width| Replacement::template("<{}>", &[Slot::digits(width)]).unwrap();
        for (left, right, merges) in [
            (digits(4), digits(4), true),
            (digits(4), digits(5), false),
            (
                Replacement::template("{}", &[Slot::wordlist(["a", "b"]).unwrap()]).unwrap(),
                Replacement::template("{}", &[Slot::wordlist(["a", "b"]).unwrap()]).unwrap(),
                true,
            ),
            (
                replacement(),
                Replacement::permanent_template("{{{{SECRET_{}}}}}", &[Slot::base32(8)]).unwrap(),
                false,
            ),
        ] {
            assert_eq!(left.merge(&right).is_ok(), merges, "{left:?} {right:?}");
        }
    }

    #[test]
    fn templates_reject_unused_slots() {
        assert!(matches!(
            Replacement::template("{}", &[Slot::base16(8), Slot::digits(4)]),
            Err(Error::InvalidTemplate)
        ));
    }

    #[test]
    fn wide_number_slots_cover_the_whole_range() {
        let mask = Replacement::template("{}", &[Slot::number(0..=u64::MAX)]).unwrap();
        let (template, _) = mask.as_template().unwrap();
        let wide = (0..64_u8).any(|byte| {
            let mut output = String::new();
            template.generate(&[byte; 32], &mut output, &mut String::new());
            output.parse::<u64>().unwrap() > u64::from(u32::MAX)
        });
        assert!(wide);
    }

    #[test]
    fn rules_match_literals_regexes_and_overlaps() {
        let mask = Replacement::mask("$1");
        let rules = Rules::new([
            (Matcher::literal("éé"), mask.clone()),
            (Matcher::literal("aba"), mask.clone()),
            (Matcher::regex(r"\d+").unwrap(), mask.clone()),
            (Matcher::regex(r"^$").unwrap(), mask),
        ]);
        let text = "ééé ababa 123 ABA";
        assert_eq!(rules.find(text), [0..4, 2..6, 7..10, 9..12, 13..16]);
        assert_eq!(
            RedactionSession::new().mask(text, &rules).unwrap(),
            "$1 $1 $1 ABA"
        );
        assert_eq!(rules.find(""), Vec::<std::ops::Range<usize>>::new());
        let error = Matcher::regex("[").unwrap_err();
        assert!(matches!(error, Error::InvalidRegex { .. }));
        let source = std::error::Error::source(&error).unwrap();
        assert!(source.is::<regex::Error>());
    }

    #[test]
    fn regex_rules_select_capture_group_one() {
        let rules = Rules::new([(
            Matcher::regex(r"nstok:([0-9a-f]{12})").unwrap(),
            Replacement::mask("{{NS_TOKEN_...}}"),
        )]);
        assert_eq!(rules.find("nstok:0123456789ab"), vec![6..18]);
        assert_eq!(
            RedactionSession::new()
                .mask("nstok:0123456789ab", &rules)
                .unwrap(),
            "nstok:{{NS_TOKEN_...}}"
        );

        let whole = Rules::new([(
            Matcher::regex(r"nstok:[0-9a-f]{12}").unwrap(),
            Replacement::mask("X"),
        )]);
        assert_eq!(whole.find("nstok:0123456789ab"), vec![0..18]);
    }

    #[test]
    fn custom_detectors_return_ranges_and_errors() {
        let custom = Matcher::custom(|text| {
            Ok(text
                .match_indices("secret")
                .map(|(start, value)| start..start + value.len())
                .collect())
        });
        let rules = Rules::new([(custom, Replacement::mask("X"))]);
        assert_eq!(rules.try_find("secret and secret").unwrap(), [0..6, 11..17]);

        let invalid = Rules::new([(
            Matcher::custom(|_| Ok(std::iter::once(1..99).collect())),
            Replacement::mask("X"),
        )]);
        assert!(matches!(
            invalid.try_find("secret"),
            Err(Error::DetectorFailed { .. })
        ));
        assert_eq!(invalid.find("secret"), Vec::<std::ops::Range<usize>>::new());
    }

    #[test]
    fn external_spans_are_validated_and_overlap_is_merged() {
        let text = "αβγδε";
        let spans = [6..8, 2..6, 0..4];
        assert_eq!(mask(text, &spans, "X").unwrap(), "XXε");
        let mut session = RedactionSession::new();
        let replacement_mask = Replacement::mask("X");
        let masked = session.mask_spans(text, &spans, &replacement_mask).unwrap();
        assert_eq!(session.restore(&masked), masked);
        assert_eq!(mask(text, &[], "X").unwrap(), text);
        for invalid in [0..1, 0..11, 2..2, std::ops::Range { start: 6, end: 2 }] {
            let spans = [0..2, invalid];
            assert!(matches!(
                mask(text, &spans, "X"),
                Err(Error::InvalidSpan { index: 1 })
            ));
            assert!(matches!(
                session.mask_spans(text, &spans, &replacement_mask),
                Err(Error::InvalidSpan { index: 1 })
            ));
        }
    }

    #[test]
    fn handles_every_chunk_boundary() {
        let mut session = RedactionSession::new();
        let rules = Rules::new([
            (Matcher::literal("secret"), replacement()),
            (Matcher::literal("privé"), replacement()),
        ]);
        let masked = session.mask("secretprivé", &rules).unwrap();
        let truncated = "{{SECRET_001122|";
        let input = format!("α<<<{masked}末 {truncated}");
        let expected = format!("α<<<secretprivé末 {truncated}");
        for boundary in input
            .char_indices()
            .map(|(index, _)| index)
            .chain([input.len()])
        {
            let output = session.restore_split(&input, boundary);
            assert_eq!(output, expected, "split at {boundary}");
        }
        let mut restorer = session.restorer();
        assert_eq!(restorer.push("prefix {{SECRET_"), "prefix ");
        assert_eq!(restorer.finish(), "{{SECRET_");
    }

    #[test]
    fn validates_template_slots_and_delimiters() {
        let bindings = [Slot::base32(8)];
        for pattern in ["<masked>", "<{{}}>", "<{}"] {
            assert!(matches!(
                Replacement::template(pattern, &bindings),
                Err(Error::InvalidTemplate)
            ));
        }
        for pattern in ["{>", "<}", "dead{>", "<}END"] {
            assert!(matches!(
                Replacement::template(pattern, &bindings),
                Err(Error::InvalidTemplate)
            ));
        }

        // Doubled braces are literal braces, as in Python format strings.
        for (pattern, open, close) in [
            ("{{SECRET_{}}}", "{SECRET_", "}"),
            ("{{{{SECRET_{}}}}}", "{{SECRET_", "}}"),
        ] {
            let replacement = Replacement::template(pattern, &bindings).unwrap();
            let rules = Rules::new([(Matcher::literal("private"), replacement)]);
            let masked = RedactionSession::new().mask("private", &rules).unwrap();
            let slot = masked
                .strip_prefix(open)
                .and_then(|rest| rest.strip_suffix(close))
                .unwrap_or_else(|| panic!("{pattern} rendered {masked}"));
            assert_eq!(slot.len(), 8, "{pattern} rendered {masked}");
        }
    }

    #[test]
    fn restores_custom_templates_across_chunks() {
        for (prefix, suffix) in [
            (r#"<redacted data=""#, r#""/>"#),
            ("[MASK(", ")]"),
            ("｜", "｜"),
        ] {
            let pattern = format!("{prefix}{{}}{suffix}");
            let rules = Rules::new([(
                Matcher::literal("privé"),
                Replacement::template(&pattern, &[Slot::base16(12)]).unwrap(),
            )]);
            let mut session = RedactionSession::new();
            let masked = session.mask("privé", &rules).unwrap();
            let hex = masked
                .strip_prefix(prefix)
                .unwrap()
                .strip_suffix(suffix)
                .unwrap();
            assert_eq!(hex.len(), 12);
            assert!(hex.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(RedactionSession::new().restore(&masked), masked);

            let broken = format!("{prefix}bad");
            let input = format!("α{broken}{masked}{masked}末");
            let expected = format!("α{broken}privéprivé末");
            for boundary in input
                .char_indices()
                .map(|(offset, _)| offset)
                .chain([input.len()])
            {
                let output = session.restore_split(&input, boundary);
                assert_eq!(output, expected, "{pattern} split at {boundary}");
            }
            let mut restorer = session.restorer();
            let mut output = String::new();
            for character in input.chars() {
                output.push_str(&restorer.push(character.encode_utf8(&mut [0; 4])));
            }
            output.push_str(&restorer.finish());
            assert_eq!(output, expected);
        }
    }
}
