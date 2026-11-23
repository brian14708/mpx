//! Literal and regex masking for UTF-8 text.
//!
//! Use [`Rules`] to detect spans, [`mask`] for permanent replacement,
//! and [`RedactionSession`] for short placeholders and aliases backed by an exchange-local lookup.
//!
//! ```
//! use mpx_redact::{Mask, Matcher, RedactionSession, Rule, Rules};
//!
//! let rule = Rule::new(Matcher::regex(r"\bID-[0-9]+\b")?, Mask::tag());
//! let rules = Rules::new([rule]);
//! let message = "Project Nightjar has ID-1234";
//!
//! let mut session = RedactionSession::new();
//! let masked = session.mask(message, &rules)?;
//! let mut restore = session.restorer();
//! let mut output = restore.push(&masked[..10]);
//! output.push_str(&restore.push(&masked[10..]));
//! output.push_str(&restore.finish());
//! assert_eq!(output, message);
//! # Ok::<(), mpx_redact::Error>(())
//! ```

mod builtins;
mod generator;
mod replacements;
mod restorer;
mod rules;
mod session;
mod template;

pub use generator::{Digits, Generator, GeneratorBinding, Hex, Number, WordList};
pub use replacements::ReplacementMap;
pub use restorer::Restorer;
pub use rules::{Mask, Matcher, Rule, Rules, mask};
pub use session::RedactionSession;
pub use template::PlaceholderTemplate;

/// An error compiling rules or masking text.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A regular expression could not be compiled.
    #[error("redaction regex could not be compiled")]
    InvalidRegex,
    /// A span was empty, out of range, reversed, or split a UTF-8 code point.
    #[error("span {index} is not a nonempty UTF-8 byte range in the input")]
    InvalidSpan {
        /// Zero-based index of the span.
        index: usize,
    },
    /// A placeholder template lacked literal delimiters.
    #[error("placeholder template must start and end with literal delimiters")]
    InvalidTemplateDelimiter,
    /// An alias template could not be compiled.
    #[error("alias template must contain valid slots and render one value")]
    InvalidAliasTemplate,
    /// An alias binding used an invalid or duplicate slot name.
    #[error(
        "alias slot names must be unique and start with an ASCII letter followed by letters, digits, or underscores"
    )]
    InvalidAliasBinding,
    /// A named slot had no generator bound to it.
    #[error("alias slot {name:?} has no generator binding")]
    UnboundAliasSlot {
        /// The slot name in the template.
        name: String,
    },
    /// A list entry could not be parsed as one value.
    #[error("alias list entry {index} must be a single value")]
    InvalidAliasValue {
        /// Zero-based position in the table.
        index: usize,
    },
    /// The placeholder template did not provide an available replacement.
    ///
    /// The diagnostic describes the candidate space and number of candidates
    /// considered. It intentionally does not include the value being masked.
    #[error(
        "could not generate an unused redaction marker from template {template:?} after {attempts} of {combinations} candidate(s)"
    )]
    ReplacementUnavailable {
        /// The configured template pattern.
        template: String,
        /// The size of the template's candidate space.
        combinations: u128,
        /// The number of candidates considered before giving up.
        attempts: u128,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_match_literals_regexes_and_overlaps() {
        let mask = Mask::literal("$1");
        let rules = Rules::new([
            Rule::new(Matcher::Literal("éé".into()), mask.clone()),
            Rule::new(Matcher::Literal("aba".into()), mask.clone()),
            Rule::new(Matcher::regex(r"\d+").unwrap(), mask.clone()),
            Rule::new(Matcher::regex(r"^$").unwrap(), mask),
        ]);
        let text = "ééé ababa 123 ABA";
        assert_eq!(rules.find(text), [0..4, 2..6, 7..10, 9..12, 13..16]);
        assert_eq!(
            RedactionSession::new().mask(text, &rules).unwrap(),
            "$1 $1 $1 ABA"
        );
        assert_eq!(rules.find(""), Vec::<std::ops::Range<usize>>::new());
        assert!(matches!(Matcher::regex("["), Err(Error::InvalidRegex)));
    }

    #[test]
    fn external_spans_are_validated_and_overlap_is_merged() {
        let text = "αβγδε";
        let spans = [6..8, 2..6, 0..4];
        assert_eq!(mask(text, &spans, "X").unwrap(), "XXε");
        let mut session = RedactionSession::new();
        let masked = session.mask_spans(text, &spans).unwrap();
        assert_eq!(session.restore(&masked), text);
        assert_eq!(mask(text, &[], "X").unwrap(), text);
        for invalid in [0..1, 0..11, 2..2, std::ops::Range { start: 6, end: 2 }] {
            let spans = [0..2, invalid];
            assert!(matches!(
                mask(text, &spans, "X"),
                Err(Error::InvalidSpan { index: 1 })
            ));
            assert!(matches!(
                session.mask_spans(text, &spans),
                Err(Error::InvalidSpan { index: 1 })
            ));
        }
    }

    #[test]
    fn handles_every_chunk_boundary() {
        let mut session = RedactionSession::new();
        let rules = Rules::new([
            Rule::new(Matcher::Literal("secret".into()), Mask::tag()),
            Rule::new(Matcher::Literal("privé".into()), Mask::tag()),
        ]);
        let masked = session.mask("secretprivé", &rules).unwrap();
        let truncated = "<|mpx-redact-001122|";
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
        assert_eq!(restorer.push("prefix <|mpx"), "prefix ");
        assert_eq!(restorer.finish(), "<|mpx");
    }

    #[test]
    fn validates_template_slots_and_delimiters() {
        let bindings = [GeneratorBinding::new("hex", Hex(6))];
        for pattern in ["<masked>", "<{{}}>", "<{{hex:6}}>"] {
            assert!(matches!(
                PlaceholderTemplate::new(pattern, &bindings),
                Err(Error::InvalidAliasTemplate)
            ));
        }
        for pattern in ["{{hex}}>", "<{{hex}}", "dead{{hex}}>", "<{{hex}}END"] {
            assert!(matches!(
                PlaceholderTemplate::new(pattern, &bindings),
                Err(Error::InvalidTemplateDelimiter)
            ));
        }
        let template = PlaceholderTemplate::new("<|mpx-redact-{{hex}}|>", &bindings).unwrap();
        assert_eq!(
            template.compiled.prefix,
            PlaceholderTemplate::default().compiled.prefix
        );
    }

    #[test]
    fn restores_custom_templates_across_chunks() {
        let rules = Rules::new([Rule::new(Matcher::Literal("privé".into()), Mask::tag())]);
        for (prefix, suffix) in [
            (r#"<redacted data=""#, r#""/>"#),
            ("[MASK(", ")]"),
            ("｜", "｜"),
        ] {
            let pattern = format!("{prefix}{{{{hex}}}}{suffix}");
            let template =
                PlaceholderTemplate::new(&pattern, &[GeneratorBinding::new("hex", Hex(12))])
                    .unwrap();
            let mut session = RedactionSession::new().with_placeholder_template(template.clone());
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
