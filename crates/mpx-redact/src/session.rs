use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    ops::Range,
};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::{
    Error, PlaceholderTemplate, ReplacementMap, Restorer, Rules,
    rules::{self, RuleMatch},
    template::joins_value,
};

const DERIVATION_CONTEXT: &[u8] = b"mpx-redact/v1\0";

/// Masks text and keeps the reversible replacements for one exchange.
#[derive(Default)]
pub struct RedactionSession {
    template: PlaceholderTemplate,
    hmac: Option<Hmac<Sha256>>,
    scope: Vec<u8>,
    strict: bool,
    probes: u64,
    reserved: HashSet<String>,
    replacements: ReplacementMap,
}

impl fmt::Debug for RedactionSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedactionSession").finish_non_exhaustive()
    }
}

impl RedactionSession {
    /// Creates an exchange with the default placeholder template.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the template for generated redaction markers.
    #[must_use]
    pub fn with_placeholder_template(mut self, template: PlaceholderTemplate) -> Self {
        self.template = template;
        self
    }

    /// Seeds generated aliases and markers with HMAC-SHA-256 using the supplied secret key.
    #[must_use]
    #[expect(clippy::missing_panics_doc, reason = "HMAC accepts keys of any length")]
    pub fn with_hmac_key(mut self, key: &[u8]) -> Self {
        self.hmac = Some(Hmac::new_from_slice(key).expect("HMAC accepts any key length"));
        self
    }

    /// Scopes keyed replacements to the supplied context.
    ///
    /// The scope is included in the versioned HMAC input and may be visible to
    /// the upstream provider. Keep the HMAC key secret.
    #[must_use]
    pub fn with_scope(mut self, scope: impl AsRef<[u8]>) -> Self {
        scope.as_ref().clone_into(&mut self.scope);
        self
    }

    /// Disables collision probing and uses only the deterministic candidate.
    ///
    /// This keeps replacements independent of the other values in the
    /// exchange, which makes them suitable for provider prompt caches. A
    /// collision returns [`Error::ReplacementUnavailable`] with diagnostics.
    #[must_use]
    pub const fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Returns a snapshot of the reversible replacements issued so far.
    #[must_use]
    pub fn replacement_map(&self) -> ReplacementMap {
        self.replacements.clone()
    }

    /// Returns the number of replacements that required collision probing.
    #[must_use]
    pub const fn probe_count(&self) -> u64 {
        self.probes
    }

    /// Reserves replacements already present in an original field.
    ///
    /// Call this on all original fields before masking a multi-field exchange.
    pub fn reserve_replacements_in(&mut self, text: &str, rules: &Rules) {
        rules.reserve_aliases_in(text, &mut self.reserved);
        self.reserve_markers(text);
        for replacement in rules.literal_replacements() {
            self.reserve_markers(replacement);
        }
    }

    fn reserve_markers(&mut self, text: &str) {
        for marker in self.template.compiled.matching_values(text) {
            if !self.reserved.contains(marker) {
                self.reserved.insert(marker.to_owned());
            }
        }
    }

    /// Masks matches from the supplied rules.
    ///
    /// # Errors
    /// Returns an error if the template cannot supply an unused replacement.
    pub fn mask(&mut self, text: &str, rules: &Rules) -> Result<String, Error> {
        self.reserve_replacements_in(text, rules);
        self.mask_reserved(text, rules)
    }

    /// Masks text whose replacements were already reserved.
    ///
    /// Use this after [`Self::reserve_replacements_in`] has seen every field of a
    /// multi-field exchange, to avoid scanning each field twice.
    ///
    /// # Errors
    /// Returns an error if the template cannot supply an unused replacement.
    pub fn mask_reserved(&mut self, text: &str, rules: &Rules) -> Result<String, Error> {
        self.replace(text, rules.find_matches(text))
    }

    /// Masks externally supplied UTF-8 byte ranges.
    ///
    /// # Errors
    /// Returns an error for invalid ranges or when no replacement can be generated.
    pub fn mask_spans(&mut self, text: &str, spans: &[Range<usize>]) -> Result<String, Error> {
        let spans = rules::tagged_spans(text, spans)?;
        self.reserve_markers(text);
        self.replace(text, spans)
    }

    /// Restores issued replacements in complete text.
    #[must_use]
    pub fn restore(&self, text: &str) -> String {
        self.restorer().restore(text)
    }

    /// Restores `text` as two chunks split at byte `at`.
    #[cfg(test)]
    pub(crate) fn restore_split(&self, text: &str, at: usize) -> String {
        let mut restorer = self.restorer();
        restorer.push(&text[..at]) + &restorer.push(&text[at..]) + &restorer.finish()
    }

    /// Creates a restorer for one field using the replacements issued so far.
    #[must_use]
    pub fn restorer(&self) -> Restorer {
        Restorer::new(self.replacement_map())
    }

    fn replacement_seed(&self, value: &str) -> [u8; 32] {
        self.hmac.clone().map_or_else(
            || Sha256::digest(value.as_bytes()).into(),
            |mut hmac| {
                hmac.update(DERIVATION_CONTEXT);
                hmac.update(&(self.scope.len() as u64).to_be_bytes());
                hmac.update(&self.scope);
                hmac.update(&(value.len() as u64).to_be_bytes());
                hmac.update(value.as_bytes());
                hmac.finalize().into_bytes().into()
            },
        )
    }

    fn replace(&mut self, text: &str, spans: Vec<RuleMatch<'_>>) -> Result<String, Error> {
        let mut pending = BTreeMap::new();
        let mut replacement = String::new();
        let output = rules::replace_matches(text, spans, |output, matched, next| {
            if let Some(replacement) = matched.mask.literal_replacement() {
                output.push_str(replacement);
                return Ok(());
            }
            let value = &text[matched.span.clone()];
            let seed = self.replacement_seed(value);
            let after = &text[matched.span.end..];
            let mut new_replacement = false;
            let mut alias_conflict = false;
            let mut available = |replacement: &str| {
                let original = pending
                    .get(replacement)
                    .map(String::as_str)
                    .or_else(|| self.replacements.get(replacement));
                new_replacement = original.is_none();
                original.map_or_else(
                    || !self.reserved.contains(replacement),
                    |original| original == value,
                )
            };
            let alias_template = matched.mask.alias_template();
            let aliased = alias_template.is_some_and(|template| {
                let valid_position = !joins_value(output.chars().rev())
                    && !joins_value(after.chars())
                    && next.is_none_or(|start| {
                        start > matched.span.end + usize::from(after.starts_with('.'))
                    });
                if !valid_position || !template.select(&seed, &mut replacement) {
                    return false;
                }
                if available(&replacement) {
                    true
                } else {
                    alias_conflict = true;
                    false
                }
            });
            if alias_conflict && self.strict {
                let template = alias_template.expect("alias conflict has an alias template");
                return Err(Error::ReplacementUnavailable {
                    template: template.pattern().to_owned(),
                    combinations: template.combinations,
                    attempts: 1,
                });
            }
            let probed = if aliased {
                false
            } else {
                match self.template.compiled.select_available(
                    &seed,
                    &mut replacement,
                    available,
                    self.strict,
                ) {
                    Ok(probed) => probed,
                    Err(attempts) => {
                        return Err(Error::ReplacementUnavailable {
                            template: self.template.compiled.pattern().to_owned(),
                            combinations: self.template.compiled.combinations,
                            attempts,
                        });
                    }
                }
            };
            if probed {
                self.probes = self.probes.saturating_add(1);
            }
            output.push_str(&replacement);
            if new_replacement {
                pending.insert(replacement.clone(), value.to_owned());
            }
            Ok(())
        })?;
        self.replacements
            .extend(pending, &self.template.compiled.prefix);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Digits, GeneratorBinding, Hex, Mask, Matcher, Number, Rule, WordList};

    use super::*;

    #[test]
    fn restores_only_replacements_issued_in_the_exchange() {
        let mut exchange = RedactionSession::new();
        let other = RedactionSession::new();
        let rules = Rules::new([Rule::new(Matcher::Literal("privé".into()), Mask::tag())]);
        let token = exchange.mask("privé", &rules).unwrap();
        let repeated = exchange.mask("privé", &rules).unwrap();
        assert_eq!(token, repeated);
        assert_eq!(other.restore(&token), token);

        let unissued = "<|mpx-redact-ffffffff|>";
        let input = format!("{unissued}{token}{token}");
        for split in 0..=input.len() {
            let output = exchange.restore_split(&input, split);
            assert_eq!(output, format!("{unissued}privéprivé"));
        }
    }

    #[test]
    fn keeps_originals_distinct_when_replacements_collide() {
        let mut exchange = RedactionSession::new();
        let mut old = String::new();
        let seed = exchange.replacement_seed("new");
        assert!(exchange.template.compiled.select(&seed, &mut old));
        exchange.replacements.extend(
            BTreeMap::from([(old.clone(), "old".to_owned())]),
            &exchange.template.compiled.prefix,
        );
        let spans = std::slice::from_ref(&(0..3));
        let new = exchange.mask_spans("new", spans).unwrap();
        assert_ne!(old, new);
        assert_eq!(new, exchange.mask_spans("new", spans).unwrap());
        assert_eq!(exchange.restore(&format!("{old} {new}")), "old new");
    }

    #[test]
    fn exhausted_custom_generators_leave_no_partial_replacement_map() {
        let template = PlaceholderTemplate::new(
            "<{{number}}>",
            &[GeneratorBinding::new("number", Number(0..=2))],
        )
        .unwrap();
        let mut exchange = RedactionSession::new().with_placeholder_template(template);
        let rules = Rules::new([
            Rule::new(Matcher::Literal("first".into()), Mask::tag()),
            Rule::new(Matcher::Literal("second".into()), Mask::tag()),
        ]);
        assert!(matches!(
            exchange.mask("<0> <1> first second", &rules),
            Err(Error::ReplacementUnavailable { .. })
        ));
        assert_eq!(exchange.restore("<2>"), "<2>");
        assert_eq!(exchange.replacement_map().entries().len(), 0);
        assert_eq!(exchange.mask("first", &rules).unwrap(), "<2>");
        assert_eq!(exchange.restore("<0> <2>"), "<0> first");
    }

    #[test]
    fn literal_mask_keeps_overlapping_text_out_of_the_restore_lookup() {
        let rules = Rules::new([
            Rule::new(Matcher::Literal("private".into()), Mask::tag()),
            Rule::new(
                Matcher::Literal("vate-ID".into()),
                Mask::literal("[removed]"),
            ),
        ]);
        let mut exchange = RedactionSession::new();
        let masked = exchange.mask("private-ID", &rules).unwrap();
        assert_eq!(masked, "[removed]");
        assert_eq!(exchange.restore(&masked), "[removed]");
        assert_eq!(exchange.replacement_map().entries().len(), 0);
    }

    #[test]
    fn replacement_map_snapshots_share_the_lookup_and_keep_their_contents() {
        let words = WordList::new(&["cedar"]).unwrap();
        let rules = Rules::new([
            Rule::new(Matcher::Literal("private".into()), Mask::tag()),
            Rule::new(
                Matcher::Literal("Nightjar".into()),
                Mask::alias(
                    "project",
                    "{{word}}",
                    &[GeneratorBinding::new("word", words)],
                )
                .unwrap(),
            ),
        ]);
        let mut exchange = RedactionSession::new();
        let tag = exchange.mask("private", &rules).unwrap();
        let first_snapshot = exchange.replacement_map();
        let masked = exchange.mask("private, Nightjar, private", &rules).unwrap();
        assert_eq!(masked, format!("{tag}, cedar, {tag}"));
        let replacements = exchange.replacement_map();
        assert_eq!(replacements.entries().len(), 2);
        assert_eq!(replacements.get(&tag), Some("private"));
        assert_eq!(replacements.get("cedar"), Some("Nightjar"));
        assert_eq!(exchange.restore(&masked), "private, Nightjar, private");
        assert_eq!(first_snapshot.entries().len(), 1);
        assert_eq!(first_snapshot.get(&tag), Some("private"));
    }

    #[test]
    fn restores_exact_aliases_at_every_stream_boundary() {
        let words = WordList::new(&["cedar"]).unwrap();
        let rule = Rule::new(
            Matcher::Literal("alice@corp.test".to_owned()),
            Mask::alias(
                "email",
                "{{word}}-{{word}}@example.com",
                &[GeneratorBinding::new("word", words)],
            )
            .unwrap(),
        );
        let rules = Rules::new([rule]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask("alice@corp.test", &rules).unwrap();
        assert_eq!(alias, "cedar-cedar@example.com");
        assert_eq!(exchange.mask("alice@corp.test", &rules).unwrap(), alias);
        let text = format!("{alias}. ({alias}) x{alias} {alias}.other");
        for boundary in 0..=text.len() {
            let output = exchange.restore_split(&text, boundary);
            assert_eq!(
                output,
                format!("alice@corp.test. (alice@corp.test) x{alias} {alias}.other")
            );
        }
    }

    #[test]
    fn seeds_word_selection_with_the_exact_match_across_exchanges() {
        let words = WordList::new(&["cedar", "maple", "birch", "willow"]).unwrap();
        let mask = Mask::alias(
            "email",
            "{{word}}-{{word}}@example.com",
            &[GeneratorBinding::new("word", words)],
        )
        .unwrap();
        let rules = Rules::new([
            Rule::new(Matcher::Literal("alice@corp.test".into()), mask.clone()),
            Rule::new(Matcher::Literal("bob@corp.test".into()), mask),
        ]);
        for _ in 0..3 {
            let mut exchange = RedactionSession::new();
            let original = "alice@corp.test, bob@corp.test";
            let masked = exchange.mask(original, &rules).unwrap();
            assert_eq!(masked, "cedar-maple@example.com, willow-willow@example.com");
            assert_eq!(exchange.restore(&masked), original);
        }
    }

    #[test]
    fn mixes_named_lists_and_generated_slots_in_one_alias() {
        let rule = Rule::new(
            Matcher::Literal("ID-1234".to_owned()),
            Mask::alias(
                "account",
                "{{first_name}}-{{digits}}-{{year}}-{{hex}}",
                &[
                    GeneratorBinding::new("first_name", WordList::new(&["Mira"]).unwrap()),
                    GeneratorBinding::new("digits", Digits(4)),
                    GeneratorBinding::new("year", Number(1970..=2025)),
                    GeneratorBinding::new("hex", Hex(40)),
                ],
            )
            .unwrap(),
        );
        let rules = Rules::new([rule]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask("ID-1234", &rules).unwrap();
        let (digits, hex) = alias
            .strip_prefix("Mira-")
            .unwrap()
            .split_once('-')
            .unwrap();
        assert_eq!(digits.len(), 4);
        assert!(digits.bytes().all(|byte| byte.is_ascii_digit()));
        let (year, hex) = hex.split_once('-').unwrap();
        assert!((1970..=2025).contains(&year.parse::<u64>().unwrap()));
        assert_eq!(hex.len(), 40);
        assert!(hex.bytes().all(|byte| b"0123456789abcdef".contains(&byte)));
        let mut restorer = exchange.restorer();
        assert_eq!(restorer.push(&alias[..7]), "");
        assert_eq!(restorer.push(&alias[7..]), "");
        assert_eq!(restorer.finish(), "ID-1234");
        let mut other = RedactionSession::new();
        assert_eq!(other.mask("ID-1234", &rules).unwrap(), alias);
    }

    #[test]
    fn falls_back_to_tags_for_aliases_already_present_or_unbounded() {
        let words = WordList::new(&["cedar"]).unwrap();
        let rule = Rule::new(
            Matcher::Literal("alice@corp.test".to_owned()),
            Mask::alias(
                "email",
                "{{word}}-{{word}}@example.com",
                &[GeneratorBinding::new("word", words)],
            )
            .unwrap(),
        );
        let rules = Rules::new([rule]);
        let mut exchange = RedactionSession::new();
        exchange.reserve_replacements_in("Use cedar-cedar@example.com.", &rules);
        let masked = exchange.mask("alice@corp.test", &rules).unwrap();
        assert!(masked.starts_with("<|mpx-redact-"));
        assert_eq!(exchange.restore(&masked), "alice@corp.test");
        assert_eq!(
            exchange.restore("cedar-cedar@example.com"),
            "cedar-cedar@example.com"
        );

        let mut exchange = RedactionSession::new();
        let masked = exchange.mask("xalice@corp.test", &rules).unwrap();
        assert!(masked.starts_with("x<|mpx-redact-"));
        assert_eq!(exchange.restore(&masked), "xalice@corp.test");
    }
}
