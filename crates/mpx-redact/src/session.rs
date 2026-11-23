use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    ops::Range,
};

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256, digest::Update};

use crate::{
    Error, Replacement, ReplacementMap, Restorer, Rules,
    prepared_aliases::PreparedAliases,
    replacement_template::{ReplacementTemplate, contains_replacement, is_alias, joins_value},
    rules::{self, RuleMatch},
};

const DERIVATION_CONTEXT: &[u8] = b"mpx-redact/v1\0";

/// Masks text and keeps the replacements for one exchange. Rejects collisions by default.
#[derive(Default)]
pub struct RedactionSession {
    hmac: Option<Hmac<Sha256>>,
    scope: Vec<u8>,
    fallback: Option<Replacement>,
    reserved: HashSet<String>,
    replacements: ReplacementMap,
}

impl fmt::Debug for RedactionSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedactionSession").finish_non_exhaustive()
    }
}

impl RedactionSession {
    /// Creates an exchange for explicitly configured masks.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Returns [`Error::ReplacementUnavailable`] on a collision. This is the default.
    #[must_use]
    pub fn with_strict(mut self) -> Self {
        self.fallback = None;
        self
    }

    /// Tries the supplied fallback when a replacement collides or an alias would
    /// join adjacent text. Fallbacks for permanent rules remain permanent.
    ///
    /// Choose a fallback that you expect to avoid collisions with original text
    /// and issued replacements. An unavailable fallback fails masking.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Replacement) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// Returns a snapshot of the replacements issued so far.
    #[must_use]
    pub fn replacement_map(&self) -> ReplacementMap {
        self.replacements.clone()
    }

    /// Records an original field and literal replacements for collision checks.
    ///
    /// Call this on all original fields before masking a multi-field exchange.
    pub fn reserve_replacements_in(&mut self, text: &str, rules: &Rules) {
        self.reserve(text, rules.literal_replacements());
    }

    fn reserve<'a>(&mut self, text: &'a str, literals: impl IntoIterator<Item = &'a str>) {
        for value in std::iter::once(text).chain(literals) {
            if !self.reserved.contains(value) {
                self.reserved.insert(value.to_owned());
            }
        }
    }

    /// Releases the original fields reserved for collision checks after masking.
    pub fn clear_reserved_replacements(&mut self) {
        self.reserved = HashSet::new();
    }

    /// Masks matches from the supplied rules.
    ///
    /// # Errors
    /// Returns an error if detection fails or a replacement cannot be applied.
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
    /// Returns an error if detection fails or a replacement cannot be applied.
    pub fn mask_reserved(&mut self, text: &str, rules: &Rules) -> Result<String, Error> {
        self.replace(text, rules.try_find_matches(text, true)?)
    }

    /// Masks externally supplied UTF-8 byte ranges.
    ///
    /// The replacement mask must be supplied explicitly.
    ///
    /// # Errors
    /// Returns an error for invalid ranges or when the replacement blocks masking
    /// or cannot be generated.
    pub fn mask_spans(
        &mut self,
        text: &str,
        spans: &[Range<usize>],
        replacement: &Replacement,
    ) -> Result<String, Error> {
        let spans = rules::masked_spans(text, spans, replacement)?;
        self.reserve(text, replacement.literal_replacement());
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

    /// Derives the base seed for `value`, shared by every template.
    ///
    /// The base seed feeds slot 0 directly, so a single-slot template costs one
    /// keyed hash per value.
    fn replacement_seed(&self, value: &str) -> [u8; 32] {
        self.hmac.clone().map_or_else(
            || {
                let mut hash = Sha256::new();
                feed_seed(&mut hash, &self.scope, value);
                hash.finalize().into()
            },
            |mut hmac| {
                feed_seed(&mut hmac, &self.scope, value);
                hmac.finalize().into_bytes().into()
            },
        )
    }

    fn replace(&mut self, text: &str, spans: Vec<RuleMatch<'_>>) -> Result<String, Error> {
        if spans.iter().any(|matched| matched.replacement.is_block()) {
            return Err(Error::Blocked);
        }
        let spans = rules::merge_matches(spans)?;
        let mut aliases = PreparedAliases::new(text, &spans, &self.reserved, |value| {
            self.replacement_seed(value)
        })?;
        let mut pending = PendingReplacements::default();
        let mut reserved_markers = ReservedMarkerSearch::new(&self.reserved);
        let mut replacement = String::new();
        let mut slot = String::new();
        let output = rules::replace_matches(text, spans, |output, matched, next| {
            let value = &text[matched.span.clone()];
            let after = &text[matched.span.end..];
            let mut current = matched.replacement;
            let mut fallback = self.fallback.as_ref();
            let mut is_primary = true;
            let mut seed = None;
            loop {
                let rejected = if let Some(literal) = current.literal_replacement() {
                    if self.permanent_available(literal, &pending) {
                        output.push_str(literal);
                        pending.record_permanent(value, literal);
                        return Ok(());
                    }
                    literal
                } else {
                    let (template, reversible) = current.as_template().ok_or(Error::Blocked)?;
                    let reversible = reversible
                        && (is_primary
                            || matched
                                .replacement
                                .as_template()
                                .is_some_and(|(_, reversible)| reversible));
                    let framed = template.framed();
                    let is_alias = reversible && !framed;
                    let prepared = if is_primary && is_alias {
                        aliases.next()
                    } else {
                        None
                    };
                    // A reversible alias must stay separate from adjacent value text.
                    let positioned = !is_alias
                        || !(joins_value(output.chars().rev())
                            || joins_value(after.chars())
                            || next.is_some_and(|start| {
                                start <= matched.span.end + usize::from(after.starts_with('.'))
                            }));
                    if positioned {
                        let (candidate, reserved_collision) = if is_primary && is_alias {
                            let alias = prepared.ok_or_else(|| unavailable(template.pattern()))?;
                            (alias.text, Some(alias.collides))
                        } else {
                            let seed = seed.get_or_insert_with(|| self.replacement_seed(value));
                            if !template.generate(seed, &mut replacement, &mut slot) {
                                return Err(unavailable(template.pattern()));
                            }
                            (replacement.as_str(), None)
                        };
                        let available = if reversible {
                            self.available(candidate, value, framed, &pending, || {
                                if let Some(collision) = reserved_collision {
                                    collision
                                } else if framed {
                                    reserved_markers.contains(candidate, template)
                                } else {
                                    self.reserved
                                        .iter()
                                        .any(|text| contains_replacement(text, candidate, false))
                                }
                            })
                        } else {
                            self.permanent_available(candidate, &pending)
                        };
                        if available {
                            output.push_str(candidate);
                            pending.record(
                                value,
                                candidate,
                                template,
                                reversible,
                                &self.replacements,
                            );
                            return Ok(());
                        }
                    }
                    template.pattern()
                };
                current = fallback.take().ok_or_else(|| unavailable(rejected))?;
                is_primary = false;
            }
        })?;
        self.commit(pending);
        Ok(output)
    }

    fn commit(&mut self, pending: PendingReplacements) {
        let mut aliases = BTreeMap::new();
        let mut markers = BTreeMap::<String, BTreeMap<String, String>>::new();
        for (replacement, pending) in pending.reversible {
            if let Some(prefix) = pending.marker_prefix {
                markers
                    .entry(prefix)
                    .or_default()
                    .insert(replacement, pending.original);
            } else {
                aliases.insert(replacement, pending.original);
            }
        }
        self.replacements.extend_aliases(aliases);
        for (prefix, entries) in markers {
            self.replacements.extend_markers(entries, &prefix);
        }
        self.replacements.extend_permanent(pending.permanent);
    }

    fn available(
        &self,
        replacement: &str,
        value: &str,
        framed: bool,
        pending: &PendingReplacements,
        reserved_collision: impl FnOnce() -> bool,
    ) -> bool {
        let original = pending
            .reversible
            .get(replacement)
            .map(|entry| entry.original.as_str())
            .or_else(|| self.replacements.get(replacement));
        if let Some(original) = original {
            return original == value;
        }
        !reserved_collision()
            && !self
                .fallback
                .as_ref()
                .and_then(Replacement::literal_replacement)
                .into_iter()
                .chain(pending.permanent.values().flatten().map(String::as_str))
                .chain(self.replacements.permanent_values())
                .any(|permanent| contains_replacement(permanent, replacement, framed))
    }

    fn permanent_available(&self, replacement: &str, pending: &PendingReplacements) -> bool {
        !pending
            .reversible
            .keys()
            .map(String::as_str)
            .chain(self.replacements.entries().map(|(issued, _)| issued))
            .any(|issued| contains_replacement(replacement, issued, !is_alias(issued)))
    }
}

struct ReservedMarkerSearch<'text, 'template> {
    texts: &'text HashSet<String>,
    markers: HashMap<&'template str, ReservedMarkers<'text>>,
    searched_marker: bool,
}

impl<'text, 'template> ReservedMarkerSearch<'text, 'template> {
    fn new(texts: &'text HashSet<String>) -> Self {
        Self {
            texts,
            markers: HashMap::new(),
            searched_marker: false,
        }
    }

    fn contains(&mut self, replacement: &str, template: &'template ReplacementTemplate) -> bool {
        if !self.searched_marker {
            self.searched_marker = true;
            return self.texts.iter().any(|text| text.contains(replacement));
        }
        let prefix = template.prefix.as_str();
        let markers = self
            .markers
            .entry(prefix)
            .or_insert_with(|| ReservedMarkers::new(self.texts, prefix, replacement.len()));
        markers.contains(replacement)
    }
}

struct ReservedMarkers<'a> {
    width: usize,
    starts: Vec<&'a str>,
    windows: HashSet<&'a str>,
}

impl<'a> ReservedMarkers<'a> {
    fn new(texts: &'a HashSet<String>, prefix: &str, width: usize) -> Self {
        let first = prefix.chars().next().expect("framed template has a prefix");
        let starts: Vec<_> = texts
            .iter()
            // Check each possible start to include overlapping prefixes.
            .flat_map(|text| text.match_indices(first).map(|(start, _)| &text[start..]))
            .filter(|text| text.starts_with(prefix))
            .collect();
        let windows = starts.iter().filter_map(|text| text.get(..width)).collect();
        Self {
            width,
            starts,
            windows,
        }
    }

    fn contains(&self, replacement: &str) -> bool {
        if replacement.len() == self.width {
            self.windows.contains(replacement)
        } else {
            // Custom slots can render different widths. Reuse the prefix positions.
            self.starts.iter().any(|text| text.starts_with(replacement))
        }
    }
}

fn feed_seed(hash: &mut impl Update, scope: &[u8], value: &str) {
    hash.update(DERIVATION_CONTEXT);
    for field in [scope, value.as_bytes()] {
        hash.update(&(field.len() as u64).to_be_bytes());
        hash.update(field);
    }
}

fn unavailable(template: &str) -> Error {
    Error::ReplacementUnavailable {
        template: template.to_owned(),
    }
}

struct PendingReplacement {
    original: String,
    marker_prefix: Option<String>,
}

#[derive(Default)]
struct PendingReplacements {
    reversible: BTreeMap<String, PendingReplacement>,
    permanent: BTreeMap<String, BTreeSet<String>>,
}

impl PendingReplacements {
    fn record(
        &mut self,
        value: &str,
        replacement: &str,
        template: &ReplacementTemplate,
        reversible: bool,
        existing: &ReplacementMap,
    ) {
        if !reversible {
            self.record_permanent(value, replacement);
        } else if existing.get(replacement).is_none() && !self.reversible.contains_key(replacement)
        {
            self.reversible.insert(
                replacement.to_owned(),
                PendingReplacement {
                    original: value.to_owned(),
                    marker_prefix: template.framed().then(|| template.prefix.clone()),
                },
            );
        }
    }

    fn record_permanent(&mut self, original: &str, replacement: &str) {
        self.permanent
            .entry(original.to_owned())
            .or_default()
            .insert(replacement.to_owned());
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matcher, Replacement, Slot};

    use super::*;

    fn replacement() -> Replacement {
        Replacement::kind("SECRET").unwrap()
    }

    #[test]
    fn batched_reserved_aliases_fall_back_or_fail_strictly() {
        let rules = Rules::new([
            (
                Matcher::literal("first"),
                Replacement::template("{}", &[Slot::literal("cèdre")]).unwrap(),
            ),
            (
                Matcher::literal("second"),
                Replacement::template("{}", &[Slot::literal("maple")]).unwrap(),
            ),
        ]);
        let input = "second first first";
        let mut exchange = RedactionSession::new().with_fallback(replacement());
        exchange.reserve_replacements_in("cèdre and maple.leaf", &rules);
        let masked = exchange.mask(input, &rules).unwrap();
        let words: Vec<_> = masked.split_whitespace().collect();
        assert_eq!(words[0], "maple");
        assert!(words[1].starts_with("{{SECRET_"));
        assert_eq!(words[1], words[2]);
        assert_eq!(exchange.replacement_map().entries().len(), 2);
        for (split, _) in masked.char_indices() {
            assert_eq!(exchange.restore_split(&masked, split), input);
        }

        let mut strict = RedactionSession::default();
        strict.reserve_replacements_in("cèdre and maple.leaf", &rules);
        assert!(matches!(
            strict.mask(input, &rules),
            Err(Error::ReplacementUnavailable { template }) if template == "{}"
        ));
        assert_eq!(strict.replacement_map().entries().len(), 0);
    }

    #[test]
    fn batched_aliases_keep_the_first_original_and_reuse_its_mapping() {
        let alias = Replacement::template("{}", &[Slot::literal("cedar")]).unwrap();
        let rules = Rules::new([
            (Matcher::literal("first"), alias.clone()),
            (Matcher::literal("second"), alias),
            (
                Matcher::literal("third"),
                Replacement::template("{}", &[Slot::literal("maple")]).unwrap(),
            ),
        ]);
        let text = "first second first third";
        let mut exchange = RedactionSession::new().with_fallback(replacement());
        let masked = exchange.mask(text, &rules).unwrap();
        let words: Vec<_> = masked.split_whitespace().collect();
        assert_eq!(words[0], "cedar");
        assert!(words[1].starts_with("{{SECRET_"));
        assert_eq!(words[0], words[2]);
        assert_eq!(words[3], "maple");
        assert_eq!(exchange.restore(&masked), text);

        exchange.reserve_replacements_in("cedar", &rules);
        assert_eq!(exchange.mask("first third", &rules).unwrap(), "cedar maple");
        assert_eq!(exchange.replacement_map().entries().len(), 3);

        let mut strict = RedactionSession::new()
            .with_fallback(replacement())
            .with_strict();
        assert!(matches!(
            strict.mask(text, &rules),
            Err(Error::ReplacementUnavailable { template }) if template == "{}"
        ));
        assert_eq!(strict.replacement_map().entries().len(), 0);
    }

    #[test]
    fn reserved_marker_index_handles_overlapping_prefixes_and_variable_widths() {
        let texts = HashSet::from(["&&&a! &&末!".to_owned()]);
        let markers = ReservedMarkers::new(&texts, "&&", "&&a!".len());
        assert!(markers.contains("&&a!"));
        assert!(markers.contains("&&末!"));
        assert!(!markers.contains("&&b!"));
        assert!(!markers.contains("&&x末!"));
    }

    #[test]
    fn restores_only_replacements_issued_in_the_exchange() {
        let mut exchange = RedactionSession::new();
        let other = RedactionSession::new();
        let rules = Rules::new([(Matcher::literal("privé"), replacement())]);
        let token = exchange.mask("privé", &rules).unwrap();
        let repeated = exchange.mask("privé", &rules).unwrap();
        assert_eq!(token, repeated);
        assert_eq!(other.restore(&token), token);

        let unissued = "{}";
        let input = format!("{unissued}{token}{token}");
        for split in 0..=input.len() {
            let output = exchange.restore_split(&input, split);
            assert_eq!(output, format!("{unissued}privéprivé"));
        }
    }

    #[test]
    fn keeps_originals_distinct_when_replacements_collide() {
        let mut exchange = RedactionSession::new().with_fallback(replacement());
        let mask = Replacement::kind("CUSTOM").unwrap();
        let (template, _) = mask.as_template().unwrap();
        let mut old = String::new();
        let seed = exchange.replacement_seed("new");
        assert!(template.generate(&seed, &mut old, &mut String::new()));
        exchange.replacements.extend_markers(
            BTreeMap::from([(old.clone(), "old".to_owned())]),
            &template.prefix,
        );
        let spans = std::slice::from_ref(&(0..3));
        let new = exchange.mask_spans("new", spans, &mask).unwrap();
        assert!(new.starts_with("{{SECRET_"));
        assert_eq!(new, exchange.mask_spans("new", spans, &mask).unwrap());
        assert_eq!(exchange.restore(&format!("{old} {new}")), "old new");
    }

    #[test]
    fn strict_collision_leaves_no_partial_replacement_map() {
        let mask = Replacement::template("<{}>", &[Slot::number(0..=0)]).unwrap();
        let rules = Rules::new([
            (Matcher::literal("first"), mask.clone()),
            (Matcher::literal("second"), mask),
        ]);
        let original = "first second";
        let mut exchange = RedactionSession::new();
        assert!(matches!(
            exchange.mask(original, &rules),
            Err(Error::ReplacementUnavailable { template }) if template == "<{}>"
        ));
        assert_eq!(exchange.restore("<0>"), "<0>");
        assert_eq!(exchange.replacement_map().entries().len(), 0);
        assert_eq!(exchange.mask("first", &rules).unwrap(), "<0>");
        assert_eq!(exchange.restore("<0>"), "first");
    }

    #[test]
    fn taken_fallback_markers_report_an_error() {
        let rules = Rules::new([(Matcher::literal("hunter2"), replacement())]);
        let marker = RedactionSession::new()
            .with_scope("s")
            .mask("hunter2", &rules)
            .unwrap();
        // History echoes the marker this value would get, bound to nothing here.
        let original = format!("earlier {marker} now hunter2");
        let mut exchange = RedactionSession::new()
            .with_scope("s")
            .with_fallback(replacement());
        assert!(matches!(
            exchange.mask(&original, &rules),
            Err(Error::ReplacementUnavailable { template }) if template == "{{SECRET_{}}}"
        ));
        assert_eq!(exchange.replacement_map().entries().len(), 0);
        assert_eq!(exchange.restore(&original), original);
    }

    #[test]
    fn literal_mask_keeps_overlapping_text_out_of_the_restore_lookup() {
        let rules = Rules::new([
            (Matcher::literal("private"), replacement()),
            (Matcher::literal("vate-ID"), Replacement::mask("[removed]")),
        ]);
        let mut exchange = RedactionSession::new();
        let masked = exchange.mask("private-ID", &rules).unwrap();
        assert_eq!(masked, "[removed]");
        assert_eq!(exchange.restore(&masked), "[removed]");
        assert_eq!(exchange.replacement_map().entries().len(), 0);
    }

    #[test]
    fn replacement_map_snapshots_share_the_lookup_and_keep_their_contents() {
        let words = Slot::wordlist(["cedar"]).unwrap();
        let rules = Rules::new([
            (Matcher::literal("private"), replacement()),
            (
                Matcher::literal("Nightjar"),
                Replacement::template("{}", &[words]).unwrap(),
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
        let words = Slot::wordlist(["cedar"]).unwrap();
        let rule = (
            Matcher::literal("alice@corp.test".to_owned()),
            Replacement::template("{}-{}@example.com", &[words.clone(), words]).unwrap(),
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
        let words = Slot::wordlist(["cedar", "maple", "birch", "willow"]).unwrap();
        let mask = Replacement::template("{}-{}@example.com", &[words.clone(), words]).unwrap();
        let rules = Rules::new([
            (Matcher::literal("alice@corp.test"), mask.clone()),
            (Matcher::literal("charlie@corp.test"), mask),
        ]);
        for _ in 0..3 {
            let mut exchange = RedactionSession::new();
            let original = "alice@corp.test, charlie@corp.test";
            let masked = exchange.mask(original, &rules).unwrap();
            assert_eq!(masked, "willow-birch@example.com, maple-maple@example.com");
            assert_eq!(exchange.restore(&masked), original);
        }
    }

    #[test]
    fn streams_aliases_across_dots_and_overlong_values() {
        let rules = Rules::new([
            (
                Matcher::literal("first"),
                Replacement::template("{}", &[Slot::literal("cedar")]).unwrap(),
            ),
            (
                Matcher::literal("second"),
                Replacement::template("{}", &[Slot::literal("cedar.maple")]).unwrap(),
            ),
        ]);
        let mut session = RedactionSession::new();
        assert_eq!(
            session.mask("first second", &rules).unwrap(),
            "cedar cedar.maple"
        );

        let mut restorer = session.restorer();
        assert_eq!(restorer.push("cedar."), "");
        assert_eq!(restorer.push(""), "");
        assert_eq!(restorer.push("maple."), "");
        assert_eq!(restorer.push("suffix "), "cedar.maple.suffix ");
        assert_eq!(restorer.push("cedar."), "");
        assert_eq!(restorer.push(". cedar.maple!"), "first.. second!");
        assert_eq!(restorer.finish(), "");
    }

    #[test]
    fn streams_aliases_as_soon_as_they_diverge() {
        let rules = Rules::new([
            (
                Matcher::literal("first"),
                Replacement::template("{}", &[Slot::literal("cèdre")]).unwrap(),
            ),
            (
                Matcher::literal("second"),
                Replacement::template("{}", &[Slot::literal("cèdre.maple")]).unwrap(),
            ),
            (
                Matcher::literal("third"),
                Replacement::template("{}", &[Slot::literal("pine")]).unwrap(),
            ),
            (
                Matcher::literal("fourth"),
                Replacement::template(".{}!", &[Slot::literal("maple")]).unwrap(),
            ),
        ]);
        let mut session = RedactionSession::new();
        session.mask("first second fourth", &rules).unwrap();

        let mut restorer = session.restorer();
        assert_eq!(restorer.push("hello cè"), "hello ");
        assert_eq!(restorer.push("X"), "cèX");
        assert_eq!(restorer.push("."), "");
        assert_eq!(restorer.push("cèdre "), ".cèdre ");
        assert_eq!(restorer.push("cèdre"), "");
        assert_eq!(restorer.push(".ma"), "");
        assert_eq!(restorer.push("ple!"), "firstfourth");
        assert_eq!(restorer.push(" cèdre.ma"), " ");
        assert_eq!(restorer.finish(), "cèdre.ma");

        let mut snapshot = session.restorer();
        assert_eq!(snapshot.push("p"), "p");
        session.mask("third", &rules).unwrap();
        assert_eq!(snapshot.push("ine "), "ine ");
        let mut restorer = session.restorer();
        assert_eq!(restorer.push("p"), "");
        assert_eq!(restorer.push("ine "), "third ");
        assert_eq!(restorer.finish(), "");
    }

    #[test]
    fn mixes_named_lists_and_generated_slots_in_one_alias() {
        let rule = (
            Matcher::literal("ID-1234".to_owned()),
            Replacement::template(
                "{}-{}-{}-{}",
                &[
                    Slot::wordlist(["Mira"]).unwrap(),
                    Slot::digits(4),
                    Slot::number(1970..=2025),
                    Slot::base16(40),
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
    fn rejects_aliases_that_cannot_be_issued() {
        let words = Slot::wordlist(["cedar"]).unwrap();
        let rule = (
            Matcher::literal("alice@corp.test".to_owned()),
            Replacement::template("{}-{}@example.com", &[words.clone(), words]).unwrap(),
        );
        let rules = Rules::new([rule]);
        let mut exchange = RedactionSession::new();
        exchange.reserve_replacements_in("Use cedar-cedar@example.com.", &rules);
        assert!(matches!(
            exchange.mask("alice@corp.test", &rules),
            Err(Error::ReplacementUnavailable { .. })
        ));
        assert_eq!(
            exchange.restore("cedar-cedar@example.com"),
            "cedar-cedar@example.com"
        );

        let mut exchange = RedactionSession::new();
        assert!(matches!(
            exchange.mask("xalice@corp.test", &rules),
            Err(Error::ReplacementUnavailable { .. })
        ));
    }

    #[test]
    fn unframed_aliases_next_to_value_text_use_the_fallback_marker() {
        let words = Slot::wordlist(["cedar"]).unwrap();
        let rules = Rules::new([(
            Matcher::literal("alice"),
            Replacement::template("{}", &[words]).unwrap(),
        )]);
        let mut exchange = RedactionSession::new().with_fallback(replacement());
        let masked = exchange.mask("alice_x and alice", &rules).unwrap();
        let (fallback, rest) = masked.split_once("_x and ").unwrap();
        assert!(fallback.starts_with("{{SECRET_"), "{masked:?}");
        assert_eq!(rest, "cedar");
        assert_eq!(exchange.restore(&masked), "alice_x and alice");
    }
}
