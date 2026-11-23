use std::{collections::HashSet, fmt, ops::Range, sync::Arc};

use regex::Regex;

use crate::{
    Error, GeneratorBinding, builtins,
    template::{AliasTemplate, is_value_char},
};

/// Reusable redaction rules.
#[derive(Clone)]
pub struct Rules {
    rules: Vec<Rule>,
}

/// Determines which spans a redaction rule matches.
#[derive(Clone)]
#[non_exhaustive]
pub enum Matcher {
    /// An exact, case-sensitive string. An empty literal matches no spans.
    Literal(String),
    /// A compiled Rust regular expression. Zero-width matches are ignored.
    Regex(Regex),
    /// Conventional email addresses with DNS-style domains.
    Email,
    /// IPv4 and IPv6 literals.
    IpAddress,
    /// Payment-card numbers with 13 to 19 digits and a valid Luhn checksum.
    PaymentCard,
    /// API keys with recognized service prefixes.
    ApiKey,
    /// PEM and PGP private-key blocks with matching begin and end labels.
    PrivateKey,
}

/// Determines what a rule emits for each matched span.
#[derive(Clone)]
pub struct Mask(MaskKind);

#[derive(Clone)]
enum MaskKind {
    Literal(String),
    Reversible(Option<Arc<AliasTemplate>>),
}

pub(crate) static TAG_MASK: Mask = Mask::tag();

/// A source matcher and the replacement to apply to its matches.
///
/// ~~~
/// use mpx_redact::{GeneratorBinding, Mask, Matcher, Rule, Rules, WordList};
///
/// let mask = Mask::alias("email", "{{first}}-{{last}}@{{domain}}", &[
///     GeneratorBinding::new("first", WordList::firstname()),
///     GeneratorBinding::new("last", WordList::lastname()),
///     GeneratorBinding::new("domain", WordList::domains()),
/// ])?;
/// let rule = Rule::new(Matcher::Email, mask);
/// let rules = Rules::new([rule]);
/// assert_eq!(rules.find("Email person@example.net"), [6..24]);
/// # Ok::<(), mpx_redact::Error>(())
/// ~~~
#[derive(Debug, Clone)]
pub struct Rule {
    matcher: Matcher,
    mask: Mask,
}

pub(crate) struct RuleMatch<'a> {
    pub(crate) span: Range<usize>,
    pub(crate) mask: &'a Mask,
}

impl fmt::Debug for Matcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(_) => f.debug_tuple("Literal").finish_non_exhaustive(),
            Self::Regex(_) => f.debug_tuple("Regex").finish_non_exhaustive(),
            Self::Email => f.write_str("Email"),
            Self::IpAddress => f.write_str("IpAddress"),
            Self::PaymentCard => f.write_str("PaymentCard"),
            Self::ApiKey => f.write_str("ApiKey"),
            Self::PrivateKey => f.write_str("PrivateKey"),
        }
    }
}

impl fmt::Debug for Mask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MaskKind::Literal(_) => f.debug_tuple("Literal").finish_non_exhaustive(),
            MaskKind::Reversible(alias) => {
                f.debug_struct("Reversible").field("alias", alias).finish()
            }
        }
    }
}

impl fmt::Debug for Rules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rules")
            .field("rule_count", &self.rules.len())
            .finish()
    }
}

impl Matcher {
    /// Compiles a Rust regular expression.
    ///
    /// # Errors
    /// Returns an error if the expression cannot compile.
    pub fn regex(pattern: &str) -> Result<Self, Error> {
        Regex::new(pattern)
            .map(Self::Regex)
            .map_err(|_| Error::InvalidRegex)
    }

    fn find(&self, text: &str, mut record: impl FnMut(Range<usize>)) {
        match self {
            Self::Literal(literal) => {
                let Some(first) = literal.chars().next() else {
                    return;
                };
                let mut cursor = 0;
                while let Some(offset) = text[cursor..].find(literal) {
                    let start = cursor + offset;
                    record(start..start + literal.len());
                    cursor = start + first.len_utf8();
                }
            }
            Self::Regex(regex) => {
                for matched in regex.find_iter(text).filter(|matched| !matched.is_empty()) {
                    record(matched.range());
                }
            }
            Self::Email => builtins::find_email(text, &mut record),
            Self::IpAddress => builtins::find_ip(text, &mut record),
            Self::PaymentCard => builtins::find_payment_card(text, &mut record),
            Self::ApiKey => builtins::find_api_key(text, &mut record),
            Self::PrivateKey => builtins::find_private_key(text, &mut record),
        }
    }
}

impl Mask {
    /// Creates a permanent mask that replaces matches with the supplied text.
    #[must_use]
    pub fn literal(replacement: impl Into<String>) -> Self {
        Self(MaskKind::Literal(replacement.into()))
    }

    /// Creates a reversible mask that emits the exchange's tag.
    #[must_use]
    pub const fn tag() -> Self {
        Self(MaskKind::Reversible(None))
    }

    /// Creates a reversible mask with a deterministic alias template.
    ///
    /// Each `{{name}}` requires a [`GeneratorBinding`]. Every slot uses a
    /// stable hash of the match. A valid template with a zero-capacity
    /// generator, such as an empty [`WordList`](crate::WordList), uses the tag.
    ///
    /// # Errors
    /// Returns an error for invalid bindings, unbound slots, or an invalid template.
    pub fn alias(
        kind: impl Into<String>,
        template: &str,
        bindings: &[GeneratorBinding<'_>],
    ) -> Result<Self, Error> {
        let alias = AliasTemplate::compile(kind.into(), template, bindings)?.map(Arc::new);
        Ok(Self(MaskKind::Reversible(alias)))
    }

    /// Returns the product of the alias slot capacities, capped at `u128::MAX`.
    ///
    /// Each slot occurrence counts separately. The number of distinct aliases
    /// can be smaller when different combinations produce the same value.
    /// Returns `None` for literal and tag-only masks.
    #[must_use]
    pub fn alias_combinations(&self) -> Option<u128> {
        self.alias_template().map(|template| template.combinations)
    }

    pub(crate) fn alias_template(&self) -> Option<&AliasTemplate> {
        match &self.0 {
            MaskKind::Reversible(alias) => alias.as_deref(),
            MaskKind::Literal(_) => None,
        }
    }

    pub(crate) fn literal_replacement(&self) -> Option<&str> {
        match &self.0 {
            MaskKind::Literal(replacement) => Some(replacement),
            MaskKind::Reversible(_) => None,
        }
    }

    fn merge<'a>(&'a self, other: &'a Self) -> &'a Self {
        match (&self.0, &other.0) {
            (MaskKind::Literal(_), _) => self,
            (_, MaskKind::Literal(_)) => other,
            (MaskKind::Reversible(Some(left)), MaskKind::Reversible(Some(right)))
                if Arc::ptr_eq(left, right) =>
            {
                self
            }
            _ => &TAG_MASK,
        }
    }
}

impl Rule {
    /// Pairs a source matcher with its replacement.
    #[must_use]
    pub const fn new(matcher: Matcher, mask: Mask) -> Self {
        Self { matcher, mask }
    }
}

impl Rules {
    /// Collects rules in the order supplied.
    #[must_use]
    pub fn new(rules: impl IntoIterator<Item = Rule>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
        }
    }

    pub(crate) fn reserve_aliases_in(&self, text: &str, reserved: &mut HashSet<String>) {
        let mut aliases = self
            .rules
            .iter()
            .filter_map(|rule| rule.mask.alias_template())
            .peekable();
        if aliases.peek().is_none() {
            return;
        }
        for value in values_in(text).chain(self.literal_replacements().flat_map(values_in)) {
            if !reserved.contains(value) && aliases.clone().any(|template| template.matches(value))
            {
                reserved.insert(value.to_owned());
            }
        }
    }

    pub(crate) fn literal_replacements(&self) -> impl Iterator<Item = &str> {
        self.rules
            .iter()
            .filter_map(|rule| rule.mask.literal_replacement())
    }

    /// Finds UTF-8 byte ranges, ordered by start and then end.
    ///
    /// Literal matches can overlap. Each regular expression uses nonoverlapping matches.
    #[must_use]
    pub fn find(&self, text: &str) -> Vec<Range<usize>> {
        self.find_matches(text)
            .into_iter()
            .map(|matched| matched.span)
            .collect()
    }

    pub(crate) fn find_matches(&self, text: &str) -> Vec<RuleMatch<'_>> {
        let mut matches = Vec::new();
        for rule in &self.rules {
            rule.matcher.find(text, |span| {
                matches.push(RuleMatch {
                    span,
                    mask: &rule.mask,
                });
            });
        }
        matches.sort_by_key(|matched| (matched.span.start, matched.span.end));
        matches
    }
}

/// Masks supplied UTF-8 byte ranges with a literal replacement.
///
/// Overlaps merge into one replacement. Adjacent ranges get separate replacements.
///
/// # Errors
///
/// Returns an error if any span is empty or invalid for the input text.
pub fn mask(text: &str, spans: &[Range<usize>], replacement: &str) -> Result<String, Error> {
    replace_matches(text, tagged_spans(text, spans)?, |output, _, _| {
        output.push_str(replacement);
        Ok(())
    })
}

pub(crate) fn tagged_spans(
    text: &str,
    spans: &[Range<usize>],
) -> Result<Vec<RuleMatch<'static>>, Error> {
    let mut matches = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().enumerate() {
        if span.is_empty() || text.get(span.clone()).is_none() {
            return Err(Error::InvalidSpan { index });
        }
        matches.push(RuleMatch {
            span: span.clone(),
            mask: &TAG_MASK,
        });
    }
    matches.sort_unstable_by_key(|matched| (matched.span.start, matched.span.end));
    Ok(matches)
}

fn values_in(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch| !is_value_char(ch) && ch != '.')
        .flat_map(|part| part.split(".."))
        .map(|part| part.trim_matches('.'))
}

pub(crate) fn replace_matches(
    text: &str,
    sorted: Vec<RuleMatch<'_>>,
    mut rewrite: impl FnMut(&mut String, RuleMatch<'_>, Option<usize>) -> Result<(), Error>,
) -> Result<String, Error> {
    let mut spans = sorted.into_iter().peekable();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(mut matched) = spans.next() {
        while let Some(overlap) = spans.next_if(|next| next.span.start < matched.span.end) {
            matched.span.end = matched.span.end.max(overlap.span.end);
            matched.mask = matched.mask.merge(overlap.mask);
        }
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
