use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, LazyLock},
};

use regex::{Regex, escape};
use sha2::{Digest, Sha256};

use crate::{Error, Generator, GeneratorBinding, Hex};

static DEFAULT: LazyLock<PlaceholderTemplate> = LazyLock::new(|| {
    PlaceholderTemplate::new(
        "<|mpx-redact-{{hex}}|>",
        &[GeneratorBinding::new("hex", Hex(6))],
    )
    .expect("valid default redaction template")
});

/// A delimited generator template used when no alias can be emitted.
///
/// The default is `<|mpx-redact-{{hex}}|>` with `hex` bound to [`Hex(6)`](Hex).
#[derive(Clone)]
pub struct PlaceholderTemplate {
    pub(crate) compiled: Arc<AliasTemplate>,
}

impl fmt::Debug for PlaceholderTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlaceholderTemplate")
            .finish_non_exhaustive()
    }
}

impl PlaceholderTemplate {
    /// Compiles a delimited template using the same generator slots as [`crate::Mask::alias`].
    ///
    /// Both ends need literal delimiters. Choose delimiters that identify each
    /// marker unambiguously, including when two markers appear next to each other.
    ///
    /// # Errors
    /// Returns an error for malformed templates, invalid or missing bindings,
    /// missing delimiters, or a generator with zero capacity.
    pub fn new(pattern: &str, bindings: &[GeneratorBinding<'_>]) -> Result<Self, Error> {
        Ok(Self {
            compiled: Arc::new(AliasTemplate::compile_framed(pattern, bindings)?),
        })
    }

    /// Returns the product of the slot capacities, capped at `u128::MAX`.
    ///
    /// Each slot occurrence counts separately. Ambiguous concatenation can
    /// cause two combinations to produce the same marker.
    #[must_use]
    pub fn combinations(&self) -> u128 {
        self.compiled.combinations
    }
}

impl Default for PlaceholderTemplate {
    fn default() -> Self {
        DEFAULT.clone()
    }
}

pub(crate) struct AliasTemplate {
    kind: String,
    pattern: String,
    pub(crate) prefix: String,
    pub(crate) combinations: u128,
    parts: Vec<AliasPart>,
    shape: Regex,
    framed: bool,
}

impl fmt::Debug for AliasTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AliasTemplate")
            .field("slot_count", &self.parts.len())
            .finish_non_exhaustive()
    }
}

struct AliasPart {
    slot: Arc<dyn Generator>,
    suffix: String,
    capacity: u128,
}

impl AliasTemplate {
    pub(crate) fn compile(
        kind: String,
        template: &str,
        bindings: &[GeneratorBinding<'_>],
    ) -> Result<Option<Self>, Error> {
        Self::compile_template(kind, template, bindings, false)
    }

    pub(crate) fn compile_framed(
        template: &str,
        bindings: &[GeneratorBinding<'_>],
    ) -> Result<Self, Error> {
        Self::compile_template("redaction".to_owned(), template, bindings, true)?
            .ok_or(Error::InvalidAliasTemplate)
    }

    fn compile_template(
        kind: String,
        template: &str,
        bindings: &[GeneratorBinding<'_>],
        framed: bool,
    ) -> Result<Option<Self>, Error> {
        let mut generators = HashMap::new();
        for binding in bindings {
            if !valid_slot_name(binding.name)
                || generators
                    .insert(binding.name, &binding.generator)
                    .is_some()
            {
                return Err(Error::InvalidAliasBinding);
            }
        }
        let mut pieces = template.split("{{");
        let prefix = pieces.next().ok_or(Error::InvalidAliasTemplate)?;
        let mut sample = prefix.to_owned();
        let mut parts = Vec::new();
        for piece in pieces {
            let (name, suffix) = piece.split_once("}}").ok_or(Error::InvalidAliasTemplate)?;
            if !valid_slot_name(name) {
                return Err(Error::InvalidAliasTemplate);
            }
            let slot = generators
                .get(name)
                .ok_or_else(|| Error::UnboundAliasSlot {
                    name: name.to_owned(),
                })?;
            sample.push('w');
            sample.push_str(suffix);
            parts.push(AliasPart {
                capacity: slot.capacity(),
                slot: Arc::clone(slot),
                suffix: suffix.to_owned(),
            });
        }
        let Some(last) = parts.last() else {
            return Err(Error::InvalidAliasTemplate);
        };
        if framed {
            if prefix.chars().next().is_none_or(is_value_char)
                || last.suffix.chars().last().is_none_or(is_value_char)
            {
                return Err(Error::InvalidTemplateDelimiter);
            }
        } else if !is_single_value(&sample) {
            return Err(Error::InvalidAliasTemplate);
        }

        let mut pattern = if framed {
            String::new()
        } else {
            String::from(r"\A")
        };
        pattern.push_str(&escape(prefix));
        for part in &parts {
            let slot_pattern = part.slot.pattern();
            if slot_pattern.is_empty() {
                return Err(Error::InvalidAliasTemplate);
            }
            pattern.push_str("(?:");
            pattern.push_str(&slot_pattern);
            pattern.push(')');
            pattern.push_str(&escape(&part.suffix));
        }
        if !framed {
            pattern.push_str(r"\z");
        }
        let shape = Regex::new(&pattern).map_err(|_| Error::InvalidAliasTemplate)?;
        let combinations = parts
            .iter()
            .fold(1_u128, |total, part| total.saturating_mul(part.capacity));
        if combinations == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            kind,
            pattern: template.to_owned(),
            prefix: prefix.to_owned(),
            combinations,
            parts,
            shape,
            framed,
        }))
    }

    pub(crate) fn select(&self, digest: &[u8; 32], output: &mut String) -> bool {
        let mut hash = Sha256::new();
        hash.update(&digest[..8]);
        hash.update(self.kind.as_bytes());
        output.clear();
        output.push_str(&self.prefix);
        for (index, part) in self.parts.iter().enumerate() {
            let mut slot_hash = hash.clone();
            slot_hash.update((index as u64).to_be_bytes());
            part.slot.generate(slot_hash.finalize().into(), output);
            output.push_str(&part.suffix);
        }
        self.framed || is_single_value(output)
    }

    pub(crate) fn select_available(
        &self,
        digest: &[u8; 32],
        output: &mut String,
        mut available: impl FnMut(&str) -> bool,
        strict: bool,
    ) -> Result<bool, u128> {
        let mut attempts: u128 = 1;
        if self.select(digest, output) && available(output) {
            return Ok(false);
        }
        if strict {
            return Err(1);
        }
        let seed = u128::from_be_bytes(*digest.first_chunk::<16>().expect("SHA-256 digest"));
        let start = seed % self.combinations;
        for mut index in (start..self.combinations).chain(0..start) {
            attempts = attempts.saturating_add(1).min(self.combinations);
            output.clear();
            output.push_str(&self.prefix);
            for part in &self.parts {
                part.slot.render(index % part.capacity, output);
                index /= part.capacity;
                output.push_str(&part.suffix);
            }
            if available(output) {
                return Ok(true);
            }
        }
        Err(attempts)
    }

    pub(crate) fn pattern(&self) -> &str {
        &self.pattern
    }

    pub(crate) fn matches(&self, value: &str) -> bool {
        self.shape.is_match(value)
    }

    pub(crate) fn matching_values<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        let mut start = 0;
        std::iter::from_fn(move || {
            let matched = self.shape.find_at(text, start)?;
            // A closing delimiter can also open another marker.
            start = text.ceil_char_boundary(matched.start() + 1);
            Some(matched.as_str())
        })
    }
}

fn valid_slot_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(crate) fn is_single_value(value: &str) -> bool {
    value
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(is_value_char))
}

pub(crate) fn is_value_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '@' | '_' | '%' | '+' | '-')
}

pub(crate) fn joins_value(mut neighbors: impl Iterator<Item = char>) -> bool {
    neighbors.next().is_some_and(|ch| {
        is_value_char(ch) || ch == '.' && neighbors.next().is_some_and(is_value_char)
    })
}
