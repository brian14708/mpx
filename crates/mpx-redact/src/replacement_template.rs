use std::{fmt, mem, ops::Range};

use crate::{Error, Slot, SlotGenerator};

pub(crate) struct ReplacementTemplate {
    pattern: String,
    pub(crate) prefix: String,
    parts: Vec<TemplatePart>,
    framed: bool,
}

impl fmt::Debug for ReplacementTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplacementTemplate")
            .field("slot_count", &self.parts.len())
            .finish_non_exhaustive()
    }
}

struct TemplatePart {
    slot: Slot,
    suffix: String,
}

impl ReplacementTemplate {
    /// Compiles a positional Python-style template.
    pub(crate) fn compile(template: &str, slots: &[Slot]) -> Result<Self, Error> {
        let (prefix, parts) = parse_template(template, slots)?;
        Self::from_parts(template.to_owned(), prefix, parts)
    }

    /// Builds the `{{KIND_abcdefgh}}` marker template.
    ///
    /// Errors name it `{{KIND_{}}}`, which reads better than its escaped
    /// Python form.
    pub(crate) fn marker(kind: &str) -> Result<Self, Error> {
        if !is_marker_kind(kind) {
            return Err(Error::InvalidTemplate);
        }
        Self::from_parts(
            format!("{{{{{kind}_{{}}}}}}"),
            format!("{{{{{kind}_"),
            vec![TemplatePart {
                slot: Slot::base32(8),
                suffix: "}}".to_owned(),
            }],
        )
    }

    /// `id` names the template in errors.
    fn from_parts(id: String, prefix: String, parts: Vec<TemplatePart>) -> Result<Self, Error> {
        let Some(last) = parts.last() else {
            return Err(Error::InvalidTemplate);
        };
        let framed = !prefix.chars().next().is_none_or(is_value_char)
            && !last.suffix.chars().last().is_none_or(is_value_char);
        if !framed {
            let mut sample = prefix.clone();
            for part in &parts {
                sample.push('w');
                sample.push_str(&part.suffix);
            }
            if !is_single_value(&sample) {
                return Err(Error::InvalidTemplate);
            }
        }

        Ok(Self {
            pattern: id,
            prefix,
            parts,
            framed,
        })
    }

    /// Renders the template from the matched value's base seed.
    ///
    /// Each slot receives the base seed and its zero-based position. The slot
    /// generator decides whether and how to mix the position into its output.
    /// Each slot renders into the empty `slot` buffer, so a generator cannot
    /// rewrite template text or earlier slots.
    pub(crate) fn generate(&self, base: &[u8; 32], output: &mut String, slot: &mut String) -> bool {
        output.clear();
        output.push_str(&self.prefix);
        for (index, part) in self.parts.iter().enumerate() {
            slot.clear();
            part.slot.generate(*base, index, slot);
            if slot.is_empty() {
                return false;
            }
            output.push_str(slot);
            output.push_str(&part.suffix);
        }
        self.framed || is_single_value(output)
    }

    /// Whether both templates always render the same replacement.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
            || self.prefix == other.prefix
                && self.parts.len() == other.parts.len()
                && self.parts.iter().zip(&other.parts).all(|(left, right)| {
                    left.suffix == right.suffix && left.slot.same_as(&right.slot)
                })
    }

    pub(crate) fn pattern(&self) -> &str {
        &self.pattern
    }

    pub(crate) const fn framed(&self) -> bool {
        self.framed
    }
}

fn is_marker_kind(kind: &str) -> bool {
    !kind.is_empty()
        && kind
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'_')
}

fn parse_template(template: &str, slots: &[Slot]) -> Result<(String, Vec<TemplatePart>), Error> {
    let mut prefix = String::new();
    let mut literal = String::new();
    let mut parts: Vec<TemplatePart> = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                literal.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                literal.push('}');
            }
            '{' if chars.peek() == Some(&'}') => {
                chars.next();
                if let Some(last) = parts.last_mut() {
                    last.suffix = mem::take(&mut literal);
                } else {
                    prefix = mem::take(&mut literal);
                }
                let slot = slots
                    .get(parts.len())
                    .filter(|slot| !slot.is_empty())
                    .ok_or(Error::InvalidTemplate)?;
                parts.push(TemplatePart {
                    slot: slot.clone(),
                    suffix: String::new(),
                });
            }
            '{' | '}' => return Err(Error::InvalidTemplate),
            character => literal.push(character),
        }
    }
    if slots.len() != parts.len() {
        return Err(Error::InvalidTemplate);
    }
    if let Some(last) = parts.last_mut() {
        last.suffix = literal;
    } else {
        prefix = literal;
    }
    Ok((prefix, parts))
}

pub(crate) fn is_single_value(value: &str) -> bool {
    value
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(is_value_char))
}

pub(crate) fn is_value_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '@' | '_' | '%' | '+' | '-')
}

/// Whether an issued replacement is an alias rather than a framed marker.
///
/// Aliases start with a value character and framed markers start with a delimiter.
pub(crate) fn is_alias(replacement: &str) -> bool {
    replacement.starts_with(is_value_char)
}

/// Whether `span` is not extended by value characters on either side.
pub(crate) fn bounded(text: &str, span: Range<usize>) -> bool {
    !joins_value(text[..span.start].chars().rev()) && !joins_value(text[span.end..].chars())
}

/// Whether `text` contains `replacement`, requiring value boundaries around aliases.
pub(crate) fn contains_replacement(text: &str, replacement: &str, framed: bool) -> bool {
    if framed {
        text.contains(replacement)
    } else {
        contains_alias(text, replacement)
    }
}

pub(crate) fn contains_alias(text: &str, alias: &str) -> bool {
    text.match_indices(alias)
        .any(|(start, _)| bounded(text, start..start + alias.len()))
}

pub(crate) fn joins_value(mut neighbors: impl Iterator<Item = char>) -> bool {
    neighbors.next().is_some_and(|ch| {
        is_value_char(ch) || ch == '.' && neighbors.next().is_some_and(is_value_char)
    })
}
