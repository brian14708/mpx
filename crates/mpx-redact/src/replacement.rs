use std::{fmt, sync::Arc};

use crate::{Error, Slot, replacement_template::ReplacementTemplate};

/// Determines whether a rule replaces matched text or blocks masking.
#[derive(Clone)]
pub struct Replacement(ReplacementKind);

#[derive(Clone)]
enum ReplacementKind {
    Block,
    Literal(String),
    Template {
        template: Arc<ReplacementTemplate>,
        reversible: bool,
    },
}

impl fmt::Debug for Replacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ReplacementKind::Block => f.write_str("Block"),
            ReplacementKind::Literal(_) => f.debug_tuple("Literal").finish_non_exhaustive(),
            ReplacementKind::Template {
                template,
                reversible,
            } => f
                .debug_struct("Template")
                .field("template", template)
                .field("reversible", reversible)
                .finish(),
        }
    }
}

impl Replacement {
    /// Stops masking with [`Error::Blocked`] whenever the rule matches.
    #[must_use]
    pub const fn block() -> Self {
        Self(ReplacementKind::Block)
    }

    /// Creates a permanent replacement for matched text.
    ///
    /// The session rejects collisions with issued reversible replacements by default.
    /// A fallback configured with [`crate::RedactionSession::with_fallback`] remains permanent.
    #[must_use]
    pub fn mask(replacement: impl Into<String>) -> Self {
        Self(ReplacementKind::Literal(replacement.into()))
    }

    /// Creates a reversible replacement with a positional Python-style template.
    ///
    /// Each `{}` consumes one slot in order. Literal braces use
    /// doubled braces, as in Python format strings.
    ///
    /// # Errors
    /// Returns an error for malformed templates or invalid slots.
    pub fn template(pattern: &str, slots: &[Slot]) -> Result<Self, Error> {
        ReplacementTemplate::compile(pattern, slots)
            .map(|template| Self::from_template(template, true))
    }

    /// Creates a permanent replacement using the same slot syntax as [`Self::template`].
    ///
    /// The session only records the original-to-mask entry.
    /// The session rejects collisions with issued reversible replacements by default.
    /// A fallback configured with [`crate::RedactionSession::with_fallback`] remains permanent.
    ///
    /// # Errors
    /// Returns an error for malformed templates or invalid slots.
    pub fn permanent_template(pattern: &str, slots: &[Slot]) -> Result<Self, Error> {
        ReplacementTemplate::compile(pattern, slots)
            .map(|template| Self::from_template(template, false))
    }

    fn from_template(template: ReplacementTemplate, reversible: bool) -> Self {
        Self(ReplacementKind::Template {
            template: Arc::new(template),
            reversible,
        })
    }

    /// Creates a Magpie-compatible marker replacement for one detector kind.
    ///
    /// The emitted form is `{{KIND_abcdefgh}}` with eight lowercase Base32
    /// characters derived from the matched value.
    ///
    /// # Errors
    /// Returns an error if `kind` is empty or contains characters other than
    /// ASCII uppercase letters and underscores.
    pub fn kind(kind: &str) -> Result<Self, Error> {
        ReplacementTemplate::marker(kind).map(|template| Self::from_template(template, true))
    }

    pub(crate) const fn is_block(&self) -> bool {
        matches!(self.0, ReplacementKind::Block)
    }

    pub(crate) fn as_template(&self) -> Option<(&ReplacementTemplate, bool)> {
        match &self.0 {
            ReplacementKind::Template {
                template,
                reversible,
            } => Some((template, *reversible)),
            _ => None,
        }
    }

    pub(crate) fn literal_replacement(&self) -> Option<&str> {
        match &self.0 {
            ReplacementKind::Literal(replacement) => Some(replacement),
            _ => None,
        }
    }

    pub(crate) fn merge<'a>(&'a self, other: &'a Self) -> Result<&'a Self, Error> {
        match (&self.0, &other.0) {
            (ReplacementKind::Literal(_), _) => Ok(self),
            (_, ReplacementKind::Literal(_)) => Ok(other),
            (
                ReplacementKind::Template {
                    template: left,
                    reversible: left_reversible,
                },
                ReplacementKind::Template {
                    template: right,
                    reversible: right_reversible,
                },
            ) if left_reversible == right_reversible && left.same_as(right) => Ok(self),
            _ => Err(Error::OverlappingReplacements),
        }
    }
}
