use std::{fmt, ops::RangeInclusive, sync::Arc};

use crate::{Base16, Base32, Digits, Error, Number, SlotGenerator, WordList};

/// Describes how to fill one `{}` in a replacement template.
#[derive(Clone)]
pub struct Slot(SlotKind);

#[derive(Clone)]
enum SlotKind {
    Literal(String),
    Builtin(Builtin),
    Custom(Arc<dyn SlotGenerator>),
}

/// A built-in generator, compared by value so equal slots match across calls.
#[derive(Clone, PartialEq, Eq)]
enum Builtin {
    Words(WordList),
    Digits(Digits),
    Base16(Base16),
    Base32(Base32),
    Number(Number),
}

impl Builtin {
    fn is_empty(&self) -> bool {
        match self {
            Self::Words(words) => words.is_empty(),
            Self::Digits(Digits(width))
            | Self::Base16(Base16(width))
            | Self::Base32(Base32(width)) => *width == 0,
            Self::Number(Number(range)) => range.is_empty(),
        }
    }

    fn generator(&self) -> &dyn SlotGenerator {
        match self {
            Self::Words(generator) => generator,
            Self::Digits(generator) => generator,
            Self::Base16(generator) => generator,
            Self::Base32(generator) => generator,
            Self::Number(generator) => generator,
        }
    }
}

impl fmt::Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SlotKind::Literal(_) => f.debug_tuple("Literal").finish_non_exhaustive(),
            SlotKind::Builtin(_) | SlotKind::Custom(_) => f.write_str("SlotGenerator(..)"),
        }
    }
}

impl Slot {
    /// Fills a slot with fixed text. The text must be nonempty.
    #[must_use]
    pub fn literal(value: impl Into<String>) -> Self {
        Self(SlotKind::Literal(value.into()))
    }

    /// Wraps a custom generator for use in a positional template.
    #[must_use]
    pub fn custom(generator: impl SlotGenerator + 'static) -> Self {
        Self(SlotKind::Custom(Arc::new(generator)))
    }

    /// Selects a value from an ordered list, keeping the first occurrence of each value.
    ///
    /// A template rejects a slot backed by an empty list.
    ///
    /// # Errors
    /// Returns an error if an entry is empty or contains characters outside a
    /// single template value.
    pub fn wordlist(values: impl IntoIterator<Item = impl Into<String>>) -> Result<Self, Error> {
        WordList::new(values).map(|words| Self(SlotKind::Builtin(Builtin::Words(words))))
    }

    /// Creates a positional generator for decimal digits with leading zeros.
    #[must_use]
    pub const fn digits(width: usize) -> Self {
        Self(SlotKind::Builtin(Builtin::Digits(Digits(width))))
    }

    /// Creates a positional generator for lowercase hexadecimal digits.
    #[must_use]
    pub const fn base16(width: usize) -> Self {
        Self(SlotKind::Builtin(Builtin::Base16(Base16(width))))
    }

    /// Creates a positional generator for lowercase Base32 digits.
    #[must_use]
    pub const fn base32(width: usize) -> Self {
        Self(SlotKind::Builtin(Builtin::Base32(Base32(width))))
    }

    /// Creates a positional generator for numbers in an inclusive range.
    #[must_use]
    pub const fn number(range: RangeInclusive<u64>) -> Self {
        Self(SlotKind::Builtin(Builtin::Number(Number(range))))
    }

    /// Whether two slots always generate the same values.
    ///
    /// Custom generators only compare equal to clones of themselves.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (SlotKind::Literal(left), SlotKind::Literal(right)) => left == right,
            (SlotKind::Builtin(left), SlotKind::Builtin(right)) => left == right,
            (SlotKind::Custom(left), SlotKind::Custom(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match &self.0 {
            SlotKind::Literal(value) => value.is_empty(),
            SlotKind::Builtin(builtin) => builtin.is_empty(),
            SlotKind::Custom(_) => false,
        }
    }
}

impl SlotGenerator for Slot {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        match &self.0 {
            SlotKind::Literal(value) => output.push_str(value),
            SlotKind::Builtin(builtin) => builtin.generator().generate(seed, slot_idx, output),
            SlotKind::Custom(generator) => generator.generate(seed, slot_idx, output),
        }
    }
}
