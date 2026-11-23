use std::{
    collections::HashSet,
    fmt::{self, Write},
    ops::RangeInclusive,
    sync::{Arc, LazyLock},
};

use sha2::{Digest, Sha256};

use crate::{Error, template::is_single_value};

/// An indexed source of values for a named template slot.
///
/// Implement [`Self::capacity`] and [`Self::render`] for ordinary values such
/// as words, numbers, and domains. The template hashes the matched text and
/// uses the generator's indices for deterministic selection and collision
/// probing. Use [`GeneratorBinding::new`] to bind the generator to a slot.
/// ```
/// use std::fmt::Write as _;
/// use mpx_redact::{Generator, GeneratorBinding, PlaceholderTemplate};
///
/// struct FiscalYear;
///
/// impl Generator for FiscalYear {
///     fn capacity(&self) -> u128 {
///         30
///     }
///
///     fn render(&self, index: u128, output: &mut String) {
///         write!(output, "FY{}", 2020 + index).unwrap();
///     }
/// }
///
/// let template = PlaceholderTemplate::new("<{{year}}>", &[
///     GeneratorBinding::new("year", FiscalYear),
/// ])?;
/// # let _ = template;
/// # Ok::<(), mpx_redact::Error>(())
/// ```
pub trait Generator: Send + Sync {
    /// Returns the size of the indexed value space used for collision probing.
    ///
    /// Use zero for an empty generator. An alias using it falls back to a tag.
    fn capacity(&self) -> u128;

    /// Returns a Rust regex fragment matching every emitted value.
    ///
    /// The template uses this fragment to reserve values already in the input.
    /// The default covers words, numbers, domains, and email-style values.
    /// Override it for framed slots that emit additional punctuation or require
    /// a narrower match. An empty fragment makes the template invalid.
    fn pattern(&self) -> String {
        String::from(r"[\p{Alphabetic}\p{N}@_%+.-]+")
    }

    /// Appends the value at an index strictly less than [`Self::capacity`].
    ///
    /// Each index should produce a distinct, nonempty value. Slots in aliases
    /// should emit a single value such as a word, number, or domain.
    fn render(&self, index: u128, output: &mut String);

    /// Appends the initial candidate using a hash of the source text and slot.
    ///
    /// The default selects an index. Implementations with wider output spaces
    /// can use the entire seed to generate the initial candidate.
    fn generate(&self, seed: [u8; 32], output: &mut String) {
        let capacity = self.capacity();
        if capacity != 0 {
            let index = if capacity <= u128::from(u64::MAX) + 1 {
                u128::from(u64::from_be_bytes(
                    *seed.first_chunk::<8>().expect("8 bytes"),
                ))
            } else {
                u128::from_be_bytes(*seed.first_chunk::<16>().expect("16 bytes"))
            };
            self.render(index % capacity, output);
        }
    }
}

/// Binds a generator to a named slot in an alias or placeholder template.
///
/// Template construction validates slot names and retains the generators it uses.
pub struct GeneratorBinding<'a> {
    pub(crate) name: &'a str,
    pub(crate) generator: Arc<dyn Generator>,
}

impl<'a> GeneratorBinding<'a> {
    /// Binds a generator to the corresponding `{{name}}` slot.
    #[must_use]
    pub fn new(name: &'a str, generator: impl Generator + 'static) -> Self {
        Self {
            name,
            generator: Arc::new(generator),
        }
    }
}

impl fmt::Debug for GeneratorBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeneratorBinding").finish_non_exhaustive()
    }
}

fn builtin_list(values: &[&str]) -> WordList {
    WordList(values.iter().map(|&value| value.to_owned()).collect())
}

static FIRSTNAMES: LazyLock<WordList> = LazyLock::new(|| {
    builtin_list(&[
        "ada", "adrian", "alex", "amina", "anika", "avery", "casey", "clara", "dana", "devon",
        "elena", "ellis", "evan", "hana", "imani", "jamie", "jules", "kai", "leila", "leo", "lina",
        "maya", "mira", "naomi", "noah", "nora", "rafael", "reese", "robin", "sam", "talia",
        "zara",
    ])
});

static LASTNAMES: LazyLock<WordList> = LazyLock::new(|| {
    builtin_list(&[
        "ahmed", "bennett", "brooks", "carter", "chen", "clark", "costa", "das", "diaz", "evans",
        "garcia", "green", "hassan", "ito", "jensen", "kim", "lewis", "martin", "moreno", "nair",
        "nguyen", "ortiz", "park", "patel", "raman", "ross", "santos", "singh", "taylor", "wong",
        "young", "zhang",
    ])
});

static DOMAINS: LazyLock<WordList> =
    LazyLock::new(|| builtin_list(&["example.com", "example.net", "example.org"]));

/// An ordered, reusable list of values for a slot in an alias template.
#[derive(Clone)]
pub struct WordList(Arc<[String]>);

impl fmt::Debug for WordList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WordList")
            .field("value_count", &self.0.len())
            .finish()
    }
}

impl WordList {
    /// Builds a list and discards duplicate values while keeping their order.
    ///
    /// # Errors
    /// Returns an error if an entry cannot render a single value.
    pub fn new(values: &[&str]) -> Result<Self, Error> {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for (index, &value) in values.iter().enumerate() {
            if !is_single_value(value) {
                return Err(Error::InvalidAliasValue { index });
            }
            if seen.insert(value) {
                unique.push(value.to_owned());
            }
        }
        Ok(Self(unique.into()))
    }

    /// The built-in first names.
    #[must_use]
    pub fn firstname() -> Self {
        FIRSTNAMES.clone()
    }

    /// The built-in last names.
    #[must_use]
    pub fn lastname() -> Self {
        LASTNAMES.clone()
    }

    /// The reserved domains `example.com`, `example.net`, and `example.org`.
    #[must_use]
    pub fn domains() -> Self {
        DOMAINS.clone()
    }
}

impl Generator for WordList {
    fn capacity(&self) -> u128 {
        self.0.len() as u128
    }

    fn render(&self, index: u128, output: &mut String) {
        output.push_str(&self.0[usize::try_from(index).expect("valid word list index")]);
    }
}

/// Generates decimal digits with leading zeros. The width must be positive.
#[derive(Debug, Clone, Copy)]
pub struct Digits(pub usize);

/// Generates lowercase hexadecimal digits. The width must be positive.
#[derive(Debug, Clone, Copy)]
pub struct Hex(pub usize);

macro_rules! fixed_width_generator {
    ($ty:ident, $alphabet:literal, $class:literal, $format:literal) => {
        impl Generator for $ty {
            fn capacity(&self) -> u128 {
                ($alphabet.len() as u128).saturating_pow(u32::try_from(self.0).unwrap_or(u32::MAX))
            }

            fn pattern(&self) -> String {
                if self.0 == 0 {
                    String::new()
                } else {
                    format!(concat!($class, "{{{}}}"), self.0)
                }
            }

            fn render(&self, index: u128, output: &mut String) {
                write!(output, $format, index = index, width = self.0)
                    .expect("writing to a String");
            }

            fn generate(&self, seed: [u8; 32], output: &mut String) {
                render_digits(seed, self.0, $alphabet, output);
            }
        }
    };
}

fixed_width_generator!(Digits, b"0123456789", "[0-9]", "{index:0width$}");
fixed_width_generator!(Hex, b"0123456789abcdef", "[0-9a-f]", "{index:0width$x}");

/// Generates a number from a nonempty inclusive range and writes it in decimal.
#[derive(Debug, Clone)]
pub struct Number(pub RangeInclusive<u64>);

impl Generator for Number {
    fn capacity(&self) -> u128 {
        if self.0.is_empty() {
            return 0;
        }
        u128::from(*self.0.end()) - u128::from(*self.0.start()) + 1
    }

    fn pattern(&self) -> String {
        if self.0.is_empty() {
            String::new()
        } else {
            String::from("[0-9]+")
        }
    }

    fn render(&self, index: u128, output: &mut String) {
        write!(output, "{}", u128::from(*self.0.start()) + index).expect("writing to a String");
    }
}

fn render_digits(mut digest: [u8; 32], mut remaining: usize, alphabet: &[u8], output: &mut String) {
    while remaining > 0 {
        let count = remaining.min(digest.len());
        for &byte in &digest[..count] {
            output.push(char::from(alphabet[usize::from(byte) % alphabet.len()]));
        }
        remaining -= count;
        if remaining > 0 {
            digest = Sha256::digest(digest).into();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Digits, Error, GeneratorBinding, Mask, Matcher, Number, RedactionSession, Rule, Rules,
        WordList,
    };

    #[test]
    fn templates_bind_built_in_and_custom_lists() {
        let original = "person@corp.example";
        let mask = Mask::alias(
            "email",
            "{{first}}-{{last}}@{{host}}",
            &[
                GeneratorBinding::new("first", WordList::firstname()),
                GeneratorBinding::new("last", WordList::lastname()),
                GeneratorBinding::new("host", WordList::domains()),
            ],
        )
        .unwrap();
        let rules = Rules::new([Rule::new(Matcher::Email, mask)]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask(original, &rules).unwrap();
        let (name, domain) = alias.split_once('@').unwrap();
        let (first, last) = name.split_once('-').unwrap();
        for (list, value) in [
            (WordList::firstname(), first),
            (WordList::lastname(), last),
            (WordList::domains(), domain),
        ] {
            assert!(list.0.iter().any(|entry| entry == value));
        }
        assert_eq!(exchange.restore(&alias), original);

        let mask = Mask::alias(
            "email",
            "{{firstname}}-{{team}}@{{domains}}",
            &[
                GeneratorBinding::new("firstname", WordList::firstname()),
                GeneratorBinding::new("team", WordList::new(&["support"]).unwrap()),
                GeneratorBinding::new("domains", WordList::new(&["mail.example.invalid"]).unwrap()),
            ],
        )
        .unwrap();
        let rules = Rules::new([Rule::new(Matcher::Email, mask)]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask(original, &rules).unwrap();
        assert!(alias.ends_with("-support@mail.example.invalid"));
        assert_eq!(exchange.restore(&alias), original);
    }

    #[test]
    fn validates_generator_bindings_and_falls_back_for_an_empty_word_list() {
        assert!(matches!(
            Mask::alias("project", "{{firstname}}", &[]),
            Err(Error::UnboundAliasSlot { name }) if name == "firstname"
        ));
        assert!(matches!(
            Mask::alias(
                "project",
                "{{digits}}",
                &[GeneratorBinding::new("digits", Digits(0))]
            ),
            Err(Error::InvalidAliasTemplate)
        ));
        assert!(matches!(
            Mask::alias(
                "project",
                "{{year}}",
                &[GeneratorBinding::new(
                    "year",
                    Number(std::ops::RangeInclusive::new(2025, 1970))
                )]
            ),
            Err(Error::InvalidAliasTemplate)
        ));

        let mask = Mask::alias(
            "project",
            "{{team}}-{{digits}}",
            &[
                GeneratorBinding::new("team", WordList::new(&[]).unwrap()),
                GeneratorBinding::new("digits", Digits(4)),
            ],
        )
        .unwrap();
        let rules = Rules::new([Rule::new(Matcher::Literal("Nightjar".into()), mask)]);
        let mut exchange = RedactionSession::new();
        let tag = exchange.mask("Nightjar", &rules).unwrap();
        assert!(tag.starts_with("<|mpx-redact-"));
        assert_eq!(exchange.restore(&tag), "Nightjar");
    }
}
