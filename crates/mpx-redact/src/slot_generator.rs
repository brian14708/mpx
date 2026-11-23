use std::{
    collections::HashSet,
    fmt::{self, Write},
    ops::RangeInclusive,
    sync::Arc,
};

use sha2::{Digest, Sha256};

use crate::{Error, replacement_template::is_single_value};

/// A deterministic source of values for a template slot.
///
/// ```
/// use std::fmt::Write as _;
/// use mpx_redact::{Replacement, Slot, SlotGenerator};
///
/// struct FiscalYear;
///
/// impl SlotGenerator for FiscalYear {
///     fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
///         let index = (usize::from(seed[0]) + slot_idx) % 30;
///         write!(output, "FY{}", 2020 + index).unwrap();
///     }
/// }
///
/// let replacement = Replacement::template("<{}>", &[Slot::custom(FiscalYear)])?;
/// # let _ = replacement;
/// # Ok::<(), mpx_redact::Error>(())
/// ```
pub trait SlotGenerator: Send + Sync {
    /// Appends a nonempty value derived from `seed` and `slot_idx`.
    ///
    /// The seed is a keyed hash of the matched text. `slot_idx` is the slot's
    /// zero-based position in its template. Implementations choose whether and
    /// how to mix the position into the generated value. The same seed and slot
    /// index must produce the same value. A generator for an unframed template
    /// must emit text that forms a single value with the rest of the template.
    /// Template rendering supplies an empty buffer for each slot.
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String);
}

/// An ordered, reusable list of values for a replacement template slot.
#[derive(Clone)]
pub struct WordList(Arc<[String]>);

impl fmt::Debug for WordList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WordList")
            .field("value_count", &self.0.len())
            .finish()
    }
}

impl PartialEq for WordList {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for WordList {}

impl WordList {
    /// Consumes the supplied values and keeps the first occurrence of each value.
    ///
    /// # Errors
    /// Returns an error if an entry cannot render a single value.
    pub fn new(values: impl IntoIterator<Item = impl Into<String>>) -> Result<Self, Error> {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for (index, value) in values.into_iter().enumerate() {
            let value = value.into();
            if !is_single_value(&value) {
                return Err(Error::InvalidWordListValue { index });
            }
            if seen.insert(value.clone()) {
                unique.push(value);
            }
        }
        Ok(Self(unique.into()))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl SlotGenerator for WordList {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        let seed = slot_seed(seed, slot_idx, 8);
        let count = u64::try_from(self.0.len()).expect("word list fits in u64");
        if count != 0 {
            let index = usize::try_from(seed_u64(&seed) % count).expect("valid word list index");
            output.push_str(&self.0[index]);
        }
    }
}

/// Generates decimal digits with leading zeros. The width must be positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digits(pub usize);

impl SlotGenerator for Digits {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        let mut seed = slot_seed(seed, slot_idx, self.0);
        let mut remaining = self.0;
        while remaining > 0 {
            let count = remaining.min(seed.len());
            for &byte in &seed[..count] {
                output.push(char::from(b'0' + byte % 10));
            }
            remaining -= count;
            if remaining > 0 {
                seed = Sha256::digest(seed).into();
            }
        }
    }
}

/// Generates a prefix of the lowercase hexadecimal encoding of the hash.
///
/// Wider outputs continue with the encoding of the SHA-256 digest of the previous block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Base16(pub usize);

impl SlotGenerator for Base16 {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        encode_prefix(
            slot_seed(seed, slot_idx, self.0.div_ceil(2)),
            self.0,
            4,
            b"0123456789abcdef",
            output,
        );
    }
}

/// Generates a prefix of the lowercase RFC 4648 Base32 encoding without padding.
///
/// Wider outputs continue with the encoding of the SHA-256 digest of the previous block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Base32(pub usize);

impl SlotGenerator for Base32 {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        encode_prefix(
            slot_seed(seed, slot_idx, self.0.saturating_mul(5).div_ceil(8)),
            self.0,
            5,
            b"abcdefghijklmnopqrstuvwxyz234567",
            output,
        );
    }
}

/// Generates a number from a nonempty inclusive range and writes it in decimal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number(pub RangeInclusive<u64>);

impl SlotGenerator for Number {
    fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
        if self.0.is_empty() {
            return;
        }
        let seed = slot_seed(seed, slot_idx, 8);
        let seed = seed_u64(&seed);
        // The full u64 range has 2^64 values, so its count overflows; every seed is then in range.
        let index = (self.0.end() - self.0.start())
            .checked_add(1)
            .map_or(seed, |count| seed % count);
        let number = self.0.start() + index;
        write!(output, "{number}").expect("writing to a String");
    }
}

const fn seed_u64(seed: &[u8; 32]) -> u64 {
    u64::from_be_bytes(*seed.first_chunk::<8>().expect("8 bytes"))
}

/// Short values use disjoint chunks of the base digest; wide or later slots
/// retain the full SHA-256 derivation.
fn slot_seed(base: [u8; 32], slot_idx: usize, needed_bytes: usize) -> [u8; 32] {
    if slot_idx == 0 {
        return base;
    }
    if needed_bytes <= 8 && slot_idx < 4 {
        let mut seed = base;
        seed.rotate_left(slot_idx * 8);
        return seed;
    }
    let mut hash = Sha256::new();
    hash.update(base);
    hash.update(
        u64::try_from(slot_idx)
            .expect("template slot index fits in u64")
            .to_be_bytes(),
    );
    hash.finalize().into()
}

fn encode_prefix(
    mut digest: [u8; 32],
    mut remaining: usize,
    bits_per_digit: u32,
    alphabet: &[u8],
    output: &mut String,
) {
    let mask = (1_u16 << bits_per_digit) - 1;
    while remaining > 0 {
        let mut buffer = 0_u16;
        let mut buffered_bits = 0;
        for byte in digest {
            buffer = (buffer << 8) | u16::from(byte);
            buffered_bits += 8;
            while buffered_bits >= bits_per_digit {
                buffered_bits -= bits_per_digit;
                let index = (buffer >> buffered_bits) & mask;
                output.push(char::from(alphabet[usize::from(index)]));
                remaining -= 1;
                if remaining == 0 {
                    return;
                }
            }
        }
        if buffered_bits > 0 {
            let index = (buffer << (bits_per_digit - buffered_bits)) & mask;
            output.push(char::from(alphabet[usize::from(index)]));
            remaining -= 1;
        }
        if remaining > 0 {
            digest = Sha256::digest(digest).into();
        }
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::slot_seed;
    use crate::{
        Base16, Base32, Error, Matcher, RedactionSession, Replacement, Rules, Slot, SlotGenerator,
    };

    #[test]
    fn templates_bind_word_lists() {
        let original = "person@corp.example";
        let first_names = Slot::wordlist(["ada", "mira"]).unwrap();
        let last_names = Slot::wordlist(["chen", "patel"]).unwrap();
        let domains = Slot::wordlist(["example.com", "example.net"]).unwrap();
        let mask =
            Replacement::template("{}-{}@{}", &[first_names.clone(), last_names, domains]).unwrap();
        let rules = Rules::new([(
            Matcher::regex(r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}").unwrap(),
            mask,
        )]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask(original, &rules).unwrap();
        let (name, domain) = alias.split_once('@').unwrap();
        let (first, last) = name.split_once('-').unwrap();
        assert!(matches!(first, "ada" | "mira"));
        assert!(matches!(last, "chen" | "patel"));
        assert!(matches!(domain, "example.com" | "example.net"));
        assert_eq!(exchange.restore(&alias), original);

        let mask = Replacement::template(
            "{}-{}@{}",
            &[
                first_names,
                Slot::wordlist(["support"]).unwrap(),
                Slot::wordlist(["mail.example.invalid"]).unwrap(),
            ],
        )
        .unwrap();
        let rules = Rules::new([(
            Matcher::regex(r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}").unwrap(),
            mask,
        )]);
        let mut exchange = RedactionSession::new();
        let alias = exchange.mask(original, &rules).unwrap();
        assert!(alias.ends_with("-support@mail.example.invalid"));
        assert_eq!(exchange.restore(&alias), original);
    }

    #[test]
    fn rejects_aliases_with_an_empty_generator() {
        assert!(matches!(
            Replacement::template("{}", &[]),
            Err(Error::InvalidTemplate)
        ));
        assert!(matches!(
            Replacement::template("{}", &[Slot::digits(0)]),
            Err(Error::InvalidTemplate)
        ));
        assert!(matches!(
            Replacement::template(
                "{}",
                &[Slot::number(std::ops::RangeInclusive::new(2025, 1970))]
            ),
            Err(Error::InvalidTemplate)
        ));

        let result = Replacement::template(
            "{}-{}",
            &[Slot::wordlist([] as [&str; 0]).unwrap(), Slot::digits(4)],
        );
        assert!(matches!(result, Err(Error::InvalidTemplate)));
    }

    #[test]
    fn base_generators_use_standard_encoding_prefixes() {
        let digest = std::array::from_fn(|index| u8::try_from(index).unwrap());
        let hex = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        let base32 = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dypq";
        assert_eq!(generated(&Base16(5), digest), &hex[..5]);
        assert_eq!(generated(&Base32(8), digest), &base32[..8]);
        assert_eq!(generated(&Base16(64), digest), hex);
        assert_eq!(generated(&Base32(52), digest), base32);
        assert_eq!(generated(&Base16(68), digest), format!("{hex}630d"));
        assert_eq!(generated(&Base32(56), digest), format!("{base32}mmg4"));
    }

    #[test]
    fn short_slots_draw_disjoint_digest_chunks() {
        let base = std::array::from_fn(|index| u8::try_from(index).unwrap());
        assert_eq!(slot_seed(base, 0, 32), base);
        for index in 1..4 {
            assert_eq!(slot_seed(base, index, 8)[..8], base[index * 8..][..8]);
        }
        assert_eq!(generated_at(&Base16(16), base, 1), "08090a0b0c0d0e0f");

        for (index, width) in [(1, 9), (4, 8)] {
            let mut hash = Sha256::new();
            hash.update(base);
            hash.update((index as u64).to_be_bytes());
            let expected: [u8; 32] = hash.finalize().into();
            assert_eq!(slot_seed(base, index, width), expected);
        }
    }

    fn generated(generator: &impl SlotGenerator, seed: [u8; 32]) -> String {
        generated_at(generator, seed, 0)
    }

    fn generated_at(generator: &impl SlotGenerator, seed: [u8; 32], slot_idx: usize) -> String {
        let mut output = String::new();
        generator.generate(seed, slot_idx, &mut output);
        output
    }
}
