//! Masks text with the built-in detectors: known positives for every detector
//! kind must be found and restored, and arbitrary text must restore and agree
//! between best-effort and strict detection.
#![no_main]

use std::sync::LazyLock;

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::redact::{assert_restores, assert_session_consistent, mask_atomically};
use mpx_redact::{RedactionSession, Rules};

static RULES: LazyLock<Rules> = LazyLock::new(Rules::builtin);

/// Whitespace that cannot join a detected value.
const SEPARATORS: [&str; 4] = ["", " ", "\t", "\n"];

#[derive(Debug)]
enum Positive {
    Email,
    Ipv4,
    Ipv6,
    UserLabel,
    HostLabel,
    UrlPassword,
    UnixUserPath,
    WindowsUserPath,
    SshUser,
    SshHost,
    ApiKey(u8),
    Jwt,
}

#[derive(Debug, Arbitrary)]
struct Input {
    /// The first byte selects the detector, so seeds can target each one.
    #[arbitrary(with = Positive::select)]
    positive: Positive,
    n: u8,
    /// Characters for generated value bodies.
    body: String,
    left: u8,
    right: u8,
    /// Arbitrary neighbors; the positive need not be detected next to them.
    neighbors: (char, char),
    text: String,
    key: Option<Vec<u8>>,
    cuts: Vec<u16>,
}

impl Positive {
    fn select(u: &mut Unstructured<'_>) -> arbitrary::Result<Self> {
        Ok(match u.arbitrary::<u8>()? % 12 {
            0 => Self::Email,
            1 => Self::Ipv4,
            2 => Self::Ipv6,
            3 => Self::UserLabel,
            4 => Self::HostLabel,
            5 => Self::UrlPassword,
            6 => Self::UnixUserPath,
            7 => Self::WindowsUserPath,
            8 => Self::SshUser,
            9 => Self::SshHost,
            10 => Self::ApiKey(u.arbitrary()?),
            _ => Self::Jwt,
        })
    }
}

/// Returns `len` characters drawn from `source` and `allowed`, padded deterministically.
fn body(source: &str, len: usize, allowed: impl Fn(char) -> bool) -> String {
    source
        .chars()
        .filter(|ch| ch.is_ascii() && allowed(*ch))
        .chain("a1B2c3D4e5F6g7H8".chars().cycle().filter(|ch| allowed(*ch)))
        .take(len)
        .collect()
}

/// Builds text containing a known secret, and the secret itself.
fn positive(input: &Input) -> (String, String) {
    let n = input.n;
    let alnum = |len| body(&input.body, len, |ch| ch.is_ascii_alphanumeric());
    let same = |secret: String| (secret.clone(), secret);
    match input.positive {
        Positive::Email => same(format!("user{n}.{}@example.org", alnum(4))),
        Positive::Ipv4 => same(format!("192.0.2.{n}")),
        Positive::Ipv6 => same(format!("2001:db8::{n:x}")),
        Positive::UserLabel => {
            let secret = format!("user{n}");
            (format!("Username: {secret}"), secret)
        }
        Positive::HostLabel => {
            let secret = format!("host-{n}.internal");
            (format!("Host Name = {secret}"), secret)
        }
        Positive::UrlPassword => {
            let secret = format!("pw{n}%40ss");
            (format!("https://alice:{secret}@localhost/db"), secret)
        }
        Positive::UnixUserPath => {
            let secret = format!("user{n}");
            (format!("/home/{secret}/src/main.rs"), secret)
        }
        Positive::WindowsUserPath => {
            let secret = format!("user{n}");
            (format!(r"C:\Users\{secret}\Desktop"), secret)
        }
        Positive::SshUser => {
            let secret = format!("deploy{n}");
            (format!("$ ssh {secret}@build{n}"), secret)
        }
        Positive::SshHost => {
            let secret = format!("build{n}");
            (format!("$ ssh deploy{n}@{secret}"), secret)
        }
        Positive::ApiKey(kind) => same(match kind % 6 {
            0 => format!("sk-proj-{}", alnum(24)),
            1 => format!("ghp_{}", alnum(36)),
            2 => format!("glpat-{}", alnum(20)),
            3 => format!(
                "AKIA{}",
                body(&input.body, 16, |ch| ch.is_ascii_uppercase()
                    || ch.is_ascii_digit())
            ),
            4 => format!("hf_{}", alnum(24)),
            _ => format!("xoxb-{}", alnum(16)),
        }),
        // The header and claims must decode to JSON; the signature is free.
        Positive::Jwt => same(format!(
            "eyJhbGciOiJIUzI1NiJ9.e30.{}",
            body(&input.body, 43, |ch| ch.is_ascii_alphanumeric()
                || "-_".contains(ch))
        )),
    }
}

fn session(key: Option<&[u8]>) -> RedactionSession {
    let session = RedactionSession::new();
    match key {
        Some(key) => session.with_hmac_key(key),
        None => session,
    }
}

fuzz_target!(|input: Input| {
    let (positive, secret) = positive(&input);
    let surrounding: String = input
        .text
        .chars()
        .take(32)
        .map(|ch| if ch.is_ascii_alphabetic() { ch } else { 'x' })
        .collect();
    let left = SEPARATORS[usize::from(input.left) % SEPARATORS.len()];
    let right = SEPARATORS[usize::from(input.right) % SEPARATORS.len()];
    let prefix = format!("prefix-{surrounding}\n{left}");
    let suffix = format!("{right}\nsuffix-{surrounding}");
    let text = format!("{prefix}{positive}{suffix}");
    // The surrounding text may contain the secret too, joined to its neighbors.
    let start = prefix.len() + positive.find(&secret).unwrap();
    assert!(
        RULES
            .try_find(&text)
            .expect("built-in detectors succeed")
            .iter()
            .any(|span| span.start <= start && span.end >= start + secret.len()),
        "known secret was not detected: {text:?}"
    );
    let mut known = session(input.key.as_deref());
    let masked = known
        .mask(&text, &RULES)
        .expect("known-positive input has no replacement collisions");
    assert!(masked.starts_with(&prefix));
    assert!(masked.ends_with(&suffix));
    assert!(
        !masked[prefix.len()..masked.len() - suffix.len()].contains(&secret),
        "known secret remained visible"
    );
    assert_restores(&known, &masked, &text, &input.cuts);
    assert_session_consistent(&known);

    // Arbitrary neighbors may legitimately suppress detection, but never break restoration.
    let (before, after) = input.neighbors;
    let glued = format!("{before}{positive}{after}");
    for text in [&glued, &input.text] {
        let spans = RULES.try_find(text).expect("built-in detectors succeed");
        assert_eq!(
            RULES.find(text),
            spans,
            "best-effort and strict detection differ"
        );
        let mut session = session(input.key.as_deref());
        if let Ok(masked) =
            mask_atomically(&mut session, false, |session| session.mask(text, &RULES))
        {
            assert_restores(&session, &masked, text, &input.cuts);
        }
        assert_session_consistent(&session);
    }
});
