//! Models reversible and permanent replacements in independently delimited fields.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::redact::{MAX_TOKENS, assert_restores, assert_session_consistent, mappings};
use mpx_redact::{Matcher, RedactionSession, Replacement, Rules, Slot};

#[derive(Debug, Arbitrary)]
enum Permanent {
    Mask,
    Template,
    /// The reversible and permanent rules render the same text.
    Collision,
}

#[derive(Debug, Arbitrary)]
enum Piece {
    Reversible,
    Permanent,
    Visible,
    Both,
}

#[derive(Debug, Arbitrary)]
struct Input {
    permanent: Permanent,
    fallback: bool,
    key: Vec<u8>,
    scope: Vec<u8>,
    fields: Vec<Vec<Piece>>,
    cuts: Vec<u16>,
}

fuzz_target!(|input: Input| {
    let collision = matches!(input.permanent, Permanent::Collision);
    let shared = || [Slot::literal("shared")];
    let rules = Rules::new([
        (
            Matcher::literal("secret-A"),
            if collision {
                Replacement::template("<{}>", &shared()).unwrap()
            } else {
                Replacement::kind("SECRET").unwrap()
            },
        ),
        (
            Matcher::literal("secret-B"),
            match input.permanent {
                Permanent::Mask => Replacement::mask("[permanent]"),
                Permanent::Template => {
                    Replacement::permanent_template("[permanent-{}]", &[Slot::base32(12)]).unwrap()
                }
                Permanent::Collision => Replacement::permanent_template("<{}>", &shared()).unwrap(),
            },
        ),
        (
            Matcher::literal("never-present"),
            Replacement::mask("[unused]"),
        ),
    ]);
    let fields: Vec<_> = input
        .fields
        .iter()
        .take(16)
        .map(|pieces| {
            let pieces: Vec<_> = pieces
                .iter()
                .take(MAX_TOKENS)
                .map(|piece| match piece {
                    Piece::Reversible => "secret-A",
                    Piece::Permanent => "secret-B",
                    Piece::Visible => "visible-末",
                    Piece::Both => "secret-A secret-B",
                })
                .collect();
            format!("|{}|", pieces.join(" | "))
        })
        .chain(std::iter::once("|secret-A|".to_owned()))
        .collect();
    let mut session = RedactionSession::new()
        .with_hmac_key(&input.key)
        .with_scope(&input.scope);
    if input.fallback {
        // A fallback for the permanent rule stays permanent.
        session = session.with_fallback(Replacement::kind("FALLBACK").unwrap());
    }
    for field in &fields {
        session.reserve_replacements_in(field, &rules);
    }
    let mut issued = Vec::with_capacity(fields.len());
    let mut permanent = None;
    for original in &fields {
        let before = mappings(&session);
        match session.mask_reserved(original, &rules) {
            Ok(masked) => {
                if permanent.is_none() {
                    permanent = session
                        .replacement_map()
                        .replacements_for("secret-B")
                        .next()
                        .map(str::to_owned);
                }
                if original.contains("secret-B") {
                    let replacement = permanent.as_deref().expect("permanent mask was recorded");
                    match input.permanent {
                        _ if replacement.starts_with("{{FALLBACK_") => {
                            assert!(collision && input.fallback, "unneeded fallback");
                        }
                        Permanent::Collision => assert_eq!(replacement, "<shared>"),
                        Permanent::Mask => assert_eq!(replacement, "[permanent]"),
                        Permanent::Template => {
                            let generated = replacement
                                .strip_prefix("[permanent-")
                                .and_then(|value| value.strip_suffix(']'))
                                .expect("permanent template shape");
                            assert_eq!(generated.len(), 12);
                            assert!(
                                generated
                                    .bytes()
                                    .all(|byte| matches!(byte, b'a'..=b'z' | b'2'..=b'7'))
                            );
                        }
                    }
                }
                let expected =
                    original.replace("secret-B", permanent.as_deref().unwrap_or("secret-B"));
                assert!(!masked.contains("secret-A"));
                assert!(!masked.contains("secret-B"));
                issued.push((masked, expected));
            }
            Err(error) => {
                assert_eq!(mappings(&session), before);
                assert!(
                    collision && matches!(error, mpx_redact::Error::ReplacementUnavailable { .. }),
                    "generated disjoint mixed rules must mask: {error}"
                );
            }
        }
    }
    for (masked, expected) in &issued {
        assert_restores(&session, masked, expected, &input.cuts);
    }
    assert_session_consistent(&session);
});
