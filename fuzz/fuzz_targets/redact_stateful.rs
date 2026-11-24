//! Exercises session operations, failures, snapshots, and the lifetime of issued outputs.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::redact::{
    assert_mask_error, assert_restores, assert_session_consistent, map_mappings, mappings,
    mask_atomically, restore_chunks,
};
use mpx_redact::{
    Error, Matcher, RedactionSession, Replacement, ReplacementMap, Restorer, Rules, Slot,
};

#[derive(Debug, Clone, Copy, Arbitrary)]
enum Text {
    P6,
    Zero,
    P6Zero,
    Pipe,
    Unicode,
}

impl Text {
    const fn as_str(self) -> &'static str {
        match self {
            Self::P6 => "P6",
            Self::Zero => "0",
            Self::P6Zero => "P6 0",
            Self::Pipe => "|b0",
            Self::Unicode => "末 6 0 α",
        }
    }
}

/// Rules for "0" and "6". `Permanent` masks "6" permanently.
#[derive(Debug, Clone, Copy, Arbitrary)]
enum RuleSet {
    Marker,
    Alias,
    Hex,
    Permanent,
}

#[derive(Debug, Arbitrary)]
enum Op {
    Reserve(Text, RuleSet),
    /// Checks outputs, then releases reservations and forgets the outputs.
    Clear,
    Fallback,
    Strict,
    Scope(u8),
    Key(Vec<u8>),
    Block,
    Mask {
        text: Text,
        rules: RuleSet,
        reserved: bool,
    },
    MaskSpan {
        text: Text,
        start: u8,
        len: u8,
        rules: RuleSet,
    },
    /// Freezes the map and opens one restorer per issued output.
    Snapshot,
}

struct Snapshot {
    map: ReplacementMap,
    entries: Vec<(String, String)>,
    restorers: Vec<(Restorer, String, String)>,
}

fn rules(set: RuleSet) -> Rules {
    let reversible = match set {
        RuleSet::Marker | RuleSet::Permanent => Replacement::kind("SECRET").unwrap(),
        RuleSet::Alias => Replacement::template("{}", &[Slot::literal("P")]).unwrap(),
        RuleSet::Hex => Replacement::template("|{}|", &[Slot::base16(8)]).unwrap(),
    };
    let six = match set {
        RuleSet::Permanent => Replacement::mask("[p]"),
        _ => reversible.clone(),
    };
    Rules::new([
        (Matcher::literal("0"), reversible),
        (Matcher::literal("6"), six),
    ])
}

fn expected(original: &str, set: RuleSet) -> String {
    match set {
        RuleSet::Permanent => original.replace('6', "[p]"),
        _ => original.to_owned(),
    }
}

fn check(session: &RedactionSession, issued: &[(String, String)], snapshots: Vec<Snapshot>) {
    for (masked, expected) in issued {
        assert_restores(session, masked, expected, &[1, 2, 3]);
    }
    for snapshot in snapshots {
        assert_eq!(
            map_mappings(&snapshot.map),
            snapshot.entries,
            "snapshot map changed"
        );
        for (restorer, masked, expected) in snapshot.restorers {
            let output = restore_chunks(restorer, masked.split_inclusive(|_| true));
            assert_eq!(output, expected, "snapshot restorer for {masked:?}");
        }
    }
    assert_session_consistent(session);
}

fuzz_target!(|input: (Vec<u8>, Vec<Op>)| {
    let (key, ops) = input;
    let new_session = || RedactionSession::new().with_hmac_key(&key).with_scope(&key);
    let stable = rules(RuleSet::Marker);
    assert_eq!(
        new_session().mask("6", &stable).unwrap(),
        new_session().mask("6", &stable).unwrap(),
        "keyed replacements must be deterministic within a scope"
    );
    let blocked = Rules::new([(Matcher::literal("blocked"), Replacement::block())]);
    let mut session = new_session();
    let mut issued: Vec<(String, String)> = Vec::new();
    let mut snapshots = Vec::new();
    for op in ops.into_iter().take(64) {
        match op {
            Op::Reserve(text, set) => session.reserve_replacements_in(text.as_str(), &rules(set)),
            Op::Clear => {
                // Outputs must stay restorable until their reservations are cleared.
                check(&session, &issued, std::mem::take(&mut snapshots));
                session.clear_reserved_replacements();
                issued.clear();
            }
            Op::Fallback => session = session.with_fallback(Replacement::kind("FALLBACK").unwrap()),
            Op::Strict => session = session.with_strict(),
            Op::Scope(scope) => session = session.with_scope([scope]),
            Op::Key(key) => session = session.with_hmac_key(&key),
            Op::Block => {
                let before = mappings(&session);
                assert!(matches!(
                    session.mask("blocked", &blocked),
                    Err(Error::Blocked)
                ));
                assert_eq!(mappings(&session), before);
            }
            Op::Mask {
                text,
                rules: set,
                reserved,
            } => {
                let original = text.as_str();
                let rules = rules(set);
                if reserved {
                    session.reserve_replacements_in(original, &rules);
                }
                let result = mask_atomically(&mut session, false, |session| {
                    if reserved {
                        session.mask_reserved(original, &rules)
                    } else {
                        session.mask(original, &rules)
                    }
                });
                if let Ok(masked) = result {
                    issued.push((masked, expected(original, set)));
                }
            }
            Op::MaskSpan {
                text,
                start,
                len,
                rules: set,
            } => {
                let original = text.as_str();
                let start = usize::from(start) % original.len();
                let span = start..(start + usize::from(len) % 4 + 1).min(original.len());
                let replacement = match set {
                    RuleSet::Permanent => Replacement::mask("[p]"),
                    _ => Replacement::kind("SPAN").unwrap(),
                };
                let before = mappings(&session);
                match session.mask_spans(original, std::slice::from_ref(&span), &replacement) {
                    Ok(masked) => {
                        let expected = match set {
                            RuleSet::Permanent => {
                                format!("{}[p]{}", &original[..span.start], &original[span.end..])
                            }
                            _ => original.to_owned(),
                        };
                        issued.push((masked, expected));
                    }
                    Err(Error::InvalidSpan { index: 0 }) => {
                        assert!(original.get(span).is_none());
                        assert_eq!(mappings(&session), before);
                    }
                    Err(error) => {
                        assert_mask_error(&error, false);
                        assert_eq!(mappings(&session), before);
                    }
                }
            }
            Op::Snapshot => {
                let map = session.replacement_map();
                snapshots.push(Snapshot {
                    entries: map_mappings(&map),
                    map,
                    restorers: issued
                        .iter()
                        .map(|(masked, expected)| {
                            (session.restorer(), masked.clone(), expected.clone())
                        })
                        .collect(),
                });
            }
        }
    }
    check(&session, &issued, snapshots);
});
