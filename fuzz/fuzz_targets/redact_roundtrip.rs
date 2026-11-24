//! Masks token-built fields with generated reversible rules, then checks that
//! every match was replaced, masking is deterministic, and restoration (whole or
//! chunked) recovers the originals and treats model-style output consistently.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::redact::{
    BuiltRules, MAX_FIELDS, NoiseToken, RuleSpec, SessionSpec, Token, assert_all_masked,
    assert_restore_consistent, assert_restores, assert_session_consistent, mappings,
    mask_atomically, render_field, render_noise,
};
use mpx_redact::RedactionSession;

#[derive(Debug, Arbitrary)]
struct Input {
    // Rules are reversible; redact_mixed models permanent replacements.
    rules: Vec<RuleSpec>,
    session: SessionSpec,
    fields: Vec<Vec<Token>>,
    noise: Vec<NoiseToken>,
    cuts: Vec<u16>,
}

type Outcome = Vec<Result<String, String>>;

fn mask_fields(session: &mut RedactionSession, built: &BuiltRules, fields: &[String]) -> Outcome {
    for field in fields {
        session.reserve_replacements_in(field, &built.rules);
    }
    fields
        .iter()
        .map(|field| {
            mask_atomically(session, built.can_fail, |session| {
                session.mask_reserved(field, &built.rules)
            })
            .map_err(|error| format!("{error:?}"))
        })
        .collect()
}

fuzz_target!(|input: Input| {
    let built = BuiltRules::new(&input.rules, false);
    let fields: Vec<String> = input
        .fields
        .iter()
        .take(MAX_FIELDS)
        .map(|tokens| render_field(tokens, &built.samples))
        .collect();

    let mut session = input.session.build(false);
    let outcome = mask_fields(&mut session, &built, &fields);

    let mut again = input.session.build(false);
    assert_eq!(
        mask_fields(&mut again, &built, &fields),
        outcome,
        "masking must be deterministic for a key and scope"
    );
    assert_eq!(mappings(&again), mappings(&session));

    for (original, masked) in fields.iter().zip(&outcome) {
        if let Ok(masked) = masked {
            assert_all_masked(&session, original, &built.rules);
            assert_restores(&session, masked, original, &input.cuts);
        }
    }
    assert_session_consistent(&session);

    // Model output echoes, truncates, and damages replacements next to arbitrary text.
    let noise = render_noise(&input.noise, &session);
    assert_restore_consistent(&session, &noise, &input.cuts);
});
