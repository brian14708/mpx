//! Public API redaction contracts.

use std::fmt::Write as _;

use mpx_redact::{
    Digits, Error, Generator, GeneratorBinding, Hex, Mask, Matcher, Number, PlaceholderTemplate,
    RedactionSession, Rule, Rules, WordList,
};

fn tagged(matcher: Matcher) -> Rules {
    Rules::new([Rule::new(matcher, Mask::tag())])
}

#[test]
fn counts_each_slot_occurrence_and_saturates_large_products() {
    assert_eq!(PlaceholderTemplate::default().combinations(), 16_777_216);
    let words: Vec<_> = (0..2048).map(|index| format!("word{index}")).collect();
    let mut entries: Vec<_> = words.iter().map(String::as_str).collect();
    entries.push(&words[0]);
    let bindings = [GeneratorBinding::new(
        "word",
        WordList::new(&entries).unwrap(),
    )];
    let alias = Mask::alias("email", "{{word}}-{{word}}@example.com", &bindings).unwrap();
    assert_eq!(alias.alias_combinations(), Some(4_194_304));
    let marker = PlaceholderTemplate::new("<{{word}}-{{word}}-{{word}}>", &bindings).unwrap();
    assert_eq!(marker.combinations(), 8_589_934_592);
    assert_eq!(Mask::tag().alias_combinations(), None);

    let wide = PlaceholderTemplate::new(
        "<{{hex}}-{{hex}}>",
        &[GeneratorBinding::new("hex", Hex(16))],
    )
    .unwrap();
    assert_eq!(wide.combinations(), u128::MAX);
}

struct Region;

impl Generator for Region {
    fn capacity(&self) -> u128 {
        2
    }

    fn render(&self, index: u128, output: &mut String) {
        output.push_str(if index == 0 { "north" } else { "sud" });
    }
}

#[test]
fn aliases_and_placeholders_use_explicit_generator_bindings() {
    let bindings = [
        GeneratorBinding::new("serial", Digits(4)),
        GeneratorBinding::new("region", Region),
    ];
    let mask = Mask::alias("account", "user-{{region}}-{{serial}}", &bindings).unwrap();
    let rules = Rules::new([Rule::new(Matcher::Email, mask)]);
    let original = "a@example.org";
    let mut session = RedactionSession::new();
    let alias = session.mask(original, &rules).unwrap();
    let (region, serial) = alias
        .strip_prefix("user-")
        .unwrap()
        .split_once('-')
        .unwrap();
    assert!(["north", "sud"].contains(&region));
    assert_eq!(serial.len(), 4);
    assert!(serial.bytes().all(|byte| byte.is_ascii_digit()));
    assert_eq!(session.restore(&alias), original);
    let template = PlaceholderTemplate::new("<{{region}}>", &bindings).unwrap();
    let mut session = RedactionSession::new().with_placeholder_template(template);
    let masked = session
        .mask("<north> a@example.org", &tagged(Matcher::Email))
        .unwrap();
    assert_eq!(masked, "<north> <sud>");
    assert_eq!(session.restore(&masked), "<north> a@example.org");
    assert!(matches!(
        PlaceholderTemplate::new("<{{serial:4}}>", &bindings),
        Err(Error::InvalidAliasTemplate)
    ));
    assert!(matches!(
        Mask::alias("account", "{{serial:4}}", &bindings),
        Err(Error::InvalidAliasTemplate)
    ));
    assert!(matches!(
        PlaceholderTemplate::new("<{{hex}}>", &[]),
        Err(Error::UnboundAliasSlot { name }) if name == "hex"
    ));
    let duplicate_bindings = [
        GeneratorBinding::new("region", Region),
        GeneratorBinding::new("region", Digits(1)),
    ];
    assert!(matches!(
        Mask::alias("account", "{{region}}", &duplicate_bindings),
        Err(Error::InvalidAliasBinding)
    ));
}

#[test]
fn keyed_replacements_are_stable_within_a_scope() {
    let rules = tagged(Matcher::Literal("secret".into()));
    let first = RedactionSession::new()
        .with_hmac_key(b"gateway-secret")
        .with_scope("session-a")
        .mask("secret", &rules)
        .unwrap();
    let repeat = RedactionSession::new()
        .with_hmac_key(b"gateway-secret")
        .with_scope("session-a")
        .mask("secret", &rules)
        .unwrap();
    let other_scope = RedactionSession::new()
        .with_hmac_key(b"gateway-secret")
        .with_scope("session-b")
        .mask("secret", &rules)
        .unwrap();
    let other_key = RedactionSession::new()
        .with_hmac_key(b"other-secret")
        .with_scope("session-a")
        .mask("secret", &rules)
        .unwrap();

    assert_eq!(first, repeat);
    assert_ne!(first, other_scope);
    assert_ne!(first, other_key);
}

#[test]
fn custom_generators_reserve_values_and_probe_collisions() {
    let (alias, template) = {
        let name = String::from("region");
        let bindings = [GeneratorBinding::new(&name, Region)];
        let alias = Mask::alias("region", "region-{{region}}", &bindings).unwrap();
        let template = PlaceholderTemplate::new("<{{region}}>", &bindings).unwrap();
        (alias, template)
    };
    let rules = Rules::new([Rule::new(Matcher::Email, alias)]);
    let mut exchange = RedactionSession::new().with_placeholder_template(template.clone());
    let first = exchange.mask("a@example.org", &rules).unwrap();
    assert!(["region-north", "region-sud"].contains(&first.as_str()));
    let mut exchange = RedactionSession::new().with_placeholder_template(template);
    let original = format!("{first}. <north> a@example.org");
    assert_eq!(
        exchange.mask(&original, &rules).unwrap(),
        format!("{first}. <north> <sud>")
    );
    assert_eq!(
        exchange.restore(&format!("{first}. <north> <sud>")),
        original
    );
    assert!(matches!(
        exchange.mask("b@example.org", &tagged(Matcher::Email)),
        Err(Error::ReplacementUnavailable { .. })
    ));
    assert_eq!(exchange.replacement_map().entries().len(), 1);
}

#[test]
fn overlapping_marker_text_reserves_all_values() {
    let template = PlaceholderTemplate::new(
        ".{{digits}}.",
        &[GeneratorBinding::new("digits", Digits(1))],
    )
    .unwrap();
    let mut exchange = RedactionSession::new().with_placeholder_template(template);
    let rules = tagged(Matcher::Literal("secret".into()));
    let original = ".0.1.2.3.4.5.6.7.8.9.";
    exchange.reserve_replacements_in(original, &rules);
    assert!(matches!(
        exchange.mask("secret", &rules),
        Err(Error::ReplacementUnavailable { .. })
    ));
    assert_eq!(exchange.restore(original), original);
}

#[test]
fn strict_mode_rejects_collisions_without_probing() {
    let template = PlaceholderTemplate::new(
        "<{{number}}>",
        &[GeneratorBinding::new("number", Number(0..=0))],
    )
    .unwrap();
    let rules = tagged(Matcher::Literal("secret".into()));
    let mut session = RedactionSession::new()
        .with_placeholder_template(template)
        .with_strict();
    let first = session.mask("secret", &rules).unwrap();
    assert_eq!(first, "<0>");
    let error = session
        .mask("secret-2", &tagged(Matcher::Literal("secret-2".into())))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::ReplacementUnavailable {
            combinations: 1,
            attempts: 1,
            ..
        }
    ));
    assert!(error.to_string().contains("template \"<{{number}}>\""));
}

#[test]
fn template_changes_keep_previously_issued_markers_restorable() {
    let rules = tagged(Matcher::regex(r"\bsecret-[0-9]+\b").unwrap());
    let mut exchange = RedactionSession::new();
    let first = exchange.mask("secret-1", &rules).unwrap();
    let template = PlaceholderTemplate::new(
        "[[hidden-{{hex}}]]",
        &[GeneratorBinding::new("hex", Hex(8))],
    )
    .unwrap();
    let mut exchange = exchange.with_placeholder_template(template);
    let second = exchange.mask("secret-2", &rules).unwrap();
    let combined = format!("α{first}{second}末");

    assert_eq!(exchange.restore(&combined), "αsecret-1secret-2末");
    for (split, _) in combined.char_indices() {
        let mut restorer = exchange.restorer();
        let output = restorer.push(&combined[..split])
            + &restorer.push(&combined[split..])
            + &restorer.finish();
        assert_eq!(output, "αsecret-1secret-2末", "split at byte {split}");
    }
}

#[test]
fn labels_do_not_hide_ipv6_addresses() {
    let rules = tagged(Matcher::IpAddress);
    let text = "source:2001:db8::1 host:::1 address 2001:db8::g";
    let matches: Vec<_> = rules
        .find(text)
        .into_iter()
        .map(|span| &text[span])
        .collect();
    assert_eq!(matches, ["2001:db8::1", "::1"]);
}

#[test]
fn a_malformed_private_key_block_does_not_hide_the_next_valid_block() {
    let rules = tagged(Matcher::PrivateKey);
    let key = "-----BEGIN RSA PRIVATE KEY-----\nsynthetic\n-----END RSA PRIVATE KEY-----";
    let text = format!("-----BEGIN PRIVATE KEY-----\nincomplete\n{key}\nordinary text");
    let matches: Vec<_> = rules
        .find(&text)
        .into_iter()
        .map(|span| &text[span])
        .collect();
    assert_eq!(matches, [key]);
}

#[test]
fn mixed_unicode_text_round_trips_under_many_chunk_partitions() {
    let rules = Rules::new([
        Rule::new(Matcher::regex(r"\bsecret-[0-9]+\b").unwrap(), Mask::tag()),
        Rule::new(
            Matcher::Email,
            Mask::alias(
                "email",
                "person-{{hex}}@example.invalid",
                &[GeneratorBinding::new("hex", Hex(12))],
            )
            .unwrap(),
        ),
    ]);
    let mut text = String::new();
    for index in 0..128 {
        write!(
            text,
            "α{index} secret-{index}, person{index}@example.org. 末 "
        )
        .unwrap();
    }
    let mut exchange = RedactionSession::new();
    let masked = exchange.mask(&text, &rules).unwrap();
    assert_eq!(exchange.replacement_map().entries().len(), 256);
    let boundaries: Vec<_> = masked
        .char_indices()
        .map(|(offset, _)| offset)
        .chain([masked.len()])
        .collect();
    for width in [1, 2, 3, 7, 11, 32, 113, 1024] {
        let mut restorer = exchange.restorer();
        let mut output = String::new();
        let mut start = 0;
        for end in boundaries.iter().copied().skip(width).step_by(width) {
            output.push_str(&restorer.push(&masked[start..end]));
            start = end;
        }
        output.push_str(&restorer.push(&masked[start..]));
        output.push_str(&restorer.finish());
        assert_eq!(output, text, "with {width}-character chunks");
    }
}
