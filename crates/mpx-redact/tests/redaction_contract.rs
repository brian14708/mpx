//! Public API redaction contracts.

use std::fmt::Write as _;

use mpx_redact::{Error, Matcher, RedactionSession, Replacement, Rules, Slot, SlotGenerator};

fn tagged(matcher: Matcher) -> Rules {
    Rules::new([(matcher, Replacement::kind("SECRET").unwrap())])
}

#[test]
fn marker_kinds_require_nonempty_uppercase_ascii_names() {
    for kind in ["", "bad", "A-B", "A B", "É", "A{B}"] {
        assert!(matches!(
            Replacement::kind(kind),
            Err(Error::InvalidTemplate)
        ));
    }
    let rules = Rules::new([(
        Matcher::literal("sample"),
        Replacement::kind("API_KEY").unwrap(),
    )]);
    let mut session = RedactionSession::new();
    let masked = session.mask("sample", &rules).unwrap();
    assert!(masked.starts_with("{{API_KEY_"));
    assert!(masked.ends_with("}}"));
    assert_eq!(session.restore(&masked), "sample");
}

#[test]
fn custom_generators_cannot_rewrite_template_text_or_earlier_slots() {
    struct Reset;

    impl SlotGenerator for Reset {
        fn generate(&self, _: [u8; 32], _: usize, output: &mut String) {
            assert_eq!(output, "");
            output.clear();
            output.push_str("second");
        }
    }

    let mask =
        Replacement::template("<{}:{}>", &[Slot::literal("first"), Slot::custom(Reset)]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), mask)]);
    let mut session = RedactionSession::new();
    let masked = session.mask("secret", &rules).unwrap();
    assert_eq!(masked, "<first:second>");
    assert_eq!(session.restore(&masked), "secret");
}

#[test]
fn templates_seed_each_slot_separately() {
    let digits =
        Replacement::permanent_template("{}_{}", &[Slot::digits(6), Slot::digits(6)]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), digits)]);
    let mut session = RedactionSession::new();
    let masked = session.mask("secret", &rules).unwrap();
    let (first, second) = masked.split_once('_').unwrap();
    for slot in [first, second] {
        assert_eq!(slot.len(), 6);
        assert!(slot.bytes().all(|byte| byte.is_ascii_digit()));
    }
    assert_ne!(first, second);
    assert_eq!(session.mask("secret", &rules).unwrap(), masked);
    assert_eq!(session.restore(&masked), masked);
}

#[test]
fn custom_generators_receive_their_slot_index() {
    struct Index;

    impl SlotGenerator for Index {
        fn generate(&self, seed: [u8; 32], slot_idx: usize, output: &mut String) {
            write!(output, "{seed:?}:{slot_idx}").unwrap();
        }
    }

    let slot = Slot::custom(Index);
    let replacement =
        Replacement::template("<{}|{}|{}>", &[slot.clone(), slot.clone(), slot]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), replacement)]);
    let masked = RedactionSession::new().mask("secret", &rules).unwrap();
    let parts: Vec<_> = masked
        .trim_start_matches('<')
        .trim_end_matches('>')
        .split('|')
        .map(|part| part.rsplit_once(':').unwrap())
        .collect();
    assert_eq!(parts.len(), 3);
    assert_eq!([parts[0].1, parts[1].1, parts[2].1], ["0", "1", "2"]);
    assert_eq!(parts[0].0, parts[1].0);
    assert_eq!(parts[1].0, parts[2].0);
}

#[test]
fn replacement_map_records_permanent_masks_in_the_forward_direction() {
    let permanent = Replacement::permanent_template("<{}>", &[Slot::literal("masked")]).unwrap();
    let rules = Rules::new([
        (Matcher::literal("first"), permanent.clone()),
        (Matcher::literal("second"), permanent),
        (Matcher::literal("third"), Replacement::mask("")),
    ]);
    let mut session = RedactionSession::new();
    assert_eq!(
        session.mask("first second third first", &rules).unwrap(),
        "<masked> <masked>  <masked>"
    );
    let snapshot = session.replacement_map();
    assert_eq!(snapshot.entries().len(), 0);
    assert_eq!(snapshot.get("<masked>"), None);
    assert_eq!(
        snapshot.forward_entries().collect::<Vec<_>>(),
        [("first", "<masked>"), ("second", "<masked>"), ("third", "")]
    );

    let reversible = session
        .mask("first", &tagged(Matcher::literal("first")))
        .unwrap();
    let map = session.replacement_map();
    assert_eq!(map.get(&reversible), Some("first"));
    assert_eq!(
        map.replacements_for("first").collect::<Vec<_>>(),
        [reversible.as_str(), "<masked>"]
    );
    assert_eq!(
        snapshot.replacements_for("first").collect::<Vec<_>>(),
        ["<masked>"]
    );
}

#[test]
fn fixed_masks_can_be_reversible_or_permanent() {
    let reversible = Replacement::template("<{}>", &[Slot::literal("fixed")]).unwrap();
    let rules = Rules::new([
        (Matcher::literal("first"), reversible.clone()),
        (Matcher::literal("second"), reversible),
        (Matcher::literal("last"), Replacement::mask("[removed]")),
    ]);
    let mut session = RedactionSession::new().with_fallback(Replacement::kind("SECRET").unwrap());
    let masked = session.mask("first second last", &rules).unwrap();
    assert!(masked.starts_with("<fixed> {{SECRET_"));
    assert_eq!(session.restore(&masked), "first second [removed]");
    let map = session.replacement_map();
    assert_eq!(map.get("<fixed>"), Some("first"));
    assert_eq!(map.get("[removed]"), None);
    assert_eq!(
        map.replacements_for("last").collect::<Vec<_>>(),
        ["[removed]"]
    );

    let permanent = Rules::new([(Matcher::literal("last"), Replacement::mask("<fixed>"))]);
    let fallback = session.mask("last", &permanent).unwrap();
    assert!(fallback.starts_with("{{SECRET_"));
    assert_eq!(session.restore(&fallback), fallback);
}

#[test]
fn permanent_templates_and_reversible_replacements_cannot_share_an_alias() {
    let reversible = Replacement::template("{}", &[Slot::literal("alias")]).unwrap();
    let permanent = Replacement::permanent_template("<{}>", &[Slot::literal("alias")]).unwrap();
    let rules = Rules::new([
        (Matcher::literal("reversible"), reversible),
        (Matcher::literal("permanent"), permanent),
    ]);
    for (input, prefix) in [
        ("permanent reversible", "<alias>"),
        ("reversible permanent", "alias"),
    ] {
        let mut session =
            RedactionSession::new().with_fallback(Replacement::kind("SECRET").unwrap());
        let masked = session.mask(input, &rules).unwrap();
        let (first, fallback) = masked.split_once(' ').unwrap();
        assert_eq!(first, prefix);
        assert!(fallback.starts_with("{{SECRET_"));
        let restored = session.restore(&masked);
        if first == "<alias>" {
            assert_eq!(restored, "<alias> reversible");
        } else {
            assert_eq!(restored, format!("reversible {fallback}"));
        }
    }

    let mut session = RedactionSession::new();
    assert!(matches!(
        session.mask("permanent reversible", &rules),
        Err(Error::ReplacementUnavailable { .. })
    ));
    assert_eq!(session.replacement_map().forward_entries().count(), 0);
    assert_eq!(session.mask("permanent", &rules).unwrap(), "<alias>");
    session.clear_reserved_replacements();
    assert!(matches!(
        session.mask("reversible", &rules),
        Err(Error::ReplacementUnavailable { .. })
    ));
}

#[test]
fn mask_spans_falls_back_when_replacement_text_is_already_present() {
    let replacement = Replacement::template("<{}>", &[Slot::number(0..=1)]).unwrap();
    let private = "private";
    let mut session = RedactionSession::new();
    let first = session
        .mask_spans(
            private,
            std::slice::from_ref(&(0..private.len())),
            &replacement,
        )
        .unwrap();

    let text = format!("{first} {private}");
    let span = first.len() + 1..text.len();
    let mut session = RedactionSession::new();
    assert!(matches!(
        session.mask_spans(&text, std::slice::from_ref(&span), &replacement),
        Err(Error::ReplacementUnavailable { template }) if template == "<{}>"
    ));
    let mut session = session.with_fallback(Replacement::kind("SECRET").unwrap());
    let masked = session
        .mask_spans(&text, std::slice::from_ref(&span), &replacement)
        .unwrap();
    assert!(masked.starts_with(&format!("{first} {{{{SECRET_")));
    assert_eq!(session.restore(&masked), text);
}

#[test]
fn block_takes_precedence_over_an_overlapping_literal() {
    let rules = Rules::new([
        (
            Matcher::literal("allowed"),
            Replacement::kind("SECRET").unwrap(),
        ),
        (Matcher::literal("secret-ID"), Replacement::mask("removed")),
        (Matcher::literal("ID"), Replacement::block()),
    ]);
    let mut session = RedactionSession::new();
    assert!(matches!(
        session.mask("allowed secret-ID", &rules),
        Err(Error::Blocked)
    ));
    assert_eq!(session.replacement_map().entries().len(), 0);
    assert_eq!(session.mask("unmatched", &rules).unwrap(), "unmatched");

    let spans = [0..2, 1..2];
    assert!(matches!(
        session.mask_spans("ID", &spans, &Replacement::block()),
        Err(Error::Blocked)
    ));
}

#[test]
fn builtin_rules_merge_overlapping_url_password_and_email() {
    let original = "https://alice:demo-pass@example.org/db";
    let rules = Rules::builtin();
    let mut session = RedactionSession::new();
    let masked = session.mask(original, &rules).unwrap();
    let suffix = masked
        .strip_prefix("https://alice:{{EMAIL_")
        .and_then(|tag| tag.strip_suffix("}}/db"))
        .unwrap();
    assert_eq!(suffix.len(), 8);
    assert!(
        suffix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || (b'2'..=b'7').contains(&byte))
    );
    assert_eq!(session.replacement_map().entries().len(), 1);
    assert_eq!(session.restore(&masked), original);
}

fn email_matcher() -> Matcher {
    Matcher::regex(
        r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}",
    )
    .unwrap()
}

#[test]
fn generates_repeated_slots_and_wide_outputs() {
    let mut entries: Vec<_> = (0..2048).map(|index| format!("word{index}")).collect();
    entries.insert(1, String::from("word0"));
    let words = Slot::wordlist(entries).unwrap();
    let mut value = String::new();
    let mut seed = [0; 32];
    seed[7] = 1;
    words.generate(seed, 0, &mut value);
    assert_eq!(value, "word1");
    let alias =
        Replacement::template("{}-{}@example.com", &[words.clone(), words.clone()]).unwrap();
    let marker =
        Replacement::template("<{}-{}-{}>", &[words.clone(), words.clone(), words]).unwrap();
    let wide = Replacement::template("<{}-{}>", &[Slot::base16(16), Slot::base16(16)]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), alias)]);
    let masked = RedactionSession::new().mask("secret", &rules).unwrap();
    assert_ne!(masked, "");
    let rules = Rules::new([(Matcher::literal("secret"), marker)]);
    assert!(RedactionSession::new().mask("secret", &rules).is_ok());
    let rules = Rules::new([(Matcher::literal("secret"), wide)]);
    assert!(RedactionSession::new().mask("secret", &rules).is_ok());
}

struct Region;

impl SlotGenerator for Region {
    fn generate(&self, seed: [u8; 32], _: usize, output: &mut String) {
        output.push_str(if seed[0].is_multiple_of(2) {
            "north"
        } else {
            "sud"
        });
    }
}

#[test]
fn aliases_and_replacements_use_explicit_generator_bindings() {
    let bindings = [Slot::custom(Region), Slot::digits(4)];
    let mask = Replacement::template("user-{}-{}", &bindings).unwrap();
    let rules = Rules::new([(email_matcher(), mask)]);
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
    assert!(matches!(
        Replacement::template("<{}>", &[]),
        Err(Error::InvalidTemplate)
    ));
}

#[test]
fn keyed_replacements_are_stable_within_a_scope() {
    let rules = tagged(Matcher::literal("secret"));
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
fn reserved_aliases_use_the_enabled_fallback_at_whole_value_boundaries() {
    let alias = Replacement::template("region-{}", &[Slot::custom(Region)]).unwrap();
    let rules = Rules::new([(email_matcher(), alias)]);
    let private = "a@example.org";
    let primary = RedactionSession::new().mask(private, &rules).unwrap();
    let default = RedactionSession::new()
        .mask(private, &tagged(Matcher::literal(private)))
        .unwrap();

    let mut exchange = RedactionSession::new().with_fallback(Replacement::kind("SECRET").unwrap());
    let original = format!("prefix-{primary} {primary}. {private} {private}");
    let masked = exchange.mask(&original, &rules).unwrap();
    assert_eq!(
        masked,
        format!("prefix-{primary} {primary}. {default} {default}")
    );
    assert_eq!(exchange.restore(&masked), original);
    assert_eq!(exchange.replacement_map().entries().len(), 1);

    let mut exchange = RedactionSession::new();
    let original = format!("prefix-{primary} {primary}.suffix {private}");
    let masked = exchange.mask(&original, &rules).unwrap();
    assert_eq!(
        masked,
        format!("prefix-{primary} {primary}.suffix {primary}")
    );
    assert_eq!(exchange.restore(&masked), original);
}

#[test]
fn a_configured_alias_fallback_checks_reserved_text_and_issued_replacements() {
    let primary = Replacement::template("<{}>", &[Slot::literal("taken")]).unwrap();
    let fallback = Replacement::template("{}", &[Slot::literal("cedar")]).unwrap();
    let rules = Rules::new([
        (Matcher::literal("first"), primary.clone()),
        (Matcher::literal("second"), primary),
    ]);

    let mut session = RedactionSession::new().with_fallback(fallback.clone());
    session.reserve_replacements_in("<taken> cedar", &rules);
    assert!(matches!(
        session.mask("first", &rules),
        Err(Error::ReplacementUnavailable { template }) if template == "{}"
    ));
    assert_eq!(session.restore("cedar"), "cedar");

    let mut session = RedactionSession::new().with_fallback(fallback);
    session.reserve_replacements_in("<taken>", &rules);
    assert_eq!(session.mask("first", &rules).unwrap(), "cedar");
    assert!(matches!(
        session.mask("second", &rules),
        Err(Error::ReplacementUnavailable { template }) if template == "{}"
    ));
    assert_eq!(session.restore("cedar"), "first");
}

#[test]
fn literal_replacements_are_reserved_for_collision_checks() {
    let mask = Replacement::template("<{}>", &[Slot::number(0..=0)]).unwrap();
    let rules = Rules::new([
        (Matcher::literal("secret"), mask),
        (Matcher::literal("public"), Replacement::mask("<0>")),
    ]);
    let mut exchange = RedactionSession::new();
    assert!(matches!(
        exchange.mask("public secret", &rules),
        Err(Error::ReplacementUnavailable { template }) if template == "<{}>"
    ));
    let mut exchange = exchange.with_fallback(Replacement::kind("SECRET").unwrap());
    let masked = exchange.mask("public secret", &rules).unwrap();
    assert!(masked.starts_with("<0> {{SECRET_"));
    assert_eq!(exchange.restore(&masked), "<0> secret");
}

#[test]
fn empty_custom_slots_fail_even_in_framed_templates() {
    struct Empty;
    impl SlotGenerator for Empty {
        fn generate(&self, _: [u8; 32], _: usize, _: &mut String) {}
    }

    let mask = Replacement::template("<{}>", &[Slot::custom(Empty)]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), mask)]);
    let mut exchange = RedactionSession::new();
    assert!(matches!(
        exchange.mask("secret", &rules),
        Err(Error::ReplacementUnavailable { .. })
    ));
    assert_eq!(exchange.replacement_map().entries().len(), 0);
}

#[test]
fn overlapping_marker_text_reserves_all_values() {
    let mask = Replacement::template(".{}.", &[Slot::digits(1)]).unwrap();
    let mut exchange = RedactionSession::new();
    let rules = Rules::new([(Matcher::literal("secret"), mask)]);
    let original = ".0.1.2.3.4.5.6.7.8.9.";
    exchange.reserve_replacements_in(original, &rules);
    assert!(matches!(
        exchange.mask("secret", &rules),
        Err(Error::ReplacementUnavailable { .. })
    ));
    assert_eq!(exchange.restore(original), original);
}

#[test]
fn collisions_with_existing_map_entries_are_rejected_by_default() {
    let mask = Replacement::template("<{}>", &[Slot::number(0..=0)]).unwrap();
    let rules = Rules::new([(Matcher::literal("secret"), mask.clone())]);
    let mut session = RedactionSession::new();
    let first = session.mask("secret", &rules).unwrap();
    assert_eq!(first, "<0>");
    let error = session
        .mask(
            "secret-2",
            &Rules::new([(Matcher::literal("secret-2"), mask)]),
        )
        .unwrap_err();
    assert!(matches!(error, Error::ReplacementUnavailable { .. }));
    assert!(error.to_string().contains("template \"<{}>\""));
}

#[test]
fn template_changes_keep_previously_issued_markers_restorable() {
    let rules = Rules::new([(
        Matcher::regex(r"\bsecret-[0-9]+\b").unwrap(),
        Replacement::kind("SECRET").unwrap(),
    )]);
    let mut exchange = RedactionSession::new();
    let first = exchange.mask("secret-1", &rules).unwrap();
    let second_rules = Rules::new([(
        Matcher::regex(r"\bsecret-[0-9]+\b").unwrap(),
        Replacement::template("[[hidden-{}]]", &[Slot::base16(8)]).unwrap(),
    )]);
    let second = exchange.mask("secret-2", &second_rules).unwrap();
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
fn builtin_rules_mask_and_restore_representative_values() {
    let rules = Rules::builtin();
    let jwt = "eyJhbGciOiJIUzI1NiJ9.e30.s8Kh1_zISiE2C5qsmPn1oaNh_S-i4aqu1Rg0TZjnFpg";
    let cases: &[(&str, &str, &[&str])] = &[
        (
            r#"（"josé smith"@example.org）"#,
            "{{EMAIL_",
            &[r#""josé smith"@example.org"#],
        ),
        (
            "Read @param. Email café@例子.世界 or alice@server.42.example",
            "{{EMAIL_",
            &["café@例子.世界", "alice@server.42.example"],
        ),
        (
            "alice@[IPv6:2001:db8::8]",
            "{{EMAIL_",
            &["alice@[IPv6:2001:db8::8]"],
        ),
        (
            "192.0.2.8:8080 source:2001:db8::1 [::ffff:192.0.2.9%eth0]:443 999.0.2.8",
            "{{IP_",
            &["192.0.2.8", "2001:db8::1", "::ffff:192.0.2.9"],
        ),
        (
            "https://alice:p%40ss@localhost/db",
            "{{PASSWORD_",
            &["p%40ss"],
        ),
        (r"C:\Users\alícé\notes.txt", "{{USERNAME_", &["alícé"]),
        (r"C:\uſers\Åsa\notes.txt", "{{USERNAME_", &["Åsa"]),
        ("Username: alice", "{{USERNAME_", &["alice"]),
        (
            "Host Name = build-01.internal",
            "{{HOSTNAME_",
            &["build-01.internal"],
        ),
        (
            "sk-proj-abcdefghijklmnopqrstuvwxyz",
            "{{API_KEY_",
            &["sk-proj-abcdefghijklmnopqrstuvwxyz"],
        ),
        (jwt, "{{TOKEN_", &[jwt]),
    ];
    for &(text, prefix, expected) in cases {
        let matches: Vec<_> = rules
            .try_find(text)
            .unwrap()
            .into_iter()
            .map(|span| &text[span])
            .collect();
        assert_eq!(matches, expected, "{text:?}");

        let mut session = RedactionSession::new();
        let masked = session.mask(text, &rules).unwrap();
        assert_eq!(masked.matches(prefix).count(), expected.len(), "{masked:?}");
        assert_eq!(session.restore(&masked), text);
    }

    let text = "ssh -p 2222 alice@server";
    let mut session = RedactionSession::new();
    let masked = session.mask(text, &rules).unwrap();
    let (user, host) = masked
        .strip_prefix("ssh -p 2222 ")
        .and_then(|target| target.split_once('@'))
        .unwrap();
    assert!(user.starts_with("{{USERNAME_"), "{masked:?}");
    assert!(host.starts_with("{{HOSTNAME_"), "{masked:?}");
    assert_eq!(session.restore(&masked), text);
}

#[test]
fn builtin_rules_leave_boundary_cases_visible() {
    let rules = Rules::builtin();
    for text in [
        "alice..bob@example.org",
        "git clone git@github.com:owner/repo.git",
        "MAC: 02:42:ac:11:00:02",
        "https://localhost?next=:demo@localhost",
        "https://docs.example.org:443@",
        "/home/../shared",
        "GET /users/settings HTTP/1.1",
        "Hostname:\nstatus=ready\nUsername:\nStatus: ready",
        "v1.0-release-notes-for-november",
        "eyJxxxxxxxx.e30.abcdefgh eyJhbGciOiJIUzI1NiJ9.eA.abcdefgh",
    ] {
        assert_eq!(RedactionSession::new().mask(text, &rules).unwrap(), text);
    }
}

#[test]
fn streaming_restores_shared_prefixes_and_flushes_partial_markers() {
    let cases = [
        ("short-alias", "{}", "cèdre"),
        ("long-alias", "{}", "cèdre.maple"),
        ("short-marker", ".{}!", "maple"),
        ("long-marker", ".{}!longer?", "maple"),
        ("bracketed-marker", "[[{}]]", "test"),
    ];
    let rules = Rules::new(cases.into_iter().map(|(original, template, value)| {
        (
            Matcher::literal(original),
            Replacement::template(template, &[Slot::literal(value)]).unwrap(),
        )
    }));
    let mut session = RedactionSession::new();
    for (original, template, value) in cases {
        assert_eq!(
            session.mask(original, &rules).unwrap(),
            template.replace("{}", value)
        );
    }

    let text = "α cèdre.maple. cèdre.maple!longer? .maple! 末 [[tes";
    let expected = "α long-alias. short-aliaslong-marker short-marker 末 [[tes";
    assert_eq!(session.restore(text), expected);
    for split in text
        .char_indices()
        .map(|(split, _)| split)
        .chain([text.len()])
    {
        let mut restorer = session.restorer();
        let mut output = restorer.push(&text[..split]);
        output.push_str(&restorer.push(""));
        output.push_str(&restorer.push(&text[split..]));
        output.push_str(&restorer.finish());
        assert_eq!(output, expected, "split at byte {split}");
    }
}

#[test]
fn mixed_unicode_text_round_trips_under_many_chunk_partitions() {
    let rules = Rules::new([
        (
            Matcher::regex(r"\bsecret-[0-9]+\b").unwrap(),
            Replacement::kind("SECRET").unwrap(),
        ),
        (
            email_matcher(),
            Replacement::template("person-{}@example.invalid", &[Slot::base16(12)]).unwrap(),
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
