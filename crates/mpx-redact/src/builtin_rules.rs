#![allow(
    clippy::non_std_lazy_statics,
    reason = "lazy_regex validates and compiles these detector patterns once"
)]

use std::{
    net::{Ipv4Addr, Ipv6Addr},
    ops::Range,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use lazy_regex::{Lazy, lazy_regex};
use memchr::{memchr, memchr_iter, memchr2, memchr3, memmem};
use regex::Regex;

use crate::{
    Matcher, Replacement,
    replacement_template::{bounded, joins_value},
};

static EMAIL: Lazy<Regex> = lazy_regex!(
    r#"(?x)
    (?:[\pL\pN\pM_%+-]+(?:\.[\pL\pN\pM_%+-]+)*|"(?:[^"\\\r\n]|\\[^\r\n])*")
    @(?:(?:[\pL\pN](?:[\pL\pN\pM-]*[\pL\pN\pM])?\.)+[\pL\pM]{2,63}
      |\[(?:(?i:IPv6):[A-Fa-f0-9:.]+|[0-9.]+)\])"#
);
static IPV4: Lazy<Regex> = lazy_regex!(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}");
static IPV6: Lazy<Regex> = lazy_regex!(
    r"(?:[A-Fa-f0-9]{1,4}:[A-Fa-f0-9]{0,4}:|::)[A-Fa-f0-9.:]*(?:%[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)?"
);
static API_KEY: Lazy<Regex> = lazy_regex!(
    r"(?x)
      sk-[A-Za-z0-9_-]{20,}
    | [A-Za-z0-9]{2,4}_sk_[A-Za-z0-9_-]{16,}
    | ark-[A-Za-z0-9_-]{16,}
    | gh[pousr]_[A-Za-z0-9]{30,}
    | github_pat_[A-Za-z0-9_]{40,}
    | glpat-[A-Za-z0-9_-]{20,}
    | AIza[0-9A-Za-z_-]{35}
    | xox[abprs]-[0-9A-Za-z-]{10,}
    | (?:sk|rk)_live_[A-Za-z0-9]{16,}
    | (?:AKIA|ASIA)[0-9A-Z]{16}
    | hf_[A-Za-z0-9]{20,}
    | gsk_[A-Za-z0-9_-]{20,}
    | xai-[A-Za-z0-9_-]{20,}
    | npm_[A-Za-z0-9_-]{20,}
    | pypi-[A-Za-z0-9_-]{20,}
    | dop_v1_[A-Za-z0-9_-]{20,}
    | SG\.[A-Za-z0-9_-]{16,}
    | (?:cfut|cfat|cfk|cfoat)_[A-Za-z0-9_-]{16,}
    | v1\.0-[A-Za-z0-9]{16,}-[A-Za-z0-9]{40,}"
);
static JWT: Lazy<Regex> =
    lazy_regex!(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{2,}\.[A-Za-z0-9_-]{8,}");
static URL_PASSWORD: Lazy<Regex> = lazy_regex!(
    r#"(?i)\b(?:https?|ftp)://[^/?#\s:@"'<>]*:([^@/?#\s"'<>]+)@(?:\[[A-Fa-f0-9:]|[\pL\pN])"#
);
// Consume URL paths and HTTP request targets without capturing a username.
static USER_PATH: Lazy<Regex> = lazy_regex!(
    r#"(?ix)
    \b(?:[a-z][a-z0-9+.-]*://[^/?\x23\s<>"']|(?-i:GET|HEAD|POST|PUT|PATCH|DELETE|OPTIONS)[\t\x20]+/)[^\s<>"']*
    |(?:/home/|[/\\]users[/\\]|documents\x20and\x20settings[/\\])(\.*[\pL\pN\pM_-][\pL\pN\pM._-]*)"#
);
// Only the directory prefixes in USER_PATH can introduce a username capture.
static USER_PATH_PREFIX: Lazy<Regex> =
    lazy_regex!(r"(?ix) /home/ | [/\\]users[/\\] | documents\x20and\x20settings[/\\]");
static USER_HOST: Lazy<Regex> = lazy_regex!(
    r"(?mx)
    (?:[$>\x23][\t\x20]*|(?:^|[\t\x20;&|])(?:[^\s;&|]*/)?(?:ssh|scp|sftp|rsync|mosh|ssh-copy-id)
        (?:[\t\x20]+-[^\s]+(?:[\t\x20]+[^\s@-][^\s@]*)?)*[\t\x20]+)
    \b([\pL_][\pL\pN\pM_.-]*)@([A-Za-z0-9][A-Za-z0-9_.-]*)\b"
);
static HOST_LABEL: Lazy<Regex> = lazy_regex!(
    r"(?im)^[\t\p{Zs}]*(?:hostname|host name|computer name|machine name|device name)[\t\p{Zs}]*[:=][\t\p{Zs}]*([\pL\pN][\pL\pN\pM_.-]*)"
);
static USER_LABEL: Lazy<Regex> = lazy_regex!(
    r"(?im)^[\t\p{Zs}]*(?:user|username|login|logname)[\t\p{Zs}]*[:=][\t\p{Zs}]*([\pL\pN\pM._-]+)"
);

/// Builds the deterministic built-in detector inventory.
pub(crate) fn rules() -> Vec<(Matcher, Replacement)> {
    let user_path = Matcher::Regex(USER_PATH.clone());
    vec![
        (
            Matcher::custom(|text| Ok(find_email(text))),
            marker("EMAIL"),
        ),
        (Matcher::custom(|text| Ok(find_ip(text))), marker("IP")),
        (
            requires(Matcher::Regex(URL_PASSWORD.clone()), b"@"),
            marker("PASSWORD"),
        ),
        (
            requires(
                Matcher::custom(move |text| {
                    if USER_PATH_PREFIX.is_match(text) {
                        user_path.find(text)
                    } else {
                        Ok(Vec::new())
                    }
                }),
                b"/\\",
            ),
            marker("USERNAME"),
        ),
        (
            requires(Matcher::custom(|text| Ok(find_user_host(text, 1))), b"@"),
            marker("USERNAME"),
        ),
        (
            requires(Matcher::custom(|text| Ok(find_user_host(text, 2))), b"@"),
            marker("HOSTNAME"),
        ),
        (
            requires(Matcher::Regex(HOST_LABEL.clone()), b":="),
            marker("HOSTNAME"),
        ),
        (
            requires(Matcher::Regex(USER_LABEL.clone()), b":="),
            marker("USERNAME"),
        ),
        (
            requires(
                Matcher::custom(|text| Ok(find_bounded(&API_KEY, text).collect())),
                b"_-AS",
            ),
            marker("API_KEY"),
        ),
        (
            Matcher::custom(|text| {
                Ok(find_bounded(&JWT, text)
                    .filter(|span| jwt_value(&text[span.clone()]))
                    .collect())
            }),
            marker("TOKEN"),
        ),
    ]
}

// Every match must contain at least one of these ASCII bytes.
fn requires(matcher: Matcher, any_of: &'static [u8]) -> Matcher {
    Matcher::custom(move |text| {
        let bytes = text.as_bytes();
        let found = any_of.chunks(3).any(|needles| match *needles {
            [first] => memchr(first, bytes).is_some(),
            [first, second] => memchr2(first, second, bytes).is_some(),
            [first, second, third] => memchr3(first, second, third, bytes).is_some(),
            _ => unreachable!("chunks contain one to three bytes"),
        });
        if found {
            matcher.find(text)
        } else {
            Ok(Vec::new())
        }
    })
}

fn marker(kind: &str) -> Replacement {
    Replacement::kind(kind).expect("built-in marker template is valid")
}

fn find_email(text: &str) -> Vec<Range<usize>> {
    let bytes = text.as_bytes();
    let has_domain = memchr_iter(b'@', bytes).any(|at| {
        let domain = &bytes[at + 1..];
        if domain.starts_with(b"[") {
            return true;
        }
        // Named domains need a dot followed by another label. Let the regex
        // resolve Unicode categories and validate label boundaries.
        let mut suffix = domain
            .iter()
            .copied()
            .skip_while(|&byte| !byte.is_ascii() || byte.is_ascii_alphanumeric() || byte == b'-');
        suffix.next() == Some(b'.')
            && suffix
                .next()
                .is_some_and(|byte| !byte.is_ascii() || byte.is_ascii_alphanumeric())
    });
    if !has_domain {
        return Vec::new();
    }
    EMAIL
        .find_iter(text)
        .filter_map(|matched| {
            let span = matched.range();
            // A dot next to an address can be punctuation. A dot following a
            // word can also be part of a malformed local part.
            let before = text[..span.start].trim_end_matches('.');
            let after = &text[span.end..];
            if joins_value(before.chars().rev()) || joins_value(after.chars()) {
                return None;
            }
            let (local, domain) = matched.as_str().rsplit_once('@')?;
            if local.eq_ignore_ascii_case("git")
                && after
                    .strip_prefix(':')
                    .and_then(|path| path.chars().next())
                    .is_some_and(|first| {
                        first.is_alphanumeric() || matches!(first, '/' | '~' | '.')
                    })
            {
                return None;
            }
            if let Some(literal) = domain
                .strip_prefix('[')
                .and_then(|domain| domain.strip_suffix(']'))
            {
                if let Some((_, ipv6)) = literal.split_once(':') {
                    ipv6.parse::<Ipv6Addr>().ok()?;
                } else {
                    literal.parse::<Ipv4Addr>().ok()?;
                }
            }
            Some(span)
        })
        .collect()
}

/// Finds the user (group 1) or host (group 2) of a shell `user@host` argument.
fn find_user_host(text: &str, group: usize) -> Vec<Range<usize>> {
    // The pattern requires a prompt character or one of these command names.
    let bytes = text.as_bytes();
    if memchr3(b'$', b'#', b'>', bytes).is_none()
        && !["ssh", "scp", "sftp", "rsync", "mosh"]
            .into_iter()
            .any(|command| memmem::find(bytes, command.as_bytes()).is_some())
    {
        return Vec::new();
    }
    USER_HOST
        .captures_iter(text)
        .filter_map(|captures| captures.get(group))
        .map(|matched| matched.range())
        .collect()
}

fn find_ip(text: &str) -> Vec<Range<usize>> {
    // Rules::builtin already sorts spans and merges an IPv4 suffix into its IPv6 match.
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    if memchr(b':', bytes).is_some() {
        spans.extend(find_ipv6(text));
    }
    if memchr(b'.', bytes).is_some() {
        spans.extend(
            find_bounded(&IPV4, text).filter(|span| text[span.clone()].parse::<Ipv4Addr>().is_ok()),
        );
    }
    spans
}

fn find_ipv6(text: &str) -> impl Iterator<Item = Range<usize>> {
    IPV6.find_iter(text).filter_map(move |matched| {
        let (address, zone) = matched
            .as_str()
            .split_once('%')
            .unwrap_or((matched.as_str(), ""));
        let mut candidate = address.trim_end_matches('.');
        let mut start = matched.start();
        if joins_value(text[..start].chars().rev()) {
            let (label_tail, address) = candidate.split_once(':')?;
            start += label_tail.len() + 1;
            candidate = address;
        }
        if candidate.parse::<Ipv6Addr>().is_err() {
            candidate = candidate.strip_suffix(':')?;
            candidate.parse::<Ipv6Addr>().ok()?;
        }
        let span = start..start + candidate.len();
        let end = if zone.is_empty() {
            span.end
        } else {
            matched.end()
        };
        bounded(text, start..end).then_some(span)
    })
}

fn jwt_value(value: &str) -> bool {
    let mut parts = value.split('.');
    let decode_json = |part: &str| {
        let decoded = URL_SAFE_NO_PAD.decode(part).ok()?;
        serde_json::from_slice::<serde_json::Value>(&decoded).ok()
    };
    let Some(header) = parts.next().and_then(decode_json) else {
        return false;
    };
    header
        .get("alg")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|alg| !alg.is_empty())
        && parts
            .next()
            .and_then(decode_json)
            .is_some_and(|claims| claims.is_object())
}

fn find_bounded(regex: &Regex, text: &str) -> impl Iterator<Item = Range<usize>> {
    regex
        .find_iter(text)
        .map(|matched| matched.range())
        .filter(move |span| bounded(text, span.clone()))
}
