use std::{
    net::{Ipv4Addr, Ipv6Addr},
    ops::Range,
    sync::LazyLock,
};

use regex::Regex;

use crate::template::joins_value;

static EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}")
        .expect("valid email pattern")
});

static IPV4: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}").expect("valid IPv4 pattern"));
static IPV6: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:[A-Fa-f0-9]{1,4}:[A-Fa-f0-9]{0,4}:|::)[A-Fa-f0-9.:]*")
        .expect("valid IPv6 pattern")
});

static CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9](?:[ -]?[0-9]){12,18}").expect("valid card pattern"));

static KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|glpat-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[0-9A-Za-z-]{10,}|(?:AKIA|ASIA)[0-9A-Z]{16})")
        .expect("valid API key pattern")
});

static PRIVATE_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----(BEGIN|END) ((?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?)-----")
        .expect("valid private-key delimiter pattern")
});

pub(crate) fn find_email(text: &str, record: &mut impl FnMut(Range<usize>)) {
    find_bounded(&EMAIL, text, record);
}

pub(crate) fn find_api_key(text: &str, record: &mut impl FnMut(Range<usize>)) {
    find_bounded(&KEY, text, record);
}

fn find_bounded(regex: &Regex, text: &str, record: &mut impl FnMut(Range<usize>)) {
    for matched in regex.find_iter(text).filter(|m| bounded(text, m.range())) {
        record(matched.range());
    }
}

pub(crate) fn find_payment_card(text: &str, record: &mut impl FnMut(Range<usize>)) {
    for matched in CARD.find_iter(text) {
        if bounded(text, matched.range()) && valid_card(matched.as_str()) {
            record(matched.range());
        }
    }
}

pub(crate) fn find_private_key(text: &str, record: &mut impl FnMut(Range<usize>)) {
    let mut opening = None;
    for groups in PRIVATE_KEY.captures_iter(text) {
        let (_, [kind, label]) = groups.extract();
        if let Some(delimiter) = groups.get(0) {
            if kind == "BEGIN" {
                opening = Some((delimiter.start(), label));
            } else if let Some((start, begin_label)) = opening
                && begin_label == label
            {
                record(start..delimiter.end());
                opening = None;
            }
        }
    }
}

fn bounded(text: &str, span: Range<usize>) -> bool {
    !joins_value(text[..span.start].chars().rev()) && !joins_value(text[span.end..].chars())
}

pub(crate) fn find_ip(text: &str, record: &mut impl FnMut(Range<usize>)) {
    let ipv6: Vec<_> = IPV6
        .find_iter(text)
        .filter_map(|matched| {
            let mut candidate = matched.as_str().trim_end_matches('.');
            let mut start = matched.start();
            // A colon after a label can be captured by the address pattern.
            if joins_value(text[..start].chars().rev()) {
                let (label_tail, address) = candidate.split_once(':')?;
                start += label_tail.len() + 1;
                candidate = address;
            }
            let span = start..start + candidate.len();
            (candidate.parse::<Ipv6Addr>().is_ok() && bounded(text, span.clone())).then_some(span)
        })
        .collect();
    let mut ipv6_ranges = ipv6.into_iter().peekable();
    for matched in IPV4.find_iter(text) {
        while ipv6_ranges
            .peek()
            .is_some_and(|span| span.end <= matched.start())
        {
            if let Some(span) = ipv6_ranges.next() {
                record(span);
            }
        }
        if !ipv6_ranges
            .peek()
            .is_some_and(|span| span.contains(&matched.start()))
            && matched.as_str().parse::<Ipv4Addr>().is_ok()
            && bounded(text, matched.range())
        {
            record(matched.range());
        }
    }
    ipv6_ranges.for_each(record);
}

fn valid_card(card: &str) -> bool {
    let digits = card.bytes().filter(u8::is_ascii_digit);
    let Some(first) = digits.clone().next() else {
        return false;
    };
    if digits.clone().all(|digit| digit == first) {
        return false;
    }
    let sum: u32 = digits
        .rev()
        .enumerate()
        .map(|(index, byte)| {
            let digit = u32::from(byte - b'0');
            if index.is_multiple_of(2) {
                digit
            } else {
                let doubled = digit * 2;
                doubled - 9 * u32::from(doubled > 9)
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use crate::{Mask, Matcher, RedactionSession, Rule, Rules};

    fn rules_for(matchers: impl IntoIterator<Item = Matcher>) -> Rules {
        Rules::new(
            matchers
                .into_iter()
                .map(|matcher| Rule::new(matcher, Mask::literal("X"))),
        )
    }

    #[test]
    fn detects_supported_shapes_and_validates_numbers() {
        let rules = rules_for([Matcher::Email, Matcher::IpAddress, Matcher::PaymentCard]);
        let text = "mail: me+work@example.com. IPv4: 192.0.2.8:8080. IPv6: [2001:db8::1] and ::ffff:192.0.2.4. card: 4111 1111 1111 1111";
        assert_eq!(
            RedactionSession::new().mask(text, &rules).unwrap(),
            "mail: X. IPv4: X:8080. IPv6: [X] and X. card: X"
        );
        let invalid = "bad addresses 999.0.2.8 and 2001:db8::g, invalid card 4111111111111112";
        assert_eq!(
            RedactionSession::new().mask(invalid, &rules).unwrap(),
            invalid
        );
    }

    #[test]
    fn detects_prefixed_api_keys_and_multiline_private_key_blocks() {
        let rules = rules_for([Matcher::ApiKey, Matcher::PrivateKey]);
        let text = format!(
            "token ghp_{}; key -----BEGIN PRIVATE KEY-----\nfixture\n-----END PRIVATE KEY-----",
            "a".repeat(36)
        );
        assert_eq!(
            RedactionSession::new().mask(&text, &rules).unwrap(),
            "token X; key X"
        );
    }
}
