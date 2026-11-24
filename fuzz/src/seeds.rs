//! Deterministic starting cases. Each target also accepts arbitrary mutations.

use crate::{
    flags::{CodecFlags, RequestFlags, SelectorFlags},
    protocol_input::{CaseKind, FIXTURES, STREAM_PREFIXES},
    selector, status_selector,
};

/// Index of `200 OK` in [`crate::STATUSES`].
const OK: usize = 3;
const RAW: u8 = CaseKind::Raw as u8;
const FIXTURE: u8 = CaseKind::Fixture as u8;
const GENERATED: u8 = CaseKind::Generated as u8;
const ROUNDTRIP_TEXTS: [(&str, &[u8]); 2] = [
    ("emoji", b"Hi \xf0\x9f\xa6\x89 \"\\\n\0"),
    ("escapes", b"quote \"\\\n\0 \xe6\x9c\xab "),
];

pub struct Seed {
    pub target: &'static str,
    pub name: String,
    pub bytes: Vec<u8>,
}

fn seed(target: &'static str, name: impl Into<String>, header: &[u8], payload: &[u8]) -> Seed {
    Seed {
        target,
        name: name.into(),
        bytes: [header, payload].concat(),
    }
}

/// Encodes a stream header with its cut-byte count and cut points.
fn stream_header(header: [u8; 4], cuts: &[u16]) -> Vec<u8> {
    let mut bytes = header.to_vec();
    bytes.push(u8::try_from(cuts.len() * 2).expect("few cuts"));
    bytes.extend(cuts.iter().flat_map(|cut| cut.to_le_bytes()));
    bytes
}

pub fn catalog() -> Vec<Seed> {
    let mut seeds = Vec::new();
    for (index, fixture) in FIXTURES.iter().enumerate() {
        let name = fixture.name;
        let native = selector(index, 0);
        let next = (index + 1) % FIXTURES.len();
        let next_stream = selector(next, SelectorFlags::STREAM.bits());
        let upstream = status_selector(index, OK);
        let codec = if index % 2 == 0 {
            0
        } else {
            CodecFlags::REJECT_LOSS.bits()
        };
        for kind in CaseKind::ALL {
            let mode = kind as u8;
            // Fixture inputs ignore the payload; Generated inputs embed it.
            let payload = |raw| match kind {
                CaseKind::Raw => raw,
                CaseKind::Fixture => &[][..],
                CaseKind::Generated => b"seed \"\\\n\0".as_slice(),
            };
            let name = format!("{name}-{}", kind.label());
            seeds.push(seed(
                "protocol_request",
                &name,
                &[mode, native, codec, 0],
                payload(fixture.request),
            ));
            seeds.push(seed(
                "protocol_response",
                &name,
                &[mode, upstream, native, codec],
                payload(fixture.response),
            ));
            let stream_payload = match kind {
                CaseKind::Generated => &[index as u8][..],
                _ => payload(fixture.stream),
            };
            seeds.push(seed(
                "protocol_stream",
                &name,
                &stream_header([mode, native, next_stream, codec], &[1, u16::MAX / 2]),
                stream_payload,
            ));
        }
        for (label, text) in ROUNDTRIP_TEXTS {
            seeds.push(seed(
                "protocol_roundtrip",
                format!("{name}-{label}"),
                &[native, selector(next, 0), codec],
                text,
            ));
        }
        seeds.push(seed(
            "protocol_request",
            format!("{name}-stream-replay-off"),
            &[
                FIXTURE,
                selector(index, SelectorFlags::STREAM.bits()),
                0,
                RequestFlags::DISABLE_REPLAY.bits(),
            ],
            &[],
        ));
        let translated = (SelectorFlags::STREAM | SelectorFlags::DISABLE_RESPONSE_REPLAY).bits();
        seeds.push(seed(
            "protocol_response",
            format!("{name}-translated-stream"),
            &[FIXTURE, upstream, selector(next, translated), codec],
            &[],
        ));
    }
    for (status, message) in ["bad request", "rate limited", "server error"]
        .into_iter()
        .enumerate()
    {
        seeds.push(seed(
            "protocol_response",
            format!("http-error-{status}"),
            &[RAW, status_selector(0, status), 0, 0],
            format!(r#"{{"error":{{"message":"{message}"}}}}"#).as_bytes(),
        ));
    }
    let gemini = selector(3, 0);
    for (prefix, name) in [(2, "double-bom"), (3, "buffered-second-bom")] {
        debug_assert!(prefix < STREAM_PREFIXES.len());
        seeds.push(seed(
            "protocol_stream",
            name,
            &stream_header([GENERATED, gemini, gemini, 0], &[1, 2, 3]),
            &[prefix as u8],
        ));
    }
    seeds.push(seed(
        "protocol_stream",
        "invalid-utf8",
        &stream_header([RAW, gemini, gemini, 0], &[1, 2, 3]),
        &[b"data: \xff\n\n".as_slice(), FIXTURES[3].stream].concat(),
    ));
    for target in [
        "redact_roundtrip",
        "redact_spans",
        "redact_mixed",
        "redact_stateful",
    ] {
        for (name, byte) in [("empty", 0), ("ones", 1), ("max", 255)] {
            seeds.push(seed(target, name, &[byte; 96], &[]));
        }
        let varied: Vec<u8> = (0..=255_u8)
            .map(|index| index.wrapping_mul(73).wrapping_add(19))
            .collect();
        seeds.push(seed(target, "varied", &varied, &[]));
    }
    for detector in 0_u8..12 {
        seeds.push(seed(
            "redact_builtin",
            format!("detector-{detector}"),
            &[detector, detector.wrapping_mul(23)],
            &[0; 96],
        ));
    }
    for kind in 0..6 {
        seeds.push(seed(
            "redact_builtin",
            format!("api-key-{kind}"),
            &[10, kind, 3],
            &[0; 96],
        ));
    }
    for place in 0..20 {
        seeds.push(seed(
            "redact_middleware",
            format!("place-{place}"),
            &[place],
            &[0; 128],
        ));
    }
    seeds
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::protocol_input::{RequestInput, ResponseInput, StreamInput};

    #[test]
    fn every_target_has_unique_nonempty_seeds() {
        let mut names = HashSet::new();
        for seed in catalog() {
            assert!(!seed.bytes.is_empty(), "{}: {}", seed.target, seed.name);
            assert!(names.insert((seed.target, seed.name)), "duplicate seed");
        }
        assert_eq!(
            names
                .iter()
                .map(|(target, _)| *target)
                .collect::<HashSet<_>>()
                .len(),
            10
        );
    }

    #[test]
    fn protocol_seeds_decode_to_their_named_mode_and_wire_data() {
        for seed in catalog() {
            let Some(expected) = CaseKind::ALL
                .into_iter()
                .find(|kind| seed.name.ends_with(&format!("-{}", kind.label())))
            else {
                continue;
            };
            let (kind, body, is_json) = match seed.target {
                "protocol_request" => {
                    let input = RequestInput::from_bytes(&seed.bytes);
                    (input.kind, input.body, true)
                }
                "protocol_response" => {
                    let input = ResponseInput::from_bytes(&seed.bytes);
                    (input.kind, input.body, true)
                }
                "protocol_stream" => {
                    let input = StreamInput::from_bytes(&seed.bytes);
                    (input.kind, input.body, false)
                }
                _ => continue,
            };
            assert_eq!(kind, expected, "{}: {}", seed.target, seed.name);
            if is_json {
                serde_json::from_slice::<serde_json::Value>(&body).unwrap();
            } else {
                assert!(
                    body.windows(6).any(|part| part == b"data: "),
                    "{}",
                    seed.name
                );
            }
        }
    }
}
