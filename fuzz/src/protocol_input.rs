//! Seedable, typed inputs for the protocol fuzz targets.
//!
//! Every input starts with a header of selector bytes. The first byte picks a
//! [`CaseKind`], and the payload after the header is used according to that kind.

use arbitrary::{Arbitrary, Unstructured};
use bytes::Bytes;
use mpx_protocol::Protocol;
use serde_json::{Value, json};

use crate::{PROTOCOLS, cuts_from_pairs};

/// How an input turns its payload into a wire body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CaseKind {
    /// The payload is the wire body.
    Raw = 0,
    /// The selected contract fixture is the wire body; the payload is ignored.
    Fixture = 1,
    /// The selected fixture, extended with payload-derived content.
    Generated = 2,
}

impl CaseKind {
    pub const ALL: [Self; 3] = [Self::Raw, Self::Fixture, Self::Generated];

    fn from_byte(byte: u8) -> Self {
        Self::ALL[usize::from(byte) % Self::ALL.len()]
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Fixture => "fixture",
            Self::Generated => "generated",
        }
    }
}

/// Contract fixtures for one protocol.
pub struct Fixture {
    pub protocol: Protocol,
    pub name: &'static str,
    pub request: &'static [u8],
    pub response: &'static [u8],
    pub stream: &'static [u8],
}

macro_rules! fixture {
    ($protocol:ident, $name:literal) => {
        Fixture {
            protocol: Protocol::$protocol,
            name: $name,
            request: include_bytes!(concat!(
                "../../crates/mpx-protocol/tests/contracts/fixtures/",
                $name,
                "-request.json"
            )),
            response: include_bytes!(concat!(
                "../../crates/mpx-protocol/tests/contracts/fixtures/",
                $name,
                "-response.json"
            )),
            // The `.sse` files end with a single newline; the contract tests append
            // the blank line that dispatches the final event, so do the same here.
            stream: concat!(
                include_str!(concat!(
                    "../../crates/mpx-protocol/tests/contracts/fixtures/",
                    $name,
                    ".sse"
                )),
                "\n"
            )
            .as_bytes(),
        }
    };
}

/// Contract fixtures, in [`PROTOCOLS`] order.
pub const FIXTURES: [Fixture; PROTOCOLS.len()] = [
    fixture!(ChatCompletions, "chat"),
    fixture!(Responses, "responses"),
    fixture!(Messages, "messages"),
    fixture!(GenerateContent, "gemini"),
    fixture!(Interactions, "interactions"),
];

/// Picks the fixture for the protocol that [`crate::protocol`] picks.
#[must_use]
pub fn fixture(selector: u8) -> &'static Fixture {
    &FIXTURES[usize::from(selector) % FIXTURES.len()]
}

/// Byte-order mark that stream decoders must skip.
macro_rules! bom {
    () => {
        "\u{feff}"
    };
}

/// Comment that marks a buffered response replayed as SSE.
macro_rules! buffered_marker {
    () => {
        ": mpx-buffered-response-v1 generate_content e30\n"
    };
}

/// Prefixes that Generated stream inputs put before the fixture, picked by the first payload byte.
pub const STREAM_PREFIXES: [&[u8]; 4] = [
    bom!().as_bytes(),
    buffered_marker!().as_bytes(),
    concat!(bom!(), bom!()).as_bytes(),
    concat!(buffered_marker!(), bom!()).as_bytes(),
];

/// Splits a zero-padded `N`-byte header from the payload.
fn header<const N: usize>(data: &[u8]) -> ([u8; N], &[u8]) {
    let mut header = [0; N];
    let len = data.len().min(N);
    header[..len].copy_from_slice(&data[..len]);
    (header, data.get(N..).unwrap_or_default())
}

/// Builds a complete JSON body for a request or response case.
fn json_body(kind: CaseKind, fixture: &'static [u8], payload: &[u8], protocol: Protocol) -> Bytes {
    match kind {
        CaseKind::Raw => Bytes::copy_from_slice(payload),
        CaseKind::Fixture => Bytes::from_static(fixture),
        CaseKind::Generated => {
            let mut value: Value =
                serde_json::from_slice(fixture).expect("contract fixture is JSON");
            let text = String::from_utf8_lossy(payload);
            value["fuzz_extension"] = json!({"text": text, "number": payload.len()});
            // Gemini vendor fields share a namespace with modeled fields.
            if protocol == Protocol::GenerateContent
                && let Some(candidate) = value
                    .get_mut("candidates")
                    .and_then(Value::as_array_mut)
                    .and_then(|candidates| candidates.first_mut())
            {
                candidate["vendorField"] = json!(text);
                candidate["usageMetadata"] = json!({});
            }
            serde_json::to_vec(&value)
                .expect("generated JSON serializes")
                .into()
        }
    }
}

/// Layout: kind, client selector, codec flags, request flags, payload.
#[derive(Debug)]
pub struct RequestInput {
    pub kind: CaseKind,
    pub client: u8,
    pub flags: u8,
    pub session: u8,
    pub body: Bytes,
}

impl RequestInput {
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let ([kind, client, flags, session], payload) = header(data);
        let kind = CaseKind::from_byte(kind);
        let fixture = fixture(client);
        let body = json_body(kind, fixture.request, payload, fixture.protocol);
        Self {
            kind,
            client,
            flags,
            session,
            body,
        }
    }
}

/// Layout: kind, upstream selector, client selector, codec flags, payload.
#[derive(Debug)]
pub struct ResponseInput {
    pub kind: CaseKind,
    pub upstream: u8,
    pub client: u8,
    pub flags: u8,
    pub body: Bytes,
}

impl ResponseInput {
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let ([kind, upstream, client, flags], payload) = header(data);
        let kind = CaseKind::from_byte(kind);
        let fixture = fixture(upstream);
        let body = json_body(kind, fixture.response, payload, fixture.protocol);
        Self {
            kind,
            upstream,
            client,
            flags,
            body,
        }
    }
}

/// Layout: kind, upstream selector, client selector, codec flags, cut-byte count,
/// cut bytes (little-endian `u16` pairs), payload.
#[derive(Debug)]
pub struct StreamInput {
    pub kind: CaseKind,
    pub upstream: u8,
    pub client: u8,
    pub flags: u8,
    pub cuts: Vec<u16>,
    pub body: Bytes,
}

impl StreamInput {
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let ([kind, upstream, client, flags, count], rest) = header(data);
        let kind = CaseKind::from_byte(kind);
        let (cuts, payload) = rest.split_at(usize::from(count % 16).min(rest.len()));
        let fixture = fixture(upstream).stream;
        let body = match kind {
            CaseKind::Raw => Bytes::copy_from_slice(payload),
            CaseKind::Fixture => Bytes::from_static(fixture),
            // A prefix the decoder must skip, then the fixture, then trailing payload bytes.
            CaseKind::Generated => {
                let (prefix, trailer) = header::<1>(payload);
                let prefix = STREAM_PREFIXES[usize::from(prefix[0]) % STREAM_PREFIXES.len()];
                [prefix, fixture, trailer].concat().into()
            }
        };
        Self {
            kind,
            upstream,
            client,
            flags,
            cuts: cuts_from_pairs(cuts),
            body,
        }
    }
}

/// Layout: origin selector, target selector, roundtrip flags, text.
#[derive(Debug)]
pub struct RoundtripInput {
    pub origin: u8,
    pub target: u8,
    pub flags: u8,
    pub text: Vec<u8>,
}

impl RoundtripInput {
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let ([origin, target, flags], text) = header(data);
        Self {
            origin,
            target,
            flags,
            text: text.to_vec(),
        }
    }
}

macro_rules! arbitrary_input {
    ($($input:ty),+ $(,)?) => {
        $(impl<'a> Arbitrary<'a> for $input {
            fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
                Ok(Self::from_bytes(u.bytes(u.len())?))
            }
        })+
    };
}

arbitrary_input!(RequestInput, ResponseInput, StreamInput, RoundtripInput);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_inputs_are_raw_and_keep_their_payload() {
        for bytes in [vec![], vec![0], vec![0, 3, 2, 1, 0xff, 0, 0xef]] {
            let input = RequestInput::from_bytes(&bytes);
            assert_eq!(input.kind, CaseKind::Raw);
            assert_eq!(&input.body[..], bytes.get(4..).unwrap_or_default());
        }
    }

    #[test]
    fn fixtures_follow_protocol_order_and_are_well_formed() {
        for (fixture, protocol) in FIXTURES.iter().zip(PROTOCOLS) {
            assert_eq!(fixture.protocol, protocol);
            serde_json::from_slice::<Value>(fixture.request).unwrap();
            serde_json::from_slice::<Value>(fixture.response).unwrap();
            assert!(fixture.stream.ends_with(b"\n\n"), "{}", fixture.name);
        }
    }

    #[test]
    fn generated_gemini_case_contains_modeled_field_collision() {
        let input = ResponseInput::from_bytes(&[2, 3, 3, 0, b'x']);
        let body: Value = serde_json::from_slice(&input.body).unwrap();
        assert!(body["candidates"][0]["usageMetadata"].is_object());
        assert_eq!(body["candidates"][0]["vendorField"], "x");
    }
}
