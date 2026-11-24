//! Shared helpers for the mpx fuzz targets.

use std::{future::Future, ops::Range};

use bytes::Bytes;
use http_body_util::BodyExt;
use mpx_protocol::{Body, ClientContext, Codec, Endpoint, LossPolicy, Protocol};

pub mod protocol_input;
pub mod redact;
pub mod seeds;

/// Every built-in protocol, indexed by fuzz selector bytes.
pub const PROTOCOLS: [Protocol; 5] = [
    Protocol::ChatCompletions,
    Protocol::Responses,
    Protocol::Messages,
    Protocol::GenerateContent,
    Protocol::Interactions,
];

/// Bit meanings shared by the protocol fuzz inputs and corpus generator.
pub mod flags {
    use bitflags::bitflags;

    bitflags! {
        #[derive(Clone, Copy)]
        pub struct CodecFlags: u8 {
            const REJECT_LOSS = 1;
            const DISABLE_HISTORY_REPAIR = 2;
            const DISABLE_REASONING_SEALS = 4;
            const DISABLE_CONTENT_FALLBACKS = 8;
            const SMALL_SEALED_IDS = 16;
        }

        #[derive(Clone, Copy)]
        pub struct RequestFlags: u8 {
            const SESSION_ID = 1;
            const DISABLE_REPLAY = 2;
        }

        #[derive(Clone, Copy)]
        pub struct SelectorFlags: u8 {
            const DISABLE_RESPONSE_REPLAY = 0x40;
            const STREAM = 0x80;
        }

        #[derive(Clone, Copy)]
        pub struct RoundtripFlags: u8 {
            const STREAM_REQUEST = 32;
            const TOOL_RESPONSE = 64;
            const TOOL_BEFORE_TEXT = 128;
        }
    }
}

/// Picks a protocol from a selector byte.
#[must_use]
pub fn protocol(selector: u8) -> Protocol {
    PROTOCOLS[usize::from(selector) % PROTOCOLS.len()]
}

/// Whether a selector byte requests streaming.
#[must_use]
pub const fn streams(selector: u8) -> bool {
    flags::SelectorFlags::from_bits_retain(selector).contains(flags::SelectorFlags::STREAM)
}

/// Builds the client context a selector byte describes.
#[must_use]
pub fn client_context(selector: u8) -> ClientContext {
    ClientContext::new(
        Endpoint::builder(protocol(selector))
            .stream(streams(selector))
            .build(),
    )
}

/// Builds a selector byte that picks `PROTOCOLS[index]` and keeps the `high` bits.
///
/// Selector bytes carry flag bits above the protocol, so `high` is rounded up
/// to a multiple of the protocol count before the index is added.
#[must_use]
pub const fn selector(index: usize, high: u8) -> u8 {
    assert!(index < PROTOCOLS.len() && high & 0x1f == 0);
    high.next_multiple_of(PROTOCOLS.len() as u8) + index as u8
}

/// Upstream statuses, indexed by the top three bits of the upstream selector.
pub const STATUSES: [http::StatusCode; 4] = [
    http::StatusCode::BAD_REQUEST,
    http::StatusCode::TOO_MANY_REQUESTS,
    http::StatusCode::INTERNAL_SERVER_ERROR,
    http::StatusCode::OK,
];

/// Picks the upstream status from the top three bits of a selector; high values are `200 OK`.
#[must_use]
pub fn status(selector: u8) -> http::StatusCode {
    STATUSES[usize::from(selector >> 5).min(STATUSES.len() - 1)]
}

/// Builds an upstream selector that picks `PROTOCOLS[index]` and `STATUSES[status]`.
#[must_use]
pub const fn status_selector(index: usize, status: usize) -> u8 {
    selector(index, (status as u8) << 5)
}

/// Returns a client-facing path that the built-in router maps to `protocol`.
#[must_use]
pub fn request_path(protocol: Protocol, stream: bool) -> &'static str {
    match protocol {
        Protocol::ChatCompletions => "/v1/chat/completions",
        Protocol::Responses => "/v1/responses",
        Protocol::Messages => "/v1/messages",
        Protocol::GenerateContent if stream => "/v1beta/models/gemini-fuzz:streamGenerateContent",
        Protocol::GenerateContent => "/v1beta/models/gemini-fuzz:generateContent",
        _ => "/v1beta/interactions",
    }
}

/// Builds a codec whose options are selected by the bits of `flags`.
#[must_use]
pub fn codec(flags: u8) -> Codec {
    let flags = flags::CodecFlags::from_bits_retain(flags);
    Codec::builder()
        .loss_policy(if flags.contains(flags::CodecFlags::REJECT_LOSS) {
            LossPolicy::Reject
        } else {
            LossPolicy::Warn
        })
        .repair_history(!flags.contains(flags::CodecFlags::DISABLE_HISTORY_REPAIR))
        .seal_reasoning_in_tool_ids(!flags.contains(flags::CodecFlags::DISABLE_REASONING_SEALS))
        .content_fallbacks(!flags.contains(flags::CodecFlags::DISABLE_CONTENT_FALLBACKS))
        .max_sealed_id_bytes(if flags.contains(flags::CodecFlags::SMALL_SEALED_IDS) {
            64
        } else {
            2048
        })
        .forward_header(http::HeaderName::from_static("x-session-id"))
        .build()
}

/// Drives a future whose I/O is entirely in memory.
pub fn block_on<F: Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// Collects an encoded body, preserving failures for the target's oracle.
pub fn collect(body: Body) -> mpx_protocol::Result<Bytes> {
    block_on(body.collect()).map(http_body_util::Collected::to_bytes)
}

/// Accepts explicit input/translation rejections while rejecting internal or I/O failures.
pub fn assert_data_error(error: &mpx_protocol::Error) {
    assert!(
        matches!(
            error,
            mpx_protocol::Error::Protocol { .. }
                | mpx_protocol::Error::InvalidRequest { .. }
                | mpx_protocol::Error::LossPolicy { .. }
        ),
        "unexpected translation failure: {error}"
    );
}

/// Splits bytes at offsets spanning the entire input, including empty chunks.
#[must_use]
pub fn split_offsets<'a>(data: &'a [u8], cuts: &[usize]) -> Vec<&'a [u8]> {
    let points = cuts.iter().map(|cut| *cut % (data.len() + 1)).collect();
    partition(data.len(), points)
        .map(|range| &data[range])
        .collect()
}

/// Splits UTF-8 text at the supplied cut points, rounded down to char boundaries.
#[must_use]
pub fn split_str<'a>(text: &'a str, cuts: &[u16]) -> Vec<&'a str> {
    let points = cuts
        .iter()
        .map(|cut| text.floor_char_boundary(usize::from(*cut) % (text.len() + 1)))
        .collect();
    partition(text.len(), points)
        .map(|range| &text[range])
        .collect()
}

/// Ranges between sorted, deduplicated `points` covering `0..len`.
fn partition(len: usize, mut points: Vec<usize>) -> impl Iterator<Item = Range<usize>> {
    points.sort_unstable();
    points.dedup();
    points.insert(0, 0);
    points.push(len);
    (1..points.len()).map(move |end| points[end - 1]..points[end])
}

/// Decodes little-endian `u16` cut points from byte pairs.
#[must_use]
pub fn cuts_from_pairs(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks(2)
        .map(|pair| u16::from_le_bytes([pair[0], *pair.get(1).unwrap_or(&0)]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_keeps_protocol_and_high_bits() {
        for (index, expected) in PROTOCOLS.into_iter().enumerate() {
            for high in (0..8).map(|bits| bits << 5) {
                let selector = selector(index, high);
                assert_eq!(protocol(selector), expected);
                assert_eq!(selector & 0xe0, high);
            }
            for (status_index, expected_status) in STATUSES.into_iter().enumerate() {
                assert_eq!(
                    status(status_selector(index, status_index)),
                    expected_status
                );
            }
        }
    }
}
