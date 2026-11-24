//! Decodes arbitrary complete upstream responses and re-encodes them for a client protocol.
#![no_main]

use http_body_util::Full;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::{
    assert_data_error, block_on, client_context, codec, collect, flags, protocol,
    protocol_input::{CaseKind, ResponseInput},
    status,
};
use mpx_protocol::{ClientContext, Codec, ResponseBody};

fuzz_target!(|input: ResponseInput| {
    let upstream_protocol = protocol(input.upstream);
    let client_protocol = protocol(input.client);
    let client_flags = flags::SelectorFlags::from_bits_retain(input.client);
    let codec = codec(input.flags);
    let status = status(input.upstream);
    let raw = || {
        http::Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Full::new(input.body.clone()))
            .expect("static response parts are valid")
    };
    let decoded = block_on(codec.decode_response(raw(), upstream_protocol));
    if input.kind != CaseKind::Raw && status.is_success() {
        assert!(
            decoded.is_ok(),
            "generated response must decode: {decoded:?}"
        );
    }
    let Ok(decoded) = decoded else { return };
    // Unedited native responses, including non-JSON HTTP errors, replay exactly.
    let native = block_on(codec.encode_response(decoded, &ClientContext::new(upstream_protocol)))
        .expect("a decoded native response replays");
    assert_eq!(native.status(), status);
    assert_eq!(
        collect(native.into_body()).expect("native replay body"),
        input.body
    );
    let mut decoded = block_on(codec.decode_response(raw(), upstream_protocol)).unwrap();
    if let ResponseBody::Complete(response) = decoded.body() {
        serde_json::to_vec(response).expect("decoded response serializes");
    }
    if client_flags.contains(flags::SelectorFlags::DISABLE_RESPONSE_REPLAY) {
        Codec::disable_response_replay(&mut decoded);
    }
    let client = client_context(input.client);
    match block_on(codec.encode_response(decoded, &client)) {
        Ok(encoded) => {
            let (parts, body) = encoded.into_parts();
            let body = collect(body).expect("complete response encoding must yield a valid body");
            // Internal bookkeeping keys in extras must never reach the wire.
            if !contains(&input.body, b"__mpx") {
                assert!(!contains(&body, b"__mpx"), "internal extra key leaked");
            }
            let decoded = block_on(codec.decode_response(
                http::Response::from_parts(parts, Full::new(body)),
                client_protocol,
            ))
            .expect("encoded response decodes in the client protocol");
            if let ResponseBody::Stream(events) = decoded.into_body() {
                block_on(events.into_events()).expect("encoded stream is readable");
            }
        }
        Err(error) => assert_data_error(&error),
    }
});

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
