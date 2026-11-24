//! Decodes arbitrary client request bodies and re-encodes them for every upstream protocol.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mpx_fuzz::{
    PROTOCOLS, assert_data_error, codec, collect, flags, protocol,
    protocol_input::{CaseKind, RequestInput},
    request_path, streams,
};
use mpx_protocol::Codec;

fuzz_target!(|input: RequestInput| {
    let client = protocol(input.client);
    let stream = streams(input.client);
    let session_flags = flags::RequestFlags::from_bits_retain(input.session);
    let codec = codec(input.flags);
    let mut raw = http::Request::post(request_path(client, stream))
        .header("content-type", "application/json")
        .body(input.body)
        .expect("static request parts are valid");
    if session_flags.contains(flags::RequestFlags::SESSION_ID) {
        raw.headers_mut().insert(
            "x-session-id",
            http::HeaderValue::from_static("fuzz-session"),
        );
    }
    let decoded = codec.decode_request(raw);
    if input.kind != CaseKind::Raw {
        assert!(
            decoded.is_ok(),
            "generated request must decode: {decoded:?}"
        );
    }
    let Ok(mut decoded) = decoded else { return };
    // The normalized IR must always serialize.
    serde_json::to_vec(decoded.body()).expect("decoded request serializes");
    if session_flags.contains(flags::RequestFlags::DISABLE_REPLAY) {
        Codec::disable_request_replay(&mut decoded);
    }
    for upstream in PROTOCOLS {
        let encoded = match codec.encode_request(&decoded, upstream) {
            Ok(encoded) => encoded,
            Err(error) => {
                assert_data_error(&error);
                continue;
            }
        };
        let (mut parts, body) = encoded.into_parts();
        parts.extensions.clear();
        let body = collect(body).expect("encoded request bodies are infallible");
        // Feed the upstream request back through the decoder as a client request.
        codec
            .decode_request(http::Request::from_parts(parts, body))
            .expect("an encoded request must decode in its target protocol");
    }
});
