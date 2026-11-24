//! Checks SSE partition invariance, truncated streams, and transport failures.
#![no_main]

use std::io;

use bytes::Bytes;
use futures_util::stream;
use http_body::Frame;
use http_body_util::StreamBody;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::{
    assert_data_error, block_on, client_context, codec, collect, protocol,
    protocol_input::{CaseKind, StreamInput},
    split_offsets,
};
use mpx_protocol::{Codec, ResponseBody, StreamEvent};

fn decode(
    codec: &Codec,
    upstream: mpx_protocol::Protocol,
    chunks: Vec<Result<Bytes, io::Error>>,
) -> Vec<StreamEvent> {
    let frames = chunks.into_iter().map(|chunk| chunk.map(Frame::data));
    let raw = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(StreamBody::new(stream::iter(frames)))
        .unwrap();
    let decoded = block_on(codec.decode_response(raw, upstream)).expect("SSE envelope decodes");
    let ResponseBody::Stream(events) = decoded.into_body() else {
        panic!("SSE response is a stream")
    };
    block_on(events.into_events()).expect("SSE failures become normalized error events")
}

fn first_event_end(body: &[u8]) -> Option<usize> {
    body.windows(2)
        .position(|pair| pair == b"\n\n")
        .map(|index| index + 2)
        .or_else(|| {
            body.windows(4)
                .position(|pair| pair == b"\r\n\r\n")
                .map(|index| index + 4)
        })
}

fuzz_target!(|input: StreamInput| {
    let bytes = input.body;
    let body = &bytes[..];
    let mut offsets: Vec<_> = input
        .cuts
        .iter()
        .map(|cut| usize::from(*cut) * body.len() / usize::from(u16::MAX))
        .collect();
    offsets.extend([
        0,
        1,
        2,
        3,
        body.len() / 2,
        body.len().saturating_sub(1),
        body.len(),
    ]);
    let codec = codec(input.flags);
    let upstream = protocol(input.upstream);
    let expected = decode(&codec, upstream, vec![Ok(bytes.clone())]);
    let chunks = split_offsets(body, &offsets)
        .into_iter()
        .map(|chunk| Ok(bytes.slice_ref(chunk)))
        .collect();
    let partitioned = decode(&codec, upstream, chunks);
    assert_eq!(partitioned, expected, "SSE chunk boundaries changed events");
    if input.kind == CaseKind::Fixture {
        assert!(
            expected
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageStop { .. }))
        );
    }

    let cut = first_event_end(body).filter(|end| *end < body.len());
    for at in [offsets.first().copied(), cut].into_iter().flatten() {
        let prefix = bytes.slice(..at.min(body.len()));
        let truncated = decode(&codec, upstream, vec![Ok(prefix.clone())]);
        let failed = decode(
            &codec,
            upstream,
            vec![Ok(prefix), Err(io::Error::other("fuzz transport failure"))],
        );
        let completed = truncated
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageStop { .. }));
        let started = truncated
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageStart { .. }));
        if started && !completed {
            assert!(
                truncated
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Error { .. })),
                "early EOF must fail"
            );
            assert!(
                failed
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Error { .. })),
                "transport failure must fail"
            );
            assert!(
                !failed
                    .iter()
                    .any(|event| matches!(event, StreamEvent::MessageStop { .. })),
                "failed stream completed"
            );
        }
    }

    let decoded = http::Response::new(ResponseBody::Stream(
        mpx_protocol::EventStream::from_events(expected),
    ));
    let client = client_context(input.client);
    // Malformed wire events can have an invalid lifecycle or tool JSON, and
    // Reject policy can fail lazily while collecting. Inspect those failures.
    match block_on(codec.encode_response(decoded, &client)) {
        Ok(encoded) => {
            if let Err(error) = collect(encoded.into_body()) {
                assert_data_error(&error);
            }
        }
        Err(error) => assert_data_error(&error),
    }
});
