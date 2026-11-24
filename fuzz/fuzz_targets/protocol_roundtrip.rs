//! Generates supported IR and checks protocol translation, replay, and SSE semantics.
#![no_main]

use bytes::Bytes;
use http_body_util::Full;
use libfuzzer_sys::fuzz_target;
use mpx_fuzz::{block_on, codec, collect, flags, protocol, protocol_input::RoundtripInput};
use mpx_protocol::{
    ClientContext, Codec, ContentBlock, Endpoint, Message, Protocol, Request, Response,
    ResponseAccumulator, ResponseBody, Role, StopReason, Usage,
};
use serde_json::{Value, json};

fn text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn assert_response(actual: &Response, expected: &Response, native: bool) {
    assert_eq!(actual.id, expected.id);
    assert_eq!(actual.model, expected.model);
    assert_eq!(actual.stop_reason, expected.stop_reason);
    assert_eq!(actual.usage.input_tokens, expected.usage.input_tokens);
    assert_eq!(actual.usage.output_tokens, expected.usage.output_tokens);
    let blocks = |response: &Response| -> Vec<(u8, String, Option<Value>)> {
        response
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text, .. } => (0, text.clone(), None),
                ContentBlock::ToolCall {
                    name, arguments, ..
                } => (
                    1,
                    name.clone(),
                    Some(arguments.value().expect("generated tool JSON")),
                ),
                block => panic!("unexpected block in supported response: {block:?}"),
            })
            .collect()
    };
    assert_eq!(blocks(actual), blocks(expected));
    if native {
        let tool_ids = |response: &Response| -> Vec<String> {
            response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolCall { id, .. } => Some(id.as_str().to_owned()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(tool_ids(actual), tool_ids(expected));
    }
}

fuzz_target!(|input: RoundtripInput| {
    let origin = protocol(input.origin);
    let target = protocol(input.target);
    let codec = codec(input.flags);
    let roundtrip_flags = flags::RoundtripFlags::from_bits_retain(input.flags);
    let value = String::from_utf8_lossy(&input.text);
    let mut request = Request::builder()
        .model("fuzz-model")
        .instructions(vec![ContentBlock::text(format!("instruction {value}"))])
        .messages(vec![
            Message::text(Role::User, format!("question {value}")),
            Message::text(Role::Assistant, format!("answer {value}")),
            Message::text(Role::User, format!("followup {value}")),
        ])
        .stream(roundtrip_flags.contains(flags::RoundtripFlags::STREAM_REQUEST))
        .build();
    request.sampling.max_output_tokens = Some(128);
    let encoded = codec
        .encode_request(&http::Request::new(request.clone()), origin)
        .expect("supported IR request encodes");
    let (mut parts, body) = encoded.into_parts();
    parts.extensions.clear();
    let mut wire: Value = serde_json::from_slice(&collect(body).unwrap()).unwrap();
    wire["fuzz_vendor_field"] = json!({"text": value});
    let wire = Bytes::from(wire.to_string());
    let mut decoded = codec
        .decode_request(http::Request::from_parts(parts, wire.clone()))
        .expect("generated wire request decodes");
    let replay = codec
        .encode_request(&decoded, origin)
        .expect("native request replays");
    assert_eq!(collect(replay.into_body()).unwrap(), wire);
    Codec::disable_request_replay(&mut decoded);
    // Unknown native fields are intentionally omitted from this supported-subset test.
    decoded.body_mut().extra = None;
    let encoded = codec
        .encode_request(&decoded, target)
        .expect("supported request translates");
    let (mut parts, body) = encoded.into_parts();
    parts.extensions.clear();
    let decoded = codec
        .decode_request(http::Request::from_parts(parts, collect(body).unwrap()))
        .expect("translated request decodes");
    assert_eq!(decoded.body().model, request.model);
    assert_eq!(decoded.body().stream, request.stream);
    assert_eq!(
        text(&decoded.body().instructions),
        text(&request.instructions)
    );
    assert_eq!(decoded.body().messages.len(), request.messages.len());
    for (actual, expected) in decoded.body().messages.iter().zip(&request.messages) {
        assert_eq!(actual.role, expected.role);
        assert_eq!(text(&actual.content), text(&expected.content));
    }
    assert_eq!(decoded.body().sampling.max_output_tokens, Some(128));

    let mut response = Response::builder()
        .id("fuzz-response")
        .model("fuzz-model")
        .content(vec![ContentBlock::text(format!("output {value}"))])
        .usage(
            Usage::builder()
                .input_tokens(u64::from(input.flags) + 1)
                .output_tokens(u64::from(input.flags) + 2)
                .build(),
        )
        .build();
    if roundtrip_flags.contains(flags::RoundtripFlags::TOOL_RESPONSE) {
        let tool = ContentBlock::tool_call(
            "call-fuzz",
            "lookup",
            json!({"value": value, "nested": [null, true, 42]}).to_string(),
        );
        if !roundtrip_flags.contains(flags::RoundtripFlags::TOOL_BEFORE_TEXT)
            || origin == Protocol::ChatCompletions
            || target == Protocol::ChatCompletions
        {
            response.content.push(tool);
        } else {
            response.content.insert(0, tool);
        }
        response.stop_reason = StopReason::ToolUse;
    }
    let mut accumulator = ResponseAccumulator::new();
    accumulator.extend(response.to_stream_events());
    assert_eq!(accumulator.finish(), response);

    let native = block_on(codec.encode_response(
        http::Response::new(ResponseBody::Complete(response.clone())),
        &ClientContext::new(origin),
    ))
    .expect("supported response encodes");
    let (parts, body) = native.into_parts();
    let bytes = collect(body).unwrap();
    for streaming in [false, true] {
        let mut decoded = block_on(codec.decode_response(
            http::Response::from_parts(parts.clone(), Full::new(bytes.clone())),
            origin,
        ))
        .expect("generated response decodes");
        Codec::disable_response_replay(&mut decoded);
        // Native envelope extensions are outside the shared supported subset.
        if let ResponseBody::Complete(response) = decoded.body_mut() {
            response.extra = None;
            response.usage.extra = None;
            response.visit_blocks_mut(&mut |block| match block {
                ContentBlock::Text { extra, .. } | ContentBlock::ToolCall { extra, .. } => {
                    *extra = None
                }
                _ => {}
            });
        }
        let mut client = ClientContext::new(Endpoint::builder(target).stream(streaming).build());
        client.include_usage = true;
        let encoded = block_on(codec.encode_response(decoded, &client))
            .expect("supported response translates");
        let (parts, body) = encoded.into_parts();
        let decoded = block_on(codec.decode_response(
            http::Response::from_parts(parts, Full::new(collect(body).unwrap())),
            target,
        ))
        .expect("translated response decodes");
        let actual = match decoded.into_body() {
            ResponseBody::Complete(response) => response,
            ResponseBody::Stream(events) => {
                let events = block_on(events.into_events()).expect("generated SSE events");
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, mpx_protocol::StreamEvent::Error { .. }))
                );
                let mut accumulator = ResponseAccumulator::new();
                accumulator.extend(events);
                accumulator.finish()
            }
            body => panic!("generated response failed: {body:?}"),
        };
        assert_response(&actual, &response, origin == target);
    }
});
