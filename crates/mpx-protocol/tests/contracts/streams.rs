use std::io;

use bytes::Bytes;
use futures_util::stream;
use http::header;
use http_body::Frame;
use http_body_util::StreamBody;
use mpx_protocol::{
    ClientContext, Codec, Endpoint, Protocol, ResponseBody, StopReason, StreamEvent,
};
use serde_json::{Value, json};

use super::{body, collect_body, sse_data};

#[tokio::test]
async fn gemini_recognizes_tool_use_from_stop_finish_reason() {
    let codec = Codec::default();
    for (finish_reason, expected) in [
        ("STOP", StopReason::ToolUse),
        ("MAX_TOKENS", StopReason::MaxTokens),
    ] {
        let value = json!({"candidates":[{
            "content":{"role":"model","parts":[{"functionCall":{"name":"weather","args":{}}}]},
            "finishReason":finish_reason
        }]});
        let wire = http::Response::new(body(value.to_string()));
        let decoded = codec
            .decode_response(wire, Protocol::GenerateContent)
            .await
            .expect("complete Gemini response decodes");
        let ResponseBody::Complete(response) = decoded.into_body() else {
            panic!("expected complete response");
        };
        assert_eq!(response.stop_reason, expected);

        let wire = http::Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(body(format!("data: {value}\n\n")))
            .expect("test response is valid");
        let decoded = codec
            .decode_response(wire, Protocol::GenerateContent)
            .await
            .expect("streaming Gemini response decodes");
        let ResponseBody::Stream(events) = decoded.into_body() else {
            panic!("expected stream");
        };
        let events = events.into_events().await.expect("events decode");
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: expected
            })
        );
    }
}

const PROTOCOLS: [Protocol; 5] = [
    Protocol::ChatCompletions,
    Protocol::Responses,
    Protocol::Messages,
    Protocol::GenerateContent,
    Protocol::Interactions,
];

const STREAMS: [&str; 5] = [
    concat!(include_str!("fixtures/chat.sse"), "\n"),
    concat!(include_str!("fixtures/responses.sse"), "\n"),
    concat!(include_str!("fixtures/messages.sse"), "\n"),
    concat!(include_str!("fixtures/gemini.sse"), "\n"),
    concat!(include_str!("fixtures/interactions.sse"), "\n"),
];

fn wire_stream(
    chunks: Vec<Result<Bytes, io::Error>>,
) -> http::Response<impl http_body::Body<Data = Bytes, Error = io::Error>> {
    let frames = stream::iter(chunks.into_iter().map(|chunk| chunk.map(Frame::data)));
    http::Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(StreamBody::new(frames))
        .expect("fixture response is valid")
}

fn client(protocol: Protocol, stream: bool) -> ClientContext {
    let mut client = ClientContext::new(Endpoint::builder(protocol).stream(stream).build());
    client.include_usage = true;
    client
}

fn wire_text_and_usage(protocol: Protocol, wire: &Value) -> (&Value, u64, u64) {
    let (text, input, output) = match protocol {
        Protocol::ChatCompletions => (
            &wire["choices"][0]["message"]["content"],
            &wire["usage"]["prompt_tokens"],
            &wire["usage"]["completion_tokens"],
        ),
        Protocol::Responses => (
            &wire["output"][0]["content"][0]["text"],
            &wire["usage"]["input_tokens"],
            &wire["usage"]["output_tokens"],
        ),
        Protocol::Messages => (
            &wire["content"][0]["text"],
            &wire["usage"]["input_tokens"],
            &wire["usage"]["output_tokens"],
        ),
        Protocol::GenerateContent => (
            &wire["candidates"][0]["content"]["parts"][0]["text"],
            &wire["usageMetadata"]["promptTokenCount"],
            &wire["usageMetadata"]["candidatesTokenCount"],
        ),
        Protocol::Interactions => (
            &wire["steps"][0]["content"][0]["text"],
            &wire["usage"]["total_input_tokens"],
            &wire["usage"]["total_output_tokens"],
        ),
        _ => unreachable!("matrix enumerates the built-in codecs"),
    };
    (
        text,
        input.as_u64().expect("input token count"),
        output.as_u64().expect("output token count"),
    )
}

#[tokio::test]
async fn text_and_usage_survive_each_protocol_pair_for_streaming_and_buffered_clients() {
    let codec = Codec::default();
    for (origin, fixture) in PROTOCOLS.into_iter().zip(STREAMS) {
        for target in PROTOCOLS {
            for streaming in [false, true] {
                let decoded = codec
                    .decode_response(
                        wire_stream(vec![Ok(Bytes::from_static(fixture.as_bytes()))]),
                        origin,
                    )
                    .await
                    .expect("wire response decodes");
                let translated = codec
                    .encode_response(decoded, &client(target, streaming))
                    .await
                    .unwrap_or_else(|error| panic!("{origin} to {target}: {error}"));
                assert_eq!(translated.status(), http::StatusCode::OK);
                let bytes = collect_body(translated.into_body()).await;
                if streaming {
                    assert_successful_stream(target, &sse_data(&bytes).await);
                } else {
                    let wire: Value =
                        serde_json::from_slice(&bytes).expect("buffered result is JSON");
                    assert_eq!(
                        wire_text_and_usage(target, &wire),
                        (&json!("Hi 🦉"), 11, 3),
                        "{origin} to {target}"
                    );
                }
            }
        }
    }
}

fn assert_successful_stream(target: Protocol, messages: &[String]) {
    let mut text = String::new();
    let mut terminal = 0;
    let mut usage = None;
    for message in messages {
        if message == "[DONE]" {
            assert!(matches!(
                target,
                Protocol::ChatCompletions | Protocol::Interactions
            ));
            continue;
        }
        let value: Value = serde_json::from_str(message).expect("stream data is JSON");
        let (delta, ended, tokens) = match target {
            Protocol::ChatCompletions => (
                value.pointer("/choices/0/delta/content"),
                value
                    .pointer("/choices/0/finish_reason")
                    .and_then(Value::as_str)
                    == Some("stop"),
                value["usage"]["prompt_tokens"]
                    .as_u64()
                    .zip(value["usage"]["completion_tokens"].as_u64()),
            ),
            Protocol::Responses => (
                (value["type"] == "response.output_text.delta").then_some(&value["delta"]),
                value["type"] == "response.completed",
                value["response"]["usage"]["input_tokens"]
                    .as_u64()
                    .zip(value["response"]["usage"]["output_tokens"].as_u64()),
            ),
            Protocol::Messages => (
                value.pointer("/delta/text"),
                value["type"] == "message_stop",
                value["usage"]["input_tokens"]
                    .as_u64()
                    .zip(value["usage"]["output_tokens"].as_u64()),
            ),
            Protocol::GenerateContent => (
                value.pointer("/candidates/0/content/parts/0/text"),
                value
                    .pointer("/candidates/0/finishReason")
                    .and_then(Value::as_str)
                    == Some("STOP"),
                value["usageMetadata"]["promptTokenCount"]
                    .as_u64()
                    .zip(value["usageMetadata"]["candidatesTokenCount"].as_u64()),
            ),
            Protocol::Interactions => (
                value.pointer("/delta/text"),
                value["event_type"] == "interaction.completed",
                value["interaction"]["usage"]["total_input_tokens"]
                    .as_u64()
                    .zip(value["interaction"]["usage"]["total_output_tokens"].as_u64()),
            ),
            _ => unreachable!("matrix enumerates the built-in codecs"),
        };
        if let Some(delta) = delta.and_then(Value::as_str) {
            text.push_str(delta);
        }
        if target == Protocol::ChatCompletions && tokens.is_some() {
            assert_eq!(terminal, 1, "Chat usage follows the terminal choice");
            assert_eq!(value["choices"], json!([]));
            assert!(usage.is_none(), "Chat emits one usage chunk");
        }
        terminal += usize::from(ended);
        usage = tokens.or(usage);
    }
    assert_eq!(text, "Hi 🦉", "{target}");
    assert_eq!(terminal, 1, "{target}");
    assert_eq!(usage, Some((11, 3)), "{target}");
    if target == Protocol::ChatCompletions {
        assert_eq!(messages.last().map(String::as_str), Some("[DONE]"));
    }
}

async fn normalized_events(
    codec: &Codec,
    protocol: Protocol,
    chunks: Vec<Result<Bytes, io::Error>>,
) -> Vec<StreamEvent> {
    let decoded = codec
        .decode_response(wire_stream(chunks), protocol)
        .await
        .expect("fixture decodes");
    let ResponseBody::Stream(events) = decoded.into_body() else {
        panic!("expected streaming response");
    };
    events.into_events().await.expect("stream yields events")
}

#[tokio::test]
async fn transport_chunk_boundaries_do_not_change_protocol_events() {
    let codec = Codec::default();
    for (protocol, fixture) in PROTOCOLS.into_iter().zip(STREAMS) {
        let bytes = Bytes::from_static(fixture.as_bytes());
        let expected = normalized_events(&codec, protocol, vec![Ok(bytes.clone())]).await;
        for offset in 1..bytes.len() {
            let chunks = vec![Ok(bytes.slice(..offset)), Ok(bytes.slice(offset..))];
            let fragmented = normalized_events(&codec, protocol, chunks).await;
            assert_eq!(fragmented, expected, "{protocol} split at byte {offset}");
        }
        let single_bytes = (0..bytes.len())
            .map(|offset| Ok(bytes.slice(offset..=offset)))
            .collect();
        let fragmented = normalized_events(&codec, protocol, single_bytes).await;
        assert_eq!(fragmented, expected, "{protocol} one byte at a time");
    }
}

// Only one leading BOM is stripped; a second one prefixes the first field name.
#[tokio::test]
async fn leading_byte_order_marks_decode_the_same_at_every_chunk_boundary() {
    const BOM: &[u8] = b"\xef\xbb\xbf";
    let codec = Codec::default();
    let fixture = STREAMS[3].as_bytes();
    let plain = normalized_events(
        &codec,
        Protocol::GenerateContent,
        vec![Ok(Bytes::from_static(fixture))],
    )
    .await;
    for boms in 1..=2 {
        let bytes = Bytes::from([BOM.repeat(boms), fixture.to_vec()].concat());
        let expected =
            normalized_events(&codec, Protocol::GenerateContent, vec![Ok(bytes.clone())]).await;
        if boms == 1 {
            assert_eq!(expected, plain);
        } else {
            assert_ne!(expected, plain, "a second BOM must not be stripped");
        }
        for first in 0..=BOM.len() * boms {
            for second in first..=BOM.len() * boms + 1 {
                let chunks = vec![
                    Ok(bytes.slice(..first)),
                    Ok(bytes.slice(first..second)),
                    Ok(bytes.slice(second..)),
                ];
                let fragmented = normalized_events(&codec, Protocol::GenerateContent, chunks).await;
                assert_eq!(
                    fragmented, expected,
                    "{boms} BOMs split at {first} and {second}"
                );
            }
        }
    }
}

// A syntactically valid prefix has neither a finish reason nor a terminal event.
#[tokio::test]
async fn early_eof_and_transport_failure_both_fail_in_each_upstream_protocol() {
    let codec = Codec::default();
    for (protocol, fixture) in PROTOCOLS.into_iter().zip(STREAMS) {
        for transport_error in [false, true] {
            let frames: Vec<_> = fixture.trim_end().split("\n\n").collect();
            let terminal_frames = match protocol {
                Protocol::ChatCompletions | Protocol::Messages | Protocol::Interactions => 2,
                Protocol::Responses | Protocol::GenerateContent => 1,
                _ => unreachable!("matrix enumerates the built-in codecs"),
            };
            let prefix = frames[..frames.len() - terminal_frames].join("\n\n") + "\n\n";
            for streaming in [false, true] {
                let mut chunks = vec![Ok(Bytes::from(prefix.clone()))];
                if transport_error {
                    chunks.push(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "upstream connection cut",
                    )));
                }
                let decoded = codec
                    .decode_response(wire_stream(chunks), protocol)
                    .await
                    .expect("prefix decodes");
                let translated = codec
                    .encode_response(decoded, &client(Protocol::Responses, streaming))
                    .await
                    .expect("translation delivers a protocol error");
                let status = translated.status();
                let bytes = collect_body(translated.into_body()).await;
                if streaming {
                    assert_eq!(status, http::StatusCode::OK);
                    let events: Vec<Value> = sse_data(&bytes)
                        .await
                        .into_iter()
                        .map(|data| serde_json::from_str(&data).expect("stream data is JSON"))
                        .collect();
                    assert!(
                        events
                            .iter()
                            .any(|event| event["type"] == "response.output_text.delta"),
                        "{protocol}"
                    );
                    assert_eq!(
                        events.last().expect("terminal event")["type"],
                        "response.failed",
                        "{protocol}"
                    );
                    assert!(
                        !events
                            .iter()
                            .any(|event| event["type"] == "response.completed"),
                        "{protocol}"
                    );
                } else {
                    assert_eq!(status, http::StatusCode::BAD_GATEWAY, "{protocol}");
                    let error: Value =
                        serde_json::from_slice(&bytes).expect("buffered error is JSON");
                    assert!(error["error"]["message"].is_string(), "{protocol}");
                }
            }
        }
    }
}

#[tokio::test]
async fn interactions_reasoning_and_function_argument_deltas_survive_buffering() {
    let codec = Codec::default();
    let fixture = concat!(
        r#"data: {"event_type":"interaction.created","interaction":{"id":"int_1","status":"in_progress"}}"#,
        "\n\n",
        r#"data: {"event_type":"step.start","index":2,"step":{"type":"thought"}}"#,
        "\n\n",
        r#"data: {"event_type":"step.delta","index":2,"delta":{"type":"thought_signature","signature":"opaque"}}"#,
        "\n\n",
        r#"data: {"event_type":"step.stop","index":2}"#,
        "\n\n",
        r#"data: {"event_type":"step.start","index":5,"step":{"type":"function_call","id":"fc_1","name":"weather","arguments":{}}}"#,
        "\n\n",
        r#"data: {"event_type":"step.delta","index":5,"delta":{"type":"arguments_delta","arguments":"{\"city\":"}}"#,
        "\n\n",
        r#"data: {"event_type":"step.delta","index":5,"delta":{"type":"arguments_delta","arguments":"\"Boston\"}"}}"#,
        "\n\n",
        r#"data: {"event_type":"step.stop","index":5}"#,
        "\n\n",
        r#"data: {"event_type":"interaction.completed","interaction":{"id":"int_1","status":"requires_action","usage":{"total_input_tokens":11,"total_cached_tokens":5}}}"#,
        "\n\n",
    );
    let decoded = codec
        .decode_response(
            wire_stream(vec![Ok(Bytes::from_static(fixture.as_bytes()))]),
            Protocol::Interactions,
        )
        .await
        .expect("stream decodes");
    let buffered = codec
        .encode_response(decoded, &client(Protocol::Interactions, false))
        .await
        .expect("stream buffers");
    let bytes = collect_body(buffered.into_body()).await;
    let actual: Value = serde_json::from_slice(&bytes).expect("body is JSON");
    assert_eq!(actual["status"], "requires_action");
    assert_eq!(actual["steps"][0]["signature"], "opaque");
    assert_eq!(actual["steps"][1]["id"], "fc_1");
    assert_eq!(actual["steps"][1]["arguments"], json!({"city": "Boston"}));
    assert_eq!(actual["usage"]["total_input_tokens"], 11);
    assert_eq!(actual["usage"]["total_cached_tokens"], 5);
}
#[tokio::test]
async fn sse_decoding_tolerates_a_bom_after_the_stream_start() {
    let codec = Codec::default();
    let event = json!({"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]});
    let data = format!("data: {event}\n\n");
    let bom = "\u{feff}";
    let comment = ": mpx-buffered-response-v1 generate_content e30\n";
    let cases: [(Vec<Vec<u8>>, bool); 6] = [
        (vec![format!("{bom}{data}").into()], true),
        (
            vec![
                b"\xef".to_vec(),
                [b"\xbb\xbf".as_slice(), data.as_bytes()].concat(),
            ],
            true,
        ),
        (vec![format!("{bom}{bom}{data}").into()], false),
        // A lone leading BOM chunk must not let the next chunk strip a second one.
        (vec![bom.into(), format!("{bom}{data}").into()], false),
        (vec![format!("{comment}{bom}{data}").into()], false),
        (
            vec![
                [comment.as_bytes(), b"\xef"].concat(),
                [b"\xbb\xbf".as_slice(), data.as_bytes()].concat(),
            ],
            false,
        ),
    ];
    for (chunks, decodes_text) in cases {
        let chunks = chunks
            .into_iter()
            .map(|chunk| Ok(Bytes::from(chunk)))
            .collect();
        let events = normalized_events(&codec, Protocol::GenerateContent, chunks).await;
        let has_text = events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: mpx_protocol::BlockDelta::Text(text),
                    ..
                } if text == "hi"
            )
        });
        assert_eq!(has_text, decodes_text, "{events:?}");
    }
}

#[tokio::test]
async fn gemini_candidate_extensions_cannot_shadow_modeled_response_fields() {
    let codec = Codec::default();
    let value = json!({"candidates":[{
        "content":{"role":"model","parts":[{"text":"hi"}]},
        "finishReason":"STOP",
        "candidates":[],
        "usageMetadata":{},
        "vendorField":1
    }]});
    for stream in [false, true] {
        let wire = http::Response::new(body(value.to_string()));
        let mut decoded = codec
            .decode_response(wire, Protocol::GenerateContent)
            .await
            .expect("complete Gemini response decodes");
        Codec::disable_response_replay(&mut decoded);
        let encoded = codec
            .encode_response(decoded, &client(Protocol::GenerateContent, stream))
            .await
            .expect("response encodes");
        let bytes = collect_body(encoded.into_body()).await;
        let chunks = if stream {
            sse_data(&bytes).await
        } else {
            vec![String::from_utf8(bytes.to_vec()).expect("UTF-8 JSON")]
        };
        for chunk in chunks {
            // Decoding rejects duplicate modeled keys.
            let wire = http::Response::new(body(chunk));
            codec
                .decode_response(wire, Protocol::GenerateContent)
                .await
                .expect("encoded chunk has no duplicate keys");
        }
    }
}

// Nested vendor fields are carried in response extras; synthesized streams
// must put them back in place instead of emitting internal bookkeeping keys.
#[tokio::test]
async fn synthesized_streams_never_emit_internal_extra_keys() {
    let codec = Codec::default();
    let cases = [
        (
            Protocol::ChatCompletions,
            json!({"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{
                "index":0,"message":{"role":"assistant","content":"hi","vendor_msg":1},
                "finish_reason":"stop","vendor_choice":2
            }]}),
        ),
        (
            Protocol::GenerateContent,
            json!({"candidates":[{
                "content":{"role":"model","parts":[{"text":"hi"}]},
                "finishReason":"STOP","vendor_candidate":3
            }],"promptFeedback":{"vendor_feedback":4}}),
        ),
    ];
    for (protocol, value) in cases {
        let wire = http::Response::new(body(value.to_string()));
        let mut decoded = codec
            .decode_response(wire, protocol)
            .await
            .expect("complete response decodes");
        Codec::disable_response_replay(&mut decoded);
        let encoded = codec
            .encode_response(decoded, &client(protocol, true))
            .await
            .expect("response encodes as a stream");
        let messages = sse_data(&collect_body(encoded.into_body()).await).await;
        assert!(
            messages.iter().all(|message| !message.contains("__mpx")),
            "{protocol}: {messages:?}"
        );
        if protocol == Protocol::ChatCompletions {
            let first: Value = serde_json::from_str(&messages[0]).expect("JSON chunk");
            assert_eq!(first["choices"][0]["vendor_choice"], 2);
            assert_eq!(first["choices"][0]["delta"]["vendor_msg"], 1);
        }
    }
}
