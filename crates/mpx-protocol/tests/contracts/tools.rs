use std::{collections::BTreeMap, convert::Infallible, fmt::Write};

use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::{FutureExt, StreamExt, stream};
use http::header;
use http_body::Frame;
use http_body_util::{BodyDataStream, StreamBody};
use mpx_protocol::{ClientContext, Codec, Endpoint, Protocol};
use serde_json::{Value, json};

use super::{body, collect_body, sse_data};

fn interleaved_tools() -> String {
    let mut stream = String::new();
    for (index, id, name, fragment) in [
        (0, Some("call_read"), Some("read_file"), "{\"pa"),
        (1, Some("call_list"), Some("list_dir"), "{\"di"),
        (0, Some("call_read"), Some("read_file"), "th\": \"a/b"),
        (1, None, None, "r\":\"src"),
        (0, None, None, ".rs\"}"),
        (1, None, None, "/x\"}"),
    ] {
        let mut call = json!({"index":index,"function":{"arguments":fragment}});
        if let Some(id) = id {
            call["id"] = json!(id);
            call["type"] = json!("function");
            call["function"]["name"] = json!(name);
        }
        let chunk = json!({"id":"chat_1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[call]}}]});
        write!(stream, "data: {chunk}\n\n").expect("String writes succeed");
    }
    stream.push_str("data: {\"id\":\"chat_1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n");
    stream
}

// Magpie's interleaved tool cases check identity and exact deltas before and after buffering.
#[tokio::test]
async fn interleaved_calls_keep_identity_and_exact_arguments() {
    let expected = [
        ("call_read", "read_file", "{\"path\": \"a/b.rs\"}"),
        ("call_list", "list_dir", "{\"dir\":\"src/x\"}"),
    ];
    let codec = Codec::default();
    for (protocol, streaming) in [
        (Protocol::Responses, true),
        (Protocol::Messages, true),
        (Protocol::ChatCompletions, false),
    ] {
        let upstream = http::Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(body(interleaved_tools()))
            .expect("test response is valid");
        let upstream = codec
            .decode_response(upstream, Protocol::ChatCompletions)
            .await
            .expect("upstream decodes");
        let client = ClientContext::new(Endpoint::builder(protocol).stream(streaming).build());
        let response = codec
            .encode_response(upstream, &client)
            .await
            .expect("response translates");
        let bytes = collect_body(response.into_body()).await;
        if !streaming {
            let value: Value = serde_json::from_slice(&bytes).expect("response is JSON");
            let choice = &value["choices"][0];
            assert_eq!(choice["finish_reason"], "tool_calls");
            let calls = choice["message"]["tool_calls"]
                .as_array()
                .expect("calls are an array");
            assert_eq!(calls.len(), expected.len());
            for (call, (id, name, args)) in calls.iter().zip(expected) {
                assert_eq!(call["id"], id);
                assert_eq!(call["function"]["name"], name);
                assert_eq!(call["function"]["arguments"], args);
            }
            continue;
        }

        assert_interleaved_tool_events(protocol, &sse_data(&bytes).await, &expected);
    }
}

fn assert_interleaved_tool_events(
    protocol: Protocol,
    events: &[String],
    expected: &[(&str, &str, &str)],
) {
    let mut calls = BTreeMap::<u64, (Value, Value, Value, String, bool)>::new();
    let mut active_anthropic = None;
    let mut anthropic_stop_reason = None;
    for data in events {
        let event: Value = serde_json::from_str(data).expect("event data is JSON");
        match event["type"].as_str() {
            Some("response.output_item.added" | "content_block_start") => {
                let responses = protocol == Protocol::Responses;
                let item = if responses {
                    &event["item"]
                } else {
                    &event["content_block"]
                };
                let index = event[if responses { "output_index" } else { "index" }]
                    .as_u64()
                    .expect("call index");
                if !responses {
                    assert_eq!(item["input"], json!({}));
                    assert!(
                        active_anthropic.replace(index).is_none(),
                        "Messages blocks must be serialized"
                    );
                }
                let call_id = &item[if responses { "call_id" } else { "id" }];
                assert!(
                    calls
                        .insert(
                            index,
                            (
                                item["id"].clone(),
                                call_id.clone(),
                                item["name"].clone(),
                                String::new(),
                                false
                            )
                        )
                        .is_none()
                );
            }
            Some(
                "response.function_call_arguments.delta"
                | "input_json_delta"
                | "content_block_delta",
            ) => {
                let responses = protocol == Protocol::Responses;
                let index = event[if responses { "output_index" } else { "index" }]
                    .as_u64()
                    .expect("delta index");
                let call = calls.get_mut(&index).expect("delta follows a start");
                if responses {
                    assert_eq!(event["item_id"], call.0);
                } else {
                    assert_eq!(Some(index), active_anthropic);
                }
                assert!(!call.4, "delta arrived after the call was closed");
                let delta = if responses {
                    &event["delta"]
                } else {
                    &event["delta"]["partial_json"]
                };
                call.3.push_str(delta.as_str().expect("argument fragment"));
            }
            Some("response.output_item.done" | "content_block_stop") => {
                let responses = protocol == Protocol::Responses;
                let index = event[if responses { "output_index" } else { "index" }]
                    .as_u64()
                    .expect("finished index");
                let call = calls.get_mut(&index).expect("stop follows a start");
                if responses {
                    assert_eq!(event["item"]["id"], call.0);
                    assert_eq!(event["item"]["arguments"], call.3);
                } else {
                    assert_eq!(active_anthropic.take(), Some(index));
                }
                assert!(!std::mem::replace(&mut call.4, true), "call stopped twice");
            }
            Some("message_delta") => {
                anthropic_stop_reason = event["delta"]["stop_reason"].as_str().map(str::to_owned);
            }
            _ => {}
        }
    }
    if protocol == Protocol::Messages {
        assert_eq!(anthropic_stop_reason.as_deref(), Some("tool_use"));
    }
    assert_eq!(calls.len(), expected.len());
    for ((_, call), (id, name, args)) in calls.iter().zip(expected.iter().copied()) {
        assert_eq!(call.1, id);
        assert_eq!(call.2, name);
        assert_eq!(call.3, args);
        assert!(call.4);
    }
}

#[tokio::test]
async fn first_argument_delta_is_visible_before_more_upstream_data_arrives() {
    let call = json!({"id":"chat_1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[
        {"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}
    ]}}]});
    let upstream = stream::iter([Ok::<_, Infallible>(Frame::data(Bytes::from(format!(
        "data: {call}\n\n"
    ))))])
    .chain(stream::pending());
    let upstream = http::Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(StreamBody::new(upstream))
        .expect("upstream is valid");
    let codec = Codec::default();
    let decoded = codec
        .decode_response(upstream, Protocol::ChatCompletions)
        .await
        .expect("upstream decodes");
    let client = ClientContext::new(Endpoint::builder(Protocol::Responses).stream(true).build());
    let downstream = codec
        .encode_response(decoded, &client)
        .await
        .expect("response encodes");
    let events = BodyDataStream::new(downstream.into_body()).eventsource();
    let mut events = std::pin::pin!(events);
    while let Some(Some(event)) = events.next().now_or_never() {
        let event = event.expect("SSE parses");
        if event.event == "response.function_call_arguments.delta" {
            let data: Value = serde_json::from_str(&event.data).expect("argument event is JSON");
            assert_eq!(data["delta"], "{\"city\":");
            return;
        }
    }
    panic!("argument delta was buffered while the upstream was pending");
}
