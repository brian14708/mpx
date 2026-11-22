use http::{HeaderValue, header};
use mpx_protocol::{
    ClientContext, Codec, Endpoint, Extra, Protocol, Response, ResponseBody, StreamEvent,
};
use serde_json::{Value, json};

use super::{body, collect_body, request, sse_data};

#[tokio::test]
async fn encoded_requests_strip_credentials_queries_and_hop_by_hop_headers() {
    let codec = Codec::default();
    let mut raw = request(
        "/v1beta/models/gemini-2:streamGenerateContent?key=secret",
        &json!({"contents":[{"role":"user","parts":[{"text":"hello"}]}]}),
    );
    raw.headers_mut()
        .insert("authorization", HeaderValue::from_static("Bearer secret"));
    raw.headers_mut()
        .insert("x-api-key", HeaderValue::from_static("secret"));
    raw.headers_mut()
        .insert("x-goog-api-key", HeaderValue::from_static("secret"));
    raw.headers_mut()
        .insert("cookie", HeaderValue::from_static("session=secret"));
    raw.headers_mut()
        .insert("connection", HeaderValue::from_static("keep-alive"));
    let decoded = codec.decode_request(raw).expect("request decodes");
    let encoded = codec
        .encode_request(&decoded, Protocol::GenerateContent)
        .expect("request encodes");
    assert_eq!(encoded.uri().query(), Some("alt=sse"));
    assert!(encoded.headers().get("authorization").is_none());
    assert!(encoded.headers().get("x-api-key").is_none());
    assert!(encoded.headers().get("x-goog-api-key").is_none());
    assert!(encoded.headers().get("cookie").is_none());
    assert!(encoded.headers().get("connection").is_none());
}

#[tokio::test]
async fn response_echo_and_same_protocol_extra_fields_survive() {
    let codec = Codec::default();
    let decoded = codec
        .decode_request(request(
            "/v1/responses",
            &json!({"model":"gpt-5","instructions":"be concise","input":"hello","metadata":{"tenant":"a"}}),
        ))
        .expect("request decodes");
    let client = decoded
        .extensions()
        .get::<ClientContext>()
        .cloned()
        .expect("client context is captured");
    let complete = Response::builder()
        .id("resp")
        .model("m")
        .content(vec![mpx_protocol::ContentBlock::text("hello")])
        .stop_reason(mpx_protocol::StopReason::EndTurn)
        .extra(Extra::with_fields(
            Protocol::Responses,
            std::iter::once(("custom".to_owned(), json!(true))).collect(),
        ))
        .build();
    let response = http::Response::new(ResponseBody::Complete(complete));
    let encoded = codec
        .encode_response(response, &client)
        .await
        .expect("response encodes");
    let value: Value =
        serde_json::from_slice(&collect_body(encoded.into_body()).await).expect("response is JSON");
    assert_eq!(value["instructions"], "be concise");
    assert_eq!(value["metadata"]["tenant"], "a");
    assert_eq!(value["custom"], true);
}

#[tokio::test]
async fn regenerated_sse_discards_original_body_and_connection_headers() {
    let codec = Codec::default();
    let client = ClientContext::new(Endpoint::builder(Protocol::Responses).stream(true).build());
    for upstream_is_stream in [false, true] {
        let wire_body = if upstream_is_stream {
            body(
                "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            )
        } else {
            body(
                r#"{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
            )
        };
        let encoding = if upstream_is_stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        let wire = http::Response::builder()
            .header(header::CONTENT_TYPE, encoding)
            .header(header::CONTENT_LENGTH, 1)
            .header(header::ETAG, "\"upstream\"")
            .header(header::CONNECTION, "x-upstream-only")
            .header("x-upstream-only", "upstream framing")
            .header("x-request-id", "request-123")
            .body(wire_body)
            .expect("test response is valid");
        let decoded = codec
            .decode_response(wire, Protocol::ChatCompletions)
            .await
            .expect("upstream response decodes");
        let encoded = codec
            .encode_response(decoded, &client)
            .await
            .expect("response encodes");
        for name in ["content-length", "etag", "connection", "x-upstream-only"] {
            assert!(
                !encoded.headers().contains_key(name),
                "unexpected header {name}"
            );
        }
        assert_eq!(encoded.headers()[header::CONTENT_TYPE], "text/event-stream");
        assert_eq!(encoded.headers()["x-request-id"], "request-123");
        let bytes = collect_body(encoded.into_body()).await;
        assert!(
            std::str::from_utf8(&bytes)
                .expect("SSE is UTF-8")
                .contains("response.completed")
        );
    }
}

#[tokio::test]
async fn responses_sse_errors_survive_streaming_and_buffering() {
    let codec = Codec::default();
    for value in [
        json!({"type":"error","code":"server_error","message":"oops"}),
        json!({"type":"response.failed","response":{"error":{"code":"server_error","message":"oops"}}}),
    ] {
        let kind = value["type"].as_str().expect("event type is a string");
        let wire = || {
            http::Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(body(format!("event: {kind}\ndata: {value}\n\n")))
                .expect("test response is valid")
        };
        let decoded = codec
            .decode_response(wire(), Protocol::Responses)
            .await
            .expect("upstream response decodes");
        let ResponseBody::Stream(events) = decoded.into_body() else {
            panic!("expected stream");
        };
        let events = events.into_events().await.expect("events decode");
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Error { error }] if error.message == "oops"
        ));

        let decoded = codec
            .decode_response(wire(), Protocol::Responses)
            .await
            .expect("upstream response decodes");
        let client = ClientContext::new(Protocol::ChatCompletions);
        let encoded = codec
            .encode_response(decoded, &client)
            .await
            .expect("response encodes");
        assert_eq!(encoded.status(), http::StatusCode::BAD_GATEWAY);
        let output: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("encoded error is JSON");
        assert_eq!(output["error"]["message"], "oops");

        for protocol in [Protocol::ChatCompletions, Protocol::Messages] {
            let decoded = codec
                .decode_response(wire(), Protocol::Responses)
                .await
                .expect("upstream response decodes");
            let client = ClientContext::new(Endpoint::builder(protocol).stream(true).build());
            let encoded = codec
                .encode_response(decoded, &client)
                .await
                .expect("response encodes");
            let bytes = collect_body(encoded.into_body()).await;
            let events = sse_data(&bytes).await;
            let [data] = events.as_slice() else {
                panic!("expected one error event: {events:?}");
            };
            let error: Value = serde_json::from_str(data).expect("error event data is JSON");
            assert_eq!(error["error"]["message"], "oops");
        }
    }
}
