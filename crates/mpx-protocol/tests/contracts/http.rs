use http::{HeaderName, HeaderValue, header};
use mpx_protocol::{
    ClientContext, Codec, Endpoint, Extra, Protocol, Response, ResponseBody, StreamEvent,
};
use serde_json::{Value, json};

use super::{body, collect_body, request, sse_data};

#[tokio::test]
async fn session_id_is_decoded_from_the_header() {
    let codec = Codec::default();
    let input = json!({"model":"gpt-5","input":"hello"});
    let raw = || request("/v1/responses", &input);
    let mut with_session = raw();
    with_session
        .headers_mut()
        .insert("X-Session-Id", HeaderValue::from_static("session-123"));

    let decoded = codec.decode_request(with_session).expect("request decodes");
    assert_eq!(decoded.body().session_id.as_deref(), Some("session-123"));
    let normalized = serde_json::to_value(decoded.body()).expect("normalized request serializes");
    assert_eq!(normalized["session_id"], "session-123");
    let encoded = codec
        .encode_request(&decoded, Protocol::Responses)
        .expect("request encodes");
    assert!(!encoded.headers().contains_key("x-session-id"));
    let encoded: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
        .expect("request body is JSON");
    assert_eq!(encoded["metadata"]["session_id"], "session-123");
    assert_eq!(encoded["input"], "hello");

    let decoded = codec.decode_request(raw()).expect("request decodes");
    assert!(decoded.body().session_id.is_none());
    let normalized = serde_json::to_value(decoded.body()).expect("normalized request serializes");
    assert!(normalized.get("session_id").is_none());

    for value in [b"".as_slice(), b"   ".as_slice(), b"\xff".as_slice()] {
        let mut with_unusable_session = raw();
        with_unusable_session.headers_mut().insert(
            "x-session-id",
            HeaderValue::from_bytes(value).expect("valid HTTP header value"),
        );
        let decoded = codec
            .decode_request(with_unusable_session)
            .expect("request decodes");
        assert!(decoded.body().session_id.is_none());
    }
}

#[test]
fn session_id_is_decoded_from_native_headers() {
    let codec = Codec::default();
    for (path, input, names) in [
        (
            "/v1/responses",
            json!({"model":"m","input":"hello"}),
            ["session_id", "session-id", "thread-id"].as_slice(),
        ),
        (
            "/v1/messages",
            json!({"model":"m","max_tokens":8,"messages":[]}),
            ["x-claude-code-session-id", "claude-code-session-id"].as_slice(),
        ),
        (
            "/v1/chat/completions",
            json!({"model":"m","messages":[]}),
            ["x-session-affinity", "x-deepseek-harness-session-id"].as_slice(),
        ),
    ] {
        for &name in names {
            let mut raw = request(path, &input);
            raw.headers_mut()
                .insert("x-session-id", HeaderValue::from_static(" "));
            raw.headers_mut()
                .insert(name, HeaderValue::from_static(" native-id "));
            let decoded = codec.decode_request(raw.clone()).expect("request decodes");
            assert_eq!(
                decoded.body().session_id.as_deref(),
                Some("native-id"),
                "{name}"
            );

            raw.headers_mut()
                .append("x-session-id", HeaderValue::from_static("explicit-id"));
            let decoded = codec.decode_request(raw).expect("request decodes");
            assert_eq!(
                decoded.body().session_id.as_deref(),
                Some("explicit-id"),
                "{name}"
            );
        }
    }
}

#[tokio::test]
async fn session_id_is_decoded_from_native_body_context() {
    let codec = Codec::default();
    let mut responses = request(
        "/v1/responses",
        &json!({"model":"m","input":"hello","client_metadata":{"session_id":"codex"}}),
    );
    responses
        .headers_mut()
        .insert("thread-id", HeaderValue::from_static("thread"));
    let decoded = codec
        .decode_request(responses.clone())
        .expect("request decodes");
    assert_eq!(decoded.body().session_id.as_deref(), Some("codex"));
    let encoded = codec
        .encode_request(&decoded, Protocol::Responses)
        .expect("request encodes");
    assert_eq!(collect_body(encoded.into_body()).await, *responses.body());

    for (metadata, expected) in [
        (
            json!({"session_id":"direct","user_id":"user_device_session_legacy"}),
            Some("direct"),
        ),
        (
            json!({"user_id":r#"{"session_id":"current","device_id":"device"}"#}),
            Some("current"),
        ),
        (
            json!({"user_id":"user_device_account__session_legacy"}),
            Some("legacy"),
        ),
        (
            json!({"user_id":r#"{"session_id":false,"device_id":"user_device_session_bad"}"#}),
            None,
        ),
        (json!({"user_id":"customer-123"}), None),
    ] {
        let raw = request(
            "/v1/messages",
            &json!({"model":"m","max_tokens":8,"messages":[],"metadata":metadata}),
        );
        let decoded = codec.decode_request(raw.clone()).expect("request decodes");
        assert_eq!(decoded.body().session_id.as_deref(), expected);
        let encoded = codec
            .encode_request(&decoded, Protocol::Messages)
            .expect("request encodes");
        assert_eq!(collect_body(encoded.into_body()).await, *raw.body());
    }
}

#[tokio::test]
async fn session_ids_round_trip_through_provider_bodies() {
    let codec = Codec::default();
    for (protocol, path, pointer, input) in [
        (
            Protocol::ChatCompletions,
            "/v1/chat/completions",
            "/metadata",
            json!({"model":"m","messages":[],"metadata":{"session_id":"wire-id","other":"keep"}}),
        ),
        (
            Protocol::Responses,
            "/v1/responses",
            "/metadata",
            json!({"model":"m","input":"hello","metadata":{"session_id":"wire-id","other":"keep"}}),
        ),
        (
            Protocol::GenerateContent,
            "/v1beta/models/gemini-2:generateContent",
            "/labels",
            json!({"labels":{"session_id":"wire-id","other":"keep"}}),
        ),
        (
            Protocol::Interactions,
            "/v1beta/interactions",
            "/webhook_config/user_metadata",
            json!({"model":"m","input":"hello","webhook_config":{"user_metadata":{"session_id":"wire-id","other":"keep"}}}),
        ),
    ] {
        let raw = || request(path, &input);
        let decoded = codec.decode_request(raw()).expect("request decodes");
        assert_eq!(decoded.body().session_id.as_deref(), Some("wire-id"));
        let unchanged = codec
            .encode_request(&decoded, protocol)
            .expect("request encodes");
        assert_eq!(collect_body(unchanged.into_body()).await, *raw().body());

        let mut with_header = raw();
        with_header
            .headers_mut()
            .insert("x-session-id", HeaderValue::from_static("header-id"));
        let mut decoded = codec.decode_request(with_header).expect("request decodes");
        assert_eq!(decoded.body().session_id.as_deref(), Some("header-id"));

        let encoded = codec
            .encode_request(&decoded, protocol)
            .expect("request encodes");
        let encoded: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("request body is JSON");
        let metadata = encoded.pointer(pointer).expect("wire carries metadata");
        assert_eq!(metadata["session_id"], "header-id");
        assert_eq!(metadata["other"], "keep");

        decoded.body_mut().session_id = None;
        let encoded = codec
            .encode_request(&decoded, protocol)
            .expect("request encodes");
        let encoded: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("request body is JSON");
        let metadata = encoded.pointer(pointer).expect("wire carries metadata");
        assert_eq!(metadata, &json!({"other":"keep"}));
    }
}

#[tokio::test]
async fn configured_session_header_uses_the_normalized_id() {
    let name = HeaderName::from_static("x-session-id");
    let codec = Codec::builder().forward_header(name.clone()).build();
    let raw = request(
        "/v1/responses",
        &json!({"model":"m","input":"hello","metadata":{"session_id":"wire-id"}}),
    );
    let mut decoded = codec.decode_request(raw).expect("request decodes");
    let encoded = codec
        .encode_request(&decoded, Protocol::GenerateContent)
        .expect("Gemini request encodes");
    assert_eq!(encoded.headers()[&name], "wire-id");
    let encoded: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
        .expect("request body is JSON");
    assert_eq!(encoded["labels"]["session_id"], "wire-id");

    let encoded = codec
        .encode_request(&decoded, Protocol::Messages)
        .expect("request encodes");
    assert_eq!(encoded.headers()[&name], "wire-id");
    let encoded: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
        .expect("request body is JSON");
    assert!(encoded.get("metadata").is_none());

    decoded.body_mut().session_id = None;
    let encoded = codec
        .encode_request(&decoded, Protocol::Messages)
        .expect("request encodes");
    assert!(!encoded.headers().contains_key(name));
}

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
