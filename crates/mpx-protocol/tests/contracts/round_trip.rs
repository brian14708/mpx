use std::convert::Infallible;

use bytes::Bytes;
use futures_util::stream;
use http::{Response as HttpResponse, header};
use http_body::Frame;
use http_body_util::StreamBody;
use mpx_protocol::{ClientContext, Codec, Protocol, Response, ResponseAccumulator};
use serde_json::{Value, json};

use super::{body, collect_body, request, sse_data};

struct WireFixture {
    protocol: Protocol,
    path: &'static str,
    request: &'static [u8],
    response: &'static [u8],
}

const WIRE_FIXTURES: [WireFixture; 5] = [
    WireFixture {
        protocol: Protocol::ChatCompletions,
        path: "/v1/chat/completions",
        request: include_bytes!("fixtures/chat-request.json"),
        response: include_bytes!("fixtures/chat-response.json"),
    },
    WireFixture {
        protocol: Protocol::Responses,
        path: "/v1/responses",
        request: include_bytes!("fixtures/responses-request.json"),
        response: include_bytes!("fixtures/responses-response.json"),
    },
    WireFixture {
        protocol: Protocol::Messages,
        path: "/v1/messages",
        request: include_bytes!("fixtures/messages-request.json"),
        response: include_bytes!("fixtures/messages-response.json"),
    },
    WireFixture {
        protocol: Protocol::GenerateContent,
        path: "/v1beta/models/m:generateContent",
        request: include_bytes!("fixtures/gemini-request.json"),
        response: include_bytes!("fixtures/gemini-response.json"),
    },
    WireFixture {
        protocol: Protocol::Interactions,
        path: "/v1/interactions",
        request: include_bytes!("fixtures/interactions-request.json"),
        response: include_bytes!("fixtures/interactions-response.json"),
    },
];

#[tokio::test]
async fn native_request_preserves_every_field() {
    let codec = Codec::default();
    for fixture in WIRE_FIXTURES {
        let protocol = fixture.protocol;
        let expected = Bytes::from_static(fixture.request);
        let wire = http::Request::post(fixture.path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(expected.clone())
            .expect("fixture request is valid");
        let decoded = codec
            .decode_request(wire)
            .unwrap_or_else(|error| panic!("{protocol} request fails to decode: {error}"));
        let replayed = codec
            .encode_request(&decoded, protocol)
            .unwrap_or_else(|error| panic!("{protocol} request fails to replay: {error}"));
        assert_eq!(
            collect_body(replayed.into_body()).await,
            expected,
            "{protocol}"
        );
    }
}

#[tokio::test]
async fn complete_native_response_survives_streaming_and_buffering() {
    let codec = Codec::default();
    for fixture in WIRE_FIXTURES {
        let protocol = fixture.protocol;
        let expected = Bytes::from_static(fixture.response);
        let client = ClientContext::new(protocol);
        let decoded = codec
            .decode_response(HttpResponse::new(body(expected.clone())), protocol)
            .await
            .unwrap_or_else(|error| panic!("{protocol} response fails to decode: {error}"));
        let replayed = codec
            .encode_response(decoded, &client)
            .await
            .unwrap_or_else(|error| panic!("{protocol} response fails to replay: {error}"));
        assert_eq!(
            collect_body(replayed.into_body()).await,
            expected,
            "{protocol}"
        );

        let decoded = codec
            .decode_response(HttpResponse::new(body(expected.clone())), protocol)
            .await
            .expect("response decodes");
        let mut streaming_client = client.clone();
        streaming_client.endpoint.stream = true;
        streaming_client.include_usage = true;
        let streamed = codec
            .encode_response(decoded, &streaming_client)
            .await
            .unwrap_or_else(|error| panic!("{protocol} response fails to stream: {error}"));
        let (parts, streamed_body) = streamed.into_parts();
        assert_eq!(
            parts.headers[header::CONTENT_TYPE],
            "text/event-stream",
            "{protocol}"
        );
        let streamed_bytes = collect_body(streamed_body).await;
        assert!(!sse_data(&streamed_bytes).await.is_empty(), "{protocol}");
        let packets = (0..streamed_bytes.len()).map(move |offset| {
            Ok::<_, Infallible>(Frame::data(streamed_bytes.slice(offset..=offset)))
        });
        let streamed = HttpResponse::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(StreamBody::new(stream::iter(packets)))
            .expect("fixture stream is valid");
        let decoded = codec
            .decode_response(streamed, protocol)
            .await
            .unwrap_or_else(|error| panic!("{protocol} SSE fails to decode: {error}"));
        let buffered = codec
            .encode_response(decoded, &client)
            .await
            .unwrap_or_else(|error| panic!("{protocol} SSE fails to buffer: {error}"));
        assert_eq!(
            collect_body(buffered.into_body()).await,
            expected,
            "{protocol}"
        );
    }
}

#[test]
fn complete_ir_survives_its_event_sequence() {
    let response: Response = serde_json::from_str(include_str!("fixtures/canonical-response.json"))
        .expect("canonical response fixture decodes");
    let mut accumulator = ResponseAccumulator::new();
    accumulator.extend(response.to_stream_events());
    assert_eq!(accumulator.finish(), response);
}

#[tokio::test]
async fn native_http_error_preserves_status_and_all_provider_fields() {
    let codec = Codec::default();
    let expected = Bytes::from_static(br#"{"error":{"message":"try again","type":"rate_limit_error","code":"slow","param":"model"},"retry_after_ms":300}"#);
    let wire = HttpResponse::builder()
        .status(http::StatusCode::TOO_MANY_REQUESTS)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body(expected.clone()))
        .expect("response is valid");
    let protocol = Protocol::ChatCompletions;
    let decoded = codec
        .decode_response(wire, protocol)
        .await
        .expect("error decodes");
    let client = ClientContext::new(protocol);
    let response = codec
        .encode_response(decoded, &client)
        .await
        .expect("error encodes");
    assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(collect_body(response.into_body()).await, expected);
}

#[tokio::test]
async fn edits_keep_unmodeled_native_fields_and_apply_the_ir_change() {
    let codec = Codec::default();
    let original = json!({
        "model":"old","input":[{"role":"user","content":"hello","vendor":{"a":false}}],
        "stream":false,"vendor":[null, 1]
    });
    let mut decoded = codec
        .decode_request(request("/v1/responses", &original))
        .expect("request decodes");
    decoded.body_mut().model = "new".to_owned();
    let output = codec
        .encode_request(&decoded, Protocol::Responses)
        .expect("request encodes");
    let actual: Value =
        serde_json::from_slice(&collect_body(output.into_body()).await).expect("request is JSON");
    let mut expected = original;
    expected["model"] = json!("new");
    assert_eq!(actual, expected);
}
