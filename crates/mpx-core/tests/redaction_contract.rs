//! End-to-end redaction contracts for the Tower gateway.

use std::fmt::Write as _;

use base64::{Engine, engine::general_purpose::STANDARD_NO_PAD};
use bytes::Bytes;
use http::{Method, Request, Response, StatusCode, Uri, header};
use http_body_util::{BodyExt, Full};
use mpx_core::{
    Ingress, IngressLayer, Redaction, RedactionLayer, Upstream, UpstreamService,
    mpx_protocol::{self, Body, ClientContext, Codec, Diagnostics, Endpoint, Protocol},
    mpx_redact::{
        Digits, GeneratorBinding, Hex, Mask, Matcher, Number, PlaceholderTemplate,
        RedactionSession, ReplacementMap, Rule, Rules, WordList,
    },
};
use serde_json::{Value, json};
use tower::{ServiceBuilder, ServiceExt, service_fn};

const SECRET: &str = "Nightjar Project";

/// Builds ingress → redaction → upstream with the default codec.
fn gateway<T>(
    transport: T,
    layer: RedactionLayer,
    protocol: Protocol,
) -> Ingress<Redaction<UpstreamService<T>>> {
    let codec = Codec::default();
    ServiceBuilder::new()
        .layer(IngressLayer::new(codec.clone()))
        .layer(layer)
        .service(
            UpstreamService::new(transport, codec).default_upstream(Upstream::new(
                protocol,
                Uri::from_static("https://upstream.example"),
            )),
        )
}

#[tokio::test]
async fn aliases_and_fallback_tags_round_trip_through_sse() {
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let request: Value = serde_json::from_slice(&bytes).unwrap();
        let text = request["messages"][0]["content"].as_str().unwrap();
        let (named, tail) = text.split_once(", ").unwrap();
        let (fallback, project) = tail.split_once(", ").unwrap();
        assert_eq!(named, "cedar-cedar@example.com");
        assert!(fallback.starts_with("<|mpx-redact-"));
        let digits = project.strip_prefix("project-cedar-").unwrap();
        assert_eq!(digits.len(), 4);
        assert!(digits.bytes().all(|byte| byte.is_ascii_digit()));
        Ok::<_, mpx_protocol::BoxError>(echo(text, true))
    });
    let words = WordList::new(&["cedar"]).unwrap();
    let email = Rule::new(
        Matcher::Email,
        Mask::alias(
            "email",
            "{{word}}-{{word}}@example.com",
            &[GeneratorBinding::new("word", words.clone())],
        )
        .unwrap(),
    );
    let project = Rule::new(
        Matcher::regex(r"Project [A-Z][a-z]+").unwrap(),
        Mask::alias(
            "project",
            "project-{{word}}-{{digits}}",
            &[
                GeneratorBinding::new("word", words),
                GeneratorBinding::new("digits", Digits(4)),
            ],
        )
        .unwrap(),
    );
    let rules = Rules::new([email, project]);
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules).with_replacement_map(),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(br#"{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":"alice@corp.test, bob@corp.test, Project Nightjar"}]}"#)))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let replacements = response.extensions().get::<ReplacementMap>().unwrap();
    assert_eq!(replacements.entries().len(), 3);
    assert_eq!(
        replacements.get("cedar-cedar@example.com"),
        Some("alice@corp.test")
    );
    assert!(replacements.entries().any(|(replacement, original)| {
        replacement.starts_with("<|mpx-redact-") && original == "bob@corp.test"
    }));
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        streamed_text(std::str::from_utf8(&body).unwrap()),
        "echo alice@corp.test, bob@corp.test, Project Nightjar"
    );
}

#[tokio::test]
async fn restores_complete_and_streamed_responses() {
    let custom = PlaceholderTemplate::new(
        r#"<redacted value="{{hex}}"/>"#,
        &[GeneratorBinding::new("hex", Hex(12))],
    )
    .unwrap();
    for (upstream_streams, client_streams, custom_template) in [
        (false, false, false),
        (true, true, false),
        (true, false, false),
        (true, true, true),
    ] {
        let (prefix, suffix, width) = if custom_template {
            (r#"<redacted value=""#, r#""/>"#, 12)
        } else {
            ("<|mpx-redact-", "|>", 6)
        };
        let transport = service_fn(move |request: Request<Body>| async move {
            let bytes = request.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let placeholder = body["messages"][1]["content"].as_str().unwrap();
            let generated = placeholder
                .strip_prefix(prefix)
                .unwrap()
                .strip_suffix(suffix)
                .unwrap();
            assert_eq!(generated.len(), width);
            assert!(generated.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(body["messages"][0]["content"], placeholder);
            assert!(!std::str::from_utf8(&bytes).unwrap().contains(SECRET));
            Ok::<_, mpx_protocol::BoxError>(echo(placeholder, upstream_streams))
        });
        let rules = Rules::new([Rule::new(Matcher::Literal(SECRET.into()), Mask::tag())]);
        let mut layer = RedactionLayer::new(rules);
        if custom_template {
            layer = layer.with_placeholder_template(custom.clone());
        }
        let gateway = gateway(transport, layer, Protocol::ChatCompletions);
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(json!({
                "model": "gpt-5",
                "stream": client_streams,
                "messages": [{"role": "system", "content": SECRET}, {"role": "user", "content": SECRET}],
            }).to_string())))
            .unwrap();
        let response = gateway.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.extensions().get::<ReplacementMap>().is_none());
        let is_sse = response.headers()[header::CONTENT_TYPE] == "text/event-stream";
        assert_eq!(is_sse, client_streams);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(!body.contains(prefix));
        if client_streams {
            assert_eq!(streamed_text(body), format!("echo {SECRET}"));
            assert!(body.contains("[DONE]"), "{body}");
        } else {
            let value: Value = serde_json::from_str(body).unwrap();
            assert_eq!(value["id"], "echo");
            assert_eq!(
                value["choices"][0]["message"]["content"],
                format!("echo {SECRET}")
            );
        }
    }
}

#[tokio::test]
async fn restores_across_protocols_and_preserves_original_marker_text() {
    const MESSAGES_STREAM: &str =
        include_str!("../../mpx-protocol/tests/contracts/fixtures/messages.sse");
    let rules = Rules::new([Rule::new(Matcher::Literal(SECRET.into()), Mask::tag())]);
    let literal = RedactionSession::new().mask(SECRET, &rules).unwrap();
    for (upstream_streams, client_streams) in [(false, false), (true, true), (true, false)] {
        let expected_literal = literal.clone();
        let transport = service_fn(move |request: Request<Body>| {
            let expected_literal = expected_literal.clone();
            async move {
                assert_eq!(request.uri().path(), "/v1/messages");
                let bytes = request.into_body().collect().await.unwrap().to_bytes();
                assert!(!std::str::from_utf8(&bytes).unwrap().contains(SECRET));
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                let marker = body["messages"][0]["content"][0]["text"].as_str().unwrap();
                assert!(marker.starts_with("<|mpx-redact-"));
                assert_ne!(marker, expected_literal);
                assert_eq!(body["messages"][1]["content"][0]["text"], expected_literal);
                let (content_type, payload) = if upstream_streams {
                    let (first, rest) = marker.split_at(10);
                    let mut payload = MESSAGES_STREAM
                        .replace("\"text\":\"Hi \"", &format!("\"text\":\"{first}\""))
                        .replace(
                            "\"text\":\"🦉\"",
                            &format!("\"text\":\"{rest} {expected_literal}\""),
                        );
                    payload.push('\n');
                    ("text/event-stream", payload)
                } else {
                    (
                        "application/json",
                        json!({
                            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                            "content": [{"type": "text", "text": format!("{marker} {expected_literal}")}],
                            "stop_reason": "end_turn", "usage": {"input_tokens": 11, "output_tokens": 3}
                        })
                        .to_string(),
                    )
                };
                Ok::<_, mpx_protocol::BoxError>(
                    Response::builder()
                        .header(header::CONTENT_TYPE, content_type)
                        .body(Full::new(Bytes::from(payload)))
                        .unwrap(),
                )
            }
        });
        let gateway = gateway(
            transport,
            RedactionLayer::new(rules.clone()),
            Protocol::Messages,
        );
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(
                json!({
                    "model": "m", "stream": client_streams, "max_tokens": 100,
                    "messages": [{"role":"user", "content":SECRET}, {"role":"assistant", "content":literal}],
                })
                .to_string(),
            )))
            .unwrap();
        let response = gateway.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        let expected = format!("{SECRET} {literal}");
        if client_streams {
            assert!(body.contains("[DONE]"), "{body}");
            assert_eq!(streamed_text(body), expected);
        } else {
            let response: Value = serde_json::from_str(body).unwrap();
            assert_eq!(response["choices"][0]["message"]["content"], expected);
        }
    }
}

#[tokio::test]
async fn generator_exhaustion_fails_before_dispatch() {
    async fn unexpected_dispatch(
        _: Request<Body>,
    ) -> Result<Response<Full<Bytes>>, mpx_protocol::BoxError> {
        panic!("generator exhaustion must reject the request before dispatch");
    }

    let rules = Rules::new([Rule::new(Matcher::Email, Mask::tag())]);
    let template =
        PlaceholderTemplate::new("<{{n}}>", &[GeneratorBinding::new("n", Number(0..=0))]).unwrap();
    let gateway = gateway(
        service_fn(unexpected_dispatch),
        RedactionLayer::new(rules).with_placeholder_template(template),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"a@example.org b@example.org"}]}"#,
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        error["error"]["message"],
        "could not generate an unused redaction marker from template \"<{{n}}>\" after 1 of 1 candidate(s)"
    );
}

#[tokio::test]
async fn probing_is_recorded_as_a_cache_stability_warning() {
    let rules = Rules::new([Rule::new(Matcher::Email, Mask::tag())]);
    let template =
        PlaceholderTemplate::new("<{{n}}>", &[GeneratorBinding::new("n", Number(0..=1))]).unwrap();
    let candidate = RedactionSession::new()
        .with_placeholder_template(template.clone())
        .mask("a@example.org", &rules)
        .unwrap();
    let expected_prefix = candidate.clone();
    let transport = service_fn(move |request: Request<Body>| {
        let expected_prefix = expected_prefix.clone();
        async move {
            let bytes = request.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let text = body["messages"][0]["content"].as_str().unwrap();
            assert!(text.starts_with(&format!("{expected_prefix} ")));
            assert_ne!(text, format!("{expected_prefix} {expected_prefix}"));
            Ok::<_, mpx_protocol::BoxError>(echo("ok", false))
        }
    });
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules)
            .with_placeholder_template(template)
            .with_replacement_map(),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(
            json!({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": format!("{candidate} a@example.org")}]
            })
            .to_string(),
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    let diagnostics = response
        .extensions()
        .get::<Diagnostics>()
        .unwrap()
        .entries();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.path == "redaction.replacement"
            && diagnostic.message.contains("collision probing")
    }));
}

fn streamed_text(body: &str) -> String {
    body.lines()
        .filter_map(|line| {
            let value: Value = serde_json::from_str(line.strip_prefix("data: ")?).ok()?;
            value["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

#[tokio::test]
async fn stream_fixture_replays_snapshot_without_redaction_layer() {
    let codec = Codec::default();
    let protocol = Protocol::ChatCompletions;
    let response = codec
        .decode_response(echo("some placeholder", true), protocol)
        .await
        .unwrap();
    let client = ClientContext::new(protocol);
    let response = codec.encode_response(response, &client).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["id"], "snapshot");
}

fn echo(placeholder: &str, stream: bool) -> Response<Full<Bytes>> {
    let id = if stream { "snapshot" } else { "echo" };
    let complete = json!({
        "id": id, "object": "chat.completion", "created": 0, "model": "gpt-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": format!("echo {placeholder}")}, "finish_reason": "stop"}]
    }).to_string();
    if !stream {
        return Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(complete)))
            .unwrap();
    }
    // A same-protocol client requesting JSON would normally replay this snapshot.
    let snapshot = STANDARD_NO_PAD.encode(complete);
    let protocol = Protocol::ChatCompletions;
    let mut sse = format!(": mpx-buffered-response-v1 {protocol} {snapshot}\n");
    let (first, rest) = placeholder.split_at(10);
    for (delta, reason) in [
        (
            json!({"role": "assistant", "content": format!("echo {first}")}),
            Value::Null,
        ),
        (json!({"content": rest}), Value::Null),
        (json!({}), json!("stop")),
    ] {
        let frame = json!({
            "id": "echo", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]
        });
        write!(sse, "data: {frame}\n\n").unwrap();
    }
    sse.push_str("data: [DONE]\n\n");
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Full::new(Bytes::from(sse)))
        .unwrap()
}

#[tokio::test]
async fn rejects_passthrough_before_dispatch() {
    async fn unexpected_dispatch(
        _: Request<Body>,
    ) -> Result<Response<Full<Bytes>>, mpx_protocol::BoxError> {
        panic!("passthrough must be rejected before dispatch");
    }

    let transport = service_fn(unexpected_dispatch);
    let gateway = gateway(
        transport,
        RedactionLayer::new(Rules::new([])),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/native")
        .extension(Endpoint::passthrough(
            Protocol::ChatCompletions,
            Method::POST,
            "/native".parse().unwrap(),
        ))
        .body(Full::new(Bytes::from_static(b"native")))
        .unwrap();
    assert_eq!(
        gateway.oneshot(request).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
}
