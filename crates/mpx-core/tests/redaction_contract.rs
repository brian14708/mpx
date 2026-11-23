//! End-to-end redaction contracts for the Tower gateway.

use std::fmt::Write as _;

use base64::{Engine, engine::general_purpose::STANDARD_NO_PAD};
use bytes::Bytes;
use http::{Method, Request, Response, StatusCode, Uri, header};
use http_body_util::{BodyExt, Full};
use mpx_core::{
    Ingress, IngressLayer, Redaction, RedactionLayer, Upstream, UpstreamService,
    mpx_protocol::{self, Body, ClientContext, Codec, Endpoint, Protocol},
    mpx_redact::{Matcher, RedactionSession, Replacement, ReplacementMap, Rules, Slot},
};
use serde_json::{Value, json};
use tower::{ServiceBuilder, ServiceExt, service_fn};

const SECRET: &str = "Nightjar Project";

fn replacement() -> Replacement {
    Replacement::kind("SECRET").unwrap()
}

fn email_matcher() -> Matcher {
    Matcher::regex(
        r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}",
    )
    .unwrap()
}

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
async fn session_id_scopes_redaction_markers_and_aliases() {
    let rules = Rules::new([
        (Matcher::literal(SECRET), replacement()),
        (
            email_matcher(),
            Replacement::template("{}@example.com", &[Slot::base16(16)]).unwrap(),
        ),
    ]);
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        let text = body["messages"][0]["content"].as_str().unwrap();
        Ok::<_, mpx_protocol::BoxError>(echo(text, false))
    });
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules)
            .with_hmac_key(b"gateway-test-secret")
            .with_replacement_map(),
        Protocol::ChatCompletions,
    );
    let send = |session_id: &str| {
        let gateway = gateway.clone();
        let text = format!("{SECRET}, alice@corp.test");
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-session-id", session_id)
            .body(Full::new(Bytes::from(
                json!({"model":"gpt-5","messages":[{"role":"user","content":text}]}).to_string(),
            )))
            .unwrap();
        async move {
            let response = gateway.oneshot(request).await.unwrap();
            let replacements = response.extensions().get::<ReplacementMap>().unwrap();
            let issued = [SECRET, "alice@corp.test"].map(|value| {
                replacements
                    .entries()
                    .find_map(|(replacement, original)| {
                        (original == value).then(|| replacement.to_owned())
                    })
                    .unwrap()
            });
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body["choices"][0]["message"]["content"],
                format!("echo {text}")
            );
            issued
        }
    };
    let [marker, alias] = send("session-a").await;
    let expected_marker = RedactionSession::new()
        .with_hmac_key(b"gateway-test-secret")
        .with_scope("session-a")
        .mask(
            SECRET,
            &Rules::new([(Matcher::literal(SECRET), replacement())]),
        )
        .unwrap();
    assert_eq!(marker, expected_marker);
    assert_eq!(send("session-a").await, [marker.clone(), alias.clone()]);
    let [other_marker, other_alias] = send("session-b").await;
    assert_ne!(marker, other_marker);
    assert_ne!(alias, other_alias);
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn redacts_structured_tool_fields_and_restores_streamed_arguments() {
    for upstream_streams in [false, true] {
        let transport = service_fn(move |request: Request<Body>| async move {
            let bytes = request.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let encoded = body.to_string();
            assert!(!encoded.contains("alice@example.org"), "{encoded}");
            let arguments = body["messages"][1]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap();
            let arguments_value: Value = serde_json::from_str(arguments).unwrap();
            assert!(
                arguments_value["email"]
                    .as_str()
                    .is_some_and(|value| value.starts_with("{{SECRET_"))
            );

            if upstream_streams {
                let split = arguments.len() / 2;
                let first = &arguments[..split];
                let second = &arguments[split..];
                let frames = [
                    json!({
                        "id": "tool-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "gpt-5",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": first}}]}, "finish_reason": null}]
                    }),
                    json!({
                        "id": "tool-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "gpt-5",
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": second}}]}, "finish_reason": null}]
                    }),
                    json!({
                        "id": "tool-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "gpt-5",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
                    }),
                ];
                let mut payload = String::new();
                for frame in frames {
                    write!(&mut payload, "data: {frame}\n\n").unwrap();
                }
                payload.push_str("data: [DONE]\n\n");
                Ok::<_, mpx_protocol::BoxError>(
                    Response::builder()
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Full::new(Bytes::from(payload)))
                        .unwrap(),
                )
            } else {
                Ok::<_, mpx_protocol::BoxError>(
                    Response::builder()
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Full::new(Bytes::from(
                            json!({
                                "id": "tool-response",
                                "object": "chat.completion",
                                "created": 0,
                                "model": "gpt-5",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": arguments}}]
                                    },
                                    "finish_reason": "tool_calls"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap(),
                )
            }
        });
        let rules = Rules::new([(email_matcher(), replacement())]);
        let gateway = gateway(
            transport,
            RedactionLayer::new(rules).with_hmac_key(b"gateway-test-secret"),
            Protocol::ChatCompletions,
        );
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(
                json!({
                    "model": "gpt-5",
                    "stream": upstream_streams,
                    "metadata": {"contact": "alice@example.org"},
                    "tools": [{"type": "function", "function": {
                        "name": "lookup",
                        "description": "Contact alice@example.org",
                        "parameters": {"type": "object", "properties": {"email": {"type": "string", "description": "alice@example.org"}}}
                    }}],
                    "messages": [
                        {"role": "user", "content": "alice@example.org"},
                        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{\"email\":\"alice@example.org\"}"}}]},
                        {"role": "tool", "tool_call_id": "call-1", "content": "alice@example.org"}
                    ]
                })
                .to_string(),
            )))
            .unwrap();
        let response = gateway.oneshot(request).await.unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        if upstream_streams {
            assert_eq!(
                streamed_tool_arguments(std::str::from_utf8(&body).unwrap()),
                r#"{"email":"alice@example.org"}"#
            );
        } else {
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                r#"{"email":"alice@example.org"}"#
            );
        }
    }
}

#[tokio::test]
async fn zero_hit_redaction_preserves_gemini_request_replay() {
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["contents"][1]["parts"][0]["functionResponse"]["response"]["sibling"],
            "keep"
        );
        Ok::<_, mpx_protocol::BoxError>(
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Full::new(Bytes::from_static(
                    br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}"#,
                )))
                .unwrap(),
        )
    });
    let rules = Rules::new([(email_matcher(), replacement())]);
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules)
            .with_hmac_key(b"gateway-test-secret")
            .with_replacement_map(),
        Protocol::GenerateContent,
    );
    let request = Request::post("/v1beta/models/gemini-2:generateContent")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"contents":[{"role":"model","parts":[{"functionCall":{"id":"call-1","name":"lookup","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"id":"call-1","name":"lookup","response":{"output":"safe","sibling":"keep"}}}]}]}"#,
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let replacements = response.extensions().get::<ReplacementMap>().unwrap();
    assert_eq!(replacements.entries().len(), 0);
}

#[tokio::test]
async fn masking_structured_tool_result_keeps_json_siblings() {
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["contents"][1]["parts"][0]["functionResponse"]["response"],
            json!({"output": "\"", "sibling": "keep"})
        );
        Ok::<_, mpx_protocol::BoxError>(
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Full::new(Bytes::from_static(
                    br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}"#,
                )))
                .unwrap(),
        )
    });
    let rules = Rules::new([(Matcher::literal("secret"), Replacement::mask("\""))]);
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules)
            .with_hmac_key(b"gateway-test-secret")
            .with_replacement_map(),
        Protocol::GenerateContent,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(
            json!({
                "model": "gpt-5",
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "{\"output\":\"secret\",\"sibling\":\"keep\"}"}
                ]
            })
            .to_string(),
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let replacements = response.extensions().get::<ReplacementMap>().unwrap();
    assert_eq!(replacements.entries().len(), 0);
    assert_eq!(
        replacements.forward_entries().collect::<Vec<_>>(),
        [("secret", "\"")]
    );
}

#[tokio::test]
async fn alias_collision_falls_back_and_restores_the_stream() {
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let request: Value = serde_json::from_slice(&bytes).unwrap();
        let text = request["messages"][0]["content"].as_str().unwrap();
        let (named, tail) = text.split_once(", ").unwrap();
        let (fallback, project) = tail.split_once(", ").unwrap();
        assert_eq!(named, "cedar-cedar@example.com");
        assert!(fallback.starts_with("{{FALLBACK_"));
        let digits = project.strip_prefix("project-cedar-").unwrap();
        assert_eq!(digits.len(), 4);
        assert!(digits.bytes().all(|byte| byte.is_ascii_digit()));
        Ok::<_, mpx_protocol::BoxError>(echo(text, true))
    });
    let words = Slot::wordlist(["cedar"]).unwrap();
    let email = (
        email_matcher(),
        Replacement::template("{}-{}@example.com", &[words.clone(), words.clone()]).unwrap(),
    );
    let project = (
        Matcher::regex(r"Project [A-Z][a-z]+").unwrap(),
        Replacement::template("project-{}-{}", &[words, Slot::digits(4)]).unwrap(),
    );
    let rules = Rules::new([email, project]);
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules)
            .with_fallback(Replacement::kind("FALLBACK").unwrap())
            .with_replacement_map(),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(br#"{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":"alice@corp.test, bob@corp.test, Project Nightjar"}]}"#)))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let replacements = response.extensions().get::<ReplacementMap>().unwrap();
    assert_eq!(
        replacements.get("cedar-cedar@example.com"),
        Some("alice@corp.test")
    );
    assert!(replacements.entries().any(|(replacement, original)| {
        replacement.starts_with("{{FALLBACK_") && original == "bob@corp.test"
    }));
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        streamed_text(std::str::from_utf8(&body).unwrap()),
        "echo alice@corp.test, bob@corp.test, Project Nightjar"
    );
}

#[tokio::test]
async fn restores_complete_and_streamed_responses() {
    for (upstream_streams, client_streams, custom_template) in [
        (false, false, false),
        (true, true, false),
        (true, false, false),
        (true, true, true),
    ] {
        let (prefix, suffix, width) = if custom_template {
            (r#"<redacted value=""#, r#""/>"#, 12)
        } else {
            ("{{SECRET_", "}}", 8)
        };
        let transport = service_fn(move |request: Request<Body>| async move {
            let bytes = request.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let replacement = body["messages"][1]["content"].as_str().unwrap();
            let generated = replacement
                .strip_prefix(prefix)
                .unwrap()
                .strip_suffix(suffix)
                .unwrap();
            assert_eq!(generated.len(), width);
            if custom_template {
                assert!(generated.bytes().all(|byte| byte.is_ascii_hexdigit()));
            } else {
                assert!(
                    generated
                        .bytes()
                        .all(|byte| byte.is_ascii_lowercase() || (b'2'..=b'7').contains(&byte))
                );
            }
            assert_eq!(body["messages"][0]["content"], replacement);
            assert!(!std::str::from_utf8(&bytes).unwrap().contains(SECRET));
            Ok::<_, mpx_protocol::BoxError>(echo(replacement, upstream_streams))
        });
        let mask = if custom_template {
            Replacement::template(r#"<redacted value="{}"/>"#, &[Slot::base16(12)]).unwrap()
        } else {
            replacement()
        };
        let layer = RedactionLayer::new(Rules::new([(Matcher::literal(SECRET), mask)]))
            .with_replacement_map();
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
        let replacements = response.extensions().get::<ReplacementMap>().unwrap();
        assert_eq!(replacements.entries().len(), 1);
        assert_eq!(replacements.entries().next().unwrap().1, SECRET);
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
    let rules = Rules::new([(Matcher::literal(SECRET), replacement())]);
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
                assert!(marker.starts_with("{{SECRET_"));
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
async fn block_rule_returns_forbidden_before_dispatch() {
    async fn unexpected_dispatch(
        _: Request<Body>,
    ) -> Result<Response<Full<Bytes>>, mpx_protocol::BoxError> {
        panic!("blocked requests must never dispatch");
    }

    let rules = Rules::new([(email_matcher(), Replacement::block())]);
    let gateway = gateway(
        service_fn(unexpected_dispatch),
        RedactionLayer::new(rules),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"a@example.org"}]}"#,
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(error["error"]["code"], "redaction_blocked");
    assert_eq!(
        error["error"]["message"],
        "request blocked by a redaction rule"
    );
}

#[tokio::test]
async fn a_collision_is_rejected_before_dispatch_by_default() {
    async fn unexpected_dispatch(
        _: Request<Body>,
    ) -> Result<Response<Full<Bytes>>, mpx_protocol::BoxError> {
        panic!("strict mode must reject a collision before dispatch");
    }

    let rules = Rules::new([(
        email_matcher(),
        Replacement::template("<{}>", &[Slot::number(0..=0)]).unwrap(),
    )]);
    let gateway = gateway(
        service_fn(unexpected_dispatch),
        RedactionLayer::new(rules),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"a@example.org b@example.org"}]}"#,
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        error["error"]["message"],
        "could not use the redaction replacement from template \"<{}>\""
    );
    assert_eq!(error["error"]["code"], "redaction_replacement_unavailable");
}

#[tokio::test]
async fn a_candidate_present_in_another_field_uses_the_configured_literal_fallback() {
    let mask = Replacement::template("<{}>", &[Slot::number(0..=0)]).unwrap();
    let rules = Rules::new([(email_matcher(), mask)]);
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        let mask = body["messages"][0]["content"].as_str().unwrap();
        assert_eq!(mask, "[removed]");
        assert_eq!(body["messages"][1]["content"], r#"{"value":"<0>"}"#);
        Ok::<_, mpx_protocol::BoxError>(echo(&format!("{mask} <0>"), false))
    });
    let gateway = gateway(
        transport,
        RedactionLayer::new(rules).with_fallback(Replacement::mask("[removed]")),
        Protocol::ChatCompletions,
    );
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(
            json!({
                "model": "gpt-5",
                "messages": [
                    {"role": "user", "content": "a@example.org"},
                    {"role": "assistant", "content": r#"{"value":"<0>"}"#}
                ]
            })
            .to_string(),
        )))
        .unwrap();
    let response = gateway.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "echo [removed] <0>"
    );
}

#[tokio::test]
async fn rejects_an_alias_and_fallback_reserved_in_a_later_tool_result() {
    async fn unexpected_dispatch(
        _: Request<Body>,
    ) -> Result<Response<Full<Bytes>>, mpx_protocol::BoxError> {
        panic!("a request with no available replacement must fail before dispatch");
    }

    let key = b"gateway-test-secret";
    let private = "private";
    let fallback = Replacement::kind("FALLBACK").unwrap();
    let marker = RedactionSession::new()
        .with_hmac_key(key)
        .mask(
            private,
            &Rules::new([(Matcher::literal(private), fallback.clone())]),
        )
        .unwrap();
    let escaped_fallback = marker.replacen('{', r"\u007b", 1);
    let history = format!(r#"{{"alias":"cedar","marker":"{escaped_fallback}"}}"#);
    let body = Bytes::from(
        json!({
            "model": "gpt-5",
            "messages": [
                {"role": "user", "content": private},
                {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": history}
            ]
        })
        .to_string(),
    );

    for (allow_fallback, rejected_template) in [(false, "{}"), (true, "{{FALLBACK_{}}}")] {
        let rules = Rules::new([(
            Matcher::literal(private),
            Replacement::template("{}", &[Slot::literal("cedar")]).unwrap(),
        )]);
        let mut layer = RedactionLayer::new(rules).with_hmac_key(key);
        if allow_fallback {
            layer = layer.with_fallback(fallback.clone());
        }
        let gateway = gateway(
            service_fn(unexpected_dispatch),
            layer,
            Protocol::ChatCompletions,
        );
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(body.clone()))
            .unwrap();
        let response = gateway.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["code"], "redaction_replacement_unavailable");
        assert_eq!(
            error["error"]["message"],
            format!("could not use the redaction replacement from template {rejected_template:?}")
        );
    }
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

fn streamed_tool_arguments(body: &str) -> String {
    body.lines()
        .filter_map(|line| {
            let data = line.strip_prefix("data: ")?;
            let value: Value = serde_json::from_str(data).ok()?;
            value["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
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
        .decode_response(echo("some replacement", true), protocol)
        .await
        .unwrap();
    let client = ClientContext::new(protocol);
    let response = codec.encode_response(response, &client).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["id"], "snapshot");
}

fn echo(replacement: &str, stream: bool) -> Response<Full<Bytes>> {
    let id = if stream { "snapshot" } else { "echo" };
    let complete = json!({
        "id": id, "object": "chat.completion", "created": 0, "model": "gpt-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": format!("echo {replacement}")}, "finish_reason": "stop"}]
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
    let (first, rest) = replacement.split_at(10);
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
