// Integration tests expose no library API, so item-level documentation would
// add noise without improving the public documentation.
#![allow(missing_docs)]

use bytes::Bytes;
use http::{Request as HttpRequest, Response as HttpResponse, Uri, header};
use http_body_util::{BodyExt, Full};
use mpx_core::{Client, Error, IngressLayer, Upstream, UpstreamService};
use mpx_protocol::{
    Body, Codec, ContentBlock, Endpoint, Protocol, Request, ResponseBody, Role, Route,
};
use tower::{Layer, Service, ServiceExt, limit::ConcurrencyLimit, service_fn};

#[tokio::test]
async fn upstream_service_rewrites_uri_and_decodes_transport_response() {
    let transport = service_fn(|request: HttpRequest<Body>| async move {
        assert_eq!(request.uri(), "https://upstream.example/base/v1/messages");
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(Full::new(Bytes::from_static(
            br#"{"id":"msg-1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#,
        ))))
    });
    let client = UpstreamService::new(ConcurrencyLimit::new(transport, 1), Codec::default())
        .default_upstream(Upstream::new(
            Protocol::Messages,
            Uri::from_static("https://upstream.example/base"),
        ));
    let request = HttpRequest::new(Request::builder().model("claude").build());
    let response = client.oneshot(request).await.expect("transport succeeds");
    let ResponseBody::Complete(response) = response.into_body() else {
        panic!("expected complete response");
    };
    assert_eq!(response.content, vec![ContentBlock::text("pong")]);
}

#[tokio::test]
async fn upstream_service_releases_transport_capacity_when_a_request_fails_before_dispatch() {
    use std::task::{Context, Poll, Waker};

    let transport = service_fn(|_: HttpRequest<Body>| async {
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(Full::new(Bytes::new())))
    });
    let mut service = UpstreamService::new(ConcurrencyLimit::new(transport, 1), Codec::default());
    let mut sibling = service.clone();
    let request = HttpRequest::new(Request::builder().model("claude").build());
    let error = service
        .ready()
        .await
        .expect("transport has capacity")
        .call(request)
        .await
        .expect_err("request is missing an upstream");
    assert!(matches!(error, Error::MissingUpstream));

    let mut cx = Context::from_waker(Waker::noop());
    assert!(matches!(sibling.poll_ready(&mut cx), Poll::Ready(Ok(()))));
    // Keep the failed service alive while checking the sibling's readiness.
    drop((service, sibling));
}

#[tokio::test]
async fn upstream_service_preserves_the_transport_error_source() {
    use std::io;

    let transport = service_fn(|_: HttpRequest<Body>| async {
        Err::<HttpResponse<Full<Bytes>>, _>(io::Error::from(io::ErrorKind::TimedOut))
    });
    let service =
        UpstreamService::new(transport, Codec::default()).default_upstream(Upstream::new(
            Protocol::Messages,
            Uri::from_static("https://upstream.example"),
        ));
    let request = HttpRequest::new(Request::builder().model("claude").build());
    let error = service
        .oneshot(request)
        .await
        .expect_err("transport times out");
    let source = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<io::Error>())
        .expect("typed transport error is preserved");
    assert_eq!(source.kind(), io::ErrorKind::TimedOut);
}

#[tokio::test]
async fn ingress_classifies_transport_failures_as_upstream_errors() {
    let transport = service_fn(|_: HttpRequest<Body>| async {
        Err::<HttpResponse<Full<Bytes>>, _>(std::io::Error::from(
            std::io::ErrorKind::ConnectionReset,
        ))
    });
    let client = Client::new(transport, Codec::default()).default_upstream(Upstream::new(
        Protocol::Messages,
        Uri::from_static("https://upstream.example"),
    ));
    let service = IngressLayer::new(Codec::default()).layer(client);
    let request = HttpRequest::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"hello"}]}"#,
        )))
        .expect("raw request is valid");
    let response = service.oneshot(request).await.expect("error is encoded");
    assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
    let value: serde_json::Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("body is valid")
            .to_bytes(),
    )
    .expect("error is JSON");
    assert_eq!(value["error"]["type"], "upstream");
}

#[tokio::test]
async fn ingress_classifies_timeout_transport_failures_as_gateway_timeout() {
    let transport = service_fn(|_: HttpRequest<Body>| async {
        Err::<HttpResponse<Full<Bytes>>, _>(std::io::Error::from(std::io::ErrorKind::TimedOut))
    });
    let client = Client::new(transport, Codec::default()).default_upstream(Upstream::new(
        Protocol::Messages,
        Uri::from_static("https://upstream.example"),
    ));
    let service = IngressLayer::new(Codec::default()).layer(client);
    let request = HttpRequest::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"hello"}]}"#,
        )))
        .expect("raw request is valid");
    let response = service.oneshot(request).await.expect("error is encoded");
    assert_eq!(response.status(), http::StatusCode::GATEWAY_TIMEOUT);
    let value: serde_json::Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("body is valid")
            .to_bytes(),
    )
    .expect("error is JSON");
    assert_eq!(value["error"]["type"], "timeout");
}

#[tokio::test]
async fn ingress_classifies_invalid_upstream_responses_as_bad_gateway() {
    let transport = service_fn(|_: HttpRequest<Body>| async {
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(Full::new(Bytes::from_static(
            b"not json",
        ))))
    });
    let client = Client::new(transport, Codec::default()).default_upstream(Upstream::new(
        Protocol::Messages,
        Uri::from_static("https://upstream.example"),
    ));
    let service = IngressLayer::new(Codec::default()).layer(client);
    let request = HttpRequest::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"hello"}]}"#,
        )))
        .expect("raw request is valid");
    let response = service.oneshot(request).await.expect("error is encoded");
    assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
    let value: serde_json::Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("body is valid")
            .to_bytes(),
    )
    .expect("error is JSON");
    assert_eq!(value["error"]["type"], "bad_response");
}

#[tokio::test]
async fn upstream_rejects_base_uri_queries_instead_of_dropping_them() {
    let transport = service_fn(|_: HttpRequest<Body>| async {
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(Full::new(Bytes::new())))
    });
    let service = Client::new(transport, Codec::default()).default_upstream(Upstream::new(
        Protocol::Messages,
        Uri::from_static("https://upstream.example/base?key=secret"),
    ));
    let request = HttpRequest::new(Request::builder().model("claude").build());
    let error = service
        .oneshot(request)
        .await
        .expect_err("base query is rejected");
    assert!(matches!(error, Error::InvalidUpstreamUri(message) if message.contains("query")));
}

#[tokio::test]
async fn ingress_translates_raw_client_http_through_model_service() {
    let codec = Codec::default();
    let inner = service_fn(|request: HttpRequest<Request>| async move {
        assert_eq!(request.body().messages[0].role, Role::User);
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(ResponseBody::Complete(
            mpx_protocol::Response::builder()
                .id("resp")
                .model("gpt-5")
                .content(vec![ContentBlock::text("hello from model")])
                .build(),
        )))
    });
    let service = IngressLayer::new(codec).layer(ConcurrencyLimit::new(inner, 1));
    let request = HttpRequest::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"gpt-5","messages":[{"role":"user","content":"hello"}]}"#,
        )))
        .expect("raw request is valid");
    let response = service.oneshot(request).await.expect("ingress succeeds");
    assert_eq!(response.status(), 200);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("response body is valid")
        .to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("response is JSON");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "hello from model"
    );
}

#[tokio::test]
async fn ingress_enforces_body_limit_with_protocol_error_shape() {
    let inner = service_fn(|_: HttpRequest<Request>| async move {
        Ok::<_, mpx_protocol::BoxError>(HttpResponse::new(ResponseBody::Error(
            mpx_protocol::ApiError::new(mpx_protocol::ApiErrorKind::Internal, "unused"),
        )))
    });
    let service = IngressLayer::new(Codec::default())
        .max_body_bytes(4)
        .layer(inner);
    let request = HttpRequest::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from_static(b"too large")))
        .expect("raw request is valid");
    let response = service.oneshot(request).await.expect("limit is handled");
    assert_eq!(response.status(), 413);
    let value: serde_json::Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("body is valid")
            .to_bytes(),
    )
    .expect("error is JSON");
    assert_eq!(value["error"]["type"], "body_too_large");
}

#[tokio::test]
async fn interaction_retrieval_preserves_resumed_events_through_transport() {
    use http::StatusCode;

    let resumed = Bytes::from_static(
        b"event: step.delta\ndata: {\"event_type\":\"step.delta\",\"event_id\":\"e2\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"hello\"}}\n\n",
    );
    let expected = resumed.clone();
    let transport = service_fn(move |request: HttpRequest<Body>| {
        let resumed = resumed.clone();
        async move {
            assert_eq!(request.method(), http::Method::GET);
            assert_eq!(
                request.uri(),
                "https://gemini.example/v1/interactions/int1?stream=true&include_input=true&last_event_id=e1"
            );
            assert!(!request.headers().contains_key("x-goog-api-key"));
            Ok::<_, mpx_protocol::BoxError>(
                HttpResponse::builder()
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Full::new(resumed))
                    .expect("upstream response is valid"),
            )
        }
    });
    let codec = Codec::default();
    let client = Client::new(transport, codec.clone()).default_upstream(Upstream::new(
        Protocol::Interactions,
        Uri::from_static("https://gemini.example"),
    ));
    let service = IngressLayer::new(codec.clone()).layer(client);
    let request = HttpRequest::get(
        "/v1beta/interactions/int1?include_input=true&stream=true&last_event_id=e1&key=hidden",
    )
    .header("x-goog-api-key", "hidden")
    .body(Bytes::new())
    .expect("resource request is valid");
    let decoded = codec
        .decode_request(request.clone())
        .expect("resource is detected");
    let encoded = codec
        .encode_request(&decoded, Protocol::Interactions)
        .expect("resource request encodes");
    assert_eq!(
        encoded.uri(),
        "/v1/interactions/int1?stream=true&include_input=true&last_event_id=e1",
    );
    assert!(codec.encode_request(&decoded, Protocol::Responses).is_err());

    let response = service
        .oneshot(request.map(Full::new))
        .await
        .expect("resource response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes(),
        expected
    );
}

#[tokio::test]
async fn interaction_cancel_and_delete_preserve_upstream_status_and_body() {
    use http::{Method, StatusCode};

    let codec = Codec::default();
    let transport = service_fn(|request: HttpRequest<Body>| async move {
        let (status, text) = match (request.method(), request.uri().path()) {
            (&Method::DELETE, "/v1/interactions/int1") => (StatusCode::NO_CONTENT, ""),
            (&Method::POST, "/v1/interactions/int1/cancel") => {
                (StatusCode::OK, r#"{"id":"int1","status":"cancelled"}"#)
            }
            other => panic!("unexpected operation {other:?}"),
        };
        Ok::<_, mpx_protocol::BoxError>(
            HttpResponse::builder()
                .status(status)
                .body(Full::new(Bytes::from_static(text.as_bytes())))
                .expect("upstream response is valid"),
        )
    });
    let client = Client::new(transport, codec.clone()).default_upstream(Upstream::new(
        Protocol::Interactions,
        Uri::from_static("https://gemini.example"),
    ));
    let service = IngressLayer::new(codec).layer(client);
    for (method, path, status, body) in [
        (
            Method::DELETE,
            "/v1/interactions/int1",
            StatusCode::NO_CONTENT,
            "",
        ),
        (
            Method::POST,
            "/v1/interactions/int1/cancel",
            StatusCode::OK,
            r#"{"id":"int1","status":"cancelled"}"#,
        ),
    ] {
        let request = HttpRequest::builder()
            .method(method)
            .uri(path)
            .body(Full::new(Bytes::new()))
            .expect("resource request is valid");
        let response = service
            .clone()
            .oneshot(request)
            .await
            .expect("resource response");
        assert_eq!(response.status(), status);
        assert_eq!(
            response
                .into_body()
                .collect()
                .await
                .expect("body")
                .to_bytes(),
            body
        );
    }
}

#[tokio::test]
async fn custom_route_forwards_native_http_through_ingress_and_client() {
    use http::{Method, StatusCode};

    let codec = Codec::builder()
        .route(Route::new(|parts| {
            (parts.method == Method::PUT && parts.uri.path() == "/files").then(|| {
                Endpoint::passthrough(
                    Protocol::Messages,
                    Method::POST,
                    http::uri::PathAndQuery::from_static("/v1/files?purpose=attachment"),
                )
            })
        }))
        .build();
    let transport = service_fn(|request: HttpRequest<Body>| async move {
        assert_eq!(request.method(), Method::POST);
        assert_eq!(
            request.uri(),
            "https://upstream.example/v1/files?purpose=attachment"
        );
        assert_eq!(
            request.headers()[header::CONTENT_TYPE],
            "application/octet-stream"
        );
        assert!(!request.headers().contains_key(header::AUTHORIZATION));
        assert_eq!(
            request
                .into_body()
                .collect()
                .await
                .expect("native request body")
                .to_bytes(),
            &[0, 255][..],
        );
        Ok::<_, mpx_protocol::BoxError>(
            HttpResponse::builder()
                .status(StatusCode::CREATED)
                .header(header::CONTENT_TYPE, "text/plain")
                .body(Full::new(Bytes::from_static(b"native response")))
                .expect("upstream response is valid"),
        )
    });
    let client = Client::new(transport, codec.clone()).default_upstream(Upstream::new(
        Protocol::Messages,
        Uri::from_static("https://upstream.example"),
    ));
    let service = IngressLayer::new(codec).layer(client);
    let request = HttpRequest::put("/files?key=hidden")
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::AUTHORIZATION, "Bearer hidden")
        .body(Full::new(Bytes::from_static(&[0, 255])))
        .expect("native request is valid");
    let response = service.oneshot(request).await.expect("native response");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/plain");
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("native response body")
            .to_bytes(),
        "native response"
    );
}
