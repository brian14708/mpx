// Integration tests expose no library API.
#![allow(missing_docs)]

mod http;
mod model;
mod requests;
mod round_trip;
mod streams;
mod tools;

use std::convert::Infallible;

use ::http::header;
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::{TryStreamExt, stream};
use http_body_util::{BodyExt, Full};
use mpx_protocol::Body;
use serde_json::Value;

fn body(bytes: impl Into<Bytes>) -> Body {
    Full::new(bytes.into())
        .map_err(|never: Infallible| match never {})
        .boxed()
}

fn request(path: &str, value: &Value) -> ::http::Request<Bytes> {
    ::http::Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Bytes::from(value.to_string()))
        .expect("test request is valid")
}

async fn collect_body(body: Body) -> Bytes {
    body.collect().await.expect("test body is valid").to_bytes()
}

async fn sse_data(bytes: &[u8]) -> Vec<String> {
    stream::iter([Ok::<_, Infallible>(bytes)])
        .eventsource()
        .map_ok(|event| event.data)
        .try_collect()
        .await
        .expect("response is valid SSE")
}
