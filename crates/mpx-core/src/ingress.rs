use std::{
    error::Error as StdError,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use bytes::Bytes;
use http_body::Body as HttpBody;
use http_body_util::{BodyExt, LengthLimitError, Limited};
use mpx_protocol::{
    ApiError, ApiErrorKind, Body, BoxError, ClientContext, Codec, Diagnostics,
    Error as ProtocolError, Protocol,
};
use tower_layer::Layer;
use tower_service::Service;

use crate::ModelService;

/// Tower layer that collects client bodies and translates raw HTTP to and from IR.
#[derive(Debug, Clone)]
pub struct IngressLayer {
    codec: Codec,
    max_body_bytes: usize,
}

impl IngressLayer {
    /// Creates an ingress layer with a 16 MiB request-body limit.
    #[must_use]
    pub const fn new(codec: Codec) -> Self {
        Self {
            codec,
            max_body_bytes: 16 * 1024 * 1024,
        }
    }

    /// Sets the maximum collected client request size.
    #[must_use]
    pub const fn max_body_bytes(mut self, limit: usize) -> Self {
        self.max_body_bytes = limit;
        self
    }
}

impl<S> Layer<S> for IngressLayer {
    type Service = Ingress<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Ingress {
            inner,
            codec: self.codec.clone(),
            max_body_bytes: self.max_body_bytes,
        }
    }
}

/// Raw-HTTP ingress service produced by [`IngressLayer`].
#[derive(Debug, Clone)]
pub struct Ingress<S> {
    inner: S,
    codec: Codec,
    max_body_bytes: usize,
}

impl<S, B> Service<http::Request<B>> for Ingress<S>
where
    S: ModelService + Send + 'static,
    S::Future: Send + 'static,
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError>,
{
    type Response = http::Response<Body>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner
            .poll_ready(cx)
            .map(|result| result.map_err(Into::into))
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let codec = self.codec.clone();
        let replacement = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, replacement);
        let limit = self.max_body_bytes;
        Box::pin(async move {
            let (mut parts, body) = request.into_parts();
            let protocol = match codec.detect(&parts) {
                Ok(endpoint) => {
                    let protocol = endpoint.protocol;
                    parts.extensions.insert(endpoint);
                    protocol
                }
                Err(ProtocolError::InvalidRequest { protocol, .. }) => protocol,
                Err(_) => Protocol::ChatCompletions,
            };
            let response = serve(&codec, &mut inner, parts, body, limit).await;
            Ok(response.unwrap_or_else(|error| codec.encode_error(protocol, &error)))
        })
    }
}

/// Runs one request through decode, the inner service, and encode.
async fn serve<S, B>(
    codec: &Codec,
    inner: &mut S,
    parts: http::request::Parts,
    body: B,
    limit: usize,
) -> Result<http::Response<Body>, ApiError>
where
    S: ModelService,
    B: HttpBody<Data = Bytes>,
    B::Error: Into<BoxError>,
{
    let bytes = Limited::new(body, limit)
        .collect()
        .await
        .map_err(|error| {
            if error.is::<LengthLimitError>() {
                ProtocolError::BodyTooLarge { limit }
            } else {
                return ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("HTTP request body error: {error}"),
                );
            }
            .to_api_error()
        })?
        .to_bytes();
    let decoded = codec
        .decode_request(http::Request::from_parts(parts, bytes))
        .map_err(|error| error.to_api_error())?;
    let client = decoded
        .extensions()
        .get::<ClientContext>()
        .cloned()
        .ok_or_else(|| ApiError::new(ApiErrorKind::Internal, "client context was not captured"))?;
    let diagnostics = decoded
        .extensions()
        .get::<Diagnostics>()
        .cloned()
        .unwrap_or_default();
    let response = inner
        .call(decoded)
        .await
        .map_err(|error| api_error(&error.into()))?;
    let (mut parts, body) = response.into_parts();
    parts.extensions.insert(diagnostics);
    codec
        .encode_response(http::Response::from_parts(parts, body), &client)
        .await
        .map_err(|error| error.to_api_error())
}

fn api_error(error: &BoxError) -> ApiError {
    if let Some(error) = error.downcast_ref::<ProtocolError>() {
        return error.to_api_error();
    }
    if let Some(crate::Error::Protocol(error)) = error.downcast_ref::<crate::Error>() {
        return error.to_api_error();
    }
    if let Some(crate::Error::UpstreamResponse(error)) = error.downcast_ref::<crate::Error>() {
        return ApiError::builder()
            .kind(ApiErrorKind::BadResponse)
            .message(error.to_string())
            .code("bad_response")
            .build();
    }
    if let Some(crate::Error::Transport(error)) = error.downcast_ref::<crate::Error>() {
        let kind = if is_timeout(error.as_ref()) {
            ApiErrorKind::Timeout
        } else {
            ApiErrorKind::Upstream
        };
        return ApiError::builder()
            .kind(kind)
            .message(error.to_string())
            .code("transport")
            .build();
    }
    ApiError::new(ApiErrorKind::Internal, error.to_string())
}

fn is_timeout(error: &(dyn StdError + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::TimedOut)
        {
            return true;
        }
        current = error.source();
    }
    false
}
