use std::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use bytes::Bytes;
use http::Uri;
use mpx_protocol::{Body, BoxError, Codec, Diagnostics, Endpoint, Protocol, Request, ResponseBody};
use tower_service::Service;

use crate::Error;

/// A selected upstream protocol and base URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upstream {
    /// Protocol used to encode requests and decode responses.
    pub protocol: Protocol,
    /// Base URI supplying scheme, authority, and an optional path prefix.
    pub base_uri: Uri,
}

impl Upstream {
    /// Creates an upstream target.
    #[must_use]
    pub const fn new(protocol: Protocol, base_uri: Uri) -> Self {
        Self { protocol, base_uri }
    }

    fn apply(&self, uri: &mut Uri) -> Result<(), Error> {
        if self.base_uri.query().is_some() {
            return Err(Error::InvalidUpstreamUri(
                "base URI must not contain a query".to_owned(),
            ));
        }
        let mut parts = self.base_uri.clone().into_parts();
        let prefix = self.base_uri.path().trim_end_matches('/');
        let suffix = uri
            .path_and_query()
            .map_or("/", http::uri::PathAndQuery::as_str);
        let path = if prefix.is_empty() {
            suffix.to_owned()
        } else {
            format!("{prefix}{suffix}")
        };
        parts.path_and_query = Some(path.parse().map_err(|error: http::uri::InvalidUri| {
            Error::InvalidUpstreamUri(error.to_string())
        })?);
        *uri =
            Uri::from_parts(parts).map_err(|error| Error::InvalidUpstreamUri(error.to_string()))?;
        Ok(())
    }
}

/// A [`crate::ModelService`] that dispatches through an injected HTTP transport.
#[derive(Debug, Clone)]
pub struct UpstreamService<C> {
    transport: C,
    codec: Codec,
    default_upstream: Option<Upstream>,
}

impl<C> UpstreamService<C> {
    /// Creates a service with no implicit upstream.
    ///
    /// Requests must then carry an [`Upstream`] extension.
    #[must_use]
    pub const fn new(transport: C, codec: Codec) -> Self {
        Self {
            transport,
            codec,
            default_upstream: None,
        }
    }

    /// Sets the upstream used when a request has no `Upstream` extension.
    #[must_use]
    pub fn default_upstream(mut self, upstream: Upstream) -> Self {
        self.default_upstream = Some(upstream);
        self
    }
}

impl<C, B> Service<http::Request<Request>> for UpstreamService<C>
where
    C: Service<http::Request<Body>, Response = http::Response<B>> + Clone,
    C::Future: Send + 'static,
    C::Error: Into<BoxError>,
    B: http_body::Body<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError>,
{
    type Response = http::Response<ResponseBody>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.transport.poll_ready(cx).map_err(transport_error)
    }

    fn call(&mut self, request: http::Request<Request>) -> Self::Future {
        let upstream = request
            .extensions()
            .get::<Upstream>()
            .or(self.default_upstream.as_ref());
        let diagnostics = request.extensions().get::<Diagnostics>().cloned();
        // Encode before the round-trip so the request IR is not held across it.
        let encoded = upstream.ok_or(Error::MissingUpstream).and_then(|upstream| {
            let mut encoded = self.codec.encode_request(&request, upstream.protocol)?;
            upstream.apply(encoded.uri_mut())?;
            let endpoint = encoded
                .extensions_mut()
                .remove::<Endpoint>()
                .unwrap_or_else(|| Endpoint::from(upstream.protocol));
            Ok((endpoint, encoded))
        });
        drop(request);
        let (endpoint, encoded) = match encoded {
            Ok(encoded) => encoded,
            Err(error) => {
                // Dropping the ready handle releases any reservation made by poll_ready.
                self.transport = self.transport.clone();
                return Box::pin(std::future::ready(Err(error)));
            }
        };
        let response = self.transport.call(encoded);
        let codec = self.codec.clone();
        Box::pin(async move {
            let mut response = response.await.map_err(transport_error)?;
            if let Some(diagnostics) = diagnostics {
                response.extensions_mut().insert(diagnostics);
            }
            codec
                .decode_response(response, endpoint)
                .await
                .map_err(Error::UpstreamResponse)
        })
    }
}

fn transport_error(error: impl Into<BoxError>) -> Error {
    Error::Transport(error.into())
}

impl fmt::Display for Upstream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.protocol, self.base_uri)
    }
}
