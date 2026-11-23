use std::{collections::HashMap, fmt, sync::Arc};

use bytes::Bytes;
use futures_util::TryStreamExt;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use http_body::{Body as HttpBody, Frame};
use http_body_util::{BodyDataStream, BodyExt, Full, StreamBody, combinators::BoxBody};
use serde_json::Value;
use sync_wrapper::SyncStream;

use crate::{
    ApiError, BoxError, ClientContext, CodecContext, Diagnostics, Endpoint, Error, LossPolicy,
    Protocol, ProtocolCodec, Request, ResponseBody, Result, Route,
    anthropic_messages::MessagesCodec,
    gemini::GenerateContentCodec,
    interactions::InteractionsCodec,
    json::nonempty,
    openai_chat::ChatCompletionsCodec,
    openai_responses::ResponsesCodec,
    protocol_codec::{CodecOptions, copy_header, is_forbidden_header},
    route::{Passthrough, detect_builtin},
    sse::{BufferedResponse, decode_event_stream, encode_event_body, prepend_buffered_response},
    stream::EventStream,
    validate::{repair_history, validate_request},
};

/// HTTP body type returned by [`Codec`] response encoders.
pub type Body = BoxBody<Bytes, Error>;

fn session_header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(name)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(|value| nonempty(value.trim()))
}

fn request_session_header(headers: &HeaderMap, protocol: Protocol) -> Option<String> {
    let native_headers: &[&str] = match protocol {
        Protocol::Responses | Protocol::ChatCompletions => &["session_id", "session-id"],
        Protocol::Messages => &["x-claude-code-session-id", "claude-code-session-id"],
        _ => &[],
    };
    std::iter::once("x-session-id")
        .chain(native_headers.iter().copied())
        .chain(["x-session-affinity", "x-deepseek-harness-session-id"])
        .find_map(|name| session_header(headers, name))
}

#[derive(Clone)]
struct OriginalBody<T> {
    protocol: Protocol,
    body: Bytes,
    value: Arc<T>,
}

#[derive(Default)]
struct CodecInner {
    codecs: HashMap<Protocol, Arc<dyn ProtocolCodec>>,
    options: Arc<CodecOptions>,
    routes: Vec<Route>,
}

impl fmt::Debug for CodecInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecInner")
            .field("protocols", &self.codecs.keys().collect::<Vec<_>>())
            .field("options", &self.options)
            .field("routes", &self.routes)
            .finish()
    }
}

/// A configured, clone-cheap registry and HTTP adapter for protocol codecs.
///
/// `Codec::default()` registers all built-in protocols. Use
/// [`Codec::builder`] to add routes, change translation policy, or replace a
/// dialect without changing middleware.
#[derive(Debug, Clone)]
pub struct Codec {
    inner: Arc<CodecInner>,
}

impl Codec {
    /// Starts a configured codec builder with all built-ins registered.
    #[must_use]
    pub fn builder() -> CodecBuilder {
        CodecBuilder::default()
    }

    /// Selects an endpoint without consuming the request body.
    pub fn detect(&self, parts: &http::request::Parts) -> Result<Endpoint> {
        let configured = parts.extensions.get::<Endpoint>().cloned().or_else(|| {
            self.inner
                .routes
                .iter()
                .find_map(|route| route.detect(parts))
        });
        if let Some(endpoint) = configured {
            return Ok(endpoint);
        }
        detect_builtin(parts)?.ok_or_else(|| Error::NotFound {
            path: parts.uri.path().to_owned(),
        })
    }

    fn protocol_codec(&self, protocol: Protocol) -> Result<&dyn ProtocolCodec> {
        self.inner
            .codecs
            .get(&protocol)
            .map(AsRef::as_ref)
            .ok_or_else(|| Error::InvalidRequest {
                protocol,
                message: "no codec is registered for this protocol".to_owned(),
            })
    }

    /// Decodes a collected client request and captures response echo context.
    pub fn decode_request(&self, request: http::Request<Bytes>) -> Result<http::Request<Request>> {
        let (mut parts, body) = request.into_parts();
        let mut endpoint = self.detect(&parts)?;
        let codec = self.protocol_codec(endpoint.protocol)?;
        let diagnostics = parts
            .extensions
            .get::<Diagnostics>()
            .cloned()
            .unwrap_or_default();
        let cx = CodecContext::new(Arc::clone(&self.inner.options), diagnostics.clone());
        let mut decoded = if let Some(route) = &endpoint.passthrough {
            if !route.allow_body && !body.is_empty() {
                return Err(Error::InvalidRequest {
                    protocol: endpoint.protocol,
                    message: "this route does not take a request body".to_owned(),
                });
            }
            Request::default()
        } else {
            codec.decode_request(&endpoint, &body, &cx)?
        };
        if let Some(model) = &endpoint.model_from_path {
            decoded.model.clone_from(model);
        }
        decoded.stream |= endpoint.stream;
        endpoint.stream = decoded.stream;
        let client = ClientContext::from_request(
            endpoint.clone(),
            &decoded,
            body.clone(),
            parts.headers.clone(),
        );
        parts.extensions.insert(endpoint);
        parts.extensions.insert(OriginalBody {
            protocol: client.endpoint.protocol,
            body,
            value: Arc::new(decoded.clone()),
        });
        // Preserve the body-derived ID in the replay baseline so a header override reaches the wire.
        decoded.session_id = request_session_header(&parts.headers, client.endpoint.protocol)
            .or(decoded.session_id)
            .or_else(|| match client.endpoint.protocol {
                Protocol::Responses => session_header(&parts.headers, "thread-id"),
                _ => None,
            });
        parts.extensions.insert(client);
        parts.extensions.insert(diagnostics);
        Ok(http::Request::from_parts(parts, decoded))
    }

    /// Discards the original wire body so encoding uses the edited request.
    ///
    /// Use this on normalized routes. Passthrough routes require their original bytes.
    pub fn disable_request_replay(request: &mut http::Request<Request>) {
        request.extensions_mut().remove::<OriginalBody<Request>>();
    }

    /// Encodes an IR request for a selected upstream protocol.
    ///
    /// The request is borrowed so retry and continuation layers can amend and
    /// encode it again. Headers are constructed from an allowlist; inbound
    /// credentials, cookies, hop-by-hop headers, and query strings are never
    /// passed through.
    pub fn encode_request(
        &self,
        request: &http::Request<Request>,
        protocol: Protocol,
    ) -> Result<http::Request<Body>> {
        let mut endpoint = Endpoint::from(protocol);
        let source = request.extensions().get::<Endpoint>();
        let source_protocol = source.map_or(protocol, |endpoint| endpoint.protocol);
        let codec = self.protocol_codec(protocol)?;
        let diagnostics = request
            .extensions()
            .get::<Diagnostics>()
            .cloned()
            .unwrap_or_default();
        let cx = CodecContext::new(Arc::clone(&self.inner.options), diagnostics.clone());
        let original = request.extensions().get::<OriginalBody<Request>>();
        let client = request.extensions().get::<ClientContext>();
        let mut encoded = if let Some(source) = source
            && let Some(route) = &source.passthrough
        {
            if source.protocol != protocol {
                return Err(Error::InvalidRequest {
                    protocol,
                    message: "a passthrough route requires an upstream of its own protocol"
                        .to_owned(),
                });
            }
            endpoint = source.clone();
            encode_passthrough(route, source, original, client)?
        } else {
            let mut prepared = request.body().clone();
            if prepared.continuation.is_some()
                && (!codec.capabilities().server_continuation || source_protocol != protocol)
            {
                return Err(Error::InvalidRequest {
                    protocol,
                    message: "server-side continuation requires its origin provider and protocol"
                        .to_owned(),
                });
            }
            if source_protocol != protocol && self.inner.options.repair_history {
                repair_history(&mut prepared, protocol, &diagnostics);
            }
            validate_request(&prepared, protocol, source_protocol, &diagnostics)?;
            let mut encoded = codec.encode_request(&prepared, &cx)?;
            if let Some(original) = original
                && original.protocol == protocol
                && codec.can_replay_request(&original.value)
            {
                if original.value.as_ref() == &prepared {
                    *encoded.body_mut() = original.body.clone();
                } else {
                    let mut options = (*self.inner.options).clone();
                    options.loss_policy = LossPolicy::Warn;
                    let baseline = CodecContext::new(Arc::new(options), Diagnostics::new());
                    if let Ok(previous) = codec.encode_request(&original.value, &baseline) {
                        *encoded.body_mut() = if previous.body() == encoded.body() {
                            original.body.clone()
                        } else {
                            preserve_original_json(&original.body, previous.body(), encoded.body())?
                        };
                    }
                }
            }
            encoded
        };
        if let Some(client) = client {
            codec.apply_header_policy(
                source_protocol,
                &client.headers,
                encoded.headers_mut(),
                &self.inner.options.forward_headers,
            );
        }
        let session_header = HeaderName::from_static("x-session-id");
        if self.inner.options.forward_headers.contains(&session_header) {
            let session_id = request
                .body()
                .session_id
                .as_deref()
                .filter(|value| !value.is_empty())
                .and_then(|value| HeaderValue::from_str(value).ok())
                .filter(|value| value.to_str().is_ok());
            if let Some(session_id) = session_id {
                encoded.headers_mut().insert(session_header, session_id);
            } else {
                encoded.headers_mut().remove(session_header);
            }
        }
        encoded.extensions_mut().insert(endpoint);
        encoded.extensions_mut().insert(diagnostics);
        Ok(encoded.map(bytes_body))
    }

    /// Decodes an upstream response using a protocol or the endpoint attached by
    /// [`Codec::encode_request`]. Use that endpoint for native passthrough routes.
    pub async fn decode_response<B>(
        &self,
        response: http::Response<B>,
        endpoint: impl Into<Endpoint>,
    ) -> Result<http::Response<ResponseBody>>
    where
        B: HttpBody<Data = Bytes> + Send + 'static,
        B::Error: Into<BoxError>,
    {
        let endpoint = endpoint.into();
        let protocol = endpoint.protocol;
        let codec = self.protocol_codec(protocol)?;
        let (mut parts, body) = response.into_parts();
        if !parts.status.is_success() {
            let bytes = body.collect().await.map_err(box_body_error)?.to_bytes();
            let error = codec.decode_error(&parts, &bytes);
            parts.extensions.insert(OriginalBody {
                protocol,
                body: bytes,
                value: Arc::new(error.clone()),
            });
            return Ok(http::Response::from_parts(
                parts,
                ResponseBody::Error(error),
            ));
        }
        if endpoint.passthrough.is_some() {
            let frames = BodyDataStream::new(body)
                .map_ok(Frame::data)
                .map_err(box_body_error);
            return Ok(http::Response::from_parts(
                parts,
                ResponseBody::Passthrough(BodyExt::boxed(StreamBody::new(SyncStream::new(frames)))),
            ));
        }
        let streaming = parts
            .headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.trim_start().starts_with("text/event-stream"));
        let diagnostics = parts
            .extensions
            .get::<Diagnostics>()
            .cloned()
            .unwrap_or_default();
        let cx = CodecContext::new(Arc::clone(&self.inner.options), diagnostics);
        if streaming {
            let buffered = BufferedResponse::new(protocol);
            let stream = decode_event_stream(body, codec.stream_decoder(), cx, buffered.clone());
            parts.extensions.insert(buffered);
            Ok(http::Response::from_parts(
                parts,
                ResponseBody::Stream(stream),
            ))
        } else {
            let bytes = body.collect().await.map_err(box_body_error)?.to_bytes();
            let cx = cx;
            let response = codec.decode_response(&parts, &bytes, &cx)?;
            parts.extensions.insert(OriginalBody {
                protocol,
                body: bytes,
                value: Arc::new(response.clone()),
            });
            Ok(http::Response::from_parts(
                parts,
                ResponseBody::Complete(response),
            ))
        }
    }

    /// Encodes a normalized response for the original client protocol.
    pub async fn encode_response(
        &self,
        response: http::Response<ResponseBody>,
        client: &ClientContext,
    ) -> Result<http::Response<Body>> {
        let codec = self.protocol_codec(client.endpoint.protocol)?;
        let (mut parts, body) = response.into_parts();
        let diagnostics = parts
            .extensions
            .get::<Diagnostics>()
            .cloned()
            .unwrap_or_default();
        let cx = CodecContext::new(Arc::clone(&self.inner.options), diagnostics);
        match body {
            ResponseBody::Passthrough(body) => {
                if client.endpoint.passthrough.is_none() {
                    return Err(Error::InvalidRequest {
                        protocol: client.endpoint.protocol,
                        message: "a passthrough response requires a passthrough client endpoint"
                            .to_owned(),
                    });
                }
                let mut response = http::Response::new(body);
                *response.status_mut() = parts.status;
                *response.extensions_mut() = parts.extensions;
                if let Some(content_type) = parts.headers.remove(header::CONTENT_TYPE) {
                    response
                        .headers_mut()
                        .insert(header::CONTENT_TYPE, content_type);
                }
                Ok(response)
            }
            ResponseBody::Complete(response) => {
                let original = parts
                    .extensions
                    .get::<OriginalBody<crate::Response>>()
                    .filter(|original| {
                        original.protocol == client.endpoint.protocol
                            && original.value.as_ref() == &response
                    })
                    .map(|original| original.body.clone());
                if client.endpoint.stream {
                    let complete_json = original.map_or_else(
                        || {
                            codec
                                .encode_response(&response, client, &cx)
                                .map(http::Response::into_body)
                        },
                        Ok,
                    )?;
                    let events = EventStream::from_events(response.to_stream_events());
                    let body = encode_event_body(events, codec.stream_encoder(client), cx);
                    return Ok(stream_response(
                        parts,
                        prepend_buffered_response(body, client.endpoint.protocol, &complete_json),
                    ));
                }
                if let Some(original) = original {
                    return Ok(replay_json_response(parts, original));
                }
                let encoded = codec.encode_response(&response, client, &cx)?;
                let (mut encoded_parts, bytes) = encoded.into_parts();
                encoded_parts.extensions = parts.extensions;
                Ok(http::Response::from_parts(encoded_parts, bytes_body(bytes)))
            }
            ResponseBody::Error(error) => {
                if let Some(original) = parts.extensions.get::<OriginalBody<ApiError>>()
                    && original.protocol == client.endpoint.protocol
                    && original.value.as_ref() == &error
                {
                    let bytes = original.body.clone();
                    let content_type = parts.headers.get(header::CONTENT_TYPE).cloned();
                    let mut response = replay_json_response(parts, bytes);
                    if let Some(content_type) = content_type {
                        response
                            .headers_mut()
                            .insert(header::CONTENT_TYPE, content_type);
                    }
                    return Ok(response);
                }
                Ok(encode_response_error(codec, parts, &error))
            }
            ResponseBody::Stream(mut events) => {
                if !client.endpoint.stream {
                    let mut accumulator = crate::ResponseAccumulator::new();
                    while let Some(event) = events.try_next().await? {
                        accumulator.push(event);
                    }
                    if let Some(error) = accumulator.error().cloned() {
                        return Ok(encode_response_error(codec, parts, &error));
                    }
                    if let Some(buffered) = parts.extensions.remove::<BufferedResponse>()
                        && buffered.protocol == client.endpoint.protocol
                        && let Some(bytes) = buffered.body.get()
                        && codec.decode_response(&parts, bytes, &cx).is_ok()
                    {
                        return Ok(replay_json_response(parts, bytes.clone()));
                    }
                    let encoded = codec.encode_response(&accumulator.finish(), client, &cx)?;
                    let (mut encoded_parts, bytes) = encoded.into_parts();
                    encoded_parts.extensions = parts.extensions;
                    return Ok(http::Response::from_parts(encoded_parts, bytes_body(bytes)));
                }
                let body = encode_event_body(events, codec.stream_encoder(client), cx);
                Ok(stream_response(parts, body))
            }
        }
    }

    /// Discards wire body snapshots so encoding uses the edited response or events.
    pub fn disable_response_replay(response: &mut http::Response<ResponseBody>) {
        let extensions = response.extensions_mut();
        extensions.remove::<OriginalBody<crate::Response>>();
        extensions.remove::<OriginalBody<ApiError>>();
        extensions.remove::<BufferedResponse>();
    }

    /// Encodes an error in a selected protocol, useful at ingress boundaries.
    #[must_use]
    pub fn encode_error(&self, protocol: Protocol, error: &ApiError) -> http::Response<Body> {
        if let Some(codec) = self.inner.codecs.get(&protocol) {
            return codec.encode_error(error).map(bytes_body);
        }
        let mut response = http::Response::new(bytes_body(Bytes::from(error.message.clone())));
        *response.status_mut() = error.status();
        response
    }
}

fn encode_passthrough(
    route: &Passthrough,
    endpoint: &Endpoint,
    original: Option<&OriginalBody<Request>>,
    client: Option<&ClientContext>,
) -> Result<http::Request<Bytes>> {
    if !route.path.path().starts_with('/') {
        return Err(Error::InvalidRequest {
            protocol: endpoint.protocol,
            message: "a passthrough target must have an origin-relative path".to_owned(),
        });
    }
    let uri = http::Uri::builder()
        .path_and_query(route.path.clone())
        .build()
        .map_err(Error::http)?;
    let mut request = http::Request::builder()
        .method(route.method.clone())
        .uri(uri)
        .header(
            header::ACCEPT,
            if endpoint.stream {
                "text/event-stream"
            } else {
                "application/json"
            },
        )
        .body(original.map_or_else(Bytes::new, |original| original.body.clone()))
        .map_err(Error::http)?;
    if let Some(client) = client {
        for name in [header::ACCEPT, header::CONTENT_TYPE] {
            copy_header(&client.headers, request.headers_mut(), name);
        }
    }
    Ok(request)
}

fn replay_json_response(parts: http::response::Parts, bytes: Bytes) -> http::Response<Body> {
    let mut response = json_response(parts.status, bytes_body(bytes));
    *response.extensions_mut() = parts.extensions;
    response
}

pub(crate) fn json_response<B>(status: StatusCode, body: B) -> http::Response<B> {
    let mut response = http::Response::new(body);
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

fn preserve_original_json(original: &[u8], previous: &[u8], updated: &[u8]) -> Result<Bytes> {
    let (Ok(mut original), Ok(previous), Ok(updated)) = (
        serde_json::from_slice::<Value>(original),
        serde_json::from_slice::<Value>(previous),
        serde_json::from_slice::<Value>(updated),
    ) else {
        return Ok(Bytes::copy_from_slice(updated));
    };
    apply_json_changes(&mut original, &previous, &updated);
    serde_json::to_vec(&original)
        .map(Bytes::from)
        .map_err(|error| Error::Internal {
            message: error.to_string(),
        })
}

fn apply_json_changes(original: &mut Value, previous: &Value, updated: &Value) {
    if previous == updated {
        return;
    }
    match (original, previous, updated) {
        (Value::Object(original), Value::Object(previous), Value::Object(updated)) => {
            for key in previous.keys() {
                if !updated.contains_key(key) {
                    original.remove(key);
                }
            }
            for (key, new) in updated {
                match (original.get_mut(key), previous.get(key)) {
                    (Some(actual), Some(old)) => apply_json_changes(actual, old, new),
                    (_, Some(old)) if old == new => {}
                    _ => {
                        original.insert(key.clone(), new.clone());
                    }
                }
            }
        }
        (Value::Array(original), Value::Array(previous), Value::Array(updated))
            if original.len() == previous.len() && previous.len() == updated.len() =>
        {
            for ((actual, old), new) in original.iter_mut().zip(previous).zip(updated) {
                apply_json_changes(actual, old, new);
            }
        }
        (original, _, updated) => original.clone_from(updated),
    }
}

fn encode_response_error(
    codec: &dyn ProtocolCodec,
    parts: http::response::Parts,
    error: &ApiError,
) -> http::Response<Body> {
    let (mut encoded_parts, bytes) = codec.encode_error(error).into_parts();
    encoded_parts.status = if parts.status.is_success() {
        error.status()
    } else {
        parts.status
    };
    encoded_parts.extensions = parts.extensions;
    http::Response::from_parts(encoded_parts, bytes_body(bytes))
}

fn stream_response(mut parts: http::response::Parts, body: Body) -> http::Response<Body> {
    let connection_headers: Vec<_> = parts
        .headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::try_from(name.trim()).ok())
        .collect();
    for name in connection_headers {
        parts.headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        header::CONTENT_ENCODING,
        header::CONTENT_LENGTH,
        header::CONTENT_RANGE,
        header::ETAG,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        parts.headers.remove(name);
    }
    for name in [
        "keep-alive",
        "proxy-connection",
        "content-md5",
        "digest",
        "content-digest",
        "repr-digest",
    ] {
        parts.headers.remove(name);
    }
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    parts
        .headers
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    http::Response::from_parts(parts, body)
}

impl Default for Codec {
    fn default() -> Self {
        CodecBuilder::default().build()
    }
}

/// Fluent configuration for [`Codec`].
#[derive(Debug, Default)]
pub struct CodecBuilder {
    codecs: Vec<Arc<dyn ProtocolCodec>>,
    options: CodecOptions,
    routes: Vec<Route>,
}

impl CodecBuilder {
    /// Replaces a built-in dialect or registers a custom implementation.
    #[must_use]
    pub fn codec(mut self, codec: Arc<dyn ProtocolCodec>) -> Self {
        self.codecs.push(codec);
        self
    }

    /// Adds a [`Route::alias`] or a custom matcher before the built-in routes.
    #[must_use]
    pub fn route(mut self, route: Route) -> Self {
        self.routes.push(route);
        self
    }

    /// Sets the required Anthropic `max_tokens` fallback.
    #[must_use]
    pub const fn default_max_tokens(mut self, value: u32) -> Self {
        self.options.default_max_tokens = value;
        self
    }

    /// Sets translation loss handling.
    #[must_use]
    pub const fn loss_policy(mut self, value: LossPolicy) -> Self {
        self.options.loss_policy = value;
        self
    }

    /// Enables or disables cross-protocol history repair.
    #[must_use]
    pub const fn repair_history(mut self, value: bool) -> Self {
        self.options.repair_history = value;
        self
    }

    /// Enables or disables reasoning seals in translated tool-call IDs.
    #[must_use]
    pub const fn seal_reasoning_in_tool_ids(mut self, value: bool) -> Self {
        self.options.seal_reasoning_in_tool_ids = value;
        self
    }

    /// Sets the maximum sealed tool-call ID size.
    #[must_use]
    pub const fn max_sealed_id_bytes(mut self, value: usize) -> Self {
        self.options.max_sealed_id_bytes = value;
        self
    }

    /// Enables or disables descriptive text content fallbacks.
    #[must_use]
    pub const fn content_fallbacks(mut self, value: bool) -> Self {
        self.options.content_fallbacks = value;
        self
    }

    /// Adds a non-credential header to the upstream forwarding allowlist.
    /// For `x-session-id`, the outgoing value comes from [`Request::session_id`].
    #[must_use]
    pub fn forward_header(mut self, name: HeaderName) -> Self {
        if !is_forbidden_header(name.as_str()) {
            self.options.forward_headers.push(name);
        }
        self
    }

    /// Builds the immutable, clone-cheap codec front.
    #[must_use]
    pub fn build(self) -> Codec {
        let mut codecs: HashMap<Protocol, Arc<dyn ProtocolCodec>> = HashMap::new();
        for codec in [
            Arc::new(ChatCompletionsCodec) as Arc<dyn ProtocolCodec>,
            Arc::new(ResponsesCodec),
            Arc::new(MessagesCodec),
            Arc::new(GenerateContentCodec),
            Arc::new(InteractionsCodec),
        ] {
            codecs.insert(codec.protocol(), codec);
        }
        for codec in self.codecs {
            codecs.insert(codec.protocol(), codec);
        }
        Codec {
            inner: Arc::new(CodecInner {
                codecs,
                options: Arc::new(self.options),
                routes: self.routes,
            }),
        }
    }
}

fn bytes_body(bytes: Bytes) -> Body {
    Full::new(bytes).map_err(|never| match never {}).boxed()
}

pub(crate) fn box_body_error(error: impl Into<BoxError>) -> Error {
    let error = error.into();
    Error::Body {
        message: error.to_string(),
    }
}
