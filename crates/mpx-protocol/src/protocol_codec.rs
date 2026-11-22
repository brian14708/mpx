use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use bon::Builder;
use bytes::Bytes;
use http::{HeaderMap, HeaderName};
use serde_json::{Map, Value};

use crate::{
    ApiError, Diagnostic, DiagnosticKind, Diagnostics, Endpoint, Error, LossPolicy, Protocol,
    Request, Response, Result, StreamDecoder, StreamEncoder,
};

/// Features a codec can provide, queryable without matching protocol names.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Builder)]
pub struct Capabilities {
    /// Whether a final assistant message can be used as a prefill.
    #[builder(default)]
    pub assistant_prefill: bool,
    /// Whether the provider supports continuation by response identifier.
    #[builder(default)]
    pub server_continuation: bool,
    /// Whether provider-hosted tools can produce `PauseTurn`.
    #[builder(default)]
    pub pause_turn: bool,
    /// Whether incremental responses are supported.
    #[builder(default = true)]
    pub streaming: bool,
    /// Whether client function tools are supported.
    #[builder(default = true)]
    pub tool_calls: bool,
    /// Whether reasoning controls and blocks are supported.
    #[builder(default)]
    pub reasoning: bool,
}

/// Client request state needed to encode its response faithfully.
#[derive(Debug, Clone)]
pub struct ClientContext {
    /// Detected client endpoint.
    pub endpoint: Endpoint,
    /// Whether Chat Completions asked for a final usage chunk.
    pub include_usage: bool,
    /// Original Responses fields echoed by `response.created` and `completed`.
    pub response_echo: Arc<Value>,
    model: String,
    request_body: Option<Arc<RequestBody>>,
    pub(crate) headers: HeaderMap,
}

#[derive(Debug)]
struct RequestBody {
    bytes: Bytes,
    json: OnceLock<Map<String, Value>>,
}

impl ClientContext {
    /// Creates minimal client state for a protocol or a configured endpoint.
    #[must_use]
    pub fn new(endpoint: impl Into<Endpoint>) -> Self {
        Self {
            endpoint: endpoint.into(),
            include_usage: false,
            response_echo: Arc::new(Value::Object(Map::default())),
            model: String::new(),
            request_body: None,
            headers: HeaderMap::new(),
        }
    }

    pub(crate) fn from_request(
        endpoint: Endpoint,
        request: &Request,
        wire: Bytes,
        headers: HeaderMap,
    ) -> Self {
        Self {
            include_usage: request.client_include_usage,
            model: request.model.clone(),
            request_body: Some(Arc::new(RequestBody {
                bytes: wire,
                json: OnceLock::new(),
            })),
            headers,
            ..Self::new(endpoint)
        }
    }

    pub(crate) fn request_model(&self) -> &str {
        if self.model.is_empty() {
            self.endpoint.model_from_path.as_deref().unwrap_or_default()
        } else {
            &self.model
        }
    }

    pub(crate) fn responses_echo(&self) -> Map<String, Value> {
        if let Some(request) = &self.request_body {
            return request
                .json
                .get_or_init(|| {
                    serde_json::from_slice::<Map<String, Value>>(&request.bytes).unwrap_or_default()
                })
                .clone();
        }
        self.response_echo.as_object().cloned().unwrap_or_default()
    }
}

/// Configuration passed to adapters by a [`crate::Codec`].
///
/// Set these options through [`crate::Codec::builder`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CodecOptions {
    /// Whether loss diagnostics are warnings or errors.
    pub loss_policy: LossPolicy,
    /// `max_tokens` supplied when Anthropic Messages omits its required value.
    pub default_max_tokens: u32,
    /// Whether to repair unmatched and split tool history for translated routes.
    pub repair_history: bool,
    /// Whether a cross-protocol reasoning seal may be carried in a call ID.
    pub seal_reasoning_in_tool_ids: bool,
    /// Maximum encoded size of a sealed tool-call identifier.
    pub max_sealed_id_bytes: usize,
    /// Whether unsupported rich content may become descriptive text.
    pub content_fallbacks: bool,
    /// Additional inbound headers safe to forward upstream.
    pub forward_headers: Vec<HeaderName>,
}

impl Default for CodecOptions {
    fn default() -> Self {
        Self {
            loss_policy: LossPolicy::default(),
            default_max_tokens: 8192,
            repair_history: true,
            seal_reasoning_in_tool_ids: true,
            max_sealed_id_bytes: 2048,
            content_fallbacks: true,
            forward_headers: Vec::new(),
        }
    }
}

/// Mutable state for one sans-IO encode or decode operation.
#[derive(Debug, Clone)]
pub struct CodecContext {
    /// Shared immutable codec configuration.
    pub options: Arc<CodecOptions>,
    /// Invocation-local shared diagnostics.
    pub diagnostics: Diagnostics,
}

impl Default for CodecContext {
    fn default() -> Self {
        Self {
            options: Arc::new(CodecOptions::default()),
            diagnostics: Diagnostics::new(),
        }
    }
}

impl CodecContext {
    /// Creates a context from configuration and a shared diagnostics handle.
    #[must_use]
    pub const fn new(options: Arc<CodecOptions>, diagnostics: Diagnostics) -> Self {
        Self {
            options,
            diagnostics,
        }
    }

    /// Records a loss, returning an error when [`LossPolicy::Reject`] is active.
    pub fn diagnose(
        &self,
        kind: DiagnosticKind,
        protocol: Protocol,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<()> {
        let path = path.into();
        let message = message.into();
        self.diagnostics.push(Diagnostic {
            kind,
            protocol,
            path: path.clone(),
            message: message.clone(),
        });
        if self.options.loss_policy == LossPolicy::Reject {
            return Err(Error::LossPolicy {
                message: format!("{path}: {message}"),
            });
        }
        Ok(())
    }
}

/// A synchronous, sans-IO protocol implementation.
///
/// Custom codecs can replace a built-in dialect through
/// [`crate::Codec::builder`]. Routes are registered separately with
/// [`crate::CodecBuilder::route`]. All methods are object-safe and
/// runtime-agnostic.
pub trait ProtocolCodec: fmt::Debug + Send + Sync + 'static {
    /// Returns the protocol family implemented by this codec.
    fn protocol(&self) -> Protocol;

    /// Returns capabilities used by translation and continuation policy.
    fn capabilities(&self) -> Capabilities {
        self.protocol().capabilities()
    }

    /// Whether the original client request can be replayed as sent.
    fn can_replay_request(&self, _request: &Request) -> bool {
        true
    }

    /// Decodes a complete JSON request body.
    fn decode_request(
        &self,
        endpoint: &Endpoint,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Request>;

    /// Encodes an IR request as a complete JSON upstream request.
    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>>;

    /// Decodes a complete JSON success response.
    fn decode_response(
        &self,
        parts: &http::response::Parts,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Response>;

    /// Decodes a non-success body without discarding its provider error data.
    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError;

    /// Encodes a complete success response for the original client.
    fn encode_response(
        &self,
        response: &Response,
        client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>>;

    /// Encodes an error in this protocol's wire shape.
    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes>;

    /// Creates a stateful decoder for upstream SSE frames.
    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;

    /// Creates a stateful encoder for the original client's SSE stream.
    fn stream_encoder(&self, client: &ClientContext) -> Box<dyn StreamEncoder>;

    /// Applies the protocol's inbound header forwarding policy.
    fn apply_header_policy(
        &self,
        source: Protocol,
        inbound: &HeaderMap,
        outbound: &mut HeaderMap,
        forward_headers: &[HeaderName],
    ) {
        if source == self.protocol() {
            for name in ["anthropic-beta", "openai-beta"] {
                copy_header(inbound, outbound, HeaderName::from_static(name));
            }
        }
        for name in forward_headers {
            if !is_forbidden_header(name.as_str()) {
                copy_header(inbound, outbound, name.clone());
            }
        }
    }
}

pub(crate) fn copy_header(source: &HeaderMap, target: &mut HeaderMap, name: HeaderName) {
    if let Some(value) = source.get(&name) {
        target.insert(name, value.clone());
    }
}

pub(crate) fn is_forbidden_header(name: &str) -> bool {
    matches!(
        name,
        "authorization"
            | "proxy-authorization"
            | "x-api-key"
            | "x-goog-api-key"
            | "cookie"
            | "set-cookie"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    ) || name.starts_with("x-gateway-")
}
