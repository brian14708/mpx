use std::{
    collections::BTreeMap,
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures_util::StreamExt;
use mpx_protocol::{
    BlockDelta, BoxError, Codec, ContentBlock, Diagnostic, DiagnosticKind, Diagnostics, Endpoint,
    Error, EventStream, Extra, Request, ResponseBody, StreamEvent, Tool, ToolArguments,
};
use mpx_redact::{PlaceholderTemplate, RedactionSession, Restorer, Rules};
use serde_json::Value;
use tower_layer::Layer;
use tower_service::Service;

use crate::ModelService;

const DEFAULT_HMAC_KEY: &str = "mpx-redact-static-key-v1";

/// Masks visible request text and restores echoed replacements in responses.
///
/// Place this layer after ingress and before upstream encoding. It covers visible text,
/// structured tool arguments and declarations, metadata, and nested tool results. Generated aliases and markers
/// use a versioned HMAC-SHA-256 derivation with the request session ID as scope. Raw passthrough
/// routes are rejected. The client must be trusted to receive the restored values,
/// since the upstream can echo any placeholder it received.
///
/// ```
/// use mpx_core::{IngressLayer, RedactionLayer, mpx_protocol::Codec, mpx_redact::{Mask, Matcher, Rule, Rules}};
/// use tower::ServiceBuilder;
///
/// let codec = Codec::default();
/// let rule = Rule::new(Matcher::regex(r"\bID-[0-9]+\b")?, Mask::tag());
/// let rules = Rules::new([rule]);
/// let layers = ServiceBuilder::new()
///     .layer(IngressLayer::new(codec))
///     .layer(RedactionLayer::new(rules));
/// // Compose with `.service(upstream)` to wrap an UpstreamService.
/// # let _ = layers;
/// # Ok::<(), mpx_core::mpx_redact::Error>(())
/// ```
#[derive(Clone)]
pub struct RedactionLayer {
    rules: Arc<Rules>,
    template: PlaceholderTemplate,
    hmac_key: Arc<[u8]>,
    strict: bool,
    record_replacements: bool,
}

impl fmt::Debug for RedactionLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedactionLayer")
            .field("rules", &self.rules)
            .field("template", &self.template)
            .field("strict", &self.strict)
            .field("record_replacements", &self.record_replacements)
            .finish_non_exhaustive()
    }
}

impl RedactionLayer {
    /// Creates a redaction layer using the supplied rules.
    #[must_use]
    pub fn new(rules: Rules) -> Self {
        Self {
            rules: Arc::new(rules),
            template: PlaceholderTemplate::default(),
            hmac_key: Arc::from(DEFAULT_HMAC_KEY.as_bytes()),
            strict: false,
            record_replacements: false,
        }
    }

    /// Uses the supplied gateway secret for generated aliases and markers.
    ///
    /// The default is a fixed built-in key. Supply a deployment secret when
    /// replacements must use a private key or remain stable across deployments.
    #[must_use]
    pub fn with_hmac_key(mut self, key: impl AsRef<[u8]>) -> Self {
        self.hmac_key = Arc::from(key.as_ref());
        self
    }

    /// Uses the given template for upstream masking and response restoration.
    #[must_use]
    pub fn with_placeholder_template(mut self, template: PlaceholderTemplate) -> Self {
        self.template = template;
        self
    }

    /// Disables collision probing so replacements stay independent of the
    /// other values in the request and remain stable for provider caches.
    #[must_use]
    pub const fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Attaches a [`mpx_redact::ReplacementMap`] to the response's Rust extensions.
    ///
    /// The map pairs the reversible tags and aliases with their original values.
    #[must_use]
    pub const fn with_replacement_map(mut self) -> Self {
        self.record_replacements = true;
        self
    }
}

impl<S> Layer<S> for RedactionLayer {
    type Service = Redaction<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Redaction {
            inner,
            layer: self.clone(),
        }
    }
}

/// Normalized model service produced by [`RedactionLayer`].
#[derive(Debug, Clone)]
pub struct Redaction<S> {
    inner: S,
    layer: RedactionLayer,
}

impl<S> Service<http::Request<Request>> for Redaction<S>
where
    S: ModelService,
    S::Future: Send + 'static,
{
    type Response = http::Response<ResponseBody>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, mut request: http::Request<Request>) -> Self::Future {
        let session = match mask_request(&self.layer, &mut request) {
            Ok(session) => session,
            Err(error) => {
                // Drop the ready handle and any reservation made by poll_ready.
                self.inner = self.inner.clone();
                return Box::pin(std::future::ready(Err(error)));
            }
        };
        let future = self.inner.call(request);
        let expose_replacements = self.layer.record_replacements;
        Box::pin(async move {
            restore_response(
                future.await.map_err(Into::into)?,
                session,
                expose_replacements,
            )
        })
    }
}

fn mask_request(
    layer: &RedactionLayer,
    request: &mut http::Request<Request>,
) -> Result<RedactionSession, BoxError> {
    if let Some(endpoint) = request.extensions().get::<Endpoint>()
        && endpoint.is_passthrough()
    {
        return Err(Error::InvalidRequest {
            protocol: endpoint.protocol,
            message: "redaction requires a normalized model endpoint".to_owned(),
        }
        .into());
    }
    let mut session = RedactionSession::new()
        .with_placeholder_template(layer.template.clone())
        .with_hmac_key(layer.hmac_key.as_ref());
    if let Some(session_id) = request.body().session_id.as_deref()
        && !session_id.trim().is_empty()
    {
        session = session.with_scope(session_id);
    }
    if layer.strict {
        session = session.with_strict();
    }
    // Reserve across every field before masking any, so no issued replacement
    // collides with text already present elsewhere in the request.
    reserve_request(&mut session, request.body(), &layer.rules);
    let changed = mask_request_body(&mut session, request.body_mut(), &layer.rules)?;
    if session.probe_count() > 0 {
        let protocol = request
            .extensions()
            .get::<Endpoint>()
            .map_or(mpx_protocol::Protocol::ChatCompletions, |endpoint| {
                endpoint.protocol
            });
        let diagnostics = request
            .extensions()
            .get::<Diagnostics>()
            .cloned()
            .unwrap_or_default();
        diagnostics.push(
            Diagnostic::builder()
            .kind(DiagnosticKind::Fallback)
            .protocol(protocol)
            .path("redaction.replacement")
            .message(format!(
                "{} replacement(s) required collision probing, which can reduce provider prompt-cache stability",
                session.probe_count()
            ))
            .build(),
        );
        request.extensions_mut().insert(diagnostics);
    }
    if changed {
        // Keep native replay when the normalized request was unchanged.
        Codec::disable_request_replay(request);
    }
    Ok(session)
}

fn reserve_request(session: &mut RedactionSession, request: &Request, rules: &Rules) {
    reserve_blocks(session, &request.instructions, rules);
    for message in &request.messages {
        reserve_blocks(session, &message.content, rules);
    }
    for tool in &request.tools {
        reserve_tool(session, tool, rules);
    }
    for value in request.metadata.values() {
        reserve_json_value(session, value, rules);
    }
    if let Some(output_format) = &request.output_format {
        if let Some(schema) = &output_format.schema {
            reserve_json_value(session, schema, rules);
        }
        reserve_extra(session, output_format.extra.as_ref(), rules);
    }
    reserve_extra(session, request.sampling.extra.as_ref(), rules);
    reserve_extra(
        session,
        request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.extra.as_ref()),
        rules,
    );
    reserve_extra(session, request.extra.as_ref(), rules);
}

fn reserve_blocks(session: &mut RedactionSession, blocks: &[ContentBlock], rules: &Rules) {
    for block in blocks {
        reserve_block(session, block, rules);
    }
}

fn reserve_block(session: &mut RedactionSession, block: &ContentBlock, rules: &Rules) {
    match block {
        ContentBlock::Text { text, .. } | ContentBlock::Refusal { text, .. } => {
            session.reserve_replacements_in(text, rules);
        }
        ContentBlock::Reasoning { text, summary, .. } => {
            session.reserve_replacements_in(text, rules);
            for item in summary {
                session.reserve_replacements_in(item, rules);
            }
        }
        ContentBlock::ToolCall { arguments, .. } => {
            reserve_json_text(session, arguments.as_str(), rules);
        }
        ContentBlock::ToolResult { content, .. } => reserve_blocks(session, content, rules),
        ContentBlock::WebSearchResult { query, results, .. } => {
            reserve_option(session, query.as_deref(), rules);
            for result in results {
                reserve_option(session, result.title.as_deref(), rules);
                reserve_option(session, result.url.as_deref(), rules);
                reserve_option(session, result.text.as_deref(), rules);
            }
        }
        ContentBlock::Native { json, .. } => reserve_json_value(session, json, rules),
        _ => {}
    }
}

fn reserve_tool(session: &mut RedactionSession, tool: &Tool, rules: &Rules) {
    match tool {
        Tool::Function {
            description,
            parameters,
            extra,
            ..
        } => {
            reserve_option(session, description.as_deref(), rules);
            reserve_json_value(session, parameters, rules);
            reserve_extra(session, extra.as_ref(), rules);
        }
        Tool::Custom {
            description,
            format,
            extra,
            ..
        } => {
            reserve_option(session, description.as_deref(), rules);
            if let Some(format) = format {
                reserve_json_value(session, format, rules);
            }
            reserve_extra(session, extra.as_ref(), rules);
        }
        Tool::Hosted { json, .. } => reserve_json_value(session, json, rules),
        _ => {}
    }
}

fn reserve_extra(session: &mut RedactionSession, extra: Option<&Extra>, rules: &Rules) {
    if let Some(extra) = extra {
        for value in extra.fields.values() {
            reserve_json_value(session, value, rules);
        }
    }
}

fn reserve_option(session: &mut RedactionSession, value: Option<&str>, rules: &Rules) {
    if let Some(value) = value {
        session.reserve_replacements_in(value, rules);
    }
}

fn reserve_json_text(session: &mut RedactionSession, text: &str, rules: &Rules) {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        reserve_json_value(session, &value, rules);
    } else {
        session.reserve_replacements_in(text, rules);
    }
}

fn reserve_json_value(session: &mut RedactionSession, value: &Value, rules: &Rules) {
    match value {
        Value::String(text) => session.reserve_replacements_in(text, rules),
        Value::Array(values) => {
            for value in values {
                reserve_json_value(session, value, rules);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                reserve_json_value(session, value, rules);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn mask_request_body(
    session: &mut RedactionSession,
    request: &mut Request,
    rules: &Rules,
) -> Result<bool, BoxError> {
    let mut changed = false;
    for block in &mut request.instructions {
        changed |= mask_block(session, block, rules)?;
    }
    for message in &mut request.messages {
        for block in &mut message.content {
            changed |= mask_block(session, block, rules)?;
        }
    }
    for tool in &mut request.tools {
        changed |= mask_tool(session, tool, rules)?;
    }
    for value in request.metadata.values_mut() {
        changed |= mask_json_value(session, value, rules)?;
    }
    if let Some(output_format) = &mut request.output_format {
        if let Some(schema) = &mut output_format.schema {
            changed |= mask_json_value(session, schema, rules)?;
        }
        changed |= mask_extra(session, output_format.extra.as_mut(), rules)?;
    }
    changed |= mask_extra(session, request.sampling.extra.as_mut(), rules)?;
    if let Some(reasoning) = &mut request.reasoning {
        changed |= mask_extra(session, reasoning.extra.as_mut(), rules)?;
    }
    changed |= mask_extra(session, request.extra.as_mut(), rules)?;
    Ok(changed)
}

fn mask_block(
    session: &mut RedactionSession,
    block: &mut ContentBlock,
    rules: &Rules,
) -> Result<bool, BoxError> {
    match block {
        ContentBlock::Text { text, .. } => mask_text_value(session, text, rules),
        ContentBlock::Refusal { text, .. } => mask_string(session, text, rules),
        ContentBlock::Reasoning { text, summary, .. } => {
            let mut changed = mask_string(session, text, rules)?;
            for item in summary {
                changed |= mask_string(session, item, rules)?;
            }
            Ok(changed)
        }
        ContentBlock::ToolCall { arguments, .. } => {
            let (masked, changed) = mask_json_text(session, arguments.as_str(), rules)?;
            if changed {
                *arguments = masked.into();
            }
            Ok(changed)
        }
        ContentBlock::ToolResult { content, .. } => {
            let mut changed = false;
            for block in content {
                changed |= mask_block(session, block, rules)?;
            }
            Ok(changed)
        }
        ContentBlock::WebSearchResult { query, results, .. } => {
            let mut changed = mask_option(session, query, rules)?;
            for result in results {
                changed |= mask_option(session, &mut result.title, rules)?;
                changed |= mask_option(session, &mut result.url, rules)?;
                changed |= mask_option(session, &mut result.text, rules)?;
            }
            Ok(changed)
        }
        ContentBlock::Native { json, .. } => mask_json_value(session, json, rules),
        ContentBlock::Image { .. } | ContentBlock::File { .. } | ContentBlock::Audio { .. } => {
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn mask_tool(
    session: &mut RedactionSession,
    tool: &mut Tool,
    rules: &Rules,
) -> Result<bool, BoxError> {
    match tool {
        Tool::Function {
            description,
            parameters,
            extra,
            ..
        } => {
            let mut changed = mask_option(session, description, rules)?;
            changed |= mask_json_value(session, parameters, rules)?;
            changed |= mask_extra(session, extra.as_mut(), rules)?;
            Ok(changed)
        }
        Tool::Custom {
            description,
            format,
            extra,
            ..
        } => {
            let mut changed = mask_option(session, description, rules)?;
            if let Some(format) = format {
                changed |= mask_json_value(session, format, rules)?;
            }
            changed |= mask_extra(session, extra.as_mut(), rules)?;
            Ok(changed)
        }
        Tool::Hosted { json, .. } => mask_json_value(session, json, rules),
        _ => Ok(false),
    }
}

fn mask_extra(
    session: &mut RedactionSession,
    extra: Option<&mut Extra>,
    rules: &Rules,
) -> Result<bool, BoxError> {
    let Some(extra) = extra else {
        return Ok(false);
    };
    let mut changed = false;
    for value in extra.fields.values_mut() {
        changed |= mask_json_value(session, value, rules)?;
    }
    Ok(changed)
}

fn mask_option(
    session: &mut RedactionSession,
    value: &mut Option<String>,
    rules: &Rules,
) -> Result<bool, BoxError> {
    value
        .as_mut()
        .map_or(Ok(false), |value| mask_string(session, value, rules))
}

fn mask_string(
    session: &mut RedactionSession,
    text: &mut String,
    rules: &Rules,
) -> Result<bool, BoxError> {
    let masked = session
        .mask_reserved(text, rules)
        .map_err(|error| -> BoxError { Box::new(error) })?;
    if masked == *text {
        return Ok(false);
    }
    text.clone_from(&masked);
    Ok(true)
}

fn mask_text_value(
    session: &mut RedactionSession,
    text: &mut String,
    rules: &Rules,
) -> Result<bool, BoxError> {
    if let Ok(mut value) = serde_json::from_str::<Value>(text)
        && matches!(value, Value::Object(_) | Value::Array(_))
    {
        let changed = mask_json_value(session, &mut value, rules)?;
        if !changed {
            return Ok(false);
        }
        let masked = serde_json::to_string(&value).map_err(|error| -> BoxError {
            Box::new(Error::Internal {
                message: error.to_string(),
            })
        })?;
        text.clone_from(&masked);
        return Ok(true);
    }
    mask_string(session, text, rules)
}

fn mask_json_text(
    session: &mut RedactionSession,
    text: &str,
    rules: &Rules,
) -> Result<(String, bool), BoxError> {
    let Ok(mut value) = serde_json::from_str::<Value>(text) else {
        let masked = session
            .mask_reserved(text, rules)
            .map_err(|error| -> BoxError { Box::new(error) })?;
        return Ok((masked.clone(), masked != text));
    };
    let changed = mask_json_value(session, &mut value, rules)?;
    if !changed {
        return Ok((text.to_owned(), false));
    }
    serde_json::to_string(&value)
        .map(|masked| (masked, true))
        .map_err(|error| -> BoxError {
            Box::new(Error::Internal {
                message: error.to_string(),
            })
        })
}

fn mask_json_value(
    session: &mut RedactionSession,
    value: &mut Value,
    rules: &Rules,
) -> Result<bool, BoxError> {
    match value {
        Value::String(text) => mask_string(session, text, rules),
        Value::Array(values) => {
            let mut changed = false;
            for value in values {
                changed |= mask_json_value(session, value, rules)?;
            }
            Ok(changed)
        }
        Value::Object(values) => {
            let mut changed = false;
            for value in values.values_mut() {
                changed |= mask_json_value(session, value, rules)?;
            }
            Ok(changed)
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(false),
    }
}

fn restore_response(
    response: http::Response<ResponseBody>,
    session: RedactionSession,
    expose_replacements: bool,
) -> Result<http::Response<ResponseBody>, BoxError> {
    if matches!(response.body(), ResponseBody::Passthrough(_)) {
        return Err(Error::Body {
            message: "redaction requires a normalized model response".to_owned(),
        }
        .into());
    }
    let replacements = session.replacement_map();
    let nothing_to_restore = replacements.entries().len() == 0;
    let (parts, body) = response.into_parts();
    let (body, changed) = match body {
        ResponseBody::Complete(mut response) => {
            let changed = if nothing_to_restore {
                false
            } else {
                restore_blocks(&mut response.content, &session)?
            };
            (ResponseBody::Complete(response), changed)
        }
        ResponseBody::Stream(events) => (
            ResponseBody::Stream(restore_stream(session, events)),
            !nothing_to_restore,
        ),
        body => (body, false),
    };
    let mut response = http::Response::from_parts(parts, body);
    if expose_replacements {
        response.extensions_mut().insert(replacements);
    }
    if changed {
        // Keep native replay when response restoration had no effect.
        Codec::disable_response_replay(&mut response);
    }
    Ok(response)
}

fn restore_blocks(
    blocks: &mut [ContentBlock],
    session: &RedactionSession,
) -> Result<bool, BoxError> {
    let mut changed = false;
    for block in blocks {
        changed |= restore_block(session, block)?;
    }
    Ok(changed)
}

fn restore_block(session: &RedactionSession, block: &mut ContentBlock) -> Result<bool, BoxError> {
    match block {
        ContentBlock::Text { text, .. } => restore_text_value(session, text),
        ContentBlock::Refusal { text, .. } => Ok(restore_string(session, text)),
        ContentBlock::Reasoning { text, summary, .. } => {
            let mut changed = restore_string(session, text);
            for item in summary {
                changed |= restore_string(session, item);
            }
            Ok(changed)
        }
        ContentBlock::ToolCall { arguments, .. } => {
            let (restored, changed) = restore_json_text(session, arguments.as_str())?;
            if changed {
                *arguments = restored.into();
            }
            Ok(changed)
        }
        ContentBlock::ToolResult { content, .. } => restore_blocks(content, session),
        ContentBlock::WebSearchResult { query, results, .. } => {
            let mut changed = restore_option(session, query);
            for result in results {
                changed |= restore_option(session, &mut result.title);
                changed |= restore_option(session, &mut result.url);
                changed |= restore_option(session, &mut result.text);
            }
            Ok(changed)
        }
        ContentBlock::Native { json, .. } => Ok(restore_json_value(session, json)),
        ContentBlock::Image { .. } | ContentBlock::File { .. } | ContentBlock::Audio { .. } => {
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn restore_string(session: &RedactionSession, text: &mut String) -> bool {
    let restored = session.restore(text);
    if restored == *text {
        return false;
    }
    text.clone_from(&restored);
    true
}

fn restore_text_value(session: &RedactionSession, text: &mut String) -> Result<bool, BoxError> {
    if let Ok(mut value) = serde_json::from_str::<Value>(text)
        && matches!(value, Value::Object(_) | Value::Array(_))
    {
        let changed = restore_json_value(session, &mut value);
        if !changed {
            return Ok(false);
        }
        let restored = serde_json::to_string(&value).map_err(|error| -> BoxError {
            Box::new(Error::Internal {
                message: error.to_string(),
            })
        })?;
        text.clone_from(&restored);
        return Ok(true);
    }
    Ok(restore_string(session, text))
}

fn restore_option(session: &RedactionSession, value: &mut Option<String>) -> bool {
    value
        .as_mut()
        .is_some_and(|value| restore_string(session, value))
}

fn restore_json_text(session: &RedactionSession, text: &str) -> Result<(String, bool), BoxError> {
    let Ok(mut value) = serde_json::from_str::<Value>(text) else {
        let restored = session.restore(text);
        return Ok((restored.clone(), restored != text));
    };
    let changed = restore_json_value(session, &mut value);
    if !changed {
        return Ok((text.to_owned(), false));
    }
    serde_json::to_string(&value)
        .map(|restored| (restored, true))
        .map_err(|error| -> BoxError {
            Box::new(Error::Internal {
                message: error.to_string(),
            })
        })
}

fn restore_json_value(session: &RedactionSession, value: &mut Value) -> bool {
    match value {
        Value::String(text) => restore_string(session, text),
        Value::Array(values) => {
            let mut changed = false;
            for value in values {
                changed |= restore_json_value(session, value);
            }
            changed
        }
        Value::Object(values) => {
            let mut changed = false;
            for value in values.values_mut() {
                changed |= restore_json_value(session, value);
            }
            changed
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

const fn block_text(block: &mut ContentBlock) -> Option<(TextKind, &mut String)> {
    match block {
        ContentBlock::Text { text, .. } => Some((TextKind::Text, text)),
        ContentBlock::Refusal { text, .. } => Some((TextKind::Refusal, text)),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Text,
    Refusal,
}

struct StreamField {
    kind: StreamFieldKind,
    restorer: Option<Restorer>,
    arguments: String,
}

impl StreamField {
    fn text(kind: TextKind, session: &RedactionSession) -> Self {
        Self {
            kind: StreamFieldKind::Text(kind),
            restorer: Some(session.restorer()),
            arguments: String::new(),
        }
    }

    const fn arguments(arguments: String) -> Self {
        Self {
            kind: StreamFieldKind::Arguments,
            restorer: None,
            arguments,
        }
    }

    const fn kind(&self) -> StreamFieldKind {
        self.kind
    }

    fn push_text(&mut self, text: &str) -> String {
        self.restorer
            .as_mut()
            .expect("text stream field has a restorer")
            .push(text)
    }

    fn push_arguments(&mut self, text: &str) {
        self.arguments.push_str(text);
    }

    fn finish(
        self,
        index: u32,
        session: &RedactionSession,
    ) -> Result<Option<StreamEvent>, BoxError> {
        let delta = match self.kind {
            StreamFieldKind::Text(kind) => {
                let text = self
                    .restorer
                    .expect("text stream field has a restorer")
                    .finish();
                if text.is_empty() {
                    return Ok(None);
                }
                match kind {
                    TextKind::Text => BlockDelta::Text(text),
                    TextKind::Refusal => BlockDelta::Refusal(text),
                }
            }
            StreamFieldKind::Arguments => {
                let (text, _) = restore_json_text(session, &self.arguments)?;
                if text.is_empty() {
                    return Ok(None);
                }
                BlockDelta::ToolArguments(text)
            }
        };
        Ok(Some(StreamEvent::BlockDelta { index, delta }))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamFieldKind {
    Text(TextKind),
    Arguments,
}

/// Ends the open field at `index`, returning its held-back tail if any.
fn take_tail(
    fields: &mut BTreeMap<u32, StreamField>,
    index: u32,
    session: &RedactionSession,
) -> Result<Option<StreamEvent>, BoxError> {
    fields
        .remove(&index)
        .map_or(Ok(None), |field| field.finish(index, session))
}

fn protocol_error(error: &BoxError) -> Error {
    Error::Body {
        message: error.to_string(),
    }
}

#[allow(clippy::too_many_lines)]
fn restore_stream(session: RedactionSession, mut events: EventStream) -> EventStream {
    EventStream::from_stream(async_stream::try_stream! {
        let mut fields: BTreeMap<u32, StreamField> = BTreeMap::new();
        while let Some(item) = events.next().await {
            let mut event = item?;
            match &mut event {
                StreamEvent::BlockStart { index, block } => {
                    if let Some(tail) = take_tail(&mut fields, *index, &session)
                        .map_err(|error| protocol_error(&error))?
                    {
                        yield tail;
                    }
                    if let Some((kind, text)) = block_text(block) {
                        let mut field = StreamField::text(kind, &session);
                        *text = field.push_text(text);
                        fields.insert(*index, field);
                    } else if let ContentBlock::ToolCall { arguments, .. } = block {
                        let initial = arguments.as_str().to_owned();
                        if !initial.is_empty() && initial != "{}" {
                            *arguments = ToolArguments::new("");
                            fields.insert(*index, StreamField::arguments(initial));
                        }
                    } else {
                        restore_block(&session, block)
                            .map_err(|error| protocol_error(&error))?;
                    }
                }
                StreamEvent::BlockDelta { index, delta } => {
                    let kind = match delta {
                        BlockDelta::Text(_) => StreamFieldKind::Text(TextKind::Text),
                        BlockDelta::Refusal(_) => StreamFieldKind::Text(TextKind::Refusal),
                        BlockDelta::ToolArguments(_) => StreamFieldKind::Arguments,
                        _ => {
                            yield event;
                            continue;
                        }
                    };
                    let tail = if fields.get(index).is_some_and(|field| field.kind() != kind) {
                        take_tail(&mut fields, *index, &session)
                            .map_err(|error| protocol_error(&error))?
                    } else {
                        None
                    };
                    if let Some(tail) = tail {
                        yield tail;
                    }
                    match delta {
                        BlockDelta::Text(text) | BlockDelta::Refusal(text) => {
                            let text_kind = match kind {
                                StreamFieldKind::Text(text_kind) => text_kind,
                                StreamFieldKind::Arguments => unreachable!(),
                            };
                            let field = fields
                                .entry(*index)
                                .or_insert_with(|| StreamField::text(text_kind, &session));
                            *text = field.push_text(text);
                            if text.is_empty() {
                                continue;
                            }
                        }
                        BlockDelta::ToolArguments(text) => {
                            let field = fields
                                .entry(*index)
                                .or_insert_with(|| StreamField::arguments(String::new()));
                            field.push_arguments(text);
                            text.clear();
                            continue;
                        }
                        _ => unreachable!(),
                    }
                }
                StreamEvent::BlockStop { index } => {
                    if let Some(tail) = take_tail(&mut fields, *index, &session)
                        .map_err(|error| protocol_error(&error))?
                    {
                        yield tail;
                    }
                }
                StreamEvent::MessageStart { .. } | StreamEvent::MessageStop { .. } => {
                    for (index, field) in std::mem::take(&mut fields) {
                        if let Some(tail) = field
                            .finish(index, &session)
                            .map_err(|error| protocol_error(&error))?
                        {
                            yield tail;
                        }
                    }
                }
                StreamEvent::Error { .. } => {
                    yield event;
                    return;
                }
                _ => {}
            }
            yield event;
        }
        for (index, field) in fields {
            if let Some(tail) = field
                .finish(index, &session)
                .map_err(|error| protocol_error(&error))?
            {
                yield tail;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;

    use super::*;
    use mpx_redact::{Mask, Matcher, Rule};

    #[tokio::test]
    async fn restores_interleaved_fields_and_flushes_truncated_tokens_at_block_end() {
        let mut session = RedactionSession::new();
        let rules = Rules::new([
            Rule::new(Matcher::Literal("first".into()), Mask::tag()),
            Rule::new(Matcher::Literal("second".into()), Mask::tag()),
        ]);
        let first = session.mask("first", &rules).unwrap();
        let second = session.mask("second", &rules).unwrap();
        let events = EventStream::from_events(vec![
            StreamEvent::BlockStart {
                index: 1,
                block: ContentBlock::text(format!("opening {}", &first[..10])),
            },
            StreamEvent::BlockStart {
                index: 2,
                block: ContentBlock::Refusal {
                    text: second[..20].to_owned(),
                    extra: None,
                },
            },
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::Text(first[10..].to_owned()),
            },
            StreamEvent::BlockDelta {
                index: 2,
                delta: BlockDelta::Refusal(format!("{} <|mpx", &second[20..])),
            },
            StreamEvent::BlockStop { index: 2 },
            StreamEvent::BlockStop { index: 1 },
        ]);
        let actual = restore_stream(session, events).into_events().await.unwrap();
        assert_eq!(
            actual,
            [
                StreamEvent::BlockStart {
                    index: 1,
                    block: ContentBlock::text("opening ")
                },
                StreamEvent::BlockStart {
                    index: 2,
                    block: ContentBlock::Refusal {
                        text: String::new(),
                        extra: None
                    }
                },
                StreamEvent::BlockDelta {
                    index: 1,
                    delta: BlockDelta::Text("first".to_owned())
                },
                StreamEvent::BlockDelta {
                    index: 2,
                    delta: BlockDelta::Refusal("second ".to_owned())
                },
                StreamEvent::BlockDelta {
                    index: 2,
                    delta: BlockDelta::Refusal("<|mpx".to_owned())
                },
                StreamEvent::BlockStop { index: 2 },
                StreamEvent::BlockStop { index: 1 },
            ]
        );
    }

    #[test]
    fn emits_safe_text_while_upstream_is_pending() {
        let events = EventStream::from_stream(async_stream::stream! {
            yield Ok(StreamEvent::BlockStart { index: 1, block: ContentBlock::text("safe <|mpx") });
            std::future::pending::<()>().await;
        });
        let mut session = RedactionSession::new();
        let rules = Rules::new([Rule::new(Matcher::Literal("private".into()), Mask::tag())]);
        let _ = session.mask("private", &rules).unwrap();
        let mut restored = restore_stream(session, events);
        assert_eq!(
            restored.next().now_or_never().unwrap().unwrap().unwrap(),
            StreamEvent::BlockStart {
                index: 1,
                block: ContentBlock::text("safe ")
            }
        );
        assert!(restored.next().now_or_never().is_none());
    }
}
