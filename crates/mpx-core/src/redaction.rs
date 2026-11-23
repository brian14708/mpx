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
    BlockDelta, BoxError, Codec, ContentBlock, Endpoint, Error, EventStream, Extra, Request,
    ResponseBody, StreamEvent, Tool, ToolArguments,
};
use mpx_redact::{RedactionSession, Replacement, Restorer, Rules};
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
/// since the upstream can echo any replacement it received.
///
/// Call [`RedactionLayer::with_replacement_map`] to attach the issued
/// [`mpx_redact::ReplacementMap`] to each response's Rust extensions.
///
/// ```
/// use mpx_core::{IngressLayer, RedactionLayer, mpx_protocol::Codec, mpx_redact::{Matcher, Replacement, Rules}};
/// use tower::ServiceBuilder;
///
/// let codec = Codec::default();
/// let rule = (
///     Matcher::regex(r"\bID-[0-9]+\b")?,
///     Replacement::kind("SECRET")?,
/// );
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
    hmac_key: Arc<[u8]>,
    fallback: Option<Replacement>,
    attach_replacement_map: bool,
}

impl fmt::Debug for RedactionLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedactionLayer")
            .field("rules", &self.rules)
            .field("fallback", &self.fallback)
            .field("attach_replacement_map", &self.attach_replacement_map)
            .finish_non_exhaustive()
    }
}

impl RedactionLayer {
    /// Creates a redaction layer that rejects replacement collisions before dispatch.
    #[must_use]
    pub fn new(rules: Rules) -> Self {
        Self {
            rules: Arc::new(rules),
            hmac_key: Arc::from(DEFAULT_HMAC_KEY.as_bytes()),
            fallback: None,
            attach_replacement_map: false,
        }
    }

    /// Uses the supplied gateway secret for generated template replacements.
    ///
    /// The default is a fixed built-in key. Supply a deployment secret when
    /// replacements must use a private key or remain stable across deployments.
    #[must_use]
    pub fn with_hmac_key(mut self, key: impl AsRef<[u8]>) -> Self {
        self.hmac_key = Arc::from(key.as_ref());
        self
    }

    /// Rejects collisions with configured replacements. This is the default.
    #[must_use]
    pub fn with_strict(mut self) -> Self {
        self.fallback = None;
        self
    }

    /// Tries the supplied fallback when a replacement collides or an alias would
    /// join adjacent text. Fallbacks for permanent rules remain permanent.
    ///
    /// Choose a fallback that you expect to avoid collisions. An unavailable
    /// fallback rejects the request before dispatch.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Replacement) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// Attaches the issued [`mpx_redact::ReplacementMap`] to each response's extensions.
    ///
    /// The map holds the original values, including the originals of permanent
    /// masks in its forward entries. Only enable this when every later layer
    /// may see the unmasked values.
    #[must_use]
    pub const fn with_replacement_map(mut self) -> Self {
        self.attach_replacement_map = true;
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
        let attach_replacement_map = self.layer.attach_replacement_map;
        Box::pin(async move {
            restore_response(
                future.await.map_err(Into::into)?,
                session,
                attach_replacement_map,
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
    let mut session = RedactionSession::new().with_hmac_key(layer.hmac_key.as_ref());
    if let Some(session_id) = request.body().session_id.as_deref()
        && !session_id.trim().is_empty()
    {
        session = session.with_scope(session_id);
    }
    if let Some(fallback) = &layer.fallback {
        session = session.with_fallback(fallback.clone());
    }
    // Reserve across every field before masking any, so no issued replacement
    // collides with text already present elsewhere in the request.
    let mut visitor = RequestFieldVisitor {
        session: &mut session,
        rules: &layer.rules,
        pass: RequestPass::Reserve,
    };
    visitor.request(request.body_mut())?;
    visitor.pass = RequestPass::Mask;
    let changed = visitor.request(request.body_mut())?;
    session.clear_reserved_replacements();
    if changed {
        // Keep native replay when the normalized request was unchanged.
        Codec::disable_request_replay(request);
    }
    Ok(session)
}

enum RequestPass {
    Reserve,
    Mask,
}

struct RequestFieldVisitor<'a> {
    session: &'a mut RedactionSession,
    rules: &'a Rules,
    pass: RequestPass,
}

impl RequestFieldVisitor<'_> {
    fn request(&mut self, request: &mut Request) -> Result<bool, BoxError> {
        let mut changed = self.blocks(&mut request.instructions)?;
        for message in &mut request.messages {
            changed |= self.blocks(&mut message.content)?;
        }
        for tool in &mut request.tools {
            changed |= self.tool(tool)?;
        }
        for value in request.metadata.values_mut() {
            changed |= self.json_value(value)?;
        }
        if let Some(output_format) = &mut request.output_format {
            if let Some(schema) = &mut output_format.schema {
                changed |= self.json_value(schema)?;
            }
            changed |= self.extra(output_format.extra.as_mut())?;
        }
        changed |= self.extra(request.sampling.extra.as_mut())?;
        if let Some(reasoning) = &mut request.reasoning {
            changed |= self.extra(reasoning.extra.as_mut())?;
        }
        changed |= self.extra(request.extra.as_mut())?;
        Ok(changed)
    }

    fn blocks(&mut self, blocks: &mut [ContentBlock]) -> Result<bool, BoxError> {
        let mut changed = false;
        for block in blocks {
            changed |= self.block(block)?;
        }
        Ok(changed)
    }

    fn block(&mut self, block: &mut ContentBlock) -> Result<bool, BoxError> {
        match block {
            ContentBlock::Text { text, .. } => {
                let value = json_container(text);
                // Also reserve the serialized form of structured text blocks.
                if value.is_some() && matches!(self.pass, RequestPass::Reserve) {
                    self.session.reserve_replacements_in(text, self.rules);
                }
                if let Some(replacement) = self.encoded_json(text, value)? {
                    *text = replacement;
                    return Ok(true);
                }
                Ok(false)
            }
            ContentBlock::Refusal { text, .. } => self.string(text),
            ContentBlock::Reasoning { text, summary, .. } => {
                let mut changed = self.string(text)?;
                for item in summary {
                    changed |= self.string(item)?;
                }
                Ok(changed)
            }
            ContentBlock::ToolCall { arguments, .. } => {
                if let Some(replacement) =
                    self.encoded_json(arguments.as_str(), arguments.value().ok())?
                {
                    *arguments = replacement.into();
                    return Ok(true);
                }
                Ok(false)
            }
            ContentBlock::ToolResult { content, .. } => self.blocks(content),
            ContentBlock::WebSearchResult { query, results, .. } => {
                let mut changed = self.optional_string(query)?;
                for result in results {
                    changed |= self.optional_string(&mut result.title)?;
                    changed |= self.optional_string(&mut result.url)?;
                    changed |= self.optional_string(&mut result.text)?;
                }
                Ok(changed)
            }
            ContentBlock::Native { json, .. } => self.json_value(json),
            _ => Ok(false),
        }
    }

    fn tool(&mut self, tool: &mut Tool) -> Result<bool, BoxError> {
        match tool {
            Tool::Function {
                description,
                parameters,
                extra,
                ..
            } => {
                let mut changed = self.optional_string(description)?;
                changed |= self.json_value(parameters)?;
                changed |= self.extra(extra.as_mut())?;
                Ok(changed)
            }
            Tool::Custom {
                description,
                format,
                extra,
                ..
            } => {
                let mut changed = self.optional_string(description)?;
                if let Some(format) = format {
                    changed |= self.json_value(format)?;
                }
                changed |= self.extra(extra.as_mut())?;
                Ok(changed)
            }
            Tool::Hosted { json, .. } => self.json_value(json),
            _ => Ok(false),
        }
    }

    fn extra(&mut self, extra: Option<&mut Extra>) -> Result<bool, BoxError> {
        let Some(extra) = extra else {
            return Ok(false);
        };
        let mut changed = false;
        for value in extra.fields.values_mut() {
            changed |= self.json_value(value)?;
        }
        Ok(changed)
    }

    fn optional_string(&mut self, text: &mut Option<String>) -> Result<bool, BoxError> {
        text.as_mut().map_or(Ok(false), |text| self.string(text))
    }

    fn string(&mut self, text: &mut String) -> Result<bool, BoxError> {
        if let Some(replacement) = self.replacement(text)? {
            *text = replacement;
            return Ok(true);
        }
        Ok(false)
    }

    fn replacement(&mut self, text: &str) -> Result<Option<String>, BoxError> {
        match self.pass {
            RequestPass::Reserve => {
                self.session.reserve_replacements_in(text, self.rules);
                Ok(None)
            }
            RequestPass::Mask => {
                let masked = self.session.mask_reserved(text, self.rules)?;
                Ok((masked != text).then_some(masked))
            }
        }
    }

    fn encoded_json(
        &mut self,
        text: &str,
        value: Option<Value>,
    ) -> Result<Option<String>, BoxError> {
        let Some(mut value) = value else {
            return self.replacement(text);
        };
        if !self.json_value(&mut value)? {
            return Ok(None);
        }
        let encoded = serde_json::to_string(&value).map_err(|error| Error::Internal {
            message: error.to_string(),
        })?;
        Ok(Some(encoded))
    }

    fn json_value(&mut self, value: &mut Value) -> Result<bool, BoxError> {
        match value {
            Value::String(text) => self.string(text),
            Value::Array(values) => {
                let mut changed = false;
                for value in values {
                    changed |= self.json_value(value)?;
                }
                Ok(changed)
            }
            Value::Object(values) => {
                let mut changed = false;
                for value in values.values_mut() {
                    changed |= self.json_value(value)?;
                }
                Ok(changed)
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => Ok(false),
        }
    }
}

/// Parses text that holds a JSON object or array, the only shapes redacted structurally.
fn json_container(text: &str) -> Option<Value> {
    serde_json::from_str::<Value>(text)
        .ok()
        .filter(|value| matches!(value, Value::Object(_) | Value::Array(_)))
}

fn restore_response(
    response: http::Response<ResponseBody>,
    session: RedactionSession,
    attach_replacement_map: bool,
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
        ResponseBody::Stream(events) if nothing_to_restore => (ResponseBody::Stream(events), false),
        ResponseBody::Stream(events) => {
            (ResponseBody::Stream(restore_stream(session, events)), true)
        }
        body => (body, false),
    };
    let mut response = http::Response::from_parts(parts, body);
    if attach_replacement_map {
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
    *text = restored;
    true
}

fn restore_text_value(session: &RedactionSession, text: &mut String) -> Result<bool, BoxError> {
    if let Some(mut value) = json_container(text) {
        let changed = restore_json_value(session, &mut value);
        if !changed {
            return Ok(false);
        }
        let restored = serde_json::to_string(&value).map_err(|error| -> BoxError {
            Box::new(Error::Internal {
                message: error.to_string(),
            })
        })?;
        *text = restored;
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
        let changed = restored != text;
        return Ok((restored, changed));
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

/// Returns the block text that later deltas of the same kind extend.
fn block_text<'a>(
    block: &'a mut ContentBlock,
    session: &RedactionSession,
) -> Option<(TextKind, &'a mut String)> {
    match block {
        ContentBlock::Text { text, .. } => Some((TextKind::Text, text)),
        ContentBlock::Refusal { text, .. } => Some((TextKind::Refusal, text)),
        ContentBlock::Reasoning { text, summary, .. } => {
            if !text.is_empty() || summary.is_empty() {
                for item in summary {
                    restore_string(session, item);
                }
                Some((TextKind::Reasoning, text))
            } else {
                // Summary deltas extend the last summary item.
                let (last, earlier) = summary.split_last_mut()?;
                for item in earlier {
                    restore_string(session, item);
                }
                Some((TextKind::ReasoningSummary, last))
            }
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Text,
    Refusal,
    Reasoning,
    ReasoningSummary,
}

impl TextKind {
    const fn delta(self, text: String) -> BlockDelta {
        match self {
            Self::Text => BlockDelta::Text(text),
            Self::Refusal => BlockDelta::Refusal(text),
            Self::Reasoning => BlockDelta::Reasoning(text),
            Self::ReasoningSummary => BlockDelta::ReasoningSummary(text),
        }
    }
}

enum StreamField {
    Text(TextKind, Restorer),
    Arguments(String),
}

impl StreamField {
    const fn kind(&self) -> StreamFieldKind {
        match self {
            Self::Text(kind, _) => StreamFieldKind::Text(*kind),
            Self::Arguments(_) => StreamFieldKind::Arguments,
        }
    }

    fn finish(
        self,
        index: u32,
        session: &RedactionSession,
    ) -> Result<Option<StreamEvent>, BoxError> {
        let delta = match self {
            Self::Text(kind, restorer) => kind.delta(restorer.finish()),
            Self::Arguments(arguments) => {
                BlockDelta::ToolArguments(restore_json_text(session, &arguments)?.0)
            }
        };
        let empty = matches!(
            &delta,
            BlockDelta::Text(text)
                | BlockDelta::Refusal(text)
                | BlockDelta::Reasoning(text)
                | BlockDelta::ReasoningSummary(text)
                | BlockDelta::ToolArguments(text) if text.is_empty()
        );
        Ok((!empty).then_some(StreamEvent::BlockDelta { index, delta }))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamFieldKind {
    Text(TextKind),
    Arguments,
}

impl StreamFieldKind {
    const fn of_delta(delta: &BlockDelta) -> Option<Self> {
        match delta {
            BlockDelta::Text(_) => Some(Self::Text(TextKind::Text)),
            BlockDelta::Refusal(_) => Some(Self::Text(TextKind::Refusal)),
            BlockDelta::Reasoning(_) => Some(Self::Text(TextKind::Reasoning)),
            BlockDelta::ReasoningSummary(_) => Some(Self::Text(TextKind::ReasoningSummary)),
            BlockDelta::ToolArguments(_) => Some(Self::Arguments),
            _ => None,
        }
    }
}

/// Returns any buffered text that must precede this event.
fn flush_closing_field(
    event: &StreamEvent,
    fields: &mut BTreeMap<u32, StreamField>,
    session: &RedactionSession,
) -> Result<Option<StreamEvent>, BoxError> {
    let index = match event {
        StreamEvent::BlockStart { index, .. } | StreamEvent::BlockStop { index } => index,
        StreamEvent::BlockDelta { index, delta }
            if fields.get(index).is_some_and(|field| {
                StreamFieldKind::of_delta(delta).is_some_and(|kind| kind != field.kind())
            }) =>
        {
            index
        }
        _ => return Ok(None),
    };
    fields
        .remove(index)
        .map_or(Ok(None), |field| field.finish(*index, session))
}

fn restore_event(
    event: &mut StreamEvent,
    fields: &mut BTreeMap<u32, StreamField>,
    session: &RedactionSession,
) -> Result<bool, BoxError> {
    match event {
        StreamEvent::BlockStart { index, block } => {
            if let Some((kind, text)) = block_text(block, session) {
                let mut restorer = session.restorer();
                *text = restorer.push(text);
                fields.insert(*index, StreamField::Text(kind, restorer));
            } else if let ContentBlock::ToolCall { arguments, .. } = block {
                let initial = arguments.as_str();
                if !initial.is_empty() && initial != "{}" {
                    fields.insert(*index, StreamField::Arguments(initial.to_owned()));
                    *arguments = ToolArguments::new("");
                }
            } else {
                restore_block(session, block)?;
            }
            Ok(true)
        }
        StreamEvent::BlockDelta { index, delta } => {
            let Some(kind) = StreamFieldKind::of_delta(delta) else {
                return Ok(true);
            };
            let field = fields.entry(*index).or_insert_with(|| match kind {
                StreamFieldKind::Text(kind) => StreamField::Text(kind, session.restorer()),
                StreamFieldKind::Arguments => StreamField::Arguments(String::new()),
            });
            match (field, delta) {
                (
                    StreamField::Text(_, restorer),
                    BlockDelta::Text(text)
                    | BlockDelta::Refusal(text)
                    | BlockDelta::Reasoning(text)
                    | BlockDelta::ReasoningSummary(text),
                ) => {
                    *text = restorer.push(text);
                    Ok(!text.is_empty())
                }
                (StreamField::Arguments(arguments), BlockDelta::ToolArguments(text)) => {
                    arguments.push_str(text);
                    Ok(false)
                }
                _ => unreachable!("the previous field was flushed before its kind changed"),
            }
        }
        _ => Ok(true),
    }
}

fn protocol_error(error: &BoxError) -> Error {
    Error::Body {
        message: error.to_string(),
    }
}

fn restore_stream(session: RedactionSession, mut events: EventStream) -> EventStream {
    EventStream::from_stream(async_stream::try_stream! {
        let mut fields = BTreeMap::<u32, StreamField>::new();
        while let Some(item) = events.next().await {
            let mut event = item?;
            if matches!(event, StreamEvent::Error { .. }) {
                yield event;
                return;
            }
            if matches!(event, StreamEvent::MessageStart { .. } | StreamEvent::MessageStop { .. }) {
                for (index, field) in std::mem::take(&mut fields) {
                    if let Some(tail) = field.finish(index, &session).map_err(|error| protocol_error(&error))? {
                        yield tail;
                    }
                }
            } else if let Some(tail) = flush_closing_field(&event, &mut fields, &session)
                .map_err(|error| protocol_error(&error))?
            {
                yield tail;
            }
            if restore_event(&mut event, &mut fields, &session).map_err(|error| protocol_error(&error))? {
                yield event;
            }
        }
        for (index, field) in fields {
            if let Some(tail) = field.finish(index, &session).map_err(|error| protocol_error(&error))? {
                yield tail;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;

    use super::*;
    use mpx_redact::Matcher;

    fn replacement() -> Replacement {
        Replacement::kind("SECRET").unwrap()
    }

    #[tokio::test]
    async fn restores_interleaved_fields_and_flushes_truncated_tokens_at_block_end() {
        let mut session = RedactionSession::new();
        let rules = Rules::new([
            (Matcher::literal("first"), replacement()),
            (Matcher::literal("second"), replacement()),
        ]);
        let first = session.mask("first", &rules).unwrap();
        let second = session.mask("second", &rules).unwrap();
        let first_split = first.len() / 2;
        let second_split = second.len() / 2;
        let events = EventStream::from_events(vec![
            StreamEvent::BlockStart {
                index: 1,
                block: ContentBlock::text(format!("opening {}", &first[..first_split])),
            },
            StreamEvent::BlockStart {
                index: 2,
                block: ContentBlock::Refusal {
                    text: second[..second_split].to_owned(),
                    extra: None,
                },
            },
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::Text(first[first_split..].to_owned()),
            },
            StreamEvent::BlockDelta {
                index: 2,
                delta: BlockDelta::Refusal(format!("{} {{{{SECRET_", &second[second_split..])),
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
                    delta: BlockDelta::Refusal("{{SECRET_".to_owned())
                },
                StreamEvent::BlockStop { index: 2 },
                StreamEvent::BlockStop { index: 1 },
            ]
        );
    }

    #[tokio::test]
    async fn restores_streamed_reasoning_and_summaries() {
        let mut session = RedactionSession::new();
        let rules = Rules::new([(Matcher::literal("private"), replacement())]);
        let marker = session.mask("private", &rules).unwrap();
        let split = marker.len() / 2;
        let events = EventStream::from_events(vec![
            StreamEvent::BlockStart {
                index: 0,
                block: ContentBlock::Reasoning {
                    text: String::new(),
                    summary: Vec::new(),
                    seal: None,
                    redacted: false,
                    extra: None,
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Reasoning(format!("think {}", &marker[..split])),
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Reasoning(marker[split..].to_owned()),
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::ReasoningSummary(format!("sum {marker}")),
            },
            StreamEvent::BlockStop { index: 0 },
        ]);
        let actual = restore_stream(session, events).into_events().await.unwrap();
        let mut reasoning = String::new();
        let mut summary = String::new();
        for event in &actual {
            match event {
                StreamEvent::BlockDelta {
                    delta: BlockDelta::Reasoning(text),
                    ..
                } => reasoning.push_str(text),
                StreamEvent::BlockDelta {
                    delta: BlockDelta::ReasoningSummary(text),
                    ..
                } => summary.push_str(text),
                _ => {}
            }
        }
        assert_eq!(reasoning, "think private");
        assert_eq!(summary, "sum private");
        assert_eq!(actual.last(), Some(&StreamEvent::BlockStop { index: 0 }));
    }

    #[test]
    fn emits_safe_text_while_upstream_is_pending() {
        let events = EventStream::from_stream(async_stream::stream! {
            yield Ok(StreamEvent::BlockStart { index: 1, block: ContentBlock::text("safe {{SECRET_") });
            std::future::pending::<()>().await;
        });
        let mut session = RedactionSession::new();
        let rules = Rules::new([(Matcher::literal("private"), replacement())]);
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
