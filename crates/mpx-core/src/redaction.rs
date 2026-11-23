use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures_util::StreamExt;
use mpx_protocol::{
    BlockDelta, BoxError, Codec, ContentBlock, Diagnostic, DiagnosticKind, Diagnostics, Endpoint,
    Error, EventStream, Request, ResponseBody, StreamEvent,
};
use mpx_redact::{PlaceholderTemplate, RedactionSession, Restorer, Rules};
use tower_layer::Layer;
use tower_service::Service;

use crate::ModelService;

/// Masks visible request text and restores echoed replacements in responses.
///
/// Place this layer after ingress and before upstream encoding. It covers text
/// and refusals, including those in nested tool results. Raw passthrough routes
/// are rejected. The client must be trusted to receive the restored values,
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
#[derive(Debug, Clone)]
pub struct RedactionLayer {
    rules: Arc<Rules>,
    template: PlaceholderTemplate,
    strict: bool,
    record_replacements: bool,
}

impl RedactionLayer {
    /// Creates a redaction layer using the supplied rules.
    #[must_use]
    pub fn new(rules: Rules) -> Self {
        Self {
            rules: Arc::new(rules),
            template: PlaceholderTemplate::default(),
            strict: false,
            record_replacements: false,
        }
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
    let mut session = RedactionSession::new().with_placeholder_template(layer.template.clone());
    if layer.strict {
        session = session.with_strict();
    }
    // Reserve across every field before masking any, so no issued replacement
    // collides with text already present elsewhere in the request.
    request.body_mut().visit_blocks_mut(&mut |block| {
        if let Some(text) = visible_text(block) {
            session.reserve_replacements_in(text, &layer.rules);
        }
    });
    let mut result = Ok(());
    request.body_mut().visit_blocks_mut(&mut |block| {
        if result.is_ok()
            && let Some(text) = visible_text(block)
        {
            result = session
                .mask_reserved(text, &layer.rules)
                .map(|masked| *text = masked);
        }
    });
    result?;
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
    Codec::disable_request_replay(request);
    Ok(session)
}

fn restore_response(
    mut response: http::Response<ResponseBody>,
    session: RedactionSession,
    expose_replacements: bool,
) -> Result<http::Response<ResponseBody>, BoxError> {
    if matches!(response.body(), ResponseBody::Passthrough(_)) {
        return Err(Error::Body {
            message: "redaction requires a normalized model response".to_owned(),
        }
        .into());
    }
    Codec::disable_response_replay(&mut response);
    let replacements = session.replacement_map();
    let nothing_to_restore = replacements.entries().len() == 0;
    if expose_replacements {
        response.extensions_mut().insert(replacements);
    }
    let (parts, body) = response.into_parts();
    let body = match body {
        ResponseBody::Complete(mut response) => {
            if !nothing_to_restore {
                response.visit_blocks_mut(&mut |block| restore_text(&session, block));
            }
            ResponseBody::Complete(response)
        }
        ResponseBody::Stream(events) => ResponseBody::Stream(restore_stream(session, events)),
        body => body,
    };
    Ok(http::Response::from_parts(parts, body))
}

fn restore_text(session: &RedactionSession, block: &mut ContentBlock) {
    if let Some(text) = visible_text(block) {
        *text = session.restore(text);
    }
}

/// The client-visible text of a block, which is what redaction covers.
fn visible_text(block: &mut ContentBlock) -> Option<&mut String> {
    block_text(block).map(|(_, text)| text)
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
    kind: TextKind,
    restorer: Restorer,
}

impl StreamField {
    fn new(kind: TextKind, session: &RedactionSession) -> Self {
        Self {
            kind,
            restorer: session.restorer(),
        }
    }

    fn finish(self, index: u32) -> Option<StreamEvent> {
        let text = self.restorer.finish();
        if text.is_empty() {
            return None;
        }
        let delta = match self.kind {
            TextKind::Text => BlockDelta::Text(text),
            TextKind::Refusal => BlockDelta::Refusal(text),
        };
        Some(StreamEvent::BlockDelta { index, delta })
    }
}

/// Ends the open field at `index`, returning its held-back tail if any.
fn take_tail(fields: &mut BTreeMap<u32, StreamField>, index: u32) -> Option<StreamEvent> {
    fields.remove(&index).and_then(|field| field.finish(index))
}

fn restore_stream(session: RedactionSession, mut events: EventStream) -> EventStream {
    EventStream::from_stream(async_stream::try_stream! {
        let mut fields: BTreeMap<u32, StreamField> = BTreeMap::new();
        while let Some(item) = events.next().await {
            let mut event = item?;
            match &mut event {
                StreamEvent::BlockStart { index, block } => {
                    if let Some(tail) = take_tail(&mut fields, *index) {
                        yield tail;
                    }
                    if let Some((kind, text)) = block_text(block) {
                        let mut field = StreamField::new(kind, &session);
                        *text = field.restorer.push(text);
                        fields.insert(*index, field);
                    } else {
                        block.visit_mut(&mut |block| restore_text(&session, block));
                    }
                }
                StreamEvent::BlockDelta { index, delta } => {
                    let (kind, text) = match delta {
                        BlockDelta::Text(text) => (TextKind::Text, text),
                        BlockDelta::Refusal(text) => (TextKind::Refusal, text),
                        _ => {
                            yield event;
                            continue;
                        }
                    };
                    if fields.get(index).is_some_and(|field| field.kind != kind)
                        && let Some(tail) = take_tail(&mut fields, *index)
                    {
                        yield tail;
                    }
                    let field = fields.entry(*index).or_insert_with(|| StreamField::new(kind, &session));
                    *text = field.restorer.push(text);
                    if text.is_empty() {
                        continue;
                    }
                }
                StreamEvent::BlockStop { index } => {
                    if let Some(tail) = take_tail(&mut fields, *index) {
                        yield tail;
                    }
                }
                StreamEvent::MessageStart { .. } | StreamEvent::MessageStop { .. } => {
                    for tail in std::mem::take(&mut fields).into_iter().filter_map(|(index, field)| field.finish(index)) {
                        yield tail;
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
        for tail in fields.into_iter().filter_map(|(index, field)| field.finish(index)) {
            yield tail;
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
