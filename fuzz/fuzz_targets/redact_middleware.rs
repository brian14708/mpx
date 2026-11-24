//! Masks generated requests through the Tower middleware and checks that the
//! secret never reaches the upstream, that restored responses equal the
//! originals, and that session scopes and upstream failures propagate.
#![no_main]

use std::collections::BTreeMap;

use arbitrary::{Arbitrary, Unstructured};
use futures_util::{StreamExt, stream};
use libfuzzer_sys::fuzz_target;
use mpx_core::RedactionLayer;
use mpx_fuzz::{block_on, split_str};
use mpx_protocol::{
    BlockDelta, BoxError, ContentBlock, EventStream, Extra, Message, OutputFormat,
    OutputFormatKind, Protocol, ReasoningParams, Request, Response, ResponseAccumulator,
    ResponseBody, Role, SearchResult, StreamEvent, Tool, ToolArguments,
};
use mpx_redact::{Matcher, RedactionSession, Replacement, ReplacementMap, Rules};
use serde_json::{Map, Value, json};
use tower::{Layer, ServiceExt, service_fn};

/// A request field that carries the secret.
#[derive(Debug, Clone, Copy, Arbitrary)]
enum Place {
    Instructions,
    Text,
    StructuredText,
    ToolCall,
    Reasoning,
    Summary,
    Refusal,
    ToolResult,
    WebSearch,
    Native,
    Metadata,
    FunctionTool,
    CustomTool,
    HostedTool,
    OutputSchema,
    OutputExtra,
    SamplingExtra,
    ReasoningExtra,
    RequestExtra,
}

#[derive(Debug, Arbitrary)]
enum SessionId {
    None,
    /// Whitespace does not scope replacements.
    Blank,
    Id(u8),
}

#[derive(Debug, Arbitrary)]
enum Upstream {
    Complete,
    Stream {
        close_blocks: bool,
    },
    /// Fails after the given number of events.
    StreamError(u8),
    Fails,
}

#[derive(Debug, Arbitrary)]
struct Input {
    /// The first byte may force a place, so seeds can target each one.
    #[arbitrary(with = Place::list)]
    places: Vec<Place>,
    secret: String,
    key: Vec<u8>,
    session_id: SessionId,
    fallback: bool,
    upstream: Upstream,
    cuts: Vec<u16>,
}

impl Place {
    const ALL: [Place; 19] = [
        Place::Instructions,
        Place::Text,
        Place::StructuredText,
        Place::ToolCall,
        Place::Reasoning,
        Place::Summary,
        Place::Refusal,
        Place::ToolResult,
        Place::WebSearch,
        Place::Native,
        Place::Metadata,
        Place::FunctionTool,
        Place::CustomTool,
        Place::HostedTool,
        Place::OutputSchema,
        Place::OutputExtra,
        Place::SamplingExtra,
        Place::ReasoningExtra,
        Place::RequestExtra,
    ];

    /// Reads a selector byte that forces one place (or none, past the end), then more places.
    fn list(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<Self>> {
        let selector = usize::from(u.arbitrary::<u8>()?) % (Self::ALL.len() + 1);
        let forced = Self::ALL.get(selector).copied();
        let rest: Vec<Self> = u.arbitrary()?;
        Ok(forced.into_iter().chain(rest).collect())
    }
}

fn structured(secret: &str) -> Value {
    // Quotes, backslashes, control characters, and a secret-bearing object key.
    json!({
        "nested": [secret, {"escaped": format!("\"\\\n{secret}"), "untouched": 42}],
        secret: "key stays unchanged",
        "boolean": true,
        "null": null,
    })
}

fn extra(secret: &str) -> Option<Extra> {
    let mut fields = Map::new();
    fields.insert("hint".to_owned(), structured(secret));
    Some(Extra::with_fields(Protocol::Responses, fields))
}

fn build_request(input: &Input, secret: &str) -> Request {
    let value = structured(secret);
    let mut request = Request::builder().model("fuzz-model").build();
    let mut blocks = Vec::new();
    for place in input.places.iter().take(24) {
        match place {
            Place::Instructions => request
                .instructions
                .push(ContentBlock::text(format!("system {secret}"))),
            Place::Text => blocks.push(ContentBlock::text(format!("text {secret}"))),
            Place::StructuredText => blocks.push(ContentBlock::text(value.to_string())),
            Place::ToolCall => {
                blocks.push(ContentBlock::tool_call(
                    "call-1",
                    "lookup",
                    value.to_string(),
                ));
            }
            Place::Reasoning => {
                let mut block = ContentBlock::reasoning();
                if let ContentBlock::Reasoning { text, summary, .. } = &mut block {
                    *text = format!("reasoning {secret}");
                    summary.push(format!("earlier {secret}"));
                }
                blocks.push(block);
            }
            Place::Summary => {
                let mut block = ContentBlock::reasoning();
                if let ContentBlock::Reasoning { summary, .. } = &mut block {
                    summary.push(format!("first {secret}"));
                    summary.push(format!("summary {secret}"));
                }
                blocks.push(block);
            }
            Place::Refusal => blocks.push(ContentBlock::Refusal {
                text: format!("refusal {secret}"),
                extra: None,
            }),
            Place::ToolResult => blocks.push(ContentBlock::ToolResult {
                call_id: "call-1".into(),
                name: Some("lookup".into()),
                content: vec![ContentBlock::text(value.to_string())],
                is_error: false,
                extra: None,
            }),
            Place::WebSearch => blocks.push(ContentBlock::WebSearchResult {
                query: Some(format!("query {secret}")),
                results: vec![
                    SearchResult::builder()
                        .title(format!("title {secret}"))
                        .url(format!("https://example.org/{secret}"))
                        .text(secret.to_owned())
                        .build(),
                ],
                extra: None,
            }),
            Place::Native => blocks.push(ContentBlock::Native {
                protocol: Protocol::Responses,
                json: value.clone(),
            }),
            Place::Metadata => {
                request.metadata.insert("nested".to_owned(), value.clone());
            }
            Place::FunctionTool => request.tools.push(Tool::Function {
                name: "lookup".to_owned(),
                description: Some(format!("describe {secret}")),
                parameters: value.clone(),
                strict: None,
                defer_loading: None,
                extra: extra(secret),
            }),
            Place::CustomTool => request.tools.push(Tool::Custom {
                name: "custom".to_owned(),
                description: Some(secret.to_owned()),
                format: Some(value.clone()),
                extra: extra(secret),
            }),
            Place::HostedTool => request.tools.push(Tool::Hosted {
                protocol: Protocol::Responses,
                json: value.clone(),
            }),
            Place::OutputSchema => {
                request.output_format = Some(
                    OutputFormat::builder()
                        .kind(OutputFormatKind::JsonSchema)
                        .schema(value.clone())
                        .build(),
                );
            }
            Place::OutputExtra => {
                request.output_format = Some(
                    OutputFormat::builder()
                        .kind(OutputFormatKind::JsonObject)
                        .maybe_extra(extra(secret))
                        .build(),
                );
            }
            Place::SamplingExtra => request.sampling.extra = extra(secret),
            Place::ReasoningExtra => {
                request.reasoning = Some(
                    ReasoningParams::builder()
                        .maybe_extra(extra(secret))
                        .build(),
                );
            }
            Place::RequestExtra => request.extra = extra(secret),
        }
    }
    let mut message = Message::text(Role::User, "");
    message.content = blocks;
    request.messages = vec![message];
    request.session_id = match input.session_id {
        SessionId::None => None,
        SessionId::Blank => Some(" \t".to_owned()),
        SessionId::Id(id) => Some(format!("scope-{id}")),
    };
    request
}

/// Collects every string value, decoding strings that hold JSON containers.
/// Object keys are not masked, so they are skipped.
fn strings(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if let Ok(inner @ (Value::Array(_) | Value::Object(_))) = serde_json::from_str(text) {
                strings(&inner, output);
            } else {
                output.push(text.clone());
            }
        }
        Value::Array(values) => values.iter().for_each(|value| strings(value, output)),
        Value::Object(values) => values.values().for_each(|value| strings(value, output)),
        _ => {}
    }
}

/// Text carried by deltas, or `None` for blocks emitted whole.
type Delta = fn(String) -> BlockDelta;

fn delta_text(block: &mut ContentBlock) -> Option<(String, Delta)> {
    Some(match block {
        ContentBlock::Text { text, .. } => (std::mem::take(text), BlockDelta::Text),
        ContentBlock::ToolCall { arguments, .. } => {
            let text = arguments.as_str().to_owned();
            *arguments = ToolArguments::new("");
            (text, BlockDelta::ToolArguments)
        }
        ContentBlock::Reasoning { text, summary, .. } if text.is_empty() && !summary.is_empty() => {
            (
                std::mem::take(summary.last_mut().unwrap()),
                BlockDelta::ReasoningSummary,
            )
        }
        ContentBlock::Reasoning { text, .. } => (std::mem::take(text), BlockDelta::Reasoning),
        ContentBlock::Refusal { text, .. } => (std::mem::take(text), BlockDelta::Refusal),
        _ => return None,
    })
}

/// Streams a response with deltas of different blocks interleaved.
fn interleaved(response: Response, cuts: &[u16], close_blocks: bool) -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::MessageStart {
        id: response.id,
        model: response.model,
        extra: None,
    }];
    let mut fields = Vec::new();
    for (index, mut block) in response.content.into_iter().enumerate() {
        let index = index as u32;
        let delta = delta_text(&mut block);
        events.push(StreamEvent::BlockStart { index, block });
        if let Some((text, delta)) = delta {
            let chunks: Vec<String> = split_str(&text, cuts)
                .into_iter()
                .map(str::to_owned)
                .collect();
            fields.push((index, delta, chunks.into_iter()));
        }
    }
    loop {
        let mut progress = false;
        for (index, delta, chunks) in &mut fields {
            if let Some(chunk) = chunks.next() {
                progress = true;
                events.push(StreamEvent::BlockDelta {
                    index: *index,
                    delta: delta(chunk),
                });
            }
        }
        if !progress {
            break;
        }
    }
    if close_blocks {
        for (index, _, _) in fields {
            events.push(StreamEvent::BlockStop { index });
        }
    }
    events.push(StreamEvent::Usage {
        usage: response.usage,
    });
    events.push(StreamEvent::MessageStop {
        stop_reason: response.stop_reason,
    });
    events
}

/// Checks that text emitted before a stream failure is a prefix of the restored text.
fn assert_partial(events: &[StreamEvent], expected: &[ContentBlock]) {
    let mut texts = BTreeMap::<u32, String>::new();
    for event in events {
        match event {
            StreamEvent::BlockStart {
                index,
                block: ContentBlock::Text { text, .. } | ContentBlock::Refusal { text, .. },
            } => texts.entry(*index).or_default().push_str(text),
            StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Text(text) | BlockDelta::Refusal(text),
            } => texts.entry(*index).or_default().push_str(text),
            _ => {}
        }
    }
    for (index, text) in texts {
        if let Some(
            ContentBlock::Text { text: full, .. } | ContentBlock::Refusal { text: full, .. },
        ) = expected.get(index as usize)
        {
            assert!(
                full.starts_with(&text),
                "partial {text:?} is not a prefix of {full:?}"
            );
        }
    }
}

fuzz_target!(|input: Input| {
    let secret = format!("secret:{}:末", input.secret);
    let request = build_request(&input, &secret);
    let expected = Response::builder()
        .id("echo")
        .model("fuzz-model")
        .content(request.messages[0].content.clone())
        .build();
    let rules = Rules::new([(
        Matcher::literal(secret.clone()),
        Replacement::kind("SECRET").unwrap(),
    )]);

    // The middleware scopes replacements to a nonblank session ID.
    let mut scoped = RedactionSession::new().with_hmac_key(&input.key);
    if let SessionId::Id(id) = input.session_id {
        scoped = scoped.with_scope(format!("scope-{id}"));
    }
    let marker = scoped.mask(&secret, &rules).expect("lone secret masks");

    let upstream_secret = secret.clone();
    let cuts = input.cuts.clone();
    let upstream = std::sync::Arc::new(input.upstream);
    let service_upstream = upstream.clone();
    let service = service_fn(move |request: http::Request<Request>| {
        let secret = upstream_secret.clone();
        let cuts = cuts.clone();
        let upstream = service_upstream.clone();
        async move {
            let mut values = Vec::new();
            strings(&serde_json::to_value(request.body()).unwrap(), &mut values);
            for value in values {
                assert!(
                    !value.contains(&secret),
                    "secret reached upstream in {value:?}"
                );
            }
            let response = Response::builder()
                .id("echo")
                .model("fuzz-model")
                .content(request.body().messages[0].content.clone())
                .build();
            let body = match *upstream {
                Upstream::Complete => ResponseBody::Complete(response),
                Upstream::Stream { close_blocks } => ResponseBody::Stream(
                    EventStream::from_events(interleaved(response, &cuts, close_blocks)),
                ),
                Upstream::StreamError(at) => {
                    let mut events: Vec<_> = interleaved(response, &cuts, false)
                        .into_iter()
                        .map(Ok)
                        .collect();
                    events.truncate(usize::from(at));
                    events.push(Err(mpx_protocol::Error::Body {
                        message: "upstream failed".to_owned(),
                    }));
                    ResponseBody::Stream(EventStream::from_stream(stream::iter(events)))
                }
                Upstream::Fails => return Err::<_, BoxError>("upstream failed".into()),
            };
            Ok(http::Response::new(body))
        }
    });
    let mut layer = RedactionLayer::new(rules)
        .with_hmac_key(&input.key)
        .with_replacement_map();
    if input.fallback {
        layer = layer.with_fallback(Replacement::kind("FALLBACK").unwrap());
    }
    let service = layer.layer(service);
    let result = block_on(service.oneshot(http::Request::new(request)));
    if matches!(*upstream, Upstream::Fails) {
        assert_eq!(result.unwrap_err().to_string(), "upstream failed");
        return;
    }
    let response = result.expect("generated middleware exchange succeeds");
    let map = response
        .extensions()
        .get::<ReplacementMap>()
        .unwrap()
        .clone();
    let masked_any = map.replacements_for(&secret).next().is_some();
    assert_eq!(
        masked_any,
        !input.places.is_empty(),
        "masking did not match placement"
    );
    if masked_any {
        assert!(
            map.replacements_for(&secret).any(|issued| issued == marker),
            "replacement {marker:?} was not scoped by the session ID"
        );
    }
    let actual = match response.into_body() {
        ResponseBody::Complete(response) => response,
        ResponseBody::Stream(mut events) => {
            let mut collected = Vec::new();
            let mut failed = false;
            while let Some(event) = block_on(events.next()) {
                match event {
                    Ok(event) => {
                        assert!(!failed, "event after stream failure");
                        collected.push(event);
                    }
                    Err(_) => failed = true,
                }
            }
            if matches!(*upstream, Upstream::StreamError(_)) {
                assert!(failed, "upstream stream failure was swallowed");
                assert_partial(&collected, &expected.content);
                return;
            }
            assert!(!failed, "restored stream failed");
            let mut accumulator = ResponseAccumulator::new();
            accumulator.extend(collected);
            accumulator.finish()
        }
        body => panic!("unexpected echo response: {body:?}"),
    };
    assert_eq!(
        actual, expected,
        "middleware changed structured values or field boundaries"
    );
});
