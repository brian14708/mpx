use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Value, json};

use crate::{
    ApiError, BlockDelta, ClientContext, CodecContext, ContentBlock, Protocol, ReasoningSeal,
    Result, SseEvent, StopReason, StreamDecoder, StreamEncoder, StreamEvent, Usage,
    anthropic_messages::{
        codec,
        wire::{self, MessagesResponse, WireUsage},
    },
    response::normalize_tool_stop_reason,
};

/// Stateful Anthropic SSE decoder.
#[derive(Debug)]
pub(crate) struct MessagesStreamDecoder {
    usage: Usage,
    stop_reason: StopReason,
    usage_emitted: bool,
    stopped: bool,
    complete_arguments: HashSet<u32>,
}

impl MessagesStreamDecoder {
    pub(crate) fn new() -> Self {
        Self {
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
            usage_emitted: false,
            stopped: false,
            complete_arguments: HashSet::new(),
        }
    }

    fn terminal_events(&mut self) -> Vec<StreamEvent> {
        if self.stopped {
            return Vec::new();
        }
        self.stopped = true;
        let mut events = Vec::new();
        if !self.usage_emitted {
            events.push(StreamEvent::Usage {
                usage: self.usage.clone(),
            });
            self.usage_emitted = true;
        }
        events.push(StreamEvent::MessageStop {
            stop_reason: self.stop_reason.clone(),
        });
        events
    }
}

impl StreamDecoder for MessagesStreamDecoder {
    fn decode(&mut self, frame: SseEvent, cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if frame.data.trim() == "[DONE]" {
            return Ok(Vec::new());
        }
        let value: Value =
            serde_json::from_str(&frame.data).map_err(|error| crate::Error::Protocol {
                protocol: Protocol::Messages,
                message: format!("invalid SSE JSON: {error}"),
            })?;
        let kind = crate::sse::event_name(&frame)
            .or_else(|| value.get("type").and_then(Value::as_str))
            .unwrap_or_default();
        match kind {
            "message_start" => {
                let message = value.get("message").cloned().unwrap_or(Value::Null);
                let response: MessagesResponse =
                    serde_json::from_value(message).map_err(wire::json_error)?;
                if let Some(usage) = response.usage {
                    self.usage = codec::decode_usage(usage);
                }
                Ok(vec![StreamEvent::MessageStart {
                    id: response.id,
                    model: response.model,
                    extra: wire::extra_fields(response.extra),
                }])
            }
            "content_block_start" => {
                let index = event_index(&value)?;
                let block = codec::decode_block(
                    value.get("content_block").cloned().unwrap_or(Value::Null),
                    cx,
                )?;
                if let ContentBlock::ToolCall { arguments, .. } = &block
                    && !arguments.as_str().is_empty()
                    && arguments.as_str() != "{}"
                {
                    self.complete_arguments.insert(index);
                }
                Ok(vec![StreamEvent::BlockStart { index, block }])
            }
            "content_block_delta" => {
                let index = event_index(&value)?;
                let delta = value.get("delta").cloned().unwrap_or(Value::Null);
                let kind = delta
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let delta = match kind {
                    "text_delta" => BlockDelta::Text(
                        delta
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    "thinking_delta" => BlockDelta::Reasoning(
                        delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    "signature_delta" => BlockDelta::Seal(ReasoningSeal::new(
                        Protocol::Messages,
                        delta
                            .get("signature")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    )),
                    "input_json_delta" if self.complete_arguments.contains(&index) => {
                        return Ok(Vec::new());
                    }
                    "input_json_delta" => BlockDelta::ToolArguments(
                        delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    _ => BlockDelta::Native {
                        protocol: Protocol::Messages,
                        json: delta,
                    },
                };
                Ok(vec![StreamEvent::BlockDelta { index, delta }])
            }
            "content_block_stop" => Ok(vec![StreamEvent::BlockStop {
                index: event_index(&value)?,
            }]),
            "message_delta" => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    let wire_reason = wire::StopReason::from(reason.to_owned());
                    self.stop_reason = (&wire_reason).into();
                }
                if let Some(usage) = value.get("usage").cloned() {
                    let usage: WireUsage =
                        serde_json::from_value(usage).map_err(wire::json_error)?;
                    self.usage.merge_nonzero(&codec::decode_usage(usage));
                }
                self.usage_emitted = true;
                Ok(vec![StreamEvent::Usage {
                    usage: self.usage.clone(),
                }])
            }
            "message_stop" => Ok(self.terminal_events()),
            "ping" => Ok(vec![StreamEvent::Ping]),
            "error" => {
                let error = value.get("error").unwrap_or(&value);
                Ok(vec![StreamEvent::Error {
                    error: ApiError {
                        kind: crate::ApiErrorKind::Upstream,
                        message: error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("upstream stream error")
                            .to_owned(),
                        code: error.get("type").and_then(Value::as_str).map(str::to_owned),
                    },
                }])
            }
            _ => Ok(Vec::new()),
        }
    }

    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.stopped {
            Ok(Vec::new())
        } else {
            Err(crate::Error::Body {
                message: "Messages stream ended before message_stop".to_owned(),
            })
        }
    }
}

fn event_index(value: &Value) -> Result<u32> {
    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
    u32::try_from(index).map_err(|_| crate::Error::Protocol {
        protocol: Protocol::Messages,
        message: format!("content block index {index} exceeds u32"),
    })
}

/// Stateful Anthropic SSE encoder with interleaving serialization.
#[derive(Debug)]
pub(crate) struct MessagesStreamEncoder {
    indices: HashMap<u32, u32>,
    next_index: u32,
    active: Option<u32>,
    pending: VecDeque<(u32, Vec<StreamEvent>)>,
    usage: Usage,
    stopped: bool,
    saw_tool_call: bool,
}

impl MessagesStreamEncoder {
    pub(crate) fn new(_client: &ClientContext) -> Self {
        Self {
            indices: HashMap::new(),
            next_index: 0,
            active: None,
            pending: VecDeque::new(),
            usage: Usage::default(),
            stopped: false,
            saw_tool_call: false,
        }
    }

    fn process(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if let StreamEvent::BlockStart { index, .. }
        | StreamEvent::BlockDelta { index, .. }
        | StreamEvent::BlockStop { index } = &event
            && self.active.is_some_and(|active| active != *index)
        {
            if let Some((_, events)) = self
                .pending
                .iter_mut()
                .find(|(queued, _)| *queued == *index)
            {
                events.push(event);
            } else {
                self.pending.push_back((*index, vec![event]));
            }
            return Ok(Vec::new());
        }
        self.encode_now(&event, cx)
    }

    fn encode_now(&mut self, event: &StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        match event {
            StreamEvent::MessageStart { id, model, extra } => {
                let mut message = json!({"id":id,"type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"usage":codec::encode_usage(&Usage::default())});
                if let Some(extra) = extra
                    .as_ref()
                    .filter(|extra| extra.protocol == Protocol::Messages)
                    && let Value::Object(map) = &mut message
                {
                    for (key, value) in &extra.fields {
                        map.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                }
                Ok(vec![frame(
                    "message_start",
                    json!({"type":"message_start","message":message}),
                )])
            }
            StreamEvent::BlockStart { index, block } => {
                let client_index = self.client_index(*index);
                self.active = Some(*index);
                let content = match block {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        seal,
                        extra,
                    } => {
                        self.saw_tool_call = true;
                        if arguments.as_str().is_empty() {
                            codec::encode_tool_use(
                                id,
                                name,
                                json!({}),
                                seal.as_ref(),
                                extra.as_ref(),
                                cx,
                            )?
                        } else {
                            codec::encode_block(block, cx)?.unwrap_or(Value::Null)
                        }
                    }
                    _ => codec::encode_block(block, cx)?.unwrap_or(Value::Null),
                };
                Ok(vec![frame(
                    "content_block_start",
                    json!({"type":"content_block_start","index":client_index,"content_block":content}),
                )])
            }
            StreamEvent::BlockDelta { index, delta } => {
                let client_index = self.client_index(*index);
                let (kind, body) = match delta {
                    BlockDelta::Text(text) => ("text_delta", json!({"text":text})),
                    BlockDelta::Reasoning(text) | BlockDelta::ReasoningSummary(text) => {
                        ("thinking_delta", json!({"thinking":text}))
                    }
                    BlockDelta::ToolArguments(text) => {
                        ("input_json_delta", json!({"partial_json":text}))
                    }
                    BlockDelta::Seal(seal) if seal.protocol == Protocol::Messages => {
                        ("signature_delta", json!({"signature":seal.value}))
                    }
                    BlockDelta::Seal(_) => return Ok(Vec::new()),
                    BlockDelta::Redacted => return Ok(Vec::new()),
                    BlockDelta::Refusal(text) => ("text_delta", json!({"text":text})),
                    BlockDelta::Native { protocol, json } if *protocol == Protocol::Messages => {
                        return Ok(vec![frame(
                            "content_block_delta",
                            json!({"type":"content_block_delta","index":client_index,"delta":json}),
                        )]);
                    }
                    BlockDelta::Native { .. } => return Ok(Vec::new()),
                };
                let mut delta = body;
                delta["type"] = Value::String(kind.to_owned());
                Ok(vec![frame(
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":client_index,"delta":delta}),
                )])
            }
            StreamEvent::BlockStop { index } => {
                let client_index = self.client_index(*index);
                if self.active == Some(*index) {
                    self.active = None;
                }
                Ok(vec![frame(
                    "content_block_stop",
                    json!({"type":"content_block_stop","index":client_index}),
                )])
            }
            StreamEvent::Usage { usage } => {
                self.usage.merge_nonzero(usage);
                Ok(Vec::new())
            }
            StreamEvent::MessageStop { stop_reason } => {
                let reason = normalize_tool_stop_reason(stop_reason.clone(), self.saw_tool_call);
                self.stopped = true;
                Ok(vec![
                    frame(
                        "message_delta",
                        json!({"type":"message_delta","delta":{"stop_reason":wire::StopReason::from(&reason).as_str()},"usage":codec::encode_usage(&self.usage)}),
                    ),
                    frame("message_stop", json!({"type":"message_stop"})),
                ])
            }
            StreamEvent::Error { error } => {
                self.stopped = true;
                Ok(vec![frame(
                    "error",
                    json!({"type":"error","error":{"type":error.code.as_deref().unwrap_or("api_error"),"message":error.message}}),
                )])
            }
            StreamEvent::Safety { .. } => Ok(Vec::new()),
            StreamEvent::ResponseMetadata { .. } => Ok(Vec::new()),
            StreamEvent::Ping => Ok(vec![frame("ping", json!({"type":"ping"}))]),
        }
    }

    fn client_index(&mut self, upstream: u32) -> u32 {
        *self.indices.entry(upstream).or_insert_with(|| {
            let index = self.next_index;
            self.next_index = self.next_index.saturating_add(1);
            index
        })
    }

    fn drain_pending(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        let mut frames = Vec::new();
        while self.active.is_none()
            && let Some((_, events)) = self.pending.pop_front()
        {
            for event in events {
                frames.extend(self.encode_now(&event, cx)?);
            }
        }
        Ok(frames)
    }

    fn terminal_frames(&mut self) -> Vec<SseEvent> {
        self.stopped = true;
        let reason = normalize_tool_stop_reason(StopReason::EndTurn, self.saw_tool_call);
        vec![
            frame(
                "message_delta",
                json!({"type":"message_delta","delta":{"stop_reason":wire::StopReason::from(&reason).as_str()},"usage":codec::encode_usage(&self.usage)}),
            ),
            frame("message_stop", json!({"type":"message_stop"})),
        ]
    }
}

impl StreamEncoder for MessagesStreamEncoder {
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.stopped {
            return Ok(Vec::new());
        }
        let mut frames = self.process(event, cx)?;
        if self.active.is_none() {
            frames.extend(self.drain_pending(cx)?);
        }
        Ok(frames)
    }

    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.stopped {
            Ok(Vec::new())
        } else {
            Ok(self.terminal_frames())
        }
    }
}

fn frame(event: &str, value: Value) -> SseEvent {
    crate::sse::named(event, value.to_string())
}
