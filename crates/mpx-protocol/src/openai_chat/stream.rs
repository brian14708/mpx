use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    BlockDelta, ClientContext, CodecContext, ContentBlock, DiagnosticKind, Error, Protocol, Result,
    SseEvent, StopReason, StreamDecoder, StreamEncoder, StreamEvent, ToolArguments, ToolCallId,
    Usage, response::normalize_tool_stop_reason, tool::synthesized_tool_call_id,
};

use super::{
    codec::{
        decode_reasoning_block, decode_usage, delta_for_native, encode_usage, make_extra,
        same_extra, stream_error,
    },
    wire::{
        ChunkWire, DeltaToolCallWire, DeltaWire, ErrorBodyWire, ErrorEnvelopeWire, FinishReason,
        FunctionCallWire, StreamChoiceWire, ToolCallKind, UsageWire, WireRole,
    },
};

#[derive(Debug, Default)]
pub(crate) struct ChatStreamDecoder {
    started: bool,
    stopped: bool,
    next_block: u32,
    ordinary: Option<OrdinaryBlock>,
    tools: BTreeMap<u32, ActiveTool>,
    saw_tool_calls: bool,
    pending_stop: Option<StopReason>,
    pending_usage: Option<Usage>,
}

#[derive(Debug)]
struct OrdinaryBlock {
    index: u32,
    kind: OrdinaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrdinaryKind {
    Text,
    Reasoning,
    Refusal,
}

#[derive(Debug)]
struct ActiveTool {
    block_index: u32,
    wire_id: Option<String>,
}

impl ChatStreamDecoder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    const fn next_index(&mut self) -> u32 {
        let index = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        index
    }

    fn ensure_started(&mut self, chunk: &ChunkWire, events: &mut Vec<StreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        let mut extra = chunk.extra.clone();
        if let Some(object) = &chunk.object {
            extra.insert("object".to_owned(), Value::String(object.clone()));
        }
        extra.insert("created".to_owned(), Value::from(chunk.created));
        events.push(StreamEvent::MessageStart {
            id: chunk.id.clone(),
            model: chunk.model.clone(),
            extra: make_extra(extra),
        });
    }

    fn close_ordinary(&mut self, events: &mut Vec<StreamEvent>) {
        if let Some(ordinary) = self.ordinary.take() {
            events.push(StreamEvent::BlockStop {
                index: ordinary.index,
            });
        }
    }

    fn close_all(&mut self, events: &mut Vec<StreamEvent>) {
        self.close_ordinary(events);
        let tools = std::mem::take(&mut self.tools);
        for (_, tool) in tools {
            events.push(StreamEvent::BlockStop {
                index: tool.block_index,
            });
        }
    }

    fn terminate(&mut self, events: &mut Vec<StreamEvent>) {
        if self.stopped {
            return;
        }
        if !self.started {
            self.stopped = true;
            return;
        }
        self.close_all(events);
        if let Some(usage) = self.pending_usage.take() {
            events.push(StreamEvent::Usage { usage });
        }
        let stop_reason = normalize_tool_stop_reason(
            self.pending_stop.take().unwrap_or(StopReason::EndTurn),
            self.saw_tool_calls,
        );
        events.push(StreamEvent::MessageStop { stop_reason });
        self.stopped = true;
    }

    fn ordinary_block(&mut self, kind: OrdinaryKind, text: String, events: &mut Vec<StreamEvent>) {
        if text.is_empty() {
            return;
        }
        if self
            .ordinary
            .as_ref()
            .is_some_and(|block| block.kind != kind)
        {
            self.close_ordinary(events);
        }
        let index = if let Some(block) = &self.ordinary {
            block.index
        } else {
            let index = self.next_index();
            let block = match kind {
                OrdinaryKind::Text => ContentBlock::text(""),
                OrdinaryKind::Reasoning => ContentBlock::reasoning(),
                OrdinaryKind::Refusal => ContentBlock::Refusal {
                    text: String::new(),
                    extra: None,
                },
            };
            events.push(StreamEvent::BlockStart { index, block });
            self.ordinary = Some(OrdinaryBlock { index, kind });
            index
        };
        let delta = match kind {
            OrdinaryKind::Text => BlockDelta::Text(text),
            OrdinaryKind::Reasoning => BlockDelta::Reasoning(text),
            OrdinaryKind::Refusal => BlockDelta::Refusal(text),
        };
        events.push(StreamEvent::BlockDelta { index, delta });
    }

    fn reasoning_value(&mut self, value: Value, events: &mut Vec<StreamEvent>) {
        let ContentBlock::Reasoning {
            text,
            summary,
            redacted,
            extra,
            ..
        } = decode_reasoning_block(value)
        else {
            return;
        };
        if self
            .ordinary
            .as_ref()
            .is_some_and(|block| block.kind != OrdinaryKind::Reasoning)
        {
            self.close_ordinary(events);
        }
        let index = if let Some(block) = &self.ordinary {
            block.index
        } else {
            let index = self.next_index();
            events.push(StreamEvent::BlockStart {
                index,
                block: ContentBlock::Reasoning {
                    text: String::new(),
                    summary: Vec::new(),
                    seal: None,
                    redacted: false,
                    extra,
                },
            });
            self.ordinary = Some(OrdinaryBlock {
                index,
                kind: OrdinaryKind::Reasoning,
            });
            index
        };
        if !text.is_empty() {
            events.push(StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Reasoning(text),
            });
        }
        if !summary.is_empty() {
            events.push(StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::ReasoningSummary(summary.join("\n")),
            });
        }
        if redacted {
            events.push(StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Redacted,
            });
        }
    }

    fn tool_call(&mut self, call: DeltaToolCallWire, events: &mut Vec<StreamEvent>) {
        self.saw_tool_calls = true;
        self.close_ordinary(events);
        let wire_index = call.index;
        let wire_id = call.id.filter(|id| !id.is_empty());
        let replace = self.tools.get(&wire_index).is_some_and(|active| {
            wire_id
                .as_ref()
                .is_some_and(|id| active.wire_id.as_ref() != Some(id))
        });
        if replace && let Some(active) = self.tools.remove(&wire_index) {
            events.push(StreamEvent::BlockStop {
                index: active.block_index,
            });
        }
        let block_index = if let Some(active) = self.tools.get(&wire_index) {
            active.block_index
        } else {
            let block_index = self.next_index();
            let id = wire_id
                .clone()
                .unwrap_or_else(|| synthesized_tool_call_id(wire_index));
            let mut extra = call.extra.clone();
            extra.extend(call.function.extra.clone());
            events.push(StreamEvent::BlockStart {
                index: block_index,
                block: ContentBlock::ToolCall {
                    id: ToolCallId::new(id),
                    name: call.function.name.clone().unwrap_or_default(),
                    arguments: ToolArguments::new(""),
                    seal: None,
                    extra: make_extra(extra),
                },
            });
            self.tools.insert(
                wire_index,
                ActiveTool {
                    block_index,
                    wire_id,
                },
            );
            block_index
        };
        if let Some(arguments) = call.function.arguments
            && !arguments.is_empty()
        {
            events.push(StreamEvent::BlockDelta {
                index: block_index,
                delta: BlockDelta::ToolArguments(arguments),
            });
        }
    }

    fn native(fields: Map<String, Value>, next_index: u32, events: &mut Vec<StreamEvent>) {
        let (block, delta) = delta_for_native(fields);
        events.push(StreamEvent::BlockStart {
            index: next_index,
            block,
        });
        events.push(StreamEvent::BlockDelta {
            index: next_index,
            delta,
        });
        events.push(StreamEvent::BlockStop { index: next_index });
    }
}

impl StreamDecoder for ChatStreamDecoder {
    fn decode(&mut self, frame: SseEvent, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.stopped {
            return Ok(Vec::new());
        }
        if frame.data.trim() == "[DONE]" {
            let mut events = Vec::new();
            self.terminate(&mut events);
            return Ok(events);
        }
        if frame.data.is_empty() || crate::sse::event_name(&frame) == Some("ping") {
            return Ok(vec![StreamEvent::Ping]);
        }
        let chunk: ChunkWire = serde_json::from_str(&frame.data)
            .map_err(|error| Error::json(Protocol::ChatCompletions, error))?;
        if let Some(error) = stream_error(&chunk) {
            self.stopped = true;
            return Ok(vec![StreamEvent::Error { error }]);
        }
        let mut events = Vec::new();
        self.ensure_started(&chunk, &mut events);
        let no_choices = chunk.choices.is_empty();

        for choice in chunk.choices {
            if choice.index != 0 {
                continue;
            }
            let delta = choice.delta;
            if let Some(text) = delta.content {
                self.ordinary_block(OrdinaryKind::Text, text, &mut events);
            }
            if let Some(reasoning) = delta.reasoning_content {
                self.ordinary_block(OrdinaryKind::Reasoning, reasoning, &mut events);
            }
            if let Some(reasoning) = delta.reasoning {
                self.reasoning_value(reasoning, &mut events);
            }
            if let Some(refusal) = delta.refusal {
                self.ordinary_block(OrdinaryKind::Refusal, refusal, &mut events);
            }
            for call in delta.tool_calls {
                self.tool_call(call, &mut events);
            }
            if !delta.extra.is_empty() {
                self.close_ordinary(&mut events);
                let index = self.next_index();
                Self::native(delta.extra, index, &mut events);
            }
            if !choice.extra.is_empty() {
                self.close_ordinary(&mut events);
                let index = self.next_index();
                Self::native(choice.extra, index, &mut events);
            }
            if let Some(finish) = choice.finish_reason {
                self.pending_stop = Some(StopReason::from(&finish));
                self.close_all(&mut events);
            }
        }

        if let Some(usage) = chunk.usage {
            self.pending_usage = Some(decode_usage(Some(usage)));
            if self.pending_stop.is_some() || no_choices {
                self.terminate(&mut events);
            }
        }
        Ok(events)
    }

    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if !self.stopped && self.pending_stop.is_none() {
            return Err(Error::Body {
                message: "Chat Completions stream ended before a finish reason or [DONE]"
                    .to_owned(),
            });
        }
        let mut events = Vec::new();
        self.terminate(&mut events);
        Ok(events)
    }
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct ChatStreamEncoder {
    include_usage: bool,
    id: String,
    model: String,
    started: bool,
    stop_emitted: bool,
    done_emitted: bool,
    next_tool: u32,
    tools: BTreeMap<u32, u32>,
    saw_tool_calls: bool,
    pending_usage: Option<Usage>,
}

impl ChatStreamEncoder {
    pub(crate) fn new(client: &ClientContext) -> Self {
        Self {
            include_usage: client.include_usage,
            id: "chatcmpl-0".to_owned(),
            model: String::new(),
            started: false,
            stop_emitted: false,
            done_emitted: false,
            next_tool: 0,
            tools: BTreeMap::new(),
            saw_tool_calls: false,
            pending_usage: None,
        }
    }

    fn start(
        &mut self,
        id: String,
        model: String,
        extra: Option<&crate::Extra>,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        if self.started {
            return Ok(Vec::new());
        }
        self.started = true;
        if !id.is_empty() {
            self.id = id;
        }
        self.model = model;
        let mut extra = same_extra(extra, cx, "extra")?;
        extra.remove("created");
        extra.remove("object");
        self.chunk(
            DeltaWire {
                role: Some(WireRole::Assistant),
                ..DeltaWire::default()
            },
            None,
            None,
            extra,
        )
        .map(|frame| vec![frame])
    }

    fn ensure_started(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        self.start(String::new(), String::new(), None, cx)
    }

    fn chunk(
        &self,
        delta: DeltaWire,
        finish_reason: Option<FinishReason>,
        usage: Option<UsageWire>,
        extra: Map<String, Value>,
    ) -> Result<SseEvent> {
        let has_finish = finish_reason.is_some();
        let chunk = ChunkWire {
            id: self.id.clone(),
            object: Some("chat.completion.chunk".to_owned()),
            created: 0,
            model: self.model.clone(),
            choices: if has_finish || usage.is_none() {
                vec![StreamChoiceWire {
                    index: 0,
                    delta,
                    finish_reason,
                    extra: Map::new(),
                }]
            } else {
                Vec::new()
            },
            usage,
            extra,
        };
        frame(&chunk)
    }

    fn terminal(&mut self, reason: &StopReason, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.stop_emitted {
            return Ok(Vec::new());
        }
        self.stop_emitted = true;
        let mut frames =
            vec![self.chunk(DeltaWire::default(), Some(reason.into()), None, Map::new())?];
        if self.include_usage {
            let usage = self.pending_usage.take().unwrap_or_default();
            frames.push(self.chunk(
                DeltaWire::default(),
                None,
                Some(encode_usage(&usage, cx)?),
                Map::new(),
            )?);
        }
        Ok(frames)
    }

    fn tool_start(
        &mut self,
        upstream_index: u32,
        id: &ToolCallId,
        name: &str,
        arguments: &ToolArguments,
        extra: Option<&crate::Extra>,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        let client_index = self.next_tool;
        self.next_tool = self.next_tool.saturating_add(1);
        self.tools.insert(upstream_index, client_index);
        self.saw_tool_calls = true;
        let id = if id.as_str().is_empty() {
            synthesized_tool_call_id(client_index)
        } else {
            id.as_str().to_owned()
        };
        let arguments = match arguments.as_str() {
            "" | "{}" => None,
            arguments => Some(arguments.to_owned()),
        };
        let extra = same_extra(extra, cx, "tool_call.extra")?;
        self.chunk(
            DeltaWire {
                tool_calls: vec![DeltaToolCallWire {
                    index: client_index,
                    id: Some(id),
                    kind: Some(ToolCallKind::Function),
                    function: FunctionCallWire {
                        name: Some(name.to_owned()),
                        arguments,
                        extra: Map::new(),
                    },
                    extra,
                }],
                ..DeltaWire::default()
            },
            None,
            None,
            Map::new(),
        )
        .map(|frame| vec![frame])
    }

    fn unsupported(&self, kind: &str, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if !cx.options.content_fallbacks {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                "stream.block",
                format!("{kind} blocks cannot be represented in a Chat Completions stream"),
            )?;
            return Ok(Vec::new());
        }
        cx.diagnose(
            DiagnosticKind::Fallback,
            Protocol::ChatCompletions,
            "stream.block",
            format!("{kind} block became descriptive text"),
        )?;
        self.chunk(
            DeltaWire {
                content: Some(format!("[Unsupported {kind} content]")),
                ..DeltaWire::default()
            },
            None,
            None,
            Map::new(),
        )
        .map(|frame| vec![frame])
    }
}

impl StreamEncoder for ChatStreamEncoder {
    #[allow(clippy::too_many_lines)]
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.done_emitted || (self.stop_emitted && !matches!(event, StreamEvent::Ping)) {
            return Ok(Vec::new());
        }
        match &event {
            StreamEvent::MessageStart { id, model, extra } => {
                self.start(id.clone(), model.clone(), extra.as_ref(), cx)
            }
            StreamEvent::BlockStart { index, block } => {
                let mut frames = self.ensure_started(cx)?;
                let encoded = match block {
                    ContentBlock::Text { text, .. } if !text.is_empty() => self
                        .chunk(
                            DeltaWire {
                                content: Some(text.clone()),
                                ..DeltaWire::default()
                            },
                            None,
                            None,
                            Map::new(),
                        )
                        .map(|frame| vec![frame])?,
                    ContentBlock::Reasoning { text, summary, .. }
                        if !text.is_empty() || !summary.is_empty() =>
                    {
                        let text = if summary.is_empty() {
                            text.clone()
                        } else {
                            format!("{text}{}", summary.join("\n"))
                        };
                        self.chunk(
                            DeltaWire {
                                reasoning_content: Some(text),
                                ..DeltaWire::default()
                            },
                            None,
                            None,
                            Map::new(),
                        )
                        .map(|frame| vec![frame])?
                    }
                    ContentBlock::Refusal { text, .. } if !text.is_empty() => self
                        .chunk(
                            DeltaWire {
                                refusal: Some(text.clone()),
                                ..DeltaWire::default()
                            },
                            None,
                            None,
                            Map::new(),
                        )
                        .map(|frame| vec![frame])?,
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        extra,
                        ..
                    } => self.tool_start(*index, id, name, arguments, extra.as_ref(), cx)?,
                    ContentBlock::Text { .. }
                    | ContentBlock::Reasoning { .. }
                    | ContentBlock::Refusal { .. }
                    | ContentBlock::Native {
                        protocol: Protocol::ChatCompletions,
                        ..
                    } => Vec::new(),
                    other => self.unsupported(other.kind(), cx)?,
                };
                frames.extend(encoded);
                Ok(frames)
            }
            StreamEvent::BlockDelta { index, delta } => {
                let mut frames = self.ensure_started(cx)?;
                let wire_delta = match delta {
                    BlockDelta::Text(text) => DeltaWire {
                        content: Some(text.clone()),
                        ..DeltaWire::default()
                    },
                    BlockDelta::Reasoning(text) | BlockDelta::ReasoningSummary(text) => DeltaWire {
                        reasoning_content: Some(text.clone()),
                        ..DeltaWire::default()
                    },
                    BlockDelta::Refusal(text) => DeltaWire {
                        refusal: Some(text.clone()),
                        ..DeltaWire::default()
                    },
                    BlockDelta::ToolArguments(arguments) => {
                        let client_index = if let Some(client_index) = self.tools.get(index) {
                            *client_index
                        } else {
                            let start = self.tool_start(
                                *index,
                                &ToolCallId::new(""),
                                "",
                                &ToolArguments::new(""),
                                None,
                                cx,
                            )?;
                            frames.extend(start);
                            self.tools.get(index).copied().unwrap_or(0)
                        };
                        DeltaWire {
                            tool_calls: vec![DeltaToolCallWire {
                                index: client_index,
                                function: FunctionCallWire {
                                    arguments: Some(arguments.clone()),
                                    ..FunctionCallWire::default()
                                },
                                ..DeltaToolCallWire::default()
                            }],
                            ..DeltaWire::default()
                        }
                    }
                    BlockDelta::Native { protocol, json }
                        if *protocol == Protocol::ChatCompletions =>
                    {
                        if let Ok(delta) = serde_json::from_value(json.clone()) {
                            delta
                        } else {
                            cx.diagnose(
                                DiagnosticKind::Dropped,
                                Protocol::ChatCompletions,
                                "stream.delta",
                                "native delta is not a Chat Completions delta object",
                            )?;
                            return Ok(frames);
                        }
                    }
                    BlockDelta::Native { .. } | BlockDelta::Seal(_) | BlockDelta::Redacted => {
                        cx.diagnose(
                            DiagnosticKind::Dropped,
                            Protocol::ChatCompletions,
                            "stream.delta",
                            "delta cannot be represented by Chat Completions",
                        )?;
                        return Ok(frames);
                    }
                };
                frames.push(self.chunk(wire_delta, None, None, Map::new())?);
                Ok(frames)
            }
            StreamEvent::BlockStop { index } => {
                self.tools.remove(index);
                Ok(Vec::new())
            }
            StreamEvent::Usage { usage } => {
                self.pending_usage = Some(usage.clone());
                Ok(Vec::new())
            }
            StreamEvent::MessageStop { stop_reason } => {
                let mut frames = self.ensure_started(cx)?;
                frames.extend(self.terminal(stop_reason, cx)?);
                Ok(frames)
            }
            StreamEvent::Error { error } => {
                self.done_emitted = true;
                let envelope = ErrorEnvelopeWire {
                    error: ErrorBodyWire {
                        message: error.message.clone(),
                        kind: Some(error.kind.as_str().to_owned()),
                        code: error.code.as_ref().map(|code| Value::String(code.clone())),
                        param: None,
                    },
                };
                Ok(vec![frame(&envelope)?])
            }
            StreamEvent::Safety { .. } => Ok(Vec::new()),
            StreamEvent::ResponseMetadata { .. } => Ok(Vec::new()),
            StreamEvent::Ping => Ok(vec![SseEvent {
                data: String::new(),
                ..SseEvent::default()
            }]),
        }
    }

    fn finish(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.done_emitted {
            return Ok(Vec::new());
        }
        self.done_emitted = true;
        let mut frames = self.ensure_started(cx)?;
        if !self.stop_emitted {
            let reason = normalize_tool_stop_reason(StopReason::EndTurn, self.saw_tool_calls);
            frames.extend(self.terminal(&reason, cx)?);
        }
        frames.push(SseEvent {
            data: "[DONE]".to_owned(),
            ..SseEvent::default()
        });
        Ok(frames)
    }
}

fn frame(value: &impl Serialize) -> Result<SseEvent> {
    let data = serde_json::to_string(value)
        .map_err(|error| Error::json(Protocol::ChatCompletions, error))?;
    Ok(SseEvent {
        data,
        ..SseEvent::default()
    })
}
