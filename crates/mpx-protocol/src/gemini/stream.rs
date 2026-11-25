use serde_json::Value;

use crate::json::nonempty;
use crate::{
    BlockDelta, CodecContext, ContentBlock, DiagnosticKind, Error, Protocol, Result, SseEvent,
    StopReason, StreamDecoder, StreamEncoder, StreamEvent, ToolArguments, Usage,
};

use super::{
    codec::{
        decode_blocks, decode_error_from_value, decode_usage, encode_error_value, encode_part,
        encode_usage, normalized_stop_reason, response_chunk, response_extra,
    },
    wire::{GenerateContentResponse, Part},
};

const PROTOCOL: Protocol = Protocol::GenerateContent;

#[derive(Debug, Default)]
pub(super) struct GeminiStreamDecoder {
    started: bool,
    stopped: bool,
    next_index: u32,
    chunk_index: usize,
    call_index: usize,
    open: Option<OpenKind>,
    saw_tool_call: bool,
    pending_usage: Option<Usage>,
}

#[derive(Debug, Clone, Copy)]
struct OpenKind {
    index: u32,
    kind: StreamingKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingKind {
    Text,
    Reasoning,
    Refusal,
}

impl StreamDecoder for GeminiStreamDecoder {
    fn decode(&mut self, frame: SseEvent, cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.stopped {
            return Ok(Vec::new());
        }
        if frame.data.trim().is_empty() {
            return Ok(vec![StreamEvent::Ping]);
        }
        if frame.data.trim() == "[DONE]" {
            return Ok(self.terminal_events());
        }
        let mut value: Value =
            serde_json::from_str(&frame.data).map_err(|error| Error::json(PROTOCOL, error))?;
        if let Some(wrapped) = value
            .as_object_mut()
            .and_then(|object| object.remove("response"))
        {
            value = wrapped;
        }
        if value
            .as_object()
            .is_some_and(|object| object.contains_key("error"))
        {
            return Ok(vec![decode_error_from_value(value)?]);
        }
        let wire: GenerateContentResponse =
            serde_json::from_value(value).map_err(|error| Error::json(PROTOCOL, error))?;
        self.decode_response(wire, cx)
    }

    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.stopped {
            Ok(Vec::new())
        } else {
            Err(Error::Body {
                message: "Gemini stream ended before its finish reason".to_owned(),
            })
        }
    }
}

impl GeminiStreamDecoder {
    fn terminal_events(&mut self) -> Vec<StreamEvent> {
        if self.stopped {
            return Vec::new();
        }
        let mut events = Vec::new();
        close_open(&mut self.open, &mut events);
        if let Some(usage) = self.pending_usage.take() {
            events.push(StreamEvent::Usage { usage });
        }
        events.push(StreamEvent::MessageStop {
            stop_reason: if self.saw_tool_call {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            },
        });
        self.stopped = true;
        events
    }

    fn decode_response(
        &mut self,
        mut wire: GenerateContentResponse,
        cx: &CodecContext,
    ) -> Result<Vec<StreamEvent>> {
        let empty_reply = !self.started
            && wire.candidates.is_empty()
            && wire
                .usage_metadata
                .as_ref()
                .is_some_and(|usage| usage.prompt_token_count == Some(0));
        let mut events = Vec::new();
        if !self.started {
            events.push(StreamEvent::MessageStart {
                id: wire.response_id.clone().unwrap_or_default(),
                model: wire.model_version.clone().unwrap_or_default(),
                extra: super::codec::make_extra(std::mem::take(&mut wire.extra)),
            });
            self.started = true;
        }

        let mut finish_reason = None;
        let blocked = wire
            .prompt_feedback
            .as_ref()
            .is_some_and(|feedback| feedback.block_reason.is_some());
        if wire.candidates.len() > 1 {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "candidates",
                "only the first Gemini streaming candidate is represented",
            )?;
        }
        let candidate = if wire.candidates.is_empty() {
            None
        } else {
            Some(wire.candidates.remove(0))
        };
        if let Some(candidate) = candidate {
            finish_reason = candidate.finish_reason;
            if let Some(content) = candidate.content {
                let position = self.chunk_index;
                self.chunk_index = self.chunk_index.saturating_add(1);
                let blocks =
                    decode_blocks(content.parts, "candidates[0].content.parts", position, cx)?;
                for block in blocks {
                    self.emit_block(block, &mut events);
                }
            }
        }

        let usage = wire.usage_metadata.map(|usage| decode_usage(Some(usage)));
        let finishing = finish_reason.is_some() || blocked || empty_reply;
        if finishing {
            close_open(&mut self.open, &mut events);
            if let Some(usage) = usage.or_else(|| self.pending_usage.take()) {
                events.push(StreamEvent::Usage { usage });
            }
        } else if usage.is_some() {
            self.pending_usage = usage;
        }
        if finishing {
            let stop_reason =
                normalized_stop_reason(finish_reason.as_ref(), blocked, self.saw_tool_call);
            events.push(StreamEvent::MessageStop { stop_reason });
            self.stopped = true;
        }
        Ok(events)
    }

    fn emit_block(&mut self, block: ContentBlock, events: &mut Vec<StreamEvent>) {
        match block {
            ContentBlock::Text { text, extra } => {
                self.emit_streaming_text(StreamingKind::Text, text, extra, None, events);
            }
            ContentBlock::Reasoning {
                text,
                summary,
                seal,
                redacted,
                extra,
            } => {
                let mut text = text;
                if !summary.is_empty() {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&summary.join("\n"));
                }
                self.emit_streaming_text(StreamingKind::Reasoning, text, extra, seal, events);
                if redacted && let Some(open) = self.open {
                    events.push(StreamEvent::BlockDelta {
                        index: open.index,
                        delta: BlockDelta::Redacted,
                    });
                }
            }
            ContentBlock::Refusal { text, extra } => {
                self.emit_streaming_text(StreamingKind::Refusal, text, extra, None, events);
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                seal,
                extra,
            } => {
                close_open(&mut self.open, events);
                let index = self.take_index();
                self.saw_tool_call = true;
                self.call_index = self.call_index.saturating_add(1);
                let id = if id.as_str().is_empty() {
                    crate::tool::synthesized_tool_call_id(self.call_index)
                } else {
                    id.into_string()
                };
                events.push(StreamEvent::BlockStart {
                    index,
                    block: ContentBlock::ToolCall {
                        id: id.into(),
                        name,
                        arguments: ToolArguments::new(""),
                        seal,
                        extra,
                    },
                });
                if !arguments.as_str().is_empty() {
                    events.push(StreamEvent::BlockDelta {
                        index,
                        delta: BlockDelta::ToolArguments(arguments.into_string()),
                    });
                }
                events.push(StreamEvent::BlockStop { index });
            }
            other => {
                close_open(&mut self.open, events);
                let index = self.take_index();
                events.push(StreamEvent::BlockStart {
                    index,
                    block: other,
                });
                events.push(StreamEvent::BlockStop { index });
            }
        }
    }

    fn emit_streaming_text(
        &mut self,
        kind: StreamingKind,
        text: String,
        extra: Option<crate::Extra>,
        seal: Option<crate::ReasoningSeal>,
        events: &mut Vec<StreamEvent>,
    ) {
        let index = match self.open {
            Some(open) if open.kind == kind => {
                if let Some(seal) = seal {
                    events.push(StreamEvent::BlockDelta {
                        index: open.index,
                        delta: BlockDelta::Seal(seal),
                    });
                }
                open.index
            }
            _ => {
                close_open(&mut self.open, events);
                let index = self.take_index();
                let block = match kind {
                    StreamingKind::Text => ContentBlock::Text {
                        text: String::new(),
                        extra,
                    },
                    StreamingKind::Reasoning => ContentBlock::Reasoning {
                        text: String::new(),
                        summary: Vec::new(),
                        seal: seal.clone(),
                        redacted: false,
                        extra,
                    },
                    StreamingKind::Refusal => ContentBlock::Refusal {
                        text: String::new(),
                        extra,
                    },
                };
                events.push(StreamEvent::BlockStart { index, block });
                self.open = Some(OpenKind { index, kind });
                index
            }
        };
        if !text.is_empty() {
            let delta = match kind {
                StreamingKind::Text => BlockDelta::Text(text),
                StreamingKind::Reasoning => BlockDelta::Reasoning(text),
                StreamingKind::Refusal => BlockDelta::Refusal(text),
            };
            events.push(StreamEvent::BlockDelta { index, delta });
        }
    }

    const fn take_index(&mut self) -> u32 {
        let index = self.next_index;
        self.next_index = self.next_index.saturating_add(1);
        index
    }
}

fn close_open(open: &mut Option<OpenKind>, events: &mut Vec<StreamEvent>) {
    if let Some(open) = open.take() {
        events.push(StreamEvent::BlockStop { index: open.index });
    }
}

#[derive(Debug, Default)]
pub(super) struct GeminiStreamEncoder {
    id: String,
    model: String,
    extra: Option<crate::Extra>,
    extra_pending: bool,
    open: Option<EncoderBlock>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    finished: bool,
    errored: bool,
}

#[derive(Debug)]
struct EncoderBlock {
    index: u32,
    block: ContentBlock,
    emitted: bool,
}

impl StreamEncoder for GeminiStreamEncoder {
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        match &event {
            StreamEvent::MessageStart { id, model, extra } => {
                let frames = self.flush_open(cx)?;
                self.id.clone_from(id);
                self.model.clone_from(model);
                self.extra.clone_from(extra);
                self.extra_pending = true;
                Ok(frames)
            }
            StreamEvent::BlockStart { index, block } => {
                let frames = self.flush_open(cx)?;
                self.open = Some(EncoderBlock {
                    index: *index,
                    block: initial_stream_block(block),
                    emitted: false,
                });
                Ok(frames)
            }
            StreamEvent::BlockDelta { index, delta } => self.encode_delta(*index, delta, cx),
            StreamEvent::BlockStop { index } => {
                if self.open.as_ref().is_some_and(|open| open.index == *index) {
                    self.flush_open(cx)
                } else {
                    Ok(Vec::new())
                }
            }
            StreamEvent::Usage { usage } => {
                self.usage = Some(usage.clone());
                Ok(Vec::new())
            }
            StreamEvent::MessageStop { stop_reason } => {
                let mut frames = self.flush_open(cx)?;
                self.stop_reason = Some(stop_reason.clone());
                if !self.finished && !self.errored {
                    frames.push(self.final_frame(cx)?);
                }
                Ok(frames)
            }
            StreamEvent::Error { error } => {
                self.errored = true;
                self.finished = true;
                Ok(vec![json_frame(&encode_error_value(error))?])
            }
            StreamEvent::Safety { .. } | StreamEvent::ResponseMetadata { .. } => Ok(Vec::new()),
            StreamEvent::Ping => Ok(vec![SseEvent {
                data: String::new(),
                ..SseEvent::default()
            }]),
        }
    }

    fn finish(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut frames = self.flush_open(cx)?;
        if !self.errored {
            frames.push(self.final_frame(cx)?);
        }
        Ok(frames)
    }
}

impl GeminiStreamEncoder {
    fn encode_delta(
        &mut self,
        index: u32,
        delta: &BlockDelta,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        if self.open.as_ref().is_none_or(|open| open.index != index) {
            return Err(Error::InvalidRequest {
                protocol: PROTOCOL,
                message: format!("stream delta references unopened block {index}"),
            });
        }
        match delta {
            BlockDelta::Text(text) => {
                self.emit_text_delta(index, text.clone(), StreamingKind::Text, cx)
            }
            BlockDelta::Reasoning(text) | BlockDelta::ReasoningSummary(text) => {
                self.emit_text_delta(index, text.clone(), StreamingKind::Reasoning, cx)
            }
            BlockDelta::Refusal(text) => {
                self.emit_text_delta(index, text.clone(), StreamingKind::Refusal, cx)
            }
            BlockDelta::ToolArguments(arguments) => {
                if let Some(open) = &mut self.open
                    && let ContentBlock::ToolCall {
                        arguments: target, ..
                    } = &mut open.block
                {
                    target.as_mut_string().push_str(arguments);
                    return Ok(Vec::new());
                }
                Err(mismatched_delta(index, "tool arguments"))
            }
            BlockDelta::Seal(seal) => {
                let should_emit = {
                    let open = self
                        .open
                        .as_mut()
                        .ok_or_else(|| mismatched_delta(index, "seal"))?;
                    let emitted = open.emitted;
                    let reasoning = matches!(open.block, ContentBlock::Reasoning { .. });
                    match &mut open.block {
                        ContentBlock::Reasoning { seal: target, .. }
                        | ContentBlock::ToolCall { seal: target, .. } => {
                            *target = Some(seal.clone());
                            Ok(emitted && reasoning)
                        }
                        _ => Err(mismatched_delta(index, "seal")),
                    }
                }?;
                if should_emit {
                    let part = Part {
                        thought_signature: Some(seal.value.clone()),
                        ..Part::default()
                    };
                    self.part_frame(part, cx).map(|frame| vec![frame])
                } else {
                    Ok(Vec::new())
                }
            }
            BlockDelta::Redacted => {
                if let Some(open) = &mut self.open
                    && let ContentBlock::Reasoning { redacted, .. } = &mut open.block
                {
                    *redacted = true;
                    return Ok(Vec::new());
                }
                Err(mismatched_delta(index, "redaction"))
            }
            BlockDelta::Native {
                protocol: PROTOCOL,
                json,
            } => {
                let part: Part = serde_json::from_value(json.clone())
                    .map_err(|error| Error::json(PROTOCOL, error))?;
                self.part_frame(part, cx).map(|frame| vec![frame])
            }
            BlockDelta::Native { .. } => {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    "stream.native",
                    "foreign native stream delta was omitted",
                )?;
                Ok(Vec::new())
            }
        }
    }

    fn emit_text_delta(
        &mut self,
        index: u32,
        text: String,
        kind: StreamingKind,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        let block = {
            let open = self
                .open
                .as_mut()
                .ok_or_else(|| mismatched_delta(index, "text"))?;
            let matches = matches!(
                (&open.block, kind),
                (ContentBlock::Text { .. }, StreamingKind::Text)
                    | (ContentBlock::Reasoning { .. }, StreamingKind::Reasoning)
                    | (ContentBlock::Refusal { .. }, StreamingKind::Refusal)
            );
            if !matches {
                return Err(mismatched_delta(index, "text kind"));
            }
            let first_emission = !open.emitted;
            open.emitted = true;
            match &mut open.block {
                ContentBlock::Text {
                    text: target,
                    extra,
                } => {
                    target.push_str(&text);
                    ContentBlock::Text {
                        text: if first_emission { target.clone() } else { text },
                        extra: extra.clone(),
                    }
                }
                ContentBlock::Reasoning {
                    text: target,
                    summary,
                    seal,
                    redacted,
                    extra,
                } => {
                    target.push_str(&text);
                    ContentBlock::Reasoning {
                        text: if first_emission { target.clone() } else { text },
                        summary: if first_emission {
                            summary.clone()
                        } else {
                            Vec::new()
                        },
                        seal: seal.clone(),
                        redacted: *redacted,
                        extra: extra.clone(),
                    }
                }
                ContentBlock::Refusal {
                    text: target,
                    extra,
                } => {
                    target.push_str(&text);
                    ContentBlock::Refusal {
                        text: if first_emission { target.clone() } else { text },
                        extra: extra.clone(),
                    }
                }
                _ => return Err(mismatched_delta(index, "text kind")),
            }
        };
        let part =
            encode_part(&block, "stream.content", cx)?.ok_or_else(|| Error::InvalidRequest {
                protocol: PROTOCOL,
                message: "text stream delta was unexpectedly omitted".to_owned(),
            })?;
        self.part_frame(part, cx).map(|frame| vec![frame])
    }

    fn flush_open(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        let Some(open) = self.open.take() else {
            return Ok(Vec::new());
        };
        if open.emitted {
            return Ok(Vec::new());
        }
        match encode_part(&open.block, "stream.content", cx)? {
            Some(part) => Ok(vec![self.part_frame(part, cx)?]),
            None => Ok(Vec::new()),
        }
    }

    fn part_frame(&mut self, part: Part, cx: &CodecContext) -> Result<SseEvent> {
        let extra = self.take_response_extra(cx)?;
        let response = response_chunk(
            vec![part],
            None,
            None,
            nonempty(&self.model),
            nonempty(&self.id),
            extra,
        );
        json_frame(&response)
    }

    fn final_frame(&mut self, cx: &CodecContext) -> Result<SseEvent> {
        self.finished = true;
        let usage = match &self.usage {
            Some(usage) => Some(encode_usage(usage, cx)?),
            None => None,
        };
        let finish_reason = self.stop_reason.as_ref().map(Into::into);
        let response = response_chunk(
            Vec::new(),
            finish_reason,
            usage,
            nonempty(&self.model),
            nonempty(&self.id),
            self.take_response_extra(cx)?,
        );
        json_frame(&response)
    }

    fn take_response_extra(&mut self, cx: &CodecContext) -> Result<serde_json::Map<String, Value>> {
        if self.extra_pending {
            self.extra_pending = false;
            response_extra(self.extra.as_ref(), cx, "stream.extra")
        } else {
            Ok(serde_json::Map::new())
        }
    }
}

fn initial_stream_block(block: &ContentBlock) -> ContentBlock {
    match block {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            seal,
            extra,
        } => ContentBlock::ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: if arguments.as_str() == "{}" {
                ToolArguments::new("")
            } else {
                arguments.clone()
            },
            seal: seal.clone(),
            extra: extra.clone(),
        },
        other => other.clone(),
    }
}

fn mismatched_delta(index: u32, kind: &str) -> Error {
    Error::InvalidRequest {
        protocol: PROTOCOL,
        message: format!("{kind} delta does not match open block {index}"),
    }
}

fn json_frame(value: &impl serde::Serialize) -> Result<SseEvent> {
    Ok(SseEvent {
        data: serde_json::to_string(value).map_err(|error| Error::json(PROTOCOL, error))?,
        ..SseEvent::default()
    })
}
