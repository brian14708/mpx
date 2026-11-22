use std::collections::BTreeMap;

use http::StatusCode;
use serde_json::{Value, json};

use crate::{
    BlockDelta, CodecContext, ContentBlock, DiagnosticKind, Extra, ReasoningSeal, Result, SseEvent,
    StreamDecoder, StreamEncoder, StreamEvent, Usage,
};

use super::{
    PROTOCOL,
    content::{decode_content, decode_step, encode_content, encode_step, is_step},
    invalid,
    response::{api_error, decode_usage, encode_usage, status_from_stop, stop_from_status},
};

#[derive(Debug, Default)]
pub(super) struct Decoder {
    steps: BTreeMap<u32, Step>,
    next_block: u32,
    started: bool,
    finished: bool,
}

#[derive(Debug, Default)]
struct Step {
    block: Option<u32>,
    model_output: bool,
    text_open: bool,
}

impl Decoder {
    fn open(&mut self, step: &mut Step, block: ContentBlock, events: &mut Vec<StreamEvent>) {
        if let Some(index) = step.block.take() {
            events.push(StreamEvent::BlockStop { index });
        }
        let index = self.next_block;
        self.next_block += 1;
        events.push(StreamEvent::BlockStart { index, block });
        step.block = Some(index);
    }

    fn start_step(&mut self, wire_index: u32, value: Value) -> Result<Vec<StreamEvent>> {
        if self.steps.contains_key(&wire_index) {
            return Err(invalid(format!("step {wire_index} is already open")));
        }
        let mut step = Step {
            model_output: value["type"] == "model_output",
            ..Step::default()
        };
        let mut events = Vec::new();
        for block in decode_step(value)? {
            step.text_open = matches!(block, ContentBlock::Text { .. });
            self.open(&mut step, block, &mut events);
        }
        self.steps.insert(wire_index, step);
        Ok(events)
    }

    fn delta(&mut self, wire_index: u32, value: Value) -> Result<Vec<StreamEvent>> {
        let mut step = self
            .steps
            .remove(&wire_index)
            .ok_or_else(|| invalid(format!("step {wire_index} is not open")))?;
        let mut events = Vec::new();
        let delta = match value["type"].as_str() {
            Some("text") => {
                if !step.text_open {
                    self.open(&mut step, ContentBlock::text(""), &mut events);
                    step.text_open = true;
                }
                Some(BlockDelta::Text(delta_string(&value, "text")?))
            }
            Some("arguments_delta") => Some(BlockDelta::ToolArguments(delta_string(
                &value,
                "arguments",
            )?)),
            Some("thought_summary") if value["content"]["type"] == "text" => Some(
                BlockDelta::ReasoningSummary(delta_string(&value["content"], "text")?),
            ),
            Some("thought_signature") if value["signature"].is_string() => Some(BlockDelta::Seal(
                ReasoningSeal::new(PROTOCOL, delta_string(&value, "signature")?),
            )),
            Some("image" | "audio" | "document" | "video") if step.model_output => {
                self.open(&mut step, decode_content(value)?, &mut events);
                step.text_open = false;
                None
            }
            _ => {
                if step.block.is_none() && step.model_output {
                    self.open(&mut step, ContentBlock::text(""), &mut events);
                    step.text_open = true;
                }
                Some(BlockDelta::Native {
                    protocol: PROTOCOL,
                    json: value,
                })
            }
        };
        if let Some(delta) = delta {
            let index = step
                .block
                .ok_or_else(|| invalid(format!("step {wire_index} has no open block")))?;
            events.push(StreamEvent::BlockDelta { index, delta });
        }
        self.steps.insert(wire_index, step);
        Ok(events)
    }

    fn completed(&mut self, interaction: &Value) -> Result<Vec<StreamEvent>> {
        let status = interaction["status"]
            .as_str()
            .ok_or_else(|| invalid("missing completion status"))?;
        let mut events = Vec::new();
        if !self.started {
            events.push(message_start(interaction));
            self.started = true;
        } else if let StreamEvent::MessageStart {
            extra: Some(extra), ..
        } = message_start(interaction)
        {
            events.push(StreamEvent::ResponseMetadata { extra });
        }
        events.extend(
            core::mem::take(&mut self.steps)
                .into_values()
                .filter_map(|step| step.block.map(|index| StreamEvent::BlockStop { index })),
        );
        if let Some(usage) = interaction["usage"].as_object() {
            events.push(StreamEvent::Usage {
                usage: decode_usage(usage.clone()),
            });
        }
        self.finished = true;
        events.push(StreamEvent::MessageStop {
            stop_reason: stop_from_status(status),
        });
        Ok(events)
    }
}

impl StreamDecoder for Decoder {
    fn decode(&mut self, frame: SseEvent, cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        if frame.data.is_empty() || crate::sse::event_name(&frame) == Some("ping") {
            return Ok(vec![StreamEvent::Ping]);
        }
        if frame.data.trim() == "[DONE]" {
            return self.finish(cx);
        }
        let mut value: Value = serde_json::from_str(&frame.data)
            .map_err(|error| crate::Error::json(PROTOCOL, error))?;
        match value["event_type"]
            .as_str()
            .or_else(|| crate::sse::event_name(&frame))
        {
            Some("interaction.created") => {
                self.started = true;
                Ok(vec![message_start(&value["interaction"])])
            }
            Some("interaction.completed") => self.completed(&value["interaction"]),
            Some("step.start") => self.start_step(step_index(&value)?, value["step"].take()),
            Some("step.delta") => self.delta(step_index(&value)?, value["delta"].take()),
            Some("step.stop") => {
                let index = step_index(&value)?;
                let mut events = Vec::new();
                if let Some(block) = self.steps.remove(&index).and_then(|step| step.block) {
                    events.push(StreamEvent::BlockStop { index: block });
                }
                if let Some(usage) = value["usage"].as_object() {
                    events.push(StreamEvent::Usage {
                        usage: decode_usage(usage.clone()),
                    });
                }
                Ok(events)
            }
            Some("error") => {
                self.finished = true;
                Ok(vec![StreamEvent::Error {
                    error: api_error(&value["error"], StatusCode::BAD_GATEWAY),
                }])
            }
            Some("ping") => Ok(vec![StreamEvent::Ping]),
            _ => Ok(Vec::new()),
        }
    }

    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        Err(invalid("stream ended before interaction.completed"))
    }
}

fn message_start(value: &Value) -> StreamEvent {
    let mut fields = value.as_object().cloned().unwrap_or_default();
    let id = super::content::string(&mut fields, "id").unwrap_or_default();
    let model = super::content::string(&mut fields, "model")
        .or_else(|| {
            fields
                .get("agent")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default();
    fields.remove("status");
    fields.remove("usage");
    fields.remove("steps");
    StreamEvent::MessageStart {
        id,
        model,
        extra: Extra::from_unknown(PROTOCOL, fields),
    }
}

fn delta_string(value: &Value, key: &str) -> Result<String> {
    value[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("expected string delta field `{key}`")))
}

fn step_index(value: &Value) -> Result<u32> {
    value["index"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid("expected a valid step index"))
}

#[derive(Debug, Default)]
pub(super) struct Encoder {
    id: String,
    model: String,
    metadata: Option<Extra>,
    blocks: BTreeMap<u32, u32>,
    next_step: u32,
    usage: Option<Usage>,
    started: bool,
    finished: bool,
}

impl Encoder {
    pub(super) fn new(model: String) -> Self {
        Self {
            model,
            ..Self::default()
        }
    }

    fn interaction(&self, status: &str, cx: &CodecContext) -> Result<Value> {
        let mut fields = super::content::fields(self.metadata.as_ref(), cx, "interaction")?;
        fields.insert("id".to_owned(), json!(self.id));
        fields.insert("status".to_owned(), json!(status));
        if !self.model.is_empty() && !fields.contains_key("agent") {
            fields.insert("model".to_owned(), json!(self.model));
        }
        Ok(Value::Object(fields))
    }

    fn start(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.started {
            return Ok(Vec::new());
        }
        self.started = true;
        Ok(vec![frame(
            "interaction.created",
            json!({"interaction": self.interaction("in_progress", cx)?}),
        )])
    }

    fn start_block(
        &mut self,
        index: u32,
        block: &ContentBlock,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        let mut frames = self.start(cx)?;
        let (step, delta) = if is_step(block) {
            let Some(mut step) = encode_step(block, cx)? else {
                return Ok(frames);
            };
            let delta = if let ContentBlock::ToolCall { arguments, .. } = block {
                step["arguments"] = json!({});
                (!matches!(arguments.as_str(), "" | "{}"))
                    .then(|| json!({"type": "arguments_delta", "arguments": arguments.as_str()}))
            } else {
                None
            };
            (step, delta)
        } else {
            let Some(content) = encode_content(block, cx)? else {
                return Ok(frames);
            };
            let delta = (content["type"] != "text" || content["text"] != "").then_some(content);
            (json!({"type": "model_output"}), delta)
        };
        let wire_index = self.next_step;
        self.next_step += 1;
        self.blocks.insert(index, wire_index);
        frames.push(frame(
            "step.start",
            json!({"index": wire_index, "step": step}),
        ));
        if let Some(delta) = delta {
            frames.push(delta_frame(wire_index, delta));
        }
        Ok(frames)
    }
}

impl StreamEncoder for Encoder {
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        match event {
            StreamEvent::MessageStart { id, model, extra } => {
                if !id.is_empty() {
                    self.id = id;
                }
                if !model.is_empty() {
                    self.model = model;
                }
                if extra.is_some() {
                    self.metadata = extra;
                }
                self.start(cx)
            }
            StreamEvent::BlockStart { index, block } => self.start_block(index, &block, cx),
            StreamEvent::BlockDelta { index, delta } => {
                let Some(index) = self.blocks.get(&index).copied() else {
                    return Ok(Vec::new());
                };
                let value = match delta {
                    BlockDelta::Text(text) | BlockDelta::Refusal(text) => {
                        json!({"type": "text", "text": text})
                    }
                    BlockDelta::Reasoning(text) | BlockDelta::ReasoningSummary(text) => {
                        json!({"type": "thought_summary", "content": {"type": "text", "text": text}})
                    }
                    BlockDelta::ToolArguments(arguments) => {
                        json!({"type": "arguments_delta", "arguments": arguments})
                    }
                    BlockDelta::Seal(seal) if seal.protocol == PROTOCOL => {
                        json!({"type": "thought_signature", "signature": seal.value})
                    }
                    BlockDelta::Native {
                        protocol: PROTOCOL,
                        json,
                    } => json,
                    BlockDelta::Redacted => return Ok(Vec::new()),
                    _ => {
                        cx.diagnose(
                            DiagnosticKind::Dropped,
                            PROTOCOL,
                            "step.delta",
                            "delta belongs to another protocol",
                        )?;
                        return Ok(Vec::new());
                    }
                };
                Ok(vec![delta_frame(index, value)])
            }
            StreamEvent::BlockStop { index } => Ok(self
                .blocks
                .remove(&index)
                .map(|wire_index| frame("step.stop", json!({"index": wire_index})))
                .into_iter()
                .collect()),
            StreamEvent::Usage { usage } => {
                self.usage
                    .get_or_insert_with(Usage::default)
                    .merge_nonzero(&usage);
                Ok(Vec::new())
            }
            StreamEvent::ResponseMetadata { extra } => {
                if let Some(current) = self.metadata.as_mut()
                    && current.protocol == extra.protocol
                {
                    current.fields.extend(extra.fields);
                } else {
                    self.metadata = Some(extra);
                }
                Ok(Vec::new())
            }
            StreamEvent::MessageStop { stop_reason } => {
                let mut frames = self.start(cx)?;
                for index in core::mem::take(&mut self.blocks).into_values() {
                    frames.push(frame("step.stop", json!({"index": index})));
                }
                let mut interaction = self.interaction(&status_from_stop(&stop_reason, cx)?, cx)?;
                if let Some(usage) = &self.usage {
                    interaction["usage"] = encode_usage(usage, cx)?;
                }
                frames.push(frame(
                    "interaction.completed",
                    json!({"interaction": interaction}),
                ));
                frames.push(crate::sse::named("done", "[DONE]"));
                self.finished = true;
                Ok(frames)
            }
            StreamEvent::Error { error } => {
                self.finished = true;
                Ok(vec![frame(
                    "error",
                    json!({"error": {
                        "code": error.code.as_deref().unwrap_or_else(|| error.kind.as_str()),
                        "message": error.message
                    }}),
                )])
            }
            StreamEvent::Safety { ratings } => {
                if !ratings.is_empty() {
                    cx.diagnose(
                        DiagnosticKind::Dropped,
                        PROTOCOL,
                        "response.safety",
                        "interactions do not expose per-candidate safety ratings",
                    )?;
                }
                Ok(Vec::new())
            }
            StreamEvent::Ping => Ok(Vec::new()),
        }
    }
}

fn frame(event: &str, mut value: Value) -> SseEvent {
    value["event_type"] = json!(event);
    crate::sse::named(event, value.to_string())
}

fn delta_frame(index: u32, value: Value) -> SseEvent {
    frame("step.delta", json!({"index": index, "delta": value}))
}
