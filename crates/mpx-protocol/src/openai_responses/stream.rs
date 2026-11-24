use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::{
    ApiError, ApiErrorKind, BlockDelta, ClientContext, CodecContext, ContentBlock, Protocol,
    ReasoningSeal, Response, Result, SseEvent, StopReason, StreamDecoder, StreamEncoder,
    StreamEvent, ToolArguments, Usage,
    openai_responses::{
        codec::{self, DecodedItem},
        wire::{self, ResponsesUsage},
    },
    response::normalize_tool_stop_reason,
};

/// Stateful Responses named-event decoder.
#[derive(Debug)]
pub(crate) struct ResponsesStreamDecoder {
    started: HashSet<u32>,
    arguments: HashMap<u32, String>,
    saw_tool_call: bool,
    stopped: bool,
}

impl ResponsesStreamDecoder {
    pub(crate) fn new() -> Self {
        Self {
            started: HashSet::new(),
            arguments: HashMap::new(),
            saw_tool_call: false,
            stopped: false,
        }
    }

    fn item_start(item: Value, index: u32, cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        let DecodedItem(_, blocks, _) = codec::decode_item(item, cx)?;
        Ok(blocks
            .into_iter()
            .map(|block| StreamEvent::BlockStart { index, block })
            .collect())
    }
}

impl StreamDecoder for ResponsesStreamDecoder {
    fn decode(&mut self, frame: SseEvent, cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        if self.stopped || frame.data.trim() == "[DONE]" {
            return Ok(Vec::new());
        }
        let value: Value =
            serde_json::from_str(&frame.data).map_err(|error| crate::Error::Protocol {
                protocol: Protocol::Responses,
                message: format!("invalid SSE JSON: {error}"),
            })?;
        let kind = crate::sse::event_name(&frame)
            .or_else(|| value.get("type").and_then(Value::as_str))
            .unwrap_or_default();
        let index = stream_index(&value);
        match kind {
            "response.created" | "response.in_progress" if kind == "response.created" => {
                let response = value.get("response").cloned().unwrap_or(Value::Null);
                let Value::Object(mut map) = response else {
                    return Ok(Vec::new());
                };
                let id = wire::take_string(&mut map, "id").unwrap_or_default();
                let model = wire::take_string(&mut map, "model").unwrap_or_default();
                Ok(vec![StreamEvent::MessageStart {
                    id,
                    model,
                    extra: wire::extra_fields(map),
                }])
            }
            "response.output_item.added" => {
                let item = value.get("item").cloned().unwrap_or(Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("message") {
                    return Ok(Vec::new());
                }
                if matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("function_call" | "custom_tool_call")
                ) {
                    self.saw_tool_call = true;
                }
                let events = Self::item_start(item, index, cx)?;
                if !events.is_empty() {
                    self.started.insert(index);
                }
                if let Some(StreamEvent::BlockStart {
                    block: ContentBlock::ToolCall { arguments, .. },
                    ..
                }) = events.first()
                {
                    self.arguments.insert(index, arguments.as_str().to_owned());
                }
                Ok(events)
            }
            "response.content_part.added" => {
                let part = value.get("part").cloned().unwrap_or(Value::Null);
                let block = codec::decode_content_part(part, cx)?;
                self.started.insert(index);
                Ok(vec![StreamEvent::BlockStart { index, block }])
            }
            "response.output_text.delta" => Ok(vec![StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Text(
                    value
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
            }]),
            "response.refusal.delta" => Ok(vec![StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Refusal(
                    value
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
            }]),
            "response.reasoning_text.delta" => Ok(vec![StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::Reasoning(
                    value
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
            }]),
            "response.reasoning_summary_text.delta" => Ok(vec![StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::ReasoningSummary(
                    value
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
            }]),
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                let delta = value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.arguments.entry(index).or_default().push_str(delta);
                Ok(vec![StreamEvent::BlockDelta {
                    index,
                    delta: BlockDelta::ToolArguments(delta.to_owned()),
                }])
            }
            "response.content_part.done" => Ok(vec![StreamEvent::BlockStop { index }]),
            "response.output_item.done" => {
                let item = value.get("item").cloned().unwrap_or(Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("message") {
                    return Ok(Vec::new());
                }
                let DecodedItem(_, blocks, _) = codec::decode_item(item, cx)?;
                let mut events = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            seal,
                            extra,
                        } => {
                            if !self.started.contains(&index) {
                                events.push(StreamEvent::BlockStart {
                                    index,
                                    block: ContentBlock::ToolCall {
                                        id,
                                        name,
                                        arguments: ToolArguments::new(""),
                                        seal,
                                        extra,
                                    },
                                });
                            }
                            let streamed = self.arguments.get(&index).map_or(0, String::len);
                            let complete = arguments.as_str();
                            if complete.len() > streamed {
                                if self
                                    .arguments
                                    .get(&index)
                                    .is_none_or(|prefix| complete.starts_with(prefix))
                                {
                                    events.push(StreamEvent::BlockDelta {
                                        index,
                                        delta: BlockDelta::ToolArguments(
                                            complete[streamed..].to_owned(),
                                        ),
                                    });
                                } else {
                                    cx.diagnose(crate::DiagnosticKind::Approximated, Protocol::Responses, "output.arguments", "done arguments diverged from streamed deltas; streamed prefix was retained")?;
                                }
                            }
                        }
                        ContentBlock::Reasoning {
                            seal: Some(seal), ..
                        } => events.push(StreamEvent::BlockDelta {
                            index,
                            delta: BlockDelta::Seal(seal),
                        }),
                        _ => {}
                    }
                }
                events.push(StreamEvent::BlockStop { index });
                self.started.insert(index);
                Ok(events)
            }
            "response.completed" | "response.incomplete" => {
                let response = value.get("response").unwrap_or(&Value::Null);
                let usage = response
                    .get("usage")
                    .cloned()
                    .map(|usage| {
                        serde_json::from_value::<ResponsesUsage>(usage).map_err(wire::json_error)
                    })
                    .transpose()?
                    .map(codec::decode_usage)
                    .unwrap_or_default();
                let reason = if kind == "response.incomplete" {
                    if response
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str)
                        == Some("max_output_tokens")
                    {
                        StopReason::MaxTokens
                    } else {
                        StopReason::Incomplete
                    }
                } else {
                    StopReason::EndTurn
                };
                let reason = normalize_tool_stop_reason(reason, self.saw_tool_call);
                self.stopped = true;
                Ok(vec![
                    StreamEvent::Usage { usage },
                    StreamEvent::MessageStop {
                        stop_reason: reason,
                    },
                ])
            }
            "response.failed" | "response.error" | "error" => {
                let error = value
                    .pointer("/response/error")
                    .or_else(|| value.get("error"))
                    .unwrap_or(&value);
                self.stopped = true;
                Ok(vec![StreamEvent::Error {
                    error: ApiError {
                        kind: ApiErrorKind::Upstream,
                        message: error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("Responses stream failed")
                            .to_owned(),
                        code: error.get("code").and_then(Value::as_str).map(str::to_owned),
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
                message: "Responses stream ended before its terminal event".to_owned(),
            })
        }
    }
}

fn stream_index(value: &Value) -> u32 {
    let output = value
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    let content = value
        .get("content_index")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    if value.get("content_index").is_some() {
        output.rotate_left(16) ^ content
    } else {
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Text,
    Refusal,
    Reasoning,
    Function,
    Custom,
    Other,
}

#[derive(Debug)]
struct OpenItem {
    output_index: u32,
    item_id: String,
    kind: ItemKind,
    text: String,
    summary: String,
    seal: Option<ReasoningSeal>,
    call_id: Option<crate::ToolCallId>,
    name: Option<String>,
}

/// Stateful Responses lifecycle encoder.
#[derive(Debug)]
pub(crate) struct ResponsesStreamEncoder {
    client: ClientContext,
    response_id: String,
    model: String,
    extra: Option<crate::Extra>,
    usage: Usage,
    sequence: u64,
    next_output: u32,
    open: HashMap<u32, OpenItem>,
    completed_items: Vec<Value>,
    stopped: bool,
}

impl ResponsesStreamEncoder {
    pub(crate) fn new(client: ClientContext) -> Self {
        Self {
            client,
            response_id: String::new(),
            model: String::new(),
            extra: None,
            usage: Usage::default(),
            sequence: 0,
            next_output: 0,
            open: HashMap::new(),
            completed_items: Vec::new(),
            stopped: false,
        }
    }

    fn event(&mut self, kind: &str, mut value: Value) -> SseEvent {
        value["type"] = Value::String(kind.to_owned());
        value["sequence_number"] = json!(self.sequence);
        self.sequence = self.sequence.saturating_add(1);
        crate::sse::named(kind, value.to_string())
    }

    fn created_frames(
        &mut self,
        id: String,
        model: String,
        extra: Option<crate::Extra>,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        self.response_id.clone_from(&id);
        self.model.clone_from(&model);
        self.extra.clone_from(&extra);
        let response = Response {
            id,
            model,
            extra,
            ..Response::default()
        };
        let mut created = codec::response_header(&response, &self.client, cx)?;
        created.insert("status".to_owned(), Value::String("in_progress".to_owned()));
        created.insert("output".to_owned(), Value::Array(Vec::new()));
        created.insert("usage".to_owned(), Value::Null);
        let first = self.event(
            "response.created",
            json!({"response":Value::Object(created.clone())}),
        );
        let second = self.event(
            "response.in_progress",
            json!({"response":Value::Object(created)}),
        );
        Ok(vec![first, second])
    }

    fn start_item(
        &mut self,
        index: u32,
        block: &ContentBlock,
        cx: &CodecContext,
    ) -> Result<Vec<SseEvent>> {
        let output_index = self.next_output;
        self.next_output = self.next_output.saturating_add(1);
        let original_id = block_extra(block)
            .and_then(|extra| extra.fields.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let (kind, prefix) = match block {
            ContentBlock::Reasoning { .. } => (ItemKind::Reasoning, "rs"),
            ContentBlock::ToolCall { extra, .. }
                if extra.as_ref().is_some_and(|extra| {
                    extra.fields.get("type").and_then(Value::as_str) == Some("custom_tool_call")
                }) =>
            {
                (ItemKind::Custom, "ctc")
            }
            ContentBlock::ToolCall { .. } => (ItemKind::Function, "fc"),
            ContentBlock::Refusal { .. } => (ItemKind::Refusal, "msg"),
            ContentBlock::Text { .. } => (ItemKind::Text, "msg"),
            _ => (ItemKind::Other, "msg"),
        };
        let item_id = original_id.unwrap_or_else(|| format!("{prefix}_{output_index}"));
        let (call_id, name) = match block {
            ContentBlock::ToolCall { id, name, .. } => (Some(id.clone()), Some(name.clone())),
            _ => (None, None),
        };
        let item = OpenItem {
            output_index,
            item_id: item_id.clone(),
            kind,
            text: String::new(),
            summary: String::new(),
            seal: None,
            call_id,
            name,
        };
        self.open.insert(index, item);
        let mut frames = Vec::new();
        match kind {
            ItemKind::Reasoning => {
                frames.push(self.event("response.output_item.added", json!({"output_index":output_index,"item":{"id":item_id,"type":"reasoning","summary":[]}})));
            }
            ItemKind::Function | ItemKind::Custom => {
                let ContentBlock::ToolCall { id, name, .. } = block else {
                    return Ok(frames);
                };
                let item = if kind == ItemKind::Custom {
                    json!({"id":item_id,"type":"custom_tool_call","call_id":id.as_str(),"name":name,"input":""})
                } else {
                    json!({"id":item_id,"type":"function_call","call_id":id.as_str(),"name":name,"arguments":"","status":"in_progress"})
                };
                frames.push(self.event(
                    "response.output_item.added",
                    json!({"output_index":output_index,"item":item}),
                ));
            }
            ItemKind::Text | ItemKind::Refusal | ItemKind::Other => {
                let item = json!({"id":item_id,"type":"message","status":"in_progress","role":"assistant","content":[]});
                frames.push(self.event(
                    "response.output_item.added",
                    json!({"output_index":output_index,"item":item}),
                ));
                let part_type = if kind == ItemKind::Refusal {
                    "refusal"
                } else {
                    "output_text"
                };
                let part = if kind == ItemKind::Refusal {
                    json!({"type":part_type,"refusal":""})
                } else {
                    json!({"type":part_type,"text":"","annotations":[]})
                };
                frames.push(self.event("response.content_part.added", json!({"item_id":item_id,"output_index":output_index,"content_index":0,"part":part})));
            }
        }
        let _ = cx;
        Ok(frames)
    }

    fn delta_item(&mut self, index: u32, delta: &BlockDelta) -> Option<SseEvent> {
        let item = self.open.get_mut(&index)?;
        let output_index = item.output_index;
        let item_id = item.item_id.clone();
        let text = match delta {
            BlockDelta::Text(text)
            | BlockDelta::Reasoning(text)
            | BlockDelta::ReasoningSummary(text)
            | BlockDelta::ToolArguments(text)
            | BlockDelta::Refusal(text) => text,
            BlockDelta::Seal(seal) => {
                item.seal = Some(seal.clone());
                return None;
            }
            BlockDelta::Redacted | BlockDelta::Native { .. } => return None,
        };
        if matches!(delta, BlockDelta::ReasoningSummary(_)) {
            item.summary.push_str(text);
        } else {
            item.text.push_str(text);
        }
        let (kind, mut value) = match (item.kind, delta) {
            (ItemKind::Text | ItemKind::Other, BlockDelta::Text(_)) => (
                "response.output_text.delta",
                json!({"item_id":item_id,"output_index":output_index,"content_index":0,"delta":text}),
            ),
            (ItemKind::Refusal, _) => (
                "response.refusal.delta",
                json!({"item_id":item_id,"output_index":output_index,"content_index":0,"delta":text}),
            ),
            (ItemKind::Reasoning, BlockDelta::ReasoningSummary(_)) => (
                "response.reasoning_summary_text.delta",
                json!({"item_id":item_id,"output_index":output_index,"summary_index":0,"delta":text}),
            ),
            (ItemKind::Reasoning, _) => (
                "response.reasoning_text.delta",
                json!({"item_id":item_id,"output_index":output_index,"content_index":0,"delta":text}),
            ),
            (ItemKind::Function, _) => (
                "response.function_call_arguments.delta",
                json!({"item_id":item_id,"output_index":output_index,"delta":text}),
            ),
            (ItemKind::Custom, _) => (
                "response.custom_tool_call_input.delta",
                json!({"item_id":item_id,"output_index":output_index,"delta":text}),
            ),
            _ => return None,
        };
        if let Value::Object(map) = &mut value {
            map.insert("type".to_owned(), Value::String(kind.to_owned()));
        }
        Some(self.event(kind, value))
    }

    fn stop_item(&mut self, index: u32) -> Vec<SseEvent> {
        let Some(item) = self.open.remove(&index) else {
            return Vec::new();
        };
        let output_index = item.output_index;
        let item_id = item.item_id.clone();
        let mut frames = Vec::new();
        let complete_item = match item.kind {
            ItemKind::Text | ItemKind::Other | ItemKind::Refusal => {
                let (done_kind, part) = if item.kind == ItemKind::Refusal {
                    (
                        "response.refusal.done",
                        json!({"type":"refusal","refusal":item.text}),
                    )
                } else {
                    (
                        "response.output_text.done",
                        json!({"type":"output_text","text":item.text,"annotations":[]}),
                    )
                };
                frames.push(self.event(done_kind, json!({"item_id":item_id,"output_index":output_index,"content_index":0,"text":item.text})));
                frames.push(self.event("response.content_part.done", json!({"item_id":item_id,"output_index":output_index,"content_index":0,"part":part})));
                let content = if item.kind == ItemKind::Refusal {
                    json!([{"type":"refusal","refusal":item.text}])
                } else {
                    json!([{"type":"output_text","text":item.text,"annotations":[]}])
                };
                json!({"id":item_id,"type":"message","status":"completed","role":"assistant","content":content})
            }
            ItemKind::Reasoning => {
                let summary = if item.summary.is_empty() {
                    Vec::new()
                } else {
                    vec![json!({"type":"summary_text","text":item.summary})]
                };
                let mut value = json!({"id":item_id,"type":"reasoning","summary":summary});
                if !item.text.is_empty() {
                    value["content"] = json!([{"type":"reasoning_text","text":item.text}]);
                }
                if let Some(seal) = item
                    .seal
                    .as_ref()
                    .filter(|seal| seal.protocol == Protocol::Responses)
                {
                    value["encrypted_content"] = Value::String(seal.value.clone());
                }
                value
            }
            ItemKind::Function | ItemKind::Custom => {
                let (kind, item_type, field) = if item.kind == ItemKind::Custom {
                    (
                        "response.custom_tool_call_input.done",
                        "custom_tool_call",
                        "input",
                    )
                } else {
                    (
                        "response.function_call_arguments.done",
                        "function_call",
                        "arguments",
                    )
                };
                frames.push(self.event(
                    kind,
                    json!({"item_id":item_id,"output_index":output_index,field:item.text}),
                ));
                // IDs and names are captured at item start, so parallel calls
                // cannot accidentally reuse the first call's identity.
                let call_id = item.call_id.unwrap_or_default();
                let name = item.name.unwrap_or_default();
                let mut value = json!({"id":item_id,"type":item_type,"call_id":call_id.as_str(),"name":name,field:item.text});
                if item.kind == ItemKind::Function {
                    value["status"] = Value::String("completed".to_owned());
                }
                value
            }
        };
        frames.push(self.event(
            "response.output_item.done",
            json!({"output_index":output_index,"item":complete_item}),
        ));
        self.completed_items.push(complete_item);
        frames
    }

    fn final_frame(&mut self, reason: StopReason, cx: &CodecContext) -> Result<SseEvent> {
        self.stopped = true;
        let response = Response {
            id: self.response_id.clone(),
            model: self.model.clone(),
            stop_reason: reason.clone(),
            usage: self.usage.clone(),
            extra: self.extra.clone(),
            ..Response::default()
        };
        let mut object = codec::response_header(&response, &self.client, cx)?;
        object.insert(
            "output".to_owned(),
            Value::Array(self.completed_items.clone()),
        );
        object.insert("error".to_owned(), Value::Null);
        object.insert("usage".to_owned(), codec::encode_usage(&self.usage));
        if reason == StopReason::MaxTokens {
            object.insert(
                "incomplete_details".to_owned(),
                json!({"reason":"max_output_tokens"}),
            );
        }
        let kind = if matches!(reason, StopReason::Incomplete | StopReason::MaxTokens) {
            "response.incomplete"
        } else {
            "response.completed"
        };
        Ok(self.event(kind, json!({"response":Value::Object(object)})))
    }
}

impl StreamEncoder for ResponsesStreamEncoder {
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        match &event {
            StreamEvent::MessageStart { id, model, extra } => {
                self.created_frames(id.clone(), model.clone(), extra.clone(), cx)
            }
            StreamEvent::BlockStart { index, block } => self.start_item(*index, block, cx),
            StreamEvent::BlockDelta { index, delta } => {
                Ok(self.delta_item(*index, delta).into_iter().collect())
            }
            StreamEvent::BlockStop { index } => Ok(self.stop_item(*index)),
            StreamEvent::Usage { usage } => {
                self.usage.merge_nonzero(usage);
                Ok(Vec::new())
            }
            StreamEvent::Safety { .. }
            | StreamEvent::ResponseMetadata { .. }
            | StreamEvent::Ping => Ok(Vec::new()),
            StreamEvent::MessageStop { stop_reason } => {
                Ok(vec![self.final_frame(stop_reason.clone(), cx)?])
            }
            StreamEvent::Error { error } => {
                let response = Response {
                    id: self.response_id.clone(),
                    model: self.model.clone(),
                    ..Response::default()
                };
                let mut object = codec::response_header(&response, &self.client, cx)?;
                object.insert("status".to_owned(), Value::String("failed".to_owned()));
                object.insert("output".to_owned(), Value::Array(Vec::new()));
                object.insert(
                    "error".to_owned(),
                    json!({"code":error.code,"message":error.message}),
                );
                self.stopped = true;
                Ok(vec![self.event(
                    "response.failed",
                    json!({"response":Value::Object(object)}),
                )])
            }
        }
    }

    fn finish(&mut self, cx: &CodecContext) -> Result<Vec<SseEvent>> {
        if self.stopped {
            Ok(Vec::new())
        } else {
            Ok(vec![self.final_frame(StopReason::EndTurn, cx)?])
        }
    }
}

const fn block_extra(block: &ContentBlock) -> Option<&crate::Extra> {
    match block {
        ContentBlock::Text { extra, .. }
        | ContentBlock::Image { extra, .. }
        | ContentBlock::File { extra, .. }
        | ContentBlock::Audio { extra, .. }
        | ContentBlock::Reasoning { extra, .. }
        | ContentBlock::ToolCall { extra, .. }
        | ContentBlock::ToolResult { extra, .. }
        | ContentBlock::WebSearchResult { extra, .. }
        | ContentBlock::Refusal { extra, .. } => extra.as_ref(),
        ContentBlock::Native { .. } => None,
    }
}
