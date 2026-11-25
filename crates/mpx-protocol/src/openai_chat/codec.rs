use bytes::Bytes;
use http::{
    HeaderValue, Method, StatusCode,
    header::{ACCEPT, CONTENT_TYPE},
};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    ApiError, ApiErrorKind, BlockDelta, ClientContext, CodecContext, ContentBlock, DiagnosticKind,
    Endpoint, Error, Extra, MediaSource, Message, OutputFormat, OutputFormatKind, Protocol,
    ProtocolCodec, ReasoningParams, Request, Response, Result, Role, SamplingParams, StopReason,
    Tool, ToolArguments, ToolCallId, ToolChoice, Usage, codec::json_response,
    request::take_session_id, tool::synthesized_tool_call_id,
};

use super::{
    ChatCompletionsCodec,
    stream::{ChatStreamDecoder, ChatStreamEncoder},
    wire::{
        AudioOutputWire, ChoiceWire, ChunkWire, ContentPartKind, ContentPartWire, ErrorBodyWire,
        ErrorEnvelopeWire, FileWire, FunctionCallWire, ImageUrlObjectWire, ImageUrlWire,
        InputAudioWire, MessageContentWire, MessageWire, RequestWire, ResponseWire, StopWire,
        StreamOptionsWire, ToolCallKind, ToolCallWire, UsageWire, WireRole,
    },
};

const URI: &str = "/v1/chat/completions";

impl ProtocolCodec for ChatCompletionsCodec {
    fn protocol(&self) -> Protocol {
        Self::protocol()
    }

    fn decode_request(
        &self,
        endpoint: &Endpoint,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Request> {
        let wire: RequestWire =
            serde_json::from_slice(body).map_err(|error| Error::json(Self::protocol(), error))?;
        let model = if wire.model.is_empty() {
            endpoint.model_from_path.clone().unwrap_or_default()
        } else {
            wire.model
        };
        let mut messages = Vec::with_capacity(wire.messages.len());
        for (index, message) in wire.messages.into_iter().enumerate() {
            messages.push(decode_message(message, cx, &format!("messages[{index}]"))?);
        }
        normalize_tool_ids(&mut messages);
        let mut instructions = Vec::new();
        while messages
            .first()
            .is_some_and(|message| matches!(message.role, Role::System | Role::Developer))
        {
            let message = messages.remove(0);
            instructions.extend(message.content);
        }
        let tools = wire
            .tools
            .into_iter()
            .enumerate()
            .map(|(index, tool)| decode_tool(tool, cx, &format!("tools[{index}]")))
            .collect::<Result<Vec<_>>>()?;
        let tool_choice = wire.tool_choice.map(decode_tool_choice);
        let output_format = wire
            .response_format
            .map(|format| decode_output_format(format, cx))
            .transpose()?
            .flatten();
        let reasoning = decode_reasoning_params(wire.reasoning_effort, wire.reasoning);
        let stop_sequences = match wire.stop {
            Some(StopWire::One(stop)) => vec![stop],
            Some(StopWire::Many(stops)) => stops,
            None => Vec::new(),
        };
        let mut metadata = wire.metadata.unwrap_or_default();
        let session_id = take_session_id(&mut metadata);
        Ok(Request {
            model,
            session_id,
            instructions,
            messages,
            tools,
            tool_choice,
            sampling: SamplingParams {
                max_output_tokens: wire.max_completion_tokens.or(wire.max_tokens),
                temperature: wire.temperature,
                top_p: wire.top_p,
                top_k: wire.top_k,
                stop_sequences,
                seed: wire.seed,
                presence_penalty: wire.presence_penalty,
                frequency_penalty: wire.frequency_penalty,
                extra: None,
                chat_legacy_max_tokens: wire.max_tokens.is_some()
                    && wire.max_completion_tokens.is_none(),
            },
            reasoning,
            output_format,
            parallel_tool_calls: wire.parallel_tool_calls,
            stream: wire.stream.unwrap_or(false) || endpoint.stream,
            metadata,
            continuation: None,
            extra: make_extra(wire.extra),
            client_include_usage: wire
                .stream_options
                .as_ref()
                .is_some_and(|options| options.include_usage),
            messages_max_tokens: false,
        })
    }

    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>> {
        if req.continuation.is_some() {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Self::protocol(),
                "continuation",
                "Chat Completions does not support server-side continuation",
            )?;
        }

        let mut messages = Vec::with_capacity(req.messages.len().saturating_add(1));
        if !req.instructions.is_empty() {
            let instructions = Message {
                role: Role::System,
                content: req.instructions.clone(),
                extra: None,
            };
            messages.extend(encode_message(&instructions, cx, "instructions")?);
        }
        for (index, message) in req.messages.iter().enumerate() {
            messages.extend(encode_message(message, cx, &format!("messages[{index}]"))?);
        }

        let mut tools = Vec::with_capacity(req.tools.len());
        for (index, tool) in req.tools.iter().enumerate() {
            if let Some(tool) = encode_tool(tool, cx, &format!("tools[{index}]"))? {
                tools.push(tool);
            }
        }
        let tool_choice = req
            .tool_choice
            .as_ref()
            .map(|choice| encode_tool_choice(choice, cx))
            .transpose()?
            .flatten();
        let response_format = req
            .output_format
            .as_ref()
            .map(|format| encode_output_format(format, cx))
            .transpose()?
            .flatten();
        let (reasoning_effort, reasoning) = encode_reasoning_params(req.reasoning.as_ref(), cx)?;
        let mut extra = same_extra(req.extra.as_ref(), cx, "extra")?;
        extra.extend(same_extra(
            req.sampling.extra.as_ref(),
            cx,
            "sampling.extra",
        )?);
        let token_limit = req.sampling.max_output_tokens;
        let legacy_token_limit = req.sampling.chat_legacy_max_tokens;
        let stop = match req.sampling.stop_sequences.as_slice() {
            [] => None,
            [stop] => Some(StopWire::One(stop.clone())),
            stops => Some(StopWire::Many(stops.to_vec())),
        };
        let stream_options = req.stream.then(|| StreamOptionsWire {
            include_usage: true,
            extra: Map::new(),
        });
        let wire = RequestWire {
            model: req.model.clone(),
            messages,
            stream: req.stream.then_some(true),
            stream_options,
            max_tokens: token_limit.filter(|_| legacy_token_limit),
            max_completion_tokens: token_limit.filter(|_| !legacy_token_limit),
            temperature: req.sampling.temperature,
            top_p: req.sampling.top_p,
            top_k: req.sampling.top_k,
            stop,
            seed: req.sampling.seed,
            presence_penalty: req.sampling.presence_penalty,
            frequency_penalty: req.sampling.frequency_penalty,
            tools,
            tool_choice,
            parallel_tool_calls: req.parallel_tool_calls,
            response_format,
            reasoning_effort,
            reasoning,
            metadata: req.openai_metadata(),
            extra,
        };
        let body = to_bytes(&wire)?;
        let mut request = http::Request::new(body);
        *request.method_mut() = Method::POST;
        *request.uri_mut() = http::Uri::from_static(URI);
        request
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        request.headers_mut().insert(
            ACCEPT,
            HeaderValue::from_static(if req.stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        Ok(request)
    }

    fn decode_response(
        &self,
        _parts: &http::response::Parts,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Response> {
        let wire: ResponseWire =
            serde_json::from_slice(body).map_err(|error| Error::json(Self::protocol(), error))?;
        if wire.choices.len() > 1 {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Self::protocol(),
                "choices",
                "only choice 0 is retained",
            )?;
        }
        let choice = wire
            .choices
            .into_iter()
            .find(|choice| choice.index == 0)
            .ok_or_else(|| Error::Protocol {
                protocol: Self::protocol(),
                message: "response does not contain choice 0".to_owned(),
            })?;
        let has_tool_calls = !choice.message.tool_calls.is_empty();
        let message = decode_message(choice.message, cx, "choices[0].message")?;
        let mut extra = wire.extra;
        if let Some(object) = wire.object {
            extra.insert("object".to_owned(), Value::String(object));
        }
        if let Some(created) = wire.created {
            extra.insert("created".to_owned(), Value::from(created));
        }
        if let Some(message_extra) = message.extra {
            insert_internal_extra(&mut extra, MESSAGE_EXTRA, message_extra);
        }
        if !choice.extra.is_empty() {
            extra.insert(CHOICE_EXTRA.to_owned(), Value::Object(choice.extra));
        }
        Ok(Response {
            id: wire.id,
            model: wire.model,
            content: message.content,
            stop_reason: choice.finish_reason.as_ref().map_or_else(
                || decode_stop_reason(None, has_tool_calls),
                StopReason::from,
            ),
            usage: decode_usage(wire.usage),
            safety: Vec::new(),
            extra: make_extra(extra),
        })
    }

    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError {
        let kind = ApiErrorKind::from_status(parts.status);
        match serde_json::from_slice::<ErrorEnvelopeWire>(body) {
            Ok(envelope) => ApiError {
                kind,
                message: if envelope.error.message.is_empty() {
                    fallback_error_message(parts.status, body)
                } else {
                    envelope.error.message
                },
                code: error_code(envelope.error.code).or(envelope.error.kind),
            },
            Err(_) => ApiError {
                kind,
                message: fallback_error_message(parts.status, body),
                code: None,
            },
        }
    }

    fn encode_response(
        &self,
        response: &Response,
        _client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>> {
        let mut message = encode_response_message(&response.content, cx)?;
        let ResponseExtra {
            fields: extra,
            message: message_extra,
            choice: choice_extra,
            created,
            object,
        } = split_response_extra(response.extra.as_ref(), cx)?;
        if let Some(fields) = message_extra {
            message.extra = fields;
        }
        let created = created.unwrap_or(0);
        let object = object.unwrap_or_else(|| "chat.completion".to_owned());
        let wire = ResponseWire {
            id: if response.id.is_empty() {
                "chatcmpl-0".to_owned()
            } else {
                response.id.clone()
            },
            object: Some(object),
            created: Some(created),
            model: response.model.clone(),
            choices: vec![ChoiceWire {
                index: 0,
                message,
                finish_reason: Some((&response.stop_reason).into()),
                extra: choice_extra,
            }],
            usage: Some(encode_usage(&response.usage, cx)?),
            extra,
        };
        Ok(json_response(StatusCode::OK, to_bytes(&wire)?))
    }

    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes> {
        let envelope = ErrorEnvelopeWire {
            error: ErrorBodyWire {
                message: error.message.clone(),
                kind: Some(error.kind.as_str().to_owned()),
                code: error.code.as_ref().map(|code| Value::String(code.clone())),
                param: None,
            },
        };
        let body = serde_json::to_vec(&envelope).map_or_else(
            |_| {
                Bytes::from_static(
                    br#"{"error":{"message":"internal error","type":"internal","code":"internal"}}"#,
                )
            },
            Bytes::from,
        );
        json_response(error.status(), body)
    }

    fn stream_decoder(&self) -> Box<dyn crate::StreamDecoder> {
        Box::new(ChatStreamDecoder::new())
    }

    fn stream_encoder(&self, client: &ClientContext) -> Box<dyn crate::StreamEncoder> {
        Box::new(ChatStreamEncoder::new(client))
    }
}

pub(crate) fn detect(parts: &http::request::Parts) -> Option<Endpoint> {
    if parts.method != Method::POST {
        return None;
    }
    let path = parts.uri.path();
    let base = path.strip_suffix("/chat/completions")?;
    if !base.is_empty() && !base.starts_with('/') {
        return None;
    }
    let model_from_path = base
        .rsplit_once("/openai/deployments/")
        .and_then(|(_, deployment)| {
            if deployment.is_empty() || deployment.contains('/') {
                return None;
            }
            percent_encoding::percent_decode_str(deployment)
                .decode_utf8()
                .ok()
                .map(std::borrow::Cow::into_owned)
        });
    Some(Endpoint {
        protocol: Protocol::ChatCompletions,
        model_from_path,
        stream: false,
        passthrough: None,
    })
}

pub(crate) fn make_extra(fields: Map<String, Value>) -> Option<Extra> {
    Extra::from_unknown(Protocol::ChatCompletions, fields)
}

pub(crate) fn same_extra(
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Map<String, Value>> {
    Extra::fields_for(extra, Protocol::ChatCompletions, cx, path)
}

fn insert_internal_extra(fields: &mut Map<String, Value>, key: &str, extra: Extra) {
    if extra.protocol == Protocol::ChatCompletions && !extra.fields.is_empty() {
        fields.insert(key.to_owned(), Value::Object(extra.fields));
    }
}

/// Internal response-extra keys for unmodeled message and choice fields.
const MESSAGE_EXTRA: &str = "__mpx_chat_message_extra";
const CHOICE_EXTRA: &str = "__mpx_chat_choice_extra";

/// Response extra split back into the wire objects it was decoded from.
pub(super) struct ResponseExtra {
    /// Top-level response fields.
    pub(super) fields: Map<String, Value>,
    pub(super) message: Option<Map<String, Value>>,
    pub(super) choice: Map<String, Value>,
    pub(super) created: Option<u64>,
    pub(super) object: Option<String>,
}

/// Splits response extra so internal message and choice keys never leak to the wire.
pub(super) fn split_response_extra(
    extra: Option<&Extra>,
    cx: &CodecContext,
) -> Result<ResponseExtra> {
    let mut fields = same_extra(extra, cx, "extra")?;
    let message = take_internal_extra(&mut fields, MESSAGE_EXTRA);
    let choice = take_internal_extra(&mut fields, CHOICE_EXTRA).unwrap_or_default();
    let created = fields.remove("created").and_then(|value| value.as_u64());
    let object = match fields.remove("object") {
        Some(Value::String(object)) => Some(object),
        _ => None,
    };
    Ok(ResponseExtra {
        fields,
        message,
        choice,
        created,
        object,
    })
}

fn take_internal_extra(fields: &mut Map<String, Value>, key: &str) -> Option<Map<String, Value>> {
    match fields.remove(key)? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

fn to_bytes(value: &impl Serialize) -> Result<Bytes> {
    serde_json::to_vec(value)
        .map(Bytes::from)
        .map_err(|error| Error::json(Protocol::ChatCompletions, error))
}

fn decode_message(wire: MessageWire, cx: &CodecContext, path: &str) -> Result<Message> {
    let role = decode_role(&wire.role, cx, path)?;
    let mut content = Vec::new();
    if role == Role::Tool {
        let result_content = decode_content(wire.content)?;
        content.push(ContentBlock::ToolResult {
            call_id: ToolCallId::new(wire.tool_call_id.unwrap_or_default()),
            name: wire.name.clone(),
            content: result_content,
            is_error: false,
            extra: None,
        });
    } else {
        if let Some(reasoning) = wire.reasoning_content {
            content.push(reasoning_block(reasoning, Vec::new(), false, Map::new()));
        }
        if let Some(reasoning) = wire.reasoning {
            content.push(decode_reasoning_block(reasoning));
        }
        content.extend(decode_content(wire.content)?);
        if let Some(refusal) = wire.refusal {
            content.push(ContentBlock::Refusal {
                text: refusal,
                extra: None,
            });
        }
        if let Some(audio) = wire.audio {
            decode_audio_output(audio, &mut content);
        }
        for (index, call) in wire.tool_calls.into_iter().enumerate() {
            content.push(decode_tool_call(call, index, cx, path)?);
        }
    }
    Ok(Message {
        role,
        content,
        extra: make_extra(wire.extra),
    })
}

fn decode_role(role: &WireRole, cx: &CodecContext, path: &str) -> Result<Role> {
    match role {
        WireRole::System => Ok(Role::System),
        WireRole::Developer => Ok(Role::Developer),
        WireRole::User => Ok(Role::User),
        WireRole::Assistant => Ok(Role::Assistant),
        WireRole::Tool | WireRole::Function => Ok(Role::Tool),
        WireRole::Unknown(role) => {
            cx.diagnose(
                DiagnosticKind::Approximated,
                Protocol::ChatCompletions,
                format!("{path}.role"),
                format!("unknown role `{role}` was treated as user"),
            )?;
            Ok(Role::User)
        }
    }
}

fn decode_content(content: Option<MessageContentWire>) -> Result<Vec<ContentBlock>> {
    match content {
        None => Ok(Vec::new()),
        Some(MessageContentWire::Text(text)) => Ok(vec![ContentBlock::text(text)]),
        Some(MessageContentWire::Parts(parts)) => {
            parts.into_iter().map(decode_content_part).collect()
        }
    }
}

fn decode_content_part(mut part: ContentPartWire) -> Result<ContentBlock> {
    match &part.kind {
        ContentPartKind::Text | ContentPartKind::OutputText => {
            if let Some(text) = part.text {
                Ok(ContentBlock::Text {
                    text,
                    extra: make_extra(part.extra),
                })
            } else {
                native_content_part(part)
            }
        }
        ContentPartKind::ImageUrl | ContentPartKind::InputImage => {
            if let Some(image) = part.image_url {
                let (url, detail, mut image_extra) = match image {
                    ImageUrlWire::Url(url) => (url, None, Map::new()),
                    ImageUrlWire::Object(object) => (object.url, object.detail, object.extra),
                };
                if let Some(detail) = detail {
                    image_extra.insert("detail".to_owned(), Value::String(detail));
                }
                part.extra.append(&mut image_extra);
                Ok(ContentBlock::Image {
                    source: decode_image_source(&url),
                    extra: make_extra(part.extra),
                })
            } else {
                native_content_part(part)
            }
        }
        ContentPartKind::InputAudio => {
            if let Some(audio) = part.input_audio {
                let mut extra = part.extra;
                extra.append(&mut audio.extra.clone());
                Ok(ContentBlock::Audio {
                    source: MediaSource::Base64 {
                        media_type: audio_media_type(&audio.format),
                        data: audio.data,
                    },
                    extra: make_extra(extra),
                })
            } else {
                native_content_part(part)
            }
        }
        ContentPartKind::File => {
            if let Some(file) = part.file {
                let source = if let Some(file_id) = file.file_id {
                    MediaSource::File {
                        file_id,
                        media_type: None,
                    }
                } else if let Some(data) = file.file_data {
                    MediaSource::Base64 {
                        media_type: "application/octet-stream".to_owned(),
                        data,
                    }
                } else {
                    return native_content_part(ContentPartWire {
                        file: Some(FileWire {
                            file_id: None,
                            file_data: None,
                            filename: file.filename,
                            extra: file.extra,
                        }),
                        ..part
                    });
                };
                let mut extra = part.extra;
                extra.extend(file.extra);
                Ok(ContentBlock::File {
                    source,
                    filename: file.filename,
                    extra: make_extra(extra),
                })
            } else {
                native_content_part(part)
            }
        }
        ContentPartKind::Refusal => Ok(ContentBlock::Refusal {
            text: part.refusal.or(part.text).unwrap_or_default(),
            extra: make_extra(part.extra),
        }),
        ContentPartKind::Reasoning | ContentPartKind::ReasoningContent => {
            if let Some(text) = part.reasoning_content.clone().or_else(|| part.text.clone()) {
                Ok(reasoning_block(text, Vec::new(), false, part.extra))
            } else if let Some(reasoning) = part.reasoning {
                Ok(decode_reasoning_block(reasoning))
            } else {
                native_content_part(part)
            }
        }
        ContentPartKind::Unknown(_) => native_content_part(part),
    }
}

fn native_content_part(part: ContentPartWire) -> Result<ContentBlock> {
    let json = serde_json::to_value(part)
        .map_err(|error| Error::json(Protocol::ChatCompletions, error))?;
    Ok(ContentBlock::Native {
        protocol: Protocol::ChatCompletions,
        json,
    })
}

fn decode_image_source(url: &str) -> MediaSource {
    if let Some((metadata, data)) = url.split_once(',')
        && let Some(media) = metadata.strip_prefix("data:")
        && media.ends_with(";base64")
    {
        let media_type = media.trim_end_matches(";base64");
        return MediaSource::Base64 {
            media_type: if media_type.is_empty() {
                "image/*".to_owned()
            } else {
                media_type.to_owned()
            },
            data: data.to_owned(),
        };
    }
    MediaSource::Url {
        url: url.to_owned(),
        media_type: None,
    }
}

fn audio_media_type(format: &str) -> String {
    match format {
        "mp3" | "mpeg" => "audio/mpeg".to_owned(),
        "" => "audio/*".to_owned(),
        other => format!("audio/{other}"),
    }
}

fn audio_format(media_type: &str) -> String {
    match media_type.strip_prefix("audio/").unwrap_or(media_type) {
        "mpeg" => "mp3".to_owned(),
        "*" => String::new(),
        other => other.to_owned(),
    }
}

fn decode_audio_output(audio: AudioOutputWire, content: &mut Vec<ContentBlock>) {
    if !audio.data.is_empty() {
        let mut extra = audio.extra;
        if let Some(format) = audio.format.clone() {
            extra.insert("format".to_owned(), Value::String(format));
        }
        content.push(ContentBlock::Audio {
            source: MediaSource::Base64 {
                media_type: audio_media_type(audio.format.as_deref().unwrap_or_default()),
                data: audio.data,
            },
            extra: make_extra(extra),
        });
    }
    if let Some(transcript) = audio.transcript {
        content.push(ContentBlock::text(transcript));
    }
}

pub(crate) fn decode_reasoning_block(value: Value) -> ContentBlock {
    match value {
        Value::String(text) => reasoning_block(text, Vec::new(), false, Map::new()),
        Value::Object(mut object) => {
            let text = take_string(&mut object, "text")
                .or_else(|| take_string(&mut object, "content"))
                .or_else(|| take_string(&mut object, "reasoning_content"))
                .unwrap_or_default();
            let summary = match object.remove("summary") {
                Some(Value::String(summary)) => vec![summary],
                Some(Value::Array(values)) => values
                    .into_iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect(),
                _ => Vec::new(),
            };
            let redacted = object
                .remove("redacted")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            reasoning_block(text, summary, redacted, object)
        }
        other => reasoning_block(other.to_string(), Vec::new(), false, Map::new()),
    }
}

fn take_string(object: &mut Map<String, Value>, key: &str) -> Option<String> {
    object.remove(key).and_then(|value| match value {
        Value::String(text) => Some(text),
        _ => None,
    })
}

pub(crate) fn reasoning_block(
    text: String,
    summary: Vec<String>,
    redacted: bool,
    extra: Map<String, Value>,
) -> ContentBlock {
    ContentBlock::Reasoning {
        text,
        summary,
        seal: None,
        redacted,
        extra: make_extra(extra),
    }
}

fn decode_tool_call(
    call: ToolCallWire,
    position: usize,
    cx: &CodecContext,
    path: &str,
) -> Result<ContentBlock> {
    if call
        .kind
        .as_ref()
        .is_some_and(|kind| !matches!(kind, ToolCallKind::Function))
    {
        cx.diagnose(
            DiagnosticKind::Approximated,
            Protocol::ChatCompletions,
            format!("{path}.tool_calls[{position}]"),
            "non-function tool call was represented as a function call",
        )?;
    }
    let mut extra = call.extra;
    extra.extend(call.function.extra);
    Ok(ContentBlock::ToolCall {
        id: ToolCallId::new(
            call.id
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| synthesized_tool_call_id(position)),
        ),
        name: call.function.name.unwrap_or_default(),
        arguments: ToolArguments::new(call.function.arguments.unwrap_or_else(|| "{}".to_owned())),
        seal: None,
        extra: make_extra(extra),
    })
}

fn normalize_tool_ids(messages: &mut [Message]) {
    let mut pending = Vec::new();
    let mut next = 0usize;
    for message in messages {
        for block in &mut message.content {
            match block {
                ContentBlock::ToolCall { id, .. } if id.as_str().is_empty() => {
                    let value = format!("call_{next}");
                    next = next.saturating_add(1);
                    *id = ToolCallId::new(value.clone());
                    pending.push(value);
                }
                ContentBlock::ToolCall { id, .. } => pending.push(id.as_str().to_owned()),
                ContentBlock::ToolResult { call_id, .. } if call_id.as_str().is_empty() => {
                    let value = pending.first().cloned().unwrap_or_else(|| {
                        let value = format!("call_{next}");
                        next = next.saturating_add(1);
                        value
                    });
                    *call_id = ToolCallId::new(value.clone());
                    if let Some(position) = pending.iter().position(|id| id == &value) {
                        pending.remove(position);
                    }
                }
                ContentBlock::ToolResult { call_id, .. } => {
                    if let Some(position) = pending.iter().position(|id| id == call_id.as_str()) {
                        pending.remove(position);
                    }
                }
                _ => {}
            }
        }
    }
}

fn encode_message(message: &Message, cx: &CodecContext, path: &str) -> Result<Vec<MessageWire>> {
    if message.role == Role::Tool {
        return encode_tool_message(message, cx, path);
    }
    if message.role == Role::User
        && message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
    {
        let mut split = Vec::new();
        message.clone().split_tool_results(&mut split);
        let mut encoded = Vec::new();
        for message in &split {
            if message.role == Role::Tool {
                encoded.extend(encode_tool_message(message, cx, path)?);
            } else {
                encoded.push(encode_regular_message(message, cx, path)?);
            }
        }
        return Ok(encoded);
    }
    Ok(vec![encode_regular_message(message, cx, path)?])
}

#[allow(clippy::too_many_lines)]
fn encode_regular_message(message: &Message, cx: &CodecContext, path: &str) -> Result<MessageWire> {
    let role = encode_role(message.role);
    let mut parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = Vec::new();
    let mut refusal = Vec::new();
    for (index, block) in message.content.iter().enumerate() {
        let block_path = format!("{path}.content[{index}]");
        match block {
            ContentBlock::Text { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::File { .. }
            | ContentBlock::Audio { .. }
            | ContentBlock::Native { .. } => {
                if let Some(part) = encode_content_part(block, cx, &block_path)? {
                    parts.push(part);
                }
            }
            ContentBlock::Reasoning {
                text,
                summary,
                seal,
                redacted,
                extra,
            } => {
                if !text.is_empty() {
                    reasoning.push(text.clone());
                }
                if !summary.is_empty() {
                    reasoning.push(summary.join("\n"));
                }
                if seal.is_some()
                    || *redacted
                    || extra.as_ref().is_some_and(|extra| !extra.fields.is_empty())
                {
                    cx.diagnose(
                        DiagnosticKind::Dropped,
                        Protocol::ChatCompletions,
                        &block_path,
                        "reasoning seals, redaction state, and extension fields are not representable in message reasoning_content",
                    )?;
                }
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                seal,
                extra,
            } => {
                if seal.is_some() {
                    cx.diagnose(
                        DiagnosticKind::Dropped,
                        Protocol::ChatCompletions,
                        &block_path,
                        "tool-call reasoning seal was omitted",
                    )?;
                }
                let call_extra = same_extra(extra.as_ref(), cx, &block_path)?;
                tool_calls.push(ToolCallWire {
                    id: Some(id.as_str().to_owned()),
                    kind: Some(ToolCallKind::Function),
                    function: FunctionCallWire {
                        name: Some(name.clone()),
                        arguments: Some(arguments.as_str().to_owned()),
                        extra: Map::new(),
                    },
                    extra: call_extra,
                });
            }
            ContentBlock::Refusal { text, .. } => refusal.push(text.clone()),
            ContentBlock::ToolResult { content, .. } => {
                cx.diagnose(
                    DiagnosticKind::Fallback,
                    Protocol::ChatCompletions,
                    &block_path,
                    "tool result outside a tool message was flattened into text",
                )?;
                parts.extend(flatten_result_text(content));
            }
            ContentBlock::WebSearchResult { .. } => {
                if let Some(part) = fallback_part(block, cx, &block_path)? {
                    parts.push(part);
                }
            }
        }
    }
    Ok(MessageWire {
        role,
        content: parts_to_content(parts),
        name: None,
        tool_call_id: None,
        tool_calls,
        refusal: join_nonempty(refusal),
        reasoning_content: join_nonempty(reasoning),
        reasoning: None,
        audio: None,
        extra: same_extra(message.extra.as_ref(), cx, &format!("{path}.extra"))?,
    })
}

fn encode_response_message(content: &[ContentBlock], cx: &CodecContext) -> Result<MessageWire> {
    let message = Message {
        role: Role::Assistant,
        content: content.to_vec(),
        extra: None,
    };
    encode_regular_message(&message, cx, "choices[0].message")
}

const fn encode_role(role: Role) -> WireRole {
    match role {
        Role::System => WireRole::System,
        Role::Developer => WireRole::Developer,
        Role::User => WireRole::User,
        Role::Assistant | Role::Tool => WireRole::Assistant,
    }
}

fn encode_tool_message(
    message: &Message,
    cx: &CodecContext,
    path: &str,
) -> Result<Vec<MessageWire>> {
    let mut messages = Vec::new();
    for (index, block) in message.content.iter().enumerate() {
        let ContentBlock::ToolResult {
            call_id,
            name,
            content,
            is_error,
            extra,
        } = block
        else {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                format!("{path}.content[{index}]"),
                "non-result content in a tool message was omitted",
            )?;
            continue;
        };
        if *is_error {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                format!("{path}.content[{index}].is_error"),
                "Chat Completions has no tool-result error flag",
            )?;
        }
        let mut message_extra = same_extra(message.extra.as_ref(), cx, &format!("{path}.extra"))?;
        message_extra.extend(same_extra(
            extra.as_ref(),
            cx,
            &format!("{path}.content[{index}]"),
        )?);
        let mut parts = Vec::new();
        for (content_index, block) in content.iter().enumerate() {
            let content_path = format!("{path}.content[{index}].content[{content_index}]");
            match block {
                ContentBlock::Reasoning { text, summary, .. } => {
                    let mut text = text.clone();
                    if !summary.is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&summary.join("\n"));
                    }
                    parts.push(text_part(text, Map::new()));
                }
                ContentBlock::Refusal { text, .. } => {
                    parts.push(text_part(text.clone(), Map::new()));
                }
                ContentBlock::ToolResult { .. }
                | ContentBlock::ToolCall { .. }
                | ContentBlock::WebSearchResult { .. } => {
                    if let Some(part) = fallback_part(block, cx, &content_path)? {
                        parts.push(part);
                    }
                }
                _ => {
                    if let Some(part) = encode_content_part(block, cx, &content_path)? {
                        parts.push(part);
                    }
                }
            }
        }
        messages.push(MessageWire {
            role: WireRole::Tool,
            content: parts_to_content(parts),
            name: name.clone().or_else(|| {
                message_extra
                    .remove("name")
                    .and_then(|value| value.as_str().map(str::to_owned))
            }),
            tool_call_id: Some(call_id.as_str().to_owned()),
            tool_calls: Vec::new(),
            refusal: None,
            reasoning_content: None,
            reasoning: None,
            audio: None,
            extra: message_extra,
        });
    }
    Ok(messages)
}

fn flatten_result_text(content: &[ContentBlock]) -> Vec<ContentPartWire> {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text, .. }
            | ContentBlock::Refusal { text, .. }
            | ContentBlock::Reasoning { text, .. } => text.clone(),
            other => format!("[{}]", other.kind()),
        })
        .map(|text| text_part(text, Map::new()))
        .collect()
}

fn encode_content_part(
    block: &ContentBlock,
    cx: &CodecContext,
    path: &str,
) -> Result<Option<ContentPartWire>> {
    match block {
        ContentBlock::Text { text, extra } => Ok(Some(text_part(
            text.clone(),
            same_extra(extra.as_ref(), cx, path)?,
        ))),
        ContentBlock::Image { source, extra } => {
            let mut fields = same_extra(extra.as_ref(), cx, path)?;
            let detail = fields
                .remove("detail")
                .and_then(|value| value.as_str().map(str::to_owned));
            let image = match source {
                MediaSource::Url { url, .. } => ImageUrlWire::Object(ImageUrlObjectWire {
                    url: url.clone(),
                    detail,
                    extra: fields,
                }),
                MediaSource::Base64 { media_type, data } => {
                    ImageUrlWire::Object(ImageUrlObjectWire {
                        url: format!("data:{media_type};base64,{data}"),
                        detail,
                        extra: fields,
                    })
                }
                MediaSource::File { file_id, .. } => {
                    cx.diagnose(
                        DiagnosticKind::Fallback,
                        Protocol::ChatCompletions,
                        path,
                        "provider file image was represented by its file identifier URL",
                    )?;
                    ImageUrlWire::Object(ImageUrlObjectWire {
                        url: file_id.clone(),
                        detail,
                        extra: fields,
                    })
                }
            };
            Ok(Some(ContentPartWire {
                kind: ContentPartKind::ImageUrl,
                image_url: Some(image),
                ..ContentPartWire::default()
            }))
        }
        ContentBlock::Audio {
            source: MediaSource::Base64 { media_type, data },
            extra,
        } => {
            let mut fields = same_extra(extra.as_ref(), cx, path)?;
            fields.remove("format");
            Ok(Some(ContentPartWire {
                kind: ContentPartKind::InputAudio,
                input_audio: Some(InputAudioWire {
                    data: data.clone(),
                    format: audio_format(media_type),
                    extra: fields,
                }),
                ..ContentPartWire::default()
            }))
        }
        ContentBlock::File {
            source,
            filename,
            extra,
        } => {
            let fields = same_extra(extra.as_ref(), cx, path)?;
            let file = match source {
                MediaSource::File { file_id, .. } => FileWire {
                    file_id: Some(file_id.clone()),
                    file_data: None,
                    filename: filename.clone(),
                    extra: fields,
                },
                MediaSource::Base64 { data, .. } => FileWire {
                    file_id: None,
                    file_data: Some(data.clone()),
                    filename: filename.clone(),
                    extra: fields,
                },
                MediaSource::Url { url, .. } => FileWire {
                    file_id: None,
                    file_data: Some(url.clone()),
                    filename: filename.clone(),
                    extra: fields,
                },
            };
            Ok(Some(ContentPartWire {
                kind: ContentPartKind::File,
                file: Some(file),
                ..ContentPartWire::default()
            }))
        }
        ContentBlock::Native { protocol, json } if *protocol == Protocol::ChatCompletions => {
            serde_json::from_value(json.clone())
                .map_or_else(|_| fallback_part(block, cx, path), |part| Ok(Some(part)))
        }
        _ => fallback_part(block, cx, path),
    }
}

fn fallback_part(
    block: &ContentBlock,
    cx: &CodecContext,
    path: &str,
) -> Result<Option<ContentPartWire>> {
    if cx.options.content_fallbacks {
        cx.diagnose(
            DiagnosticKind::Fallback,
            Protocol::ChatCompletions,
            path,
            format!("{} content became descriptive text", block.kind()),
        )?;
        Ok(Some(text_part(
            format!("[Unsupported {} content]", block.kind()),
            Map::new(),
        )))
    } else {
        cx.diagnose(
            DiagnosticKind::Dropped,
            Protocol::ChatCompletions,
            path,
            format!("{} content is not supported", block.kind()),
        )?;
        Ok(None)
    }
}

fn text_part(text: String, extra: Map<String, Value>) -> ContentPartWire {
    ContentPartWire {
        kind: ContentPartKind::Text,
        text: Some(text),
        extra,
        ..ContentPartWire::default()
    }
}

fn parts_to_content(parts: Vec<ContentPartWire>) -> Option<MessageContentWire> {
    if parts.is_empty() {
        return None;
    }
    let plain_text = parts.iter().all(|part| {
        matches!(part.kind, ContentPartKind::Text) && part.extra.is_empty() && part.text.is_some()
    });
    if plain_text {
        let text = parts
            .into_iter()
            .filter_map(|part| part.text)
            .collect::<String>();
        Some(MessageContentWire::Text(text))
    } else {
        Some(MessageContentWire::Parts(parts))
    }
}

fn join_nonempty(values: Vec<String>) -> Option<String> {
    let values: Vec<_> = values
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect();
    (!values.is_empty()).then(|| values.join("\n"))
}

fn decode_tool(value: Value, cx: &CodecContext, path: &str) -> Result<Tool> {
    let Some(object) = value.as_object() else {
        return Ok(Tool::Hosted {
            protocol: Protocol::ChatCompletions,
            json: value,
        });
    };
    match object.get("type").and_then(Value::as_str) {
        Some("function") => {
            let function = object.get("function").and_then(Value::as_object);
            let function = function.ok_or_else(|| Error::Protocol {
                protocol: Protocol::ChatCompletions,
                message: format!("{path}.function must be an object"),
            })?;
            let mut extra = object
                .iter()
                .filter(|(key, _)| key.as_str() != "type" && key.as_str() != "function")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<Map<_, _>>();
            extra.extend(
                function
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(
                            key.as_str(),
                            "name" | "description" | "parameters" | "strict"
                        )
                    })
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
            Ok(Tool::Function {
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                description: function
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                strict: function.get("strict").and_then(Value::as_bool),
                defer_loading: object.get("defer_loading").and_then(Value::as_bool),
                extra: make_extra(extra),
            })
        }
        Some("custom") => {
            let custom = object
                .get("custom")
                .and_then(Value::as_object)
                .ok_or_else(|| Error::Protocol {
                    protocol: Protocol::ChatCompletions,
                    message: format!("{path}.custom must be an object"),
                })?;
            let extra = object
                .iter()
                .filter(|(key, _)| key.as_str() != "type" && key.as_str() != "custom")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok(Tool::Custom {
                name: custom
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                description: custom
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                format: custom.get("format").cloned(),
                extra: make_extra(extra),
            })
        }
        _ => {
            cx.diagnostics.push(crate::Diagnostic {
                kind: DiagnosticKind::Approximated,
                protocol: Protocol::ChatCompletions,
                path: path.to_owned(),
                message: "unknown tool declaration was retained as native content".to_owned(),
            });
            Ok(Tool::Hosted {
                protocol: Protocol::ChatCompletions,
                json: value,
            })
        }
    }
}

fn encode_tool(tool: &Tool, cx: &CodecContext, path: &str) -> Result<Option<Value>> {
    match tool {
        Tool::Function {
            name,
            description,
            parameters,
            strict,
            defer_loading,
            extra,
        } => {
            let extra = same_extra(extra.as_ref(), cx, path)?;
            let mut function = Map::new();
            function.insert("name".to_owned(), Value::String(name.clone()));
            if let Some(description) = description {
                function.insert("description".to_owned(), Value::String(description.clone()));
            }
            function.insert("parameters".to_owned(), parameters.clone());
            if let Some(strict) = strict {
                function.insert("strict".to_owned(), Value::Bool(*strict));
            }
            for (key, value) in extra {
                if key != "type" && key != "function" && key != "defer_loading" {
                    function.insert(key, value);
                }
            }
            let mut object = Map::new();
            object.insert("type".to_owned(), Value::String("function".to_owned()));
            object.insert("function".to_owned(), Value::Object(function));
            if let Some(defer_loading) = defer_loading {
                object.insert("defer_loading".to_owned(), Value::Bool(*defer_loading));
            }
            Ok(Some(Value::Object(object)))
        }
        Tool::Custom {
            name,
            description,
            format,
            extra,
        } => {
            let mut custom = same_extra(extra.as_ref(), cx, path)?;
            custom.insert("name".to_owned(), Value::String(name.clone()));
            if let Some(description) = description {
                custom.insert("description".to_owned(), Value::String(description.clone()));
            }
            if let Some(format) = format {
                custom.insert("format".to_owned(), format.clone());
            }
            Ok(Some(json!({"type": "custom", "custom": custom})))
        }
        Tool::Hosted { protocol, json } if *protocol == Protocol::ChatCompletions => {
            Ok(Some(json.clone()))
        }
        Tool::Hosted { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                path,
                "hosted tool belongs to another protocol",
            )?;
            Ok(None)
        }
    }
}

fn decode_tool_choice(value: Value) -> ToolChoice {
    let named = value
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    match value.as_str() {
        Some("auto") => ToolChoice::Auto,
        Some("none") => ToolChoice::None,
        Some("required") => ToolChoice::Required,
        _ => named.map_or(
            ToolChoice::Native {
                protocol: Protocol::ChatCompletions,
                json: value,
            },
            ToolChoice::Named,
        ),
    }
}

fn encode_tool_choice(choice: &ToolChoice, cx: &CodecContext) -> Result<Option<Value>> {
    match choice {
        ToolChoice::Auto => Ok(Some(Value::String("auto".to_owned()))),
        ToolChoice::None => Ok(Some(Value::String("none".to_owned()))),
        ToolChoice::Required => Ok(Some(Value::String("required".to_owned()))),
        ToolChoice::Named(name) => Ok(Some(json!({
            "type": "function",
            "function": {"name": name}
        }))),
        ToolChoice::Native { protocol, json } if *protocol == Protocol::ChatCompletions => {
            Ok(Some(json.clone()))
        }
        ToolChoice::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                "tool_choice",
                "native tool choice belongs to another protocol",
            )?;
            Ok(None)
        }
    }
}

fn decode_output_format(value: Value, cx: &CodecContext) -> Result<Option<OutputFormat>> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "text" => Ok(Some(OutputFormat {
            kind: OutputFormatKind::Text,
            schema: None,
            name: None,
            strict: None,
            extra: object_extra(value, &["type"]),
        })),
        "json_object" => Ok(Some(OutputFormat {
            kind: OutputFormatKind::JsonObject,
            schema: None,
            name: None,
            strict: None,
            extra: object_extra(value, &["type"]),
        })),
        "json_schema" => {
            let schema_object = value.get("json_schema").and_then(Value::as_object);
            Ok(Some(OutputFormat {
                kind: OutputFormatKind::JsonSchema,
                schema: schema_object
                    .and_then(|object| object.get("schema"))
                    .cloned(),
                name: schema_object
                    .and_then(|object| object.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                strict: schema_object
                    .and_then(|object| object.get("strict"))
                    .and_then(Value::as_bool),
                extra: object_extra(value, &["type"]),
            }))
        }
        _ => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::ChatCompletions,
                "response_format",
                "unknown response format was omitted",
            )?;
            Ok(None)
        }
    }
}

fn object_extra(value: Value, known: &[&str]) -> Option<Extra> {
    let Value::Object(object) = value else {
        return None;
    };
    make_extra(
        object
            .into_iter()
            .filter(|(key, _)| !known.contains(&key.as_str()))
            .collect(),
    )
}

fn encode_output_format(format: &OutputFormat, cx: &CodecContext) -> Result<Option<Value>> {
    let mut extra = same_extra(format.extra.as_ref(), cx, "response_format.extra")?;
    let value = match format.kind {
        OutputFormatKind::Text => {
            extra.insert("type".to_owned(), Value::String("text".to_owned()));
            Value::Object(extra)
        }
        OutputFormatKind::JsonObject => {
            extra.insert("type".to_owned(), Value::String("json_object".to_owned()));
            Value::Object(extra)
        }
        OutputFormatKind::JsonSchema => {
            let mut schema = match extra.remove("json_schema") {
                Some(Value::Object(object)) => object,
                _ => Map::new(),
            };
            if let Some(name) = &format.name {
                schema.insert("name".to_owned(), Value::String(name.clone()));
            }
            if let Some(value) = &format.schema {
                schema.insert("schema".to_owned(), value.clone());
            }
            if let Some(strict) = format.strict {
                schema.insert("strict".to_owned(), Value::Bool(strict));
            }
            extra.insert("type".to_owned(), Value::String("json_schema".to_owned()));
            extra.insert("json_schema".to_owned(), Value::Object(schema));
            Value::Object(extra)
        }
    };
    Ok(Some(value))
}

fn decode_reasoning_params(
    effort: Option<String>,
    reasoning: Option<Value>,
) -> Option<ReasoningParams> {
    if effort.is_none() && reasoning.is_none() {
        return None;
    }
    let mut object = match reasoning {
        Some(Value::Object(object)) => object,
        Some(Value::String(value)) if effort.is_none() => {
            return Some(ReasoningParams {
                effort: Some(value),
                budget_tokens: None,
                summary: None,
                extra: None,
            });
        }
        Some(other) => {
            let mut fields = Map::new();
            fields.insert("value".to_owned(), other);
            return Some(ReasoningParams {
                effort,
                budget_tokens: None,
                summary: None,
                extra: make_extra(fields),
            });
        }
        None => Map::new(),
    };
    let object_effort = take_string(&mut object, "effort");
    let budget_tokens = object
        .remove("budget_tokens")
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok());
    let summary = take_string(&mut object, "summary");
    Some(ReasoningParams {
        effort: effort.or(object_effort),
        budget_tokens,
        summary,
        extra: make_extra(object),
    })
}

fn encode_reasoning_params(
    reasoning: Option<&ReasoningParams>,
    cx: &CodecContext,
) -> Result<(Option<String>, Option<Value>)> {
    let Some(reasoning) = reasoning else {
        return Ok((None, None));
    };
    let mut object = same_extra(reasoning.extra.as_ref(), cx, "reasoning.extra")?;
    if let Some(budget) = reasoning.budget_tokens {
        object.insert("budget_tokens".to_owned(), Value::from(budget));
    }
    if let Some(summary) = &reasoning.summary {
        object.insert("summary".to_owned(), Value::String(summary.clone()));
    }
    Ok((
        reasoning.effort.clone(),
        (!object.is_empty()).then_some(Value::Object(object)),
    ))
}

pub(crate) fn decode_usage(usage: Option<UsageWire>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    let cached_input_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens);
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .map_or(0, |details| details.reasoning_tokens);
    let mut extra = usage.extra;
    if let Some(total) = usage.total_tokens {
        extra.insert("total_tokens".to_owned(), Value::from(total));
    }
    if let Some(details) = usage.prompt_tokens_details
        && !details.extra.is_empty()
    {
        extra.insert(
            "prompt_tokens_details".to_owned(),
            Value::Object(details.extra),
        );
    }
    if let Some(details) = usage.completion_tokens_details
        && !details.extra.is_empty()
    {
        extra.insert(
            "completion_tokens_details".to_owned(),
            Value::Object(details.extra),
        );
    }
    Usage {
        input_tokens: usage.prompt_tokens.saturating_sub(cached_input_tokens),
        output_tokens: usage.completion_tokens,
        cached_input_tokens,
        cache_creation_input_tokens: 0,
        reasoning_tokens,
        extra: make_extra(extra),
    }
}

pub(crate) fn encode_usage(usage: &Usage, cx: &CodecContext) -> Result<UsageWire> {
    let mut extra = same_extra(usage.extra.as_ref(), cx, "usage.extra")?;
    let total_tokens = extra
        .remove("total_tokens")
        .and_then(|value| value.as_u64());
    let prompt_extra = extra
        .remove("prompt_tokens_details")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let completion_extra = extra
        .remove("completion_tokens_details")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if usage.cache_creation_input_tokens != 0 {
        cx.diagnose(
            DiagnosticKind::Dropped,
            Protocol::ChatCompletions,
            "usage.cache_creation_input_tokens",
            "Chat Completions has no cache-creation token count",
        )?;
    }
    Ok(UsageWire {
        prompt_tokens: usage.input_tokens.saturating_add(usage.cached_input_tokens),
        completion_tokens: usage.output_tokens,
        total_tokens: Some(total_tokens.unwrap_or_else(|| {
            usage
                .input_tokens
                .saturating_add(usage.cached_input_tokens)
                .saturating_add(usage.output_tokens)
        })),
        prompt_tokens_details: (usage.cached_input_tokens != 0 || !prompt_extra.is_empty())
            .then_some(super::wire::PromptTokensDetailsWire {
                cached_tokens: usage.cached_input_tokens,
                extra: prompt_extra,
            }),
        completion_tokens_details: (usage.reasoning_tokens != 0 || !completion_extra.is_empty())
            .then_some(super::wire::CompletionTokensDetailsWire {
                reasoning_tokens: usage.reasoning_tokens,
                extra: completion_extra,
            }),
        extra,
    })
}

pub(crate) fn decode_stop_reason(finish_reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    let reason = match finish_reason {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls" | "function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some("refusal") => StopReason::Refusal,
        Some("pause") => StopReason::PauseTurn,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::EndTurn,
    };
    crate::response::normalize_tool_stop_reason(reason, has_tool_calls)
}

fn error_code(code: Option<Value>) -> Option<String> {
    code.and_then(|value| match value {
        Value::String(code) => Some(code),
        Value::Number(code) => Some(code.to_string()),
        _ => None,
    })
}

fn fallback_error_message(status: StatusCode, body: &[u8]) -> String {
    if !body.is_empty() {
        let text = String::from_utf8_lossy(body);
        if !text.trim().is_empty() {
            return text.into_owned();
        }
    }
    status
        .canonical_reason()
        .unwrap_or("Chat Completions request failed")
        .to_owned()
}

pub(crate) fn stream_error(chunk: &ChunkWire) -> Option<ApiError> {
    let value = chunk.extra.get("error")?;
    let error: ErrorBodyWire = serde_json::from_value(value.clone()).ok()?;
    Some(ApiError {
        kind: ApiErrorKind::Upstream,
        message: error.message,
        code: error_code(error.code).or(error.kind),
    })
}

pub(crate) fn delta_for_native(fields: Map<String, Value>) -> (ContentBlock, BlockDelta) {
    let json = Value::Object(fields);
    (
        ContentBlock::Native {
            protocol: Protocol::ChatCompletions,
            json: json.clone(),
        },
        BlockDelta::Native {
            protocol: Protocol::ChatCompletions,
            json,
        },
    )
}
