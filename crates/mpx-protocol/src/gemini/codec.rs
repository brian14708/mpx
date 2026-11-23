use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Map, Value, json};

use crate::json::{merge_missing, nonempty};
use crate::{
    ApiError, ApiErrorKind, CodecContext, ContentBlock, DiagnosticKind, Endpoint, Error, Extra,
    MediaSource, Message, OutputFormat, OutputFormatKind, Protocol, ReasoningParams, ReasoningSeal,
    Request, Response, Result, Role, SafetyRating, SamplingParams, StopReason, Tool, ToolArguments,
    ToolCallId, ToolChoice, Usage, request::take_session_id,
};

use super::wire::{
    Blob, Candidate, Content, ErrorEnvelope, FileData, FinishReason, FunctionCall,
    FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, FunctionResponse,
    GenerateContentRequest, GenerateContentResponse, GenerationConfig, Part, ProviderError,
    ResponseMimeType, SafetyRating as WireSafetyRating, ThinkingConfig, Tool as WireTool,
    ToolConfig, UsageMetadata,
};

const PROTOCOL: Protocol = Protocol::GenerateContent;

pub(super) fn decode_request(
    endpoint: &Endpoint,
    body: &[u8],
    cx: &CodecContext,
) -> Result<Request> {
    let wire: GenerateContentRequest =
        serde_json::from_slice(body).map_err(|error| Error::json(PROTOCOL, error))?;
    let mut messages = Vec::with_capacity(wire.contents.len());
    for (message_index, content) in wire.contents.into_iter().enumerate() {
        messages.push(decode_message(
            content,
            message_index,
            &format!("contents[{message_index}]"),
            cx,
        )?);
    }
    let instructions = match wire.system_instruction {
        Some(content) => decode_blocks(content.parts, "systemInstruction.parts", 0, cx)?,
        None => Vec::new(),
    };
    let tools = decode_tools(wire.tools);
    link_tool_results(&mut messages)?;
    let tool_choice = match wire.tool_config {
        Some(config) => Some(decode_tool_config(config)?),
        None => None,
    };
    let (sampling, reasoning, output_format) = wire
        .generation_config
        .map(decode_generation_config)
        .transpose()?
        .unwrap_or_default();
    let model = endpoint
        .model_from_path
        .clone()
        .or(wire.model)
        .unwrap_or_default();
    let mut extra = wire.extra;
    let session_id = extra
        .get_mut("labels")
        .and_then(Value::as_object_mut)
        .and_then(take_session_id);
    if session_id.is_some() && extra["labels"].as_object().is_some_and(Map::is_empty) {
        extra.remove("labels");
    }
    Ok(Request {
        model,
        session_id,
        instructions,
        messages,
        tools,
        tool_choice,
        sampling,
        reasoning,
        output_format,
        parallel_tool_calls: None,
        stream: endpoint.stream,
        metadata: Map::new(),
        continuation: None,
        extra: make_extra(extra),
        client_include_usage: false,
        messages_max_tokens: false,
    })
}

fn link_tool_results(messages: &mut [Message]) -> Result<()> {
    let mut names = HashMap::new();
    let mut explicit_results = HashSet::new();
    // Reserve calls with explicit results before matching the remaining results by name.
    for block in messages.iter().flat_map(|message| &message.content) {
        match block {
            ContentBlock::ToolCall { id, name, .. } => {
                names.insert(id.clone(), name.clone());
            }
            ContentBlock::ToolResult { call_id, .. } if !call_id.as_str().is_empty() => {
                explicit_results.insert(call_id.clone());
            }
            _ => {}
        }
    }
    let mut pending = HashMap::<String, VecDeque<ToolCallId>>::new();
    for message in messages {
        if message.role == Role::Assistant {
            pending.clear();
        }
        for block in &mut message.content {
            match block {
                ContentBlock::ToolCall { id, name, .. } if !explicit_results.contains(id) => {
                    pending
                        .entry(name.clone())
                        .or_default()
                        .push_back(id.clone());
                }
                ContentBlock::ToolResult { call_id, name, .. } => {
                    if call_id.as_str().is_empty()
                        && let Some(name) = name.as_ref()
                        && let Some(id) = pending.get_mut(name).and_then(VecDeque::pop_front)
                    {
                        *call_id = id;
                    }
                    if name.as_ref().is_none_or(String::is_empty) {
                        *name = names.get(call_id).cloned().filter(|name| !name.is_empty());
                    }
                    if name.is_none() {
                        return Err(Error::Protocol {
                            protocol: PROTOCOL,
                            message: format!(
                                "Gemini functionResponse `{}` has no recoverable name",
                                call_id.as_str()
                            ),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn encode_request(req: &Request, cx: &CodecContext) -> Result<GenerateContentRequest> {
    if req.continuation.is_some() {
        return Err(Error::InvalidRequest {
            protocol: PROTOCOL,
            message: "Gemini generateContent does not support server-side continuation".to_owned(),
        });
    }
    if req.parallel_tool_calls.is_some() {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "parallel_tool_calls",
            "Gemini has no parallel-function-calling switch",
        )?;
    }
    if !req.metadata.is_empty() {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "metadata",
            "Gemini has no protocol-independent metadata field",
        )?;
    }

    let mut contents = Vec::with_capacity(req.messages.len());
    for (message_index, message) in req.messages.iter().enumerate() {
        contents.push(encode_message(message, message_index, cx)?);
    }
    let system_instruction = if req.instructions.is_empty() {
        None
    } else {
        Some(Content {
            role: None,
            parts: encode_blocks(&req.instructions, "systemInstruction.parts", cx)?,
            extra: Map::new(),
        })
    };
    let generation_config = encode_generation_config(req, cx)?;
    let mut extra = take_extra(req.extra.as_ref(), cx, "extra")?;
    if let Some(session_id) = &req.session_id
        && is_label_value(session_id)
        && let Some(labels) = extra
            .entry("labels")
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
    {
        labels.insert("session_id".to_owned(), Value::String(session_id.clone()));
    }
    // A model field is not normally present in a Gemini request. Retain one
    // supplied by same-protocol extensions, but use the path as the authority.
    let model = extra.remove("model").and_then(|value| {
        if endpoint_model_is_useful(&value) {
            Some(req.model.clone())
        } else {
            None
        }
    });
    Ok(GenerateContentRequest {
        contents,
        system_instruction,
        tools: encode_tools(&req.tools, cx)?,
        tool_config: req
            .tool_choice
            .as_ref()
            .map(|choice| encode_tool_choice(choice, cx))
            .transpose()?,
        generation_config,
        model,
        extra,
    })
}

fn is_label_value(value: &str) -> bool {
    !value.is_empty()
        && value.chars().take(64).count() <= 63
        && value.chars().all(|character| {
            (character.is_alphabetic() && !character.is_uppercase())
                || character.is_numeric()
                || matches!(character, '_' | '-')
        })
}

fn endpoint_model_is_useful(value: &Value) -> bool {
    value.as_str().is_some_and(|model| !model.is_empty())
}

pub(super) fn decode_response(body: &[u8], cx: &CodecContext) -> Result<Response> {
    let wire: GenerateContentResponse =
        serde_json::from_slice(body).map_err(|error| Error::json(PROTOCOL, error))?;
    response_from_wire(wire, cx)
}

pub(super) fn response_from_wire(
    mut wire: GenerateContentResponse,
    cx: &CodecContext,
) -> Result<Response> {
    if wire.candidates.len() > 1 {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "candidates",
            "only the first Gemini candidate is represented by the canonical response",
        )?;
    }
    let candidate = if wire.candidates.is_empty() {
        None
    } else {
        Some(wire.candidates.remove(0))
    };
    let mut response_extra = wire.extra;
    let mut content = Vec::new();
    let mut finish_reason = None;
    let mut safety = Vec::new();
    if let Some(candidate) = candidate {
        merge_missing(&mut response_extra, candidate.extra);
        finish_reason = candidate.finish_reason;
        safety = decode_safety(candidate.safety_ratings);
        if let Some(candidate_content) = candidate.content {
            content = decode_blocks(
                candidate_content.parts,
                "candidates[0].content.parts",
                0,
                cx,
            )?;
        }
    }
    let mut blocked = false;
    if let Some(feedback) = wire.prompt_feedback {
        merge_missing(&mut response_extra, feedback.extra);
        blocked = feedback.block_reason.is_some();
        safety.extend(decode_safety(feedback.safety_ratings));
    }
    let has_tool_call = content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolCall { .. }));
    let stop_reason = normalized_stop_reason(finish_reason.as_ref(), blocked, has_tool_call);
    Ok(Response {
        id: wire.response_id.unwrap_or_default(),
        model: wire.model_version.unwrap_or_default(),
        content,
        stop_reason,
        usage: decode_usage(wire.usage_metadata),
        safety,
        extra: make_extra(response_extra),
    })
}

pub(super) fn normalized_stop_reason(
    finish_reason: Option<&FinishReason>,
    blocked: bool,
    has_tool_call: bool,
) -> StopReason {
    let reason = match finish_reason {
        Some(reason) => StopReason::from(reason),
        None if blocked => StopReason::ContentFilter,
        None => StopReason::EndTurn,
    };
    crate::response::normalize_tool_stop_reason(reason, !blocked && has_tool_call)
}

pub(super) fn encode_response(
    response: &Response,
    cx: &CodecContext,
) -> Result<GenerateContentResponse> {
    let stop_reason: Option<FinishReason> = Some((&response.stop_reason).into());
    Ok(GenerateContentResponse {
        candidates: vec![Candidate {
            content: Some(Content {
                role: Some(super::wire::WireRole::Model),
                parts: encode_blocks(&response.content, "candidates[0].content.parts", cx)?,
                extra: Map::new(),
            }),
            finish_reason: stop_reason,
            safety_ratings: encode_safety(&response.safety, cx)?,
            index: Some(0),
            extra: Map::new(),
        }],
        prompt_feedback: None,
        usage_metadata: Some(encode_usage(&response.usage, cx)?),
        model_version: nonempty(&response.model),
        response_id: nonempty(&response.id),
        extra: take_extra(response.extra.as_ref(), cx, "extra")?,
    })
}

fn decode_message(
    content: Content,
    message_index: usize,
    path: &str,
    cx: &CodecContext,
) -> Result<Message> {
    let role = match content.role {
        Some(super::wire::WireRole::Model) => Role::Assistant,
        Some(super::wire::WireRole::Function) => Role::Tool,
        Some(super::wire::WireRole::User) | None => Role::User,
        Some(super::wire::WireRole::Unknown(_)) => {
            cx.diagnose(
                DiagnosticKind::Approximated,
                PROTOCOL,
                format!("{path}.role"),
                "unrecognized Gemini role was treated as user",
            )?;
            Role::User
        }
    };
    Ok(Message {
        role,
        content: decode_blocks(content.parts, &format!("{path}.parts"), message_index, cx)?,
        extra: make_extra(content.extra),
    })
}

fn encode_message(message: &Message, message_index: usize, cx: &CodecContext) -> Result<Content> {
    let role = match message.role {
        Role::Assistant => super::wire::WireRole::Model,
        Role::User | Role::Tool => super::wire::WireRole::User,
        Role::System | Role::Developer => {
            cx.diagnose(
                DiagnosticKind::Approximated,
                PROTOCOL,
                format!("messages[{message_index}].role"),
                "an in-conversation system role was encoded as Gemini user",
            )?;
            super::wire::WireRole::User
        }
    };
    Ok(Content {
        role: Some(role),
        parts: encode_blocks(
            &message.content,
            &format!("contents[{message_index}].parts"),
            cx,
        )?,
        extra: take_extra(
            message.extra.as_ref(),
            cx,
            &format!("messages[{message_index}].extra"),
        )?,
    })
}

pub(super) fn decode_blocks(
    parts: Vec<Part>,
    path: &str,
    message_index: usize,
    cx: &CodecContext,
) -> Result<Vec<ContentBlock>> {
    let mut blocks = Vec::with_capacity(parts.len());
    for (part_index, part) in parts.into_iter().enumerate() {
        let position = format!("{message_index}_{part_index}");
        if let Some(block) = decode_part(part, &position, &format!("{path}[{part_index}]"), cx)? {
            blocks.push(block);
        }
    }
    Ok(blocks)
}

#[allow(clippy::too_many_lines)]
pub(super) fn decode_part(
    part: Part,
    deterministic_position: &str,
    path: &str,
    cx: &CodecContext,
) -> Result<Option<ContentBlock>> {
    let primary_count = usize::from(part.text.is_some())
        + usize::from(part.inline_data.is_some())
        + usize::from(part.file_data.is_some())
        + usize::from(part.function_call.is_some())
        + usize::from(part.function_response.is_some());
    if primary_count > 1 {
        return Err(Error::Protocol {
            protocol: PROTOCOL,
            message: format!("{path} contains more than one content variant"),
        });
    }

    let thought_signature = part
        .thought_signature
        .map(|value| ReasoningSeal::new(PROTOCOL, value));
    let part_extra = part.extra;
    let block = if let Some(text) = part.text {
        if part.thought.unwrap_or(false) {
            ContentBlock::Reasoning {
                text,
                summary: Vec::new(),
                seal: thought_signature,
                redacted: false,
                extra: make_extra(part_extra),
            }
        } else {
            if let Some(seal) = thought_signature {
                cx.diagnose(
                    DiagnosticKind::Approximated,
                    PROTOCOL,
                    format!("{path}.thoughtSignature"),
                    "visible text and its signature were represented as reasoning to retain both",
                )?;
                return Ok(Some(ContentBlock::Reasoning {
                    text,
                    summary: Vec::new(),
                    seal: Some(seal),
                    redacted: false,
                    extra: make_extra(part_extra),
                }));
            }
            ContentBlock::Text {
                text,
                extra: make_extra(part_extra),
            }
        }
    } else if let Some(blob) = part.inline_data {
        let source = MediaSource::Base64 {
            media_type: blob.mime_type,
            data: blob.data,
        };
        media_block(source, merge_unknown(part_extra, blob.extra))
    } else if let Some(file) = part.file_data {
        let uri = file.file_uri;
        let media_type = nonempty(&file.mime_type);
        let source = if uri.starts_with("http://") || uri.starts_with("https://") {
            MediaSource::Url {
                url: uri,
                media_type,
            }
        } else {
            MediaSource::File {
                file_id: uri,
                media_type,
            }
        };
        media_block(source, merge_unknown(part_extra, file.extra))
    } else if let Some(call) = part.function_call {
        let id = match call.id {
            Some(id) if !id.is_empty() => id,
            _ => {
                let id = crate::tool::synthesized_tool_call_id(deterministic_position);
                cx.diagnose(
                    DiagnosticKind::Synthesized,
                    PROTOCOL,
                    format!("{path}.functionCall.id"),
                    format!("synthesized deterministic function-call ID `{id}`"),
                )?;
                id
            }
        };
        ContentBlock::ToolCall {
            id: ToolCallId::new(id),
            name: call.name,
            arguments: ToolArguments::new(call.args.to_string()),
            seal: thought_signature,
            extra: make_extra(merge_unknown(part_extra, call.extra)),
        }
    } else if let Some(response) = part.function_response {
        let call_id = response.id.unwrap_or_default();
        let is_error = response
            .response
            .as_object()
            .is_some_and(|object| object.contains_key("error"));
        let content = decode_function_result(
            &response.response,
            &response.parts,
            deterministic_position,
            path,
            cx,
        )?;
        ContentBlock::ToolResult {
            call_id: ToolCallId::new(call_id),
            name: response.name,
            content,
            is_error,
            extra: make_extra(merge_unknown(part_extra, response.extra)),
        }
    } else if thought_signature.is_some() || part.thought.unwrap_or(false) {
        ContentBlock::Reasoning {
            text: String::new(),
            summary: Vec::new(),
            seal: thought_signature,
            redacted: false,
            extra: make_extra(part_extra),
        }
    } else {
        let value = Part {
            extra: part_extra,
            ..Part::default()
        };
        let json = serde_json::to_value(value).map_err(|error| Error::json(PROTOCOL, error))?;
        if json.as_object().is_some_and(Map::is_empty) {
            return Ok(None);
        }
        ContentBlock::Native {
            protocol: PROTOCOL,
            json,
        }
    };
    Ok(Some(block))
}

fn media_block(source: MediaSource, extra: Map<String, Value>) -> ContentBlock {
    let media_type = match &source {
        MediaSource::Url { media_type, .. } | MediaSource::File { media_type, .. } => {
            media_type.as_deref()
        }
        MediaSource::Base64 { media_type, .. } => Some(media_type.as_str()),
    };
    let extra = make_extra(extra);
    if media_type.is_some_and(|value| value.starts_with("image/")) {
        ContentBlock::Image { source, extra }
    } else if media_type.is_some_and(|value| value.starts_with("audio/")) {
        ContentBlock::Audio { source, extra }
    } else {
        ContentBlock::File {
            source,
            filename: None,
            extra,
        }
    }
}

fn decode_function_result(
    response: &Value,
    parts: &[Part],
    deterministic_position: &str,
    path: &str,
    cx: &CodecContext,
) -> Result<Vec<ContentBlock>> {
    let mut content = Vec::new();
    if let Some(parts) = response
        .as_object()
        .filter(|object| object.len() == 1)
        .and_then(|object| object.get("content"))
        .and_then(Value::as_array)
    {
        content.reserve(parts.len());
        for (index, value) in parts.iter().enumerate() {
            let part: Part = serde_json::from_value(value.clone())
                .map_err(|error| Error::json(PROTOCOL, error))?;
            if let Some(block) = decode_part(
                part,
                &format!("{deterministic_position}_{index}"),
                &format!("{path}.functionResponse.response.content[{index}]"),
                cx,
            )? {
                content.push(block);
            }
        }
    }
    if content.is_empty() && !response.is_null() {
        let text = response
            .get("output")
            .and_then(Value::as_str)
            .map_or_else(|| response.to_string(), str::to_owned);
        content.push(ContentBlock::text(text));
    }
    for (index, part) in parts.iter().cloned().enumerate() {
        if let Some(block) = decode_part(
            part,
            &format!("{deterministic_position}_part_{index}"),
            &format!("{path}.functionResponse.parts[{index}]"),
            cx,
        )? {
            content.push(block);
        }
    }
    Ok(content)
}

pub(super) fn encode_blocks(
    blocks: &[ContentBlock],
    path: &str,
    cx: &CodecContext,
) -> Result<Vec<Part>> {
    let mut parts = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        if let Some(part) = encode_part(block, &format!("{path}[{index}]"), cx)? {
            parts.push(part);
        }
    }
    Ok(parts)
}

#[allow(clippy::too_many_lines)]
pub(super) fn encode_part(
    block: &ContentBlock,
    path: &str,
    cx: &CodecContext,
) -> Result<Option<Part>> {
    let mut part = Part::default();
    match block {
        ContentBlock::Text { text, extra } => {
            part.text = Some(text.clone());
            part.extra = take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?;
        }
        ContentBlock::Reasoning {
            text,
            summary,
            seal,
            redacted,
            extra,
        } => {
            if !summary.is_empty() {
                cx.diagnose(
                    DiagnosticKind::Approximated,
                    PROTOCOL,
                    format!("{path}.summary"),
                    "Gemini reasoning summaries were encoded as thought text",
                )?;
            }
            if *redacted {
                cx.diagnose(
                    DiagnosticKind::Approximated,
                    PROTOCOL,
                    format!("{path}.redacted"),
                    "Gemini has no explicit redacted-thought marker",
                )?;
            }
            let mut value = text.clone();
            if !summary.is_empty() {
                if !value.is_empty() {
                    value.push('\n');
                }
                value.push_str(&summary.join("\n"));
            }
            part.text = Some(value);
            part.thought = Some(true);
            part.thought_signature = encode_seal(seal.as_ref(), cx, path)?;
            part.extra = take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?;
        }
        ContentBlock::Image { source, extra }
        | ContentBlock::File { source, extra, .. }
        | ContentBlock::Audio { source, extra } => {
            encode_media(source, &mut part);
            match &mut part.inline_data {
                Some(blob) => {
                    blob.extra = take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?;
                }
                None => {
                    if let Some(file) = &mut part.file_data {
                        file.extra = take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?;
                    }
                }
            }
            if matches!(
                block,
                ContentBlock::File {
                    filename: Some(_),
                    ..
                }
            ) {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    format!("{path}.filename"),
                    "Gemini fileData has no filename field",
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
            let args = if arguments.as_str().is_empty() {
                Value::Object(Map::new())
            } else {
                serde_json::from_str(arguments.as_str()).map_err(|error| Error::InvalidRequest {
                    protocol: PROTOCOL,
                    message: format!("{path}.arguments is not valid JSON: {error}"),
                })?
            };
            part.thought_signature = encode_seal(seal.as_ref(), cx, path)?;
            part.function_call = Some(FunctionCall {
                id: nonempty(id.as_str()),
                name: name.clone(),
                args,
                extra: take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?,
            });
        }
        ContentBlock::ToolResult {
            call_id,
            name,
            content,
            is_error,
            extra,
        } => {
            let Some(name) = name.as_ref().filter(|name| !name.is_empty()) else {
                return Err(Error::InvalidRequest {
                    protocol: PROTOCOL,
                    message: format!(
                        "{path}.name is required for Gemini functionResponse `{}`",
                        call_id.as_str()
                    ),
                });
            };
            let (response, parts) = encode_function_result(content, *is_error, path, cx)?;
            part.function_response = Some(FunctionResponse {
                id: nonempty(call_id.as_str()),
                name: Some(name.clone()),
                response,
                parts,
                extra: take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?,
            });
        }
        ContentBlock::Refusal { text, extra } => {
            cx.diagnose(
                DiagnosticKind::Fallback,
                PROTOCOL,
                path,
                "a refusal block was encoded as visible Gemini text",
            )?;
            part.text = Some(text.clone());
            part.extra = take_extra(extra.as_ref(), cx, &format!("{path}.extra"))?;
        }
        ContentBlock::Native {
            protocol: PROTOCOL,
            json,
        } => {
            let native: Part = serde_json::from_value(json.clone())
                .map_err(|error| Error::json(PROTOCOL, error))?;
            return Ok(Some(native));
        }
        ContentBlock::Native { .. } | ContentBlock::WebSearchResult { .. } => {
            if cx.options.content_fallbacks {
                cx.diagnose(
                    DiagnosticKind::Fallback,
                    PROTOCOL,
                    path,
                    "unsupported rich content was encoded as JSON text",
                )?;
                part.text = Some(
                    serde_json::to_value(block)
                        .map_err(|error| Error::json(PROTOCOL, error))?
                        .to_string(),
                );
            } else {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    path,
                    "unsupported rich content was omitted",
                )?;
                return Ok(None);
            }
        }
    }
    Ok(Some(part))
}

fn encode_media(source: &MediaSource, part: &mut Part) {
    match source {
        MediaSource::Base64 { media_type, data } => {
            part.inline_data = Some(Blob {
                mime_type: media_type.clone(),
                data: data.clone(),
                extra: Map::new(),
            });
        }
        MediaSource::Url { url, media_type } => {
            part.file_data = Some(FileData {
                mime_type: media_type.clone().unwrap_or_default(),
                file_uri: url.clone(),
                extra: Map::new(),
            });
        }
        MediaSource::File {
            file_id,
            media_type,
        } => {
            part.file_data = Some(FileData {
                mime_type: media_type.clone().unwrap_or_default(),
                file_uri: file_id.clone(),
                extra: Map::new(),
            });
        }
    }
}

fn encode_function_result(
    content: &[ContentBlock],
    is_error: bool,
    path: &str,
    cx: &CodecContext,
) -> Result<(Value, Vec<Part>)> {
    let (mut value, parts) = if let [ContentBlock::Text { text, .. }] = content {
        (
            serde_json::from_str(text).unwrap_or_else(|_| json!({ "output": text })),
            Vec::new(),
        )
    } else {
        let parts = encode_blocks(content, &format!("{path}.content"), cx)?;
        let response_parts = parts
            .iter()
            .filter_map(|part| part.text.clone())
            .collect::<Vec<_>>();
        (json!({ "output": response_parts.join("\n") }), parts)
    };
    if is_error
        && value
            .as_object()
            .is_none_or(|object| !object.contains_key("error"))
    {
        value = json!({ "error": value });
    }
    Ok((value, parts))
}

fn decode_tools(tools: Vec<WireTool>) -> Vec<Tool> {
    let mut decoded = Vec::new();
    for tool in tools {
        for mut declaration in tool.function_declarations {
            let legacy_schema = declaration.parameters.is_some();
            let parameters = match (declaration.parameters, declaration.parameters_json_schema) {
                (Some(parameters), Some(json_schema)) => {
                    declaration
                        .extra
                        .insert("parametersJsonSchema".to_owned(), json_schema);
                    parameters
                }
                (Some(parameters), None) | (None, Some(parameters)) => parameters,
                (None, None) => Value::Object(Map::new()),
            };
            decoded.push(Tool::Function {
                name: declaration.name,
                description: declaration.description,
                parameters,
                strict: None,
                defer_loading: None,
                extra: schema_extra(declaration.extra, legacy_schema),
            });
        }
        if !tool.extra.is_empty() {
            decoded.push(Tool::Hosted {
                protocol: PROTOCOL,
                json: Value::Object(tool.extra),
            });
        }
    }
    decoded
}

fn encode_tools(tools: &[Tool], cx: &CodecContext) -> Result<Vec<WireTool>> {
    let mut wire = Vec::new();
    let mut declarations = Vec::new();
    let flush = |wire: &mut Vec<WireTool>, declarations: &mut Vec<FunctionDeclaration>| {
        if !declarations.is_empty() {
            wire.push(WireTool {
                function_declarations: std::mem::take(declarations),
                extra: Map::new(),
            });
        }
    };
    for (index, tool) in tools.iter().enumerate() {
        match tool {
            Tool::Function {
                name,
                description,
                parameters,
                strict,
                defer_loading,
                extra,
            } => {
                if strict.is_some() || defer_loading.is_some() {
                    cx.diagnose(
                        DiagnosticKind::Dropped,
                        PROTOCOL,
                        format!("tools[{index}]"),
                        "Gemini function declarations do not support strict or deferred loading",
                    )?;
                }
                let (parameters, parameters_json_schema) = if has_legacy_schema(extra.as_ref()) {
                    (Some(parameters.clone()), None)
                } else if parameters.is_null() || parameters.as_object().is_some_and(Map::is_empty)
                {
                    (None, None)
                } else {
                    (None, Some(parameters.clone()))
                };
                declarations.push(FunctionDeclaration {
                    name: name.clone(),
                    description: description.clone(),
                    parameters,
                    parameters_json_schema,
                    extra: take_extra(extra.as_ref(), cx, &format!("tools[{index}].extra"))?,
                });
            }
            Tool::Hosted {
                protocol: PROTOCOL,
                json,
            } => {
                flush(&mut wire, &mut declarations);
                let hosted: WireTool = serde_json::from_value(json.clone())
                    .map_err(|error| Error::json(PROTOCOL, error))?;
                wire.push(hosted);
            }
            Tool::Custom { .. } | Tool::Hosted { .. } => {
                flush(&mut wire, &mut declarations);
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    format!("tools[{index}]"),
                    "tool declaration is not a Gemini function declaration",
                )?;
            }
        }
    }
    flush(&mut wire, &mut declarations);
    Ok(wire)
}

fn decode_tool_config(config: ToolConfig) -> Result<ToolChoice> {
    if !config.extra.is_empty() {
        return native_tool_choice(config);
    }
    let Some(calling) = config.function_calling_config else {
        return Ok(ToolChoice::Auto);
    };
    if !calling.extra.is_empty() {
        return native_tool_choice(ToolConfig {
            function_calling_config: Some(calling),
            extra: Map::new(),
        });
    }
    let mode = calling
        .mode
        .as_ref()
        .map_or("AUTO", FunctionCallingMode::as_str);
    match (mode, calling.allowed_function_names.as_slice()) {
        ("AUTO", []) => Ok(ToolChoice::Auto),
        ("NONE", []) => Ok(ToolChoice::None),
        ("ANY", []) => Ok(ToolChoice::Required),
        ("ANY", [name]) => Ok(ToolChoice::Named(name.clone())),
        _ => native_tool_choice(ToolConfig {
            function_calling_config: Some(calling),
            extra: Map::new(),
        }),
    }
}

fn native_tool_choice(config: ToolConfig) -> Result<ToolChoice> {
    let json = serde_json::to_value(config).map_err(|error| Error::json(PROTOCOL, error))?;
    Ok(ToolChoice::Native {
        protocol: PROTOCOL,
        json,
    })
}

fn encode_tool_choice(choice: &ToolChoice, cx: &CodecContext) -> Result<ToolConfig> {
    let (mode, names) = match choice {
        ToolChoice::Auto => ("AUTO", Vec::new()),
        ToolChoice::None => ("NONE", Vec::new()),
        ToolChoice::Required => ("ANY", Vec::new()),
        ToolChoice::Named(name) => ("ANY", vec![name.clone()]),
        ToolChoice::Native {
            protocol: PROTOCOL,
            json,
        } => {
            return serde_json::from_value(json.clone())
                .map_err(|error| Error::json(PROTOCOL, error));
        }
        ToolChoice::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "tool_choice",
                "foreign native tool choice was replaced with AUTO",
            )?;
            ("AUTO", Vec::new())
        }
    };
    Ok(ToolConfig {
        function_calling_config: Some(FunctionCallingConfig {
            mode: Some(match mode {
                "AUTO" => FunctionCallingMode::Auto,
                "ANY" => FunctionCallingMode::Any,
                "NONE" => FunctionCallingMode::None,
                _ => FunctionCallingMode::Unknown(mode.to_owned()),
            }),
            allowed_function_names: names,
            extra: Map::new(),
        }),
        extra: Map::new(),
    })
}

fn decode_generation_config(
    config: GenerationConfig,
) -> Result<(
    SamplingParams,
    Option<ReasoningParams>,
    Option<OutputFormat>,
)> {
    let reasoning = config
        .thinking_config
        .map(decode_thinking_config)
        .transpose()?;
    let legacy_schema = config.response_schema.is_some();
    let schema_value = config.response_schema.or(config.response_json_schema);
    let mut output_extra = Map::new();
    let output_format = match config
        .response_mime_type
        .as_ref()
        .map(ResponseMimeType::as_str)
    {
        Some("application/json") => Some(OutputFormat {
            kind: if schema_value.is_some() {
                OutputFormatKind::JsonSchema
            } else {
                OutputFormatKind::JsonObject
            },
            schema: schema_value,
            name: None,
            strict: None,
            extra: schema_extra(Map::new(), legacy_schema),
        }),
        Some("text/plain") => Some(OutputFormat {
            kind: OutputFormatKind::Text,
            schema: schema_value,
            name: None,
            strict: None,
            extra: schema_extra(Map::new(), legacy_schema),
        }),
        Some(mime) => {
            output_extra.insert(
                "responseMimeType".to_owned(),
                Value::String(mime.to_owned()),
            );
            if let Some(schema) = schema_value {
                let key = if legacy_schema {
                    "responseSchema"
                } else {
                    "responseJsonSchema"
                };
                output_extra.insert(key.to_owned(), schema);
            }
            None
        }
        None => schema_value.map(|schema| OutputFormat {
            kind: OutputFormatKind::JsonSchema,
            schema: Some(schema),
            name: None,
            strict: None,
            extra: schema_extra(Map::new(), legacy_schema),
        }),
    };
    let mut generation_extra = config.extra;
    merge_missing(&mut generation_extra, output_extra);
    let sampling = SamplingParams {
        max_output_tokens: config.max_output_tokens,
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        stop_sequences: config.stop_sequences,
        seed: config.seed,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        extra: make_extra(generation_extra),
        chat_legacy_max_tokens: false,
    };
    Ok((sampling, reasoning, output_format))
}

fn decode_thinking_config(thinking: ThinkingConfig) -> Result<ReasoningParams> {
    let mut extra = thinking.extra;
    let budget_tokens = match thinking.thinking_budget {
        Some(-1) => {
            extra.insert("thinkingBudget".to_owned(), Value::from(-1));
            None
        }
        Some(budget) => Some(u32::try_from(budget).map_err(|_| Error::InvalidRequest {
            protocol: PROTOCOL,
            message: format!("invalid Gemini thinkingBudget: {budget}"),
        })?),
        None => None,
    };
    Ok(ReasoningParams {
        effort: thinking.thinking_level,
        budget_tokens,
        summary: thinking.include_thoughts.map(|include| {
            if include {
                "auto".to_owned()
            } else {
                "none".to_owned()
            }
        }),
        extra: make_extra(extra),
    })
}

fn encode_generation_config(req: &Request, cx: &CodecContext) -> Result<Option<GenerationConfig>> {
    if req.sampling == SamplingParams::default()
        && req.reasoning.is_none()
        && req.output_format.is_none()
    {
        return Ok(None);
    }
    let sampling = &req.sampling;
    let mut config = GenerationConfig {
        max_output_tokens: sampling.max_output_tokens,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        stop_sequences: sampling.stop_sequences.clone(),
        seed: sampling.seed,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        extra: take_extra(sampling.extra.as_ref(), cx, "sampling.extra")?,
        ..GenerationConfig::default()
    };
    if let Some(reasoning) = &req.reasoning {
        let mut extra = take_extra(reasoning.extra.as_ref(), cx, "reasoning.extra")?;
        let native_budget = extra
            .remove("thinkingBudget")
            .and_then(|value| value.as_i64());
        config.thinking_config = Some(ThinkingConfig {
            thinking_budget: reasoning.budget_tokens.map(i64::from).or(native_budget),
            thinking_level: reasoning.effort.clone(),
            include_thoughts: reasoning.summary.as_ref().map(|summary| summary != "none"),
            extra,
        });
    }
    if let Some(output) = &req.output_format {
        if output.name.is_some() || output.strict.is_some() {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "output_format",
                "Gemini structured output has no schema name or strict switch",
            )?;
        }
        match output.kind {
            OutputFormatKind::Text => config.response_mime_type = Some(ResponseMimeType::Text),
            OutputFormatKind::JsonObject | OutputFormatKind::JsonSchema => {
                config.response_mime_type = Some(ResponseMimeType::Json);
            }
        }
        if matches!(output.kind, OutputFormatKind::JsonSchema) {
            if has_legacy_schema(output.extra.as_ref()) {
                config.response_schema.clone_from(&output.schema);
            } else {
                config.response_json_schema.clone_from(&output.schema);
            }
        }
        merge_missing(
            &mut config.extra,
            take_extra(output.extra.as_ref(), cx, "output_format.extra")?,
        );
    }
    Ok(Some(config))
}

pub(super) fn decode_usage(usage: Option<UsageMetadata>) -> Usage {
    usage.map_or_else(Usage::default, |usage| {
        let cached = usage.cached_content_token_count.unwrap_or(0);
        Usage {
            input_tokens: usage.prompt_token_count.unwrap_or(0).saturating_sub(cached),
            output_tokens: usage.candidates_token_count.unwrap_or(0),
            cached_input_tokens: cached,
            cache_creation_input_tokens: 0,
            reasoning_tokens: usage.thoughts_token_count.unwrap_or(0),
            extra: make_extra(usage.extra),
        }
    })
}

pub(super) fn encode_usage(usage: &Usage, cx: &CodecContext) -> Result<UsageMetadata> {
    if usage.cache_creation_input_tokens != 0 {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "usage.cache_creation_input_tokens",
            "Gemini does not report cache-creation tokens",
        )?;
    }
    Ok(UsageMetadata {
        prompt_token_count: Some(usage.input_tokens.saturating_add(usage.cached_input_tokens)),
        candidates_token_count: Some(usage.output_tokens),
        cached_content_token_count: Some(usage.cached_input_tokens),
        thoughts_token_count: Some(usage.reasoning_tokens),
        extra: take_extra(usage.extra.as_ref(), cx, "usage.extra")?,
    })
}

pub(super) fn decode_safety(ratings: Vec<WireSafetyRating>) -> Vec<SafetyRating> {
    ratings
        .into_iter()
        .map(|rating| SafetyRating {
            category: rating.category,
            probability: rating.probability,
            blocked: rating.blocked.unwrap_or(false),
            extra: make_extra(rating.extra),
        })
        .collect()
}

fn encode_safety(ratings: &[SafetyRating], cx: &CodecContext) -> Result<Vec<WireSafetyRating>> {
    ratings
        .iter()
        .enumerate()
        .map(|(index, rating)| {
            Ok(WireSafetyRating {
                category: rating.category.clone(),
                probability: rating.probability.clone(),
                blocked: Some(rating.blocked),
                extra: take_extra(rating.extra.as_ref(), cx, &format!("safety[{index}].extra"))?,
            })
        })
        .collect()
}

pub(super) fn decode_error_from_value(value: Value) -> Result<crate::StreamEvent> {
    let envelope: ErrorEnvelope =
        serde_json::from_value(value).map_err(|error| Error::json(PROTOCOL, error))?;
    Ok(crate::StreamEvent::Error {
        error: api_error_from_provider(envelope.error, None),
    })
}

pub(super) fn decode_error(parts: &http::response::Parts, body: &[u8]) -> ApiError {
    match serde_json::from_slice::<ErrorEnvelope>(body) {
        Ok(envelope) => api_error_from_provider(envelope.error, Some(parts.status)),
        Err(_) => ApiError {
            kind: ApiErrorKind::from_status(parts.status),
            message: String::from_utf8_lossy(body).into_owned(),
            code: None,
        },
    }
}

fn api_error_from_provider(
    provider: ProviderError,
    http_status: Option<http::StatusCode>,
) -> ApiError {
    let code = provider.code;
    let status = provider.status;
    ApiError {
        kind: status
            .as_deref()
            .map(error_kind_from_status)
            .or_else(|| http_status.map(ApiErrorKind::from_status))
            .unwrap_or(ApiErrorKind::Upstream),
        message: provider.message,
        code: status.or_else(|| code.map(|code| code.to_string())),
    }
}

pub(super) fn encode_error_value(error: &ApiError) -> ErrorEnvelope {
    ErrorEnvelope {
        error: ProviderError {
            code: Some(error.status().as_u16()),
            message: error.message.clone(),
            status: Some(
                error
                    .code
                    .clone()
                    .unwrap_or_else(|| error_status(error.kind).to_owned()),
            ),
            extra: Map::new(),
        },
    }
}

fn error_kind_from_status(status: &str) -> ApiErrorKind {
    match status {
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => ApiErrorKind::InvalidRequest,
        "UNAUTHENTICATED" => ApiErrorKind::Authentication,
        "PERMISSION_DENIED" => ApiErrorKind::Permission,
        "NOT_FOUND" => ApiErrorKind::NotFound,
        "RESOURCE_EXHAUSTED" => ApiErrorKind::RateLimit,
        "DEADLINE_EXCEEDED" => ApiErrorKind::Timeout,
        "UNAVAILABLE" => ApiErrorKind::Overloaded,
        "UNIMPLEMENTED" => ApiErrorKind::Unsupported,
        _ => ApiErrorKind::Upstream,
    }
}

const fn error_status(kind: ApiErrorKind) -> &'static str {
    match kind {
        ApiErrorKind::InvalidRequest | ApiErrorKind::Unsupported => "INVALID_ARGUMENT",
        ApiErrorKind::Authentication => "UNAUTHENTICATED",
        ApiErrorKind::Permission => "PERMISSION_DENIED",
        ApiErrorKind::NotFound => "NOT_FOUND",
        ApiErrorKind::RateLimit | ApiErrorKind::BodyTooLarge => "RESOURCE_EXHAUSTED",
        ApiErrorKind::Overloaded => "UNAVAILABLE",
        ApiErrorKind::Timeout => "DEADLINE_EXCEEDED",
        ApiErrorKind::BadResponse | ApiErrorKind::Upstream | ApiErrorKind::Internal => "INTERNAL",
    }
}

pub(super) fn make_extra(fields: Map<String, Value>) -> Option<Extra> {
    Extra::from_unknown(PROTOCOL, fields)
}

fn schema_extra(fields: Map<String, Value>, legacy_schema: bool) -> Option<Extra> {
    if legacy_schema {
        let mut extra = Extra::with_fields(PROTOCOL, fields);
        extra.gemini_legacy_schema = true;
        Some(extra)
    } else {
        make_extra(fields)
    }
}

fn has_legacy_schema(extra: Option<&Extra>) -> bool {
    extra.is_some_and(|extra| extra.protocol == PROTOCOL && extra.gemini_legacy_schema)
}

pub(super) fn take_extra(
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Map<String, Value>> {
    Extra::fields_for(extra, PROTOCOL, cx, path)
}

fn merge_unknown(mut first: Map<String, Value>, second: Map<String, Value>) -> Map<String, Value> {
    merge_missing(&mut first, second);
    first
}

fn encode_seal(
    seal: Option<&ReasoningSeal>,
    cx: &CodecContext,
    path: &str,
) -> Result<Option<String>> {
    match seal {
        Some(seal) if seal.protocol == PROTOCOL => Ok(Some(seal.value.clone())),
        Some(_) => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                format!("{path}.seal"),
                "a reasoning seal from another protocol was omitted",
            )?;
            Ok(None)
        }
        None => Ok(None),
    }
}

pub(super) fn response_chunk(
    parts: Vec<Part>,
    finish_reason: Option<FinishReason>,
    usage: Option<UsageMetadata>,
    model: Option<String>,
    id: Option<String>,
    extra: Map<String, Value>,
) -> GenerateContentResponse {
    GenerateContentResponse {
        candidates: vec![Candidate {
            content: if parts.is_empty() {
                None
            } else {
                Some(Content {
                    role: Some(super::wire::WireRole::Model),
                    parts,
                    extra: Map::new(),
                })
            },
            finish_reason,
            safety_ratings: Vec::new(),
            index: Some(0),
            extra: Map::new(),
        }],
        prompt_feedback: None,
        usage_metadata: usage,
        model_version: model,
        response_id: id,
        extra,
    }
}
