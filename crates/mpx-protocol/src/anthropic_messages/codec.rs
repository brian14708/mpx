use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::{StatusCode, header};
use serde_json::{Map, Value, json};

use crate::json::{insert_option, merge_missing, nonempty, take_object};
use crate::{
    ApiError, ApiErrorKind, ClientContext, CodecContext, ContentBlock, DiagnosticKind, Endpoint,
    Error, Extra, MediaSource, Message, OutputFormat, OutputFormatKind, Protocol, ProtocolCodec,
    ReasoningParams, ReasoningSeal, Request, Response, Result, Role, SamplingParams, SearchResult,
    StopReason, StreamDecoder, StreamEncoder, Tool, ToolArguments, ToolCallId, ToolChoice, Usage,
    anthropic_messages::{
        stream::{MessagesStreamDecoder, MessagesStreamEncoder},
        wire::{self, MessagesRequest, MessagesResponse, WireMessage, WireUsage},
    },
    codec::json_response,
    request::session_id_field,
    response::normalize_tool_stop_reason,
    tool::synthesized_tool_call_id,
};

/// Sans-IO codec for Anthropic Messages.
#[derive(Debug, Clone)]
pub(crate) struct MessagesCodec;

fn claude_session_id(user_id: &Value) -> Option<String> {
    if let Some(fields) = user_id.as_object() {
        return session_id_field(fields);
    }
    let user_id = user_id.as_str()?.trim();
    if user_id.starts_with('{') {
        let fields = serde_json::from_str(user_id).ok()?;
        return session_id_field(&fields);
    }
    // Older Claude Code clients append the session ID to the user identifier.
    let (user, session_id) = user_id.strip_prefix("user_")?.rsplit_once("_session_")?;
    if user.is_empty() {
        return None;
    }
    nonempty(session_id.trim())
}

impl ProtocolCodec for MessagesCodec {
    fn protocol(&self) -> Protocol {
        Protocol::Messages
    }

    fn can_replay_request(&self, request: &Request) -> bool {
        request.messages_max_tokens
    }

    fn decode_request(
        &self,
        _endpoint: &Endpoint,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Request> {
        let wire: MessagesRequest =
            serde_json::from_slice(body).map_err(|error| Error::json(Protocol::Messages, error))?;
        let instructions = match wire.system {
            Some(value) => decode_content(value, cx)?,
            None => Vec::new(),
        };
        let mut messages = Vec::new();
        for message in wire.messages {
            let role = match message.role {
                wire::WireRole::User => Role::User,
                wire::WireRole::Assistant => Role::Assistant,
                wire::WireRole::System => Role::System,
                wire::WireRole::Developer => Role::Developer,
                wire::WireRole::Unknown(_) => Role::User,
            };
            Message {
                role,
                content: decode_content(message.content, cx)?,
                extra: wire::extra_fields(message.extra),
            }
            .split_tool_results(&mut messages);
        }
        let mut tools = Vec::new();
        for tool in wire.tools.unwrap_or_default() {
            tools.push(decode_tool(tool));
        }
        let (tool_choice, parallel_from_choice) = match wire.tool_choice {
            Some(value) => decode_tool_choice(value),
            None => (None, None),
        };
        let (mut reasoning, output_config_extra) = decode_reasoning(wire.thinking);
        let (output_format, effort, mut output_config) = decode_output_config(wire.output_config);
        if let Some(effort) = effort {
            reasoning
                .get_or_insert_with(ReasoningParams::default)
                .effort = Some(effort);
        }
        if let Some(extra) = output_config_extra {
            merge_missing(&mut output_config, extra);
        }
        let mut extra = wire.extra;
        if !output_config.is_empty() {
            extra.insert("output_config".to_owned(), Value::Object(output_config));
        }
        let metadata = wire.metadata.unwrap_or_default();
        let session_id = session_id_field(&metadata)
            .or_else(|| metadata.get("user_id").and_then(claude_session_id));
        Ok(Request {
            model: wire.model,
            session_id,
            instructions,
            messages,
            tools,
            tool_choice,
            sampling: SamplingParams {
                max_output_tokens: wire.max_tokens,
                temperature: wire.temperature,
                top_p: wire.top_p,
                top_k: wire.top_k,
                stop_sequences: wire.stop_sequences.unwrap_or_default(),
                seed: None,
                presence_penalty: None,
                frequency_penalty: None,
                extra: None,
                chat_legacy_max_tokens: false,
            },
            reasoning,
            output_format,
            parallel_tool_calls: parallel_from_choice,
            stream: wire.stream.unwrap_or(false),
            metadata,
            continuation: None,
            extra: wire::extra_fields(extra),
            client_include_usage: false,
            messages_max_tokens: wire.max_tokens.is_some(),
        })
    }

    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>> {
        if req.continuation.is_some() {
            return Err(Error::InvalidRequest {
                protocol: Protocol::Messages,
                message: "Messages does not support previous_response_id continuation".to_owned(),
            });
        }
        let mut object = request_extra(req.extra.as_ref(), cx)?;
        object.insert("model".to_owned(), Value::String(req.model.clone()));
        let max_tokens = if let Some(value) = req.sampling.max_output_tokens {
            value
        } else {
            cx.diagnose(
                DiagnosticKind::Synthesized,
                Protocol::Messages,
                "max_tokens",
                "required max_tokens was missing; the configured default was used",
            )?;
            cx.options.default_max_tokens
        };
        object.insert("max_tokens".to_owned(), json!(max_tokens));
        object.insert("stream".to_owned(), Value::Bool(req.stream));
        insert_option(
            &mut object,
            "temperature",
            req.sampling.temperature.map(Value::from),
        );
        insert_option(&mut object, "top_p", req.sampling.top_p.map(Value::from));
        insert_option(&mut object, "top_k", req.sampling.top_k.map(Value::from));
        if !req.sampling.stop_sequences.is_empty() {
            object.insert(
                "stop_sequences".to_owned(),
                json!(req.sampling.stop_sequences),
            );
        }
        if !req.metadata.is_empty() {
            object.insert("metadata".to_owned(), Value::Object(req.metadata.clone()));
        }

        let mut encoded_tools = Vec::new();
        for tool in &req.tools {
            if let Some(tool) = encode_tool(tool, cx)? {
                encoded_tools.push(tool);
            }
        }
        if !encoded_tools.is_empty() {
            object.insert("tools".to_owned(), Value::Array(encoded_tools));
        }
        if let Some(choice) =
            encode_tool_choice(req.tool_choice.as_ref(), req.parallel_tool_calls, cx)?
        {
            object.insert("tool_choice".to_owned(), choice);
        }
        if let Some(reasoning) = &req.reasoning {
            let thinking = encode_reasoning(reasoning, max_tokens, cx)?;
            object.insert("thinking".to_owned(), thinking);
            if reasoning.effort.is_some() {
                let mut output = take_object(&mut object, "output_config");
                output.insert(
                    "effort".to_owned(),
                    Value::String(reasoning.effort.clone().unwrap_or_default()),
                );
                object.insert("output_config".to_owned(), Value::Object(output));
            }
        }
        if let Some(format) = &req.output_format {
            let mut output = take_object(&mut object, "output_config");
            output.insert("format".to_owned(), encode_output_format(format));
            object.insert("output_config".to_owned(), Value::Object(output));
        }

        if !req.instructions.is_empty() {
            let mut content = Vec::new();
            for block in &req.instructions {
                if let Some(block) = encode_block(block, cx)? {
                    content.push(block);
                }
            }
            object.insert("system".to_owned(), Value::Array(content));
        }
        let messages = encode_messages(req, cx)?;
        object.insert("messages".to_owned(), Value::Array(messages));

        let bytes = Bytes::from(Value::Object(object).to_string());
        let target = if req.stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        http::Request::builder()
            .method(http::Method::POST)
            .uri("/v1/messages")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, target)
            .header("anthropic-version", "2023-06-01")
            .body(bytes)
            .map_err(Error::http)
    }

    fn decode_response(
        &self,
        _parts: &http::response::Parts,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Response> {
        let wire: MessagesResponse =
            serde_json::from_slice(body).map_err(|error| Error::json(Protocol::Messages, error))?;
        let mut content = Vec::new();
        for block in wire.content {
            content.push(decode_block(block, cx)?);
        }
        let mut stop_reason = wire
            .stop_reason
            .as_ref()
            .map(StopReason::from)
            .unwrap_or_default();
        stop_reason = normalize_tool_stop_reason(
            stop_reason,
            content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall { .. })),
        );
        Ok(Response {
            id: wire.id,
            model: wire.model,
            content,
            stop_reason,
            usage: decode_usage(wire.usage.unwrap_or_default()),
            safety: Vec::new(),
            extra: wire::extra_fields(wire.extra),
        })
    }

    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError {
        let value: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
        let error = value.get("error").unwrap_or(&value);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| fallback_error_message(parts.status), str::to_owned);
        let code = error.get("type").and_then(Value::as_str).map(str::to_owned);
        ApiError {
            kind: error_kind(parts.status, code.as_deref()),
            message,
            code,
        }
    }

    fn encode_response(
        &self,
        response: &Response,
        _client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>> {
        let mut content = Vec::new();
        for block in &response.content {
            if let Some(block) = encode_block(block, cx)? {
                content.push(block);
            }
        }
        let mut object = response_extra(response.extra.as_ref(), cx)?;
        object.insert("id".to_owned(), Value::String(response.id.clone()));
        object.insert("type".to_owned(), Value::String("message".to_owned()));
        object.insert("role".to_owned(), Value::String("assistant".to_owned()));
        object.insert("model".to_owned(), Value::String(response.model.clone()));
        object.insert("content".to_owned(), Value::Array(content));
        object.insert(
            "stop_reason".to_owned(),
            Value::String(
                wire::StopReason::from(&response.stop_reason)
                    .as_str()
                    .to_owned(),
            ),
        );
        object.insert("usage".to_owned(), encode_usage(&response.usage));
        Ok(json_response(
            StatusCode::OK,
            Bytes::from(Value::Object(object).to_string()),
        ))
    }

    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes> {
        json_response(
            error.status(),
            Bytes::from(
                json!({
                    "type": "error",
                    "error": {
                        "type": error.code.as_deref().unwrap_or(match error.kind {
                            ApiErrorKind::RateLimit => "rate_limit_error",
                            ApiErrorKind::Authentication => "authentication_error",
                            ApiErrorKind::Permission => "permission_error",
                            ApiErrorKind::NotFound => "not_found_error",
                            ApiErrorKind::Overloaded => "overloaded_error",
                            _ => "invalid_request_error",
                        }),
                        "message": error.message,
                    }
                })
                .to_string(),
            ),
        )
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(MessagesStreamDecoder::new())
    }

    fn stream_encoder(&self, client: &ClientContext) -> Box<dyn StreamEncoder> {
        Box::new(MessagesStreamEncoder::new(client))
    }
}

pub(crate) fn decode_content(value: Value, cx: &CodecContext) -> Result<Vec<ContentBlock>> {
    match value {
        Value::String(text) => Ok(vec![ContentBlock::text(text)]),
        Value::Array(blocks) => blocks
            .into_iter()
            .map(|block| decode_block(block, cx))
            .collect(),
        Value::Null => Ok(Vec::new()),
        other => Err(Error::Protocol {
            protocol: Protocol::Messages,
            message: format!("message content must be text or an array, got {other}"),
        }),
    }
}

pub(crate) fn decode_block(value: Value, cx: &CodecContext) -> Result<ContentBlock> {
    let Value::Object(mut map) = value else {
        return Ok(ContentBlock::Native {
            protocol: Protocol::Messages,
            json: value,
        });
    };
    let kind = wire::take_string(&mut map, "type").unwrap_or_default();
    match kind.as_str() {
        "text" => Ok(ContentBlock::Text {
            text: wire::take_string(&mut map, "text").unwrap_or_default(),
            extra: wire::extra_fields(map),
        }),
        "image" | "document" => {
            let filename = if kind == "document" {
                wire::take_string(&mut map, "title")
            } else {
                None
            };
            let source = decode_media_source(map.remove("source").unwrap_or(Value::Null));
            let block = if kind == "image" {
                ContentBlock::Image {
                    source,
                    extra: wire::extra_fields(map),
                }
            } else {
                ContentBlock::File {
                    source,
                    filename,
                    extra: wire::extra_fields(map),
                }
            };
            Ok(block)
        }
        "thinking" => Ok(ContentBlock::Reasoning {
            text: wire::take_string(&mut map, "thinking")
                .or_else(|| wire::take_string(&mut map, "text"))
                .unwrap_or_default(),
            summary: Vec::new(),
            seal: wire::take_string(&mut map, "signature")
                .map(|value| ReasoningSeal::new(Protocol::Messages, value)),
            redacted: false,
            extra: wire::extra_fields(map),
        }),
        "redacted_thinking" => Ok(ContentBlock::Reasoning {
            text: String::new(),
            summary: Vec::new(),
            seal: wire::take_string(&mut map, "data")
                .map(|value| ReasoningSeal::new(Protocol::Messages, value)),
            redacted: true,
            extra: wire::extra_fields(map),
        }),
        "tool_use" | "server_tool_use" => {
            let id = ToolCallId::new(
                wire::take_string(&mut map, "id")
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| synthesized_tool_call_id(0)),
            );
            let (id, seal) = decode_call_id(id);
            let input = map.remove("input").unwrap_or_else(|| json!({}));
            Ok(ContentBlock::ToolCall {
                id,
                name: wire::take_string(&mut map, "name").unwrap_or_default(),
                arguments: ToolArguments::new(input.to_string()),
                seal,
                extra: wire::extra_fields(map),
            })
        }
        "tool_result" => {
            let content = decode_content(map.remove("content").unwrap_or(Value::Null), cx)?;
            let call_id = ToolCallId::new(
                wire::take_string(&mut map, "tool_use_id")
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| synthesized_tool_call_id(0)),
            );
            let (call_id, _) = decode_call_id(call_id);
            Ok(ContentBlock::ToolResult {
                call_id,
                name: wire::take_string(&mut map, "name"),
                content,
                is_error: wire::take_bool(&mut map, "is_error").unwrap_or(false),
                extra: wire::extra_fields(map),
            })
        }
        "web_search_tool_result" => {
            let query = wire::take_string(&mut map, "query");
            let results = match map.remove("content") {
                Some(Value::Array(values)) => {
                    values.into_iter().map(decode_search_result).collect()
                }
                _ => Vec::new(),
            };
            Ok(ContentBlock::WebSearchResult {
                query,
                results,
                extra: wire::extra_fields(map),
            })
        }
        _ => {
            map.insert("type".to_owned(), Value::String(kind));
            Ok(ContentBlock::Native {
                protocol: Protocol::Messages,
                json: Value::Object(map),
            })
        }
    }
}

fn decode_search_result(value: Value) -> SearchResult {
    let Value::Object(mut map) = value else {
        return SearchResult {
            title: None,
            url: None,
            text: Some(value.to_string()),
            extra: None,
        };
    };
    let _kind = map.remove("type");
    SearchResult {
        title: wire::take_string(&mut map, "title"),
        url: wire::take_string(&mut map, "url"),
        text: wire::take_string(&mut map, "snippet")
            .or_else(|| wire::take_string(&mut map, "text")),
        extra: wire::extra_fields(map),
    }
}

fn decode_media_source(value: Value) -> MediaSource {
    let Value::Object(mut map) = value else {
        return MediaSource::Base64 {
            media_type: "application/octet-stream".to_owned(),
            data: String::new(),
        };
    };
    match wire::take_string(&mut map, "type").as_deref() {
        Some("url") => MediaSource::Url {
            url: wire::take_string(&mut map, "url").unwrap_or_default(),
            media_type: wire::take_string(&mut map, "media_type"),
        },
        Some("file") => MediaSource::File {
            file_id: wire::take_string(&mut map, "file_id").unwrap_or_default(),
            media_type: wire::take_string(&mut map, "media_type"),
        },
        _ => MediaSource::Base64 {
            media_type: wire::take_string(&mut map, "media_type")
                .unwrap_or_else(|| "application/octet-stream".to_owned()),
            data: wire::take_string(&mut map, "data").unwrap_or_default(),
        },
    }
}

pub(crate) fn encode_block(block: &ContentBlock, cx: &CodecContext) -> Result<Option<Value>> {
    match block {
        ContentBlock::Text { text, extra } => Ok(Some(with_extra(
            json!({"type":"text","text":text}),
            extra.as_ref(),
            cx,
            "content.text",
        )?)),
        ContentBlock::Image { source, extra } => Ok(Some(with_extra(
            json!({"type":"image","source":encode_media_source(source)}),
            extra.as_ref(),
            cx,
            "content.image",
        )?)),
        ContentBlock::File {
            source,
            filename,
            extra,
        } => {
            let mut value = json!({"type":"document","source":encode_media_source(source)});
            if let Some(filename) = filename {
                value["title"] = Value::String(filename.clone());
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "content.document",
            )?))
        }
        ContentBlock::Audio { .. } => fallback_block(block, cx),
        ContentBlock::Reasoning {
            text,
            summary,
            seal,
            redacted,
            extra,
        } => {
            if *redacted {
                let Some(seal) = seal else {
                    return fallback_block(block, cx);
                };
                return Ok(Some(with_extra(
                    json!({"type":"redacted_thinking","data":seal.value}),
                    extra.as_ref(),
                    cx,
                    "content.redacted_thinking",
                )?));
            }
            if let Some(seal) = seal
                && seal.protocol != Protocol::Messages
            {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    Protocol::Messages,
                    "content.thinking.signature",
                    "cross-protocol reasoning seal cannot be emitted as an Anthropic signature",
                )?;
            }
            let text = if text.is_empty() {
                summary.join("\n")
            } else {
                text.clone()
            };
            let mut value = json!({"type":"thinking","thinking":text});
            if let Some(seal) = seal
                .as_ref()
                .filter(|seal| seal.protocol == Protocol::Messages)
            {
                value["signature"] = Value::String(seal.value.clone());
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "content.thinking",
            )?))
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            seal,
            extra,
        } => {
            let input: Value = serde_json::from_str(arguments.as_str()).map_err(|error| {
                Error::InvalidRequest {
                    protocol: Protocol::Messages,
                    message: format!("tool call `{id}` arguments are not JSON: {error}"),
                }
            })?;
            encode_tool_use(id, name, input, seal.as_ref(), extra.as_ref(), cx).map(Some)
        }
        ContentBlock::ToolResult {
            call_id,
            name,
            content,
            is_error,
            extra,
        } => encode_tool_result(
            call_id,
            name.as_deref(),
            content,
            *is_error,
            extra.as_ref(),
            None,
            cx,
        )
        .map(Some),
        ContentBlock::WebSearchResult {
            query,
            results,
            extra,
        } => {
            let results: Vec<Value> = results
                .iter()
                .map(|result| {
                    let mut value = json!({"type":"web_search_result"});
                    if let Some(title) = &result.title {
                        value["title"] = Value::String(title.clone());
                    }
                    if let Some(url) = &result.url {
                        value["url"] = Value::String(url.clone());
                    }
                    if let Some(text) = &result.text {
                        value["snippet"] = Value::String(text.clone());
                    }
                    value
                })
                .collect();
            let mut value = json!({"type":"web_search_tool_result","content":results});
            if let Some(query) = query {
                value["query"] = Value::String(query.clone());
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "content.web_search_tool_result",
            )?))
        }
        ContentBlock::Refusal { text, extra } => {
            cx.diagnose(
                DiagnosticKind::Approximated,
                Protocol::Messages,
                "content.refusal",
                "refusal was represented as a text block",
            )?;
            Ok(Some(with_extra(
                json!({"type":"text","text":text}),
                extra.as_ref(),
                cx,
                "content.refusal",
            )?))
        }
        ContentBlock::Native { protocol, json } if *protocol == Protocol::Messages => {
            Ok(Some(json.clone()))
        }
        ContentBlock::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Messages,
                "content.native",
                "native block belongs to another protocol",
            )?;
            Ok(None)
        }
    }
}

pub(super) fn encode_tool_use(
    id: &ToolCallId,
    name: &str,
    input: Value,
    seal: Option<&ReasoningSeal>,
    extra: Option<&Extra>,
    cx: &CodecContext,
) -> Result<Value> {
    let id = encode_call_id(id, seal, cx)?;
    with_extra(
        json!({"type":"tool_use","id":id,"name":name,"input":input}),
        extra,
        cx,
        "content.tool_use",
    )
}

fn encode_tool_result(
    call_id: &ToolCallId,
    _name: Option<&str>,
    content: &[ContentBlock],
    is_error: bool,
    extra: Option<&Extra>,
    seal: Option<&ReasoningSeal>,
    cx: &CodecContext,
) -> Result<Value> {
    let mut encoded = Vec::new();
    for block in content {
        if let Some(block) = encode_block(block, cx)? {
            encoded.push(block);
        }
    }
    let call_id = encode_call_id(call_id, seal, cx)?;
    with_extra(
        json!({"type":"tool_result","tool_use_id":call_id,"content":encoded,"is_error":is_error}),
        extra,
        cx,
        "content.tool_result",
    )
}

fn fallback_block(block: &ContentBlock, cx: &CodecContext) -> Result<Option<Value>> {
    if !cx.options.content_fallbacks {
        cx.diagnose(
            DiagnosticKind::Dropped,
            Protocol::Messages,
            "content",
            format!("unsupported {} block was dropped", block.kind()),
        )?;
        return Ok(None);
    }
    cx.diagnose(
        DiagnosticKind::Fallback,
        Protocol::Messages,
        "content",
        format!("{} block became descriptive text", block.kind()),
    )?;
    Ok(Some(
        json!({"type":"text","text":format!("[{} content]", block.kind())}),
    ))
}

fn encode_media_source(source: &MediaSource) -> Value {
    match source {
        MediaSource::Url { url, media_type } => {
            let mut value = json!({"type":"url","url":url});
            if let Some(media_type) = media_type {
                value["media_type"] = Value::String(media_type.clone());
            }
            value
        }
        MediaSource::Base64 { media_type, data } => {
            json!({"type":"base64","media_type":media_type,"data":data})
        }
        MediaSource::File {
            file_id,
            media_type,
        } => {
            let mut value = json!({"type":"file","file_id":file_id});
            if let Some(media_type) = media_type {
                value["media_type"] = Value::String(media_type.clone());
            }
            value
        }
    }
}

fn encode_messages(req: &Request, cx: &CodecContext) -> Result<Vec<Value>> {
    let mut seals = HashMap::new();
    for message in &req.messages {
        for block in &message.content {
            if let ContentBlock::ToolCall {
                id,
                seal: Some(seal),
                ..
            } = block
            {
                seals.insert(id.clone(), seal.clone());
            }
        }
    }
    let mut messages: Vec<WireMessage> = Vec::new();
    for message in &req.messages {
        let role = match message.role {
            Role::Assistant => wire::WireRole::Assistant,
            Role::System | Role::Developer => {
                cx.diagnose(
                    DiagnosticKind::Approximated,
                    Protocol::Messages,
                    "messages.role",
                    "in-conversation system/developer content was encoded as a user message",
                )?;
                wire::WireRole::User
            }
            Role::User | Role::Tool => wire::WireRole::User,
        };
        let mut content = Vec::new();
        for block in &message.content {
            if let ContentBlock::ToolResult {
                call_id,
                name,
                content: result_content,
                is_error,
                extra,
            } = block
            {
                content.push(encode_tool_result(
                    call_id,
                    name.as_deref(),
                    result_content,
                    *is_error,
                    extra.as_ref(),
                    seals.get(call_id),
                    cx,
                )?);
            } else if let Some(block) = encode_block(block, cx)? {
                content.push(block);
            }
        }
        let extra = match &message.extra {
            Some(extra) if extra.protocol == Protocol::Messages => extra.fields.clone(),
            Some(_) => {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    Protocol::Messages,
                    "messages.extra",
                    "message extension belongs to another protocol",
                )?;
                Map::new()
            }
            None => Map::new(),
        };
        if matches!(role, wire::WireRole::User)
            && messages
                .last()
                .is_some_and(|last| matches!(last.role, wire::WireRole::User))
            && let Some(last) = messages.last_mut()
        {
            if let Some(array) = last.content.as_array_mut() {
                array.extend(content);
            }
            for (key, value) in extra {
                last.extra.entry(key).or_insert(value);
            }
            continue;
        }
        messages.push(WireMessage {
            role,
            content: Value::Array(content),
            extra,
        });
    }
    messages
        .into_iter()
        .map(|message| serde_json::to_value(message).map_err(wire::json_error))
        .collect()
}

fn decode_tool(value: Value) -> Tool {
    let Value::Object(mut map) = value else {
        return Tool::Hosted {
            protocol: Protocol::Messages,
            json: value,
        };
    };
    let kind = wire::take_string(&mut map, "type");
    let is_function =
        kind.as_deref().is_none_or(|kind| kind == "custom") && map.contains_key("input_schema");
    if is_function {
        Tool::Function {
            name: wire::take_string(&mut map, "name").unwrap_or_default(),
            description: wire::take_string(&mut map, "description"),
            parameters: map.remove("input_schema").unwrap_or_else(|| json!({})),
            strict: wire::take_bool(&mut map, "strict"),
            defer_loading: wire::take_bool(&mut map, "defer_loading"),
            extra: wire::extra_fields(map),
        }
    } else {
        if let Some(kind) = kind {
            map.insert("type".to_owned(), Value::String(kind));
        }
        Tool::Hosted {
            protocol: Protocol::Messages,
            json: Value::Object(map),
        }
    }
}

fn encode_tool(tool: &Tool, cx: &CodecContext) -> Result<Option<Value>> {
    match tool {
        Tool::Function {
            name,
            description,
            parameters,
            strict,
            defer_loading,
            extra,
        } => {
            let mut value = json!({"name":name,"input_schema":parameters});
            if let Some(description) = description {
                value["description"] = Value::String(description.clone());
            }
            if let Some(strict) = strict {
                value["strict"] = Value::Bool(*strict);
            }
            if let Some(defer_loading) = defer_loading {
                value["defer_loading"] = Value::Bool(*defer_loading);
            }
            Ok(Some(with_extra(value, extra.as_ref(), cx, "tools")?))
        }
        Tool::Custom {
            name, description, ..
        } => {
            cx.diagnose(
                DiagnosticKind::Fallback,
                Protocol::Messages,
                "tools.custom",
                "free-form tool became a function taking one input string",
            )?;
            Ok(Some(
                json!({"type":"custom","name":name,"description":description,"input_schema":{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}}),
            ))
        }
        Tool::Hosted { protocol, json } if *protocol == Protocol::Messages => {
            Ok(Some(json.clone()))
        }
        Tool::Hosted { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Messages,
                "tools.hosted",
                "hosted tool belongs to another protocol",
            )?;
            Ok(None)
        }
    }
}

fn decode_tool_choice(value: Value) -> (Option<ToolChoice>, Option<bool>) {
    let Value::Object(mut map) = value else {
        return (None, None);
    };
    let kind = wire::take_string(&mut map, "type").unwrap_or_default();
    let parallel = wire::take_bool(&mut map, "disable_parallel_tool_use").map(|value| !value);
    let choice = match kind.as_str() {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "any" => ToolChoice::Required,
        "tool" => ToolChoice::Named(wire::take_string(&mut map, "name").unwrap_or_default()),
        _ => {
            map.insert("type".to_owned(), Value::String(kind));
            ToolChoice::Native {
                protocol: Protocol::Messages,
                json: Value::Object(map),
            }
        }
    };
    (Some(choice), parallel)
}

fn encode_tool_choice(
    choice: Option<&ToolChoice>,
    parallel: Option<bool>,
    cx: &CodecContext,
) -> Result<Option<Value>> {
    let mut value = match choice {
        Some(ToolChoice::Auto) => json!({"type":"auto"}),
        Some(ToolChoice::None) => json!({"type":"none"}),
        Some(ToolChoice::Required) => json!({"type":"any"}),
        Some(ToolChoice::Named(name)) => json!({"type":"tool","name":name}),
        Some(ToolChoice::Native { protocol, json }) if *protocol == Protocol::Messages => {
            json.clone()
        }
        Some(ToolChoice::Native { .. }) => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Messages,
                "tool_choice",
                "native tool choice belongs to another protocol",
            )?;
            json!({"type":"auto"})
        }
        None if parallel.is_some() => json!({"type":"auto"}),
        None => return Ok(None),
    };
    if let Some(parallel) = parallel {
        value["disable_parallel_tool_use"] = Value::Bool(!parallel);
    }
    Ok(Some(value))
}

fn decode_reasoning(value: Option<Value>) -> (Option<ReasoningParams>, Option<Map<String, Value>>) {
    let Some(Value::Object(mut map)) = value else {
        return (None, None);
    };
    let budget = map
        .remove("budget_tokens")
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok());
    (
        Some(ReasoningParams {
            effort: None,
            budget_tokens: budget,
            summary: None,
            extra: wire::extra_fields(map),
        }),
        None,
    )
}

fn encode_reasoning(
    reasoning: &ReasoningParams,
    max_tokens: u32,
    cx: &CodecContext,
) -> Result<Value> {
    let mut map = reasoning
        .extra
        .as_ref()
        .filter(|extra| extra.protocol == Protocol::Messages)
        .map_or_else(Map::new, |extra| extra.fields.clone());
    if let Some(extra) = reasoning
        .extra
        .as_ref()
        .filter(|extra| extra.protocol != Protocol::Messages)
        && !extra.fields.is_empty()
    {
        cx.diagnose(
            DiagnosticKind::Dropped,
            Protocol::Messages,
            "thinking",
            "reasoning extension belongs to another protocol",
        )?;
    }
    if let Some(mut budget) = reasoning.budget_tokens {
        let cap = max_tokens.saturating_sub(1024);
        if budget > cap {
            cx.diagnose(
                DiagnosticKind::Approximated,
                Protocol::Messages,
                "thinking.budget_tokens",
                format!("thinking budget {budget} was capped at {cap}"),
            )?;
            budget = cap;
        }
        map.insert("type".to_owned(), Value::String("enabled".to_owned()));
        map.insert("budget_tokens".to_owned(), json!(budget));
    } else {
        map.entry("type".to_owned())
            .or_insert_with(|| Value::String("adaptive".to_owned()));
    }
    Ok(Value::Object(map))
}

fn decode_output_config(
    value: Option<Value>,
) -> (Option<OutputFormat>, Option<String>, Map<String, Value>) {
    let Some(Value::Object(mut map)) = value else {
        return (None, None, Map::new());
    };
    let effort = wire::take_string(&mut map, "effort");
    let format = map.remove("format").and_then(decode_output_format);
    (format, effort, map)
}

fn decode_output_format(value: Value) -> Option<OutputFormat> {
    let Value::Object(mut map) = value else {
        return None;
    };
    let kind = match wire::take_string(&mut map, "type").as_deref() {
        Some("json_schema") => OutputFormatKind::JsonSchema,
        Some("json_object") => OutputFormatKind::JsonObject,
        _ => OutputFormatKind::Text,
    };
    Some(OutputFormat {
        kind,
        schema: map.remove("schema"),
        name: wire::take_string(&mut map, "name"),
        strict: wire::take_bool(&mut map, "strict"),
        extra: wire::extra_fields(map),
    })
}

fn encode_output_format(format: &OutputFormat) -> Value {
    let mut map = format
        .extra
        .as_ref()
        .filter(|extra| extra.protocol == Protocol::Messages)
        .map_or_else(Map::new, |extra| extra.fields.clone());
    map.insert(
        "type".to_owned(),
        Value::String(
            match format.kind {
                OutputFormatKind::Text => "text",
                OutputFormatKind::JsonObject => "json_object",
                OutputFormatKind::JsonSchema => "json_schema",
            }
            .to_owned(),
        ),
    );
    if let Some(schema) = &format.schema {
        map.insert("schema".to_owned(), schema.clone());
    }
    if let Some(name) = &format.name {
        map.insert("name".to_owned(), Value::String(name.clone()));
    }
    if let Some(strict) = format.strict {
        map.insert("strict".to_owned(), Value::Bool(strict));
    }
    Value::Object(map)
}

pub(crate) fn decode_usage(usage: WireUsage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cached_input_tokens: usage.cache_read_input_tokens.unwrap_or(0),
        cache_creation_input_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
        reasoning_tokens: 0,
        extra: wire::extra_fields(usage.extra),
    }
}

pub(crate) fn encode_usage(usage: &Usage) -> Value {
    let mut map = usage
        .extra
        .as_ref()
        .filter(|extra| extra.protocol == Protocol::Messages)
        .map_or_else(Map::new, |extra| extra.fields.clone());
    map.insert("input_tokens".to_owned(), json!(usage.input_tokens));
    map.insert("output_tokens".to_owned(), json!(usage.output_tokens));
    if usage.cached_input_tokens != 0 {
        map.insert(
            "cache_read_input_tokens".to_owned(),
            json!(usage.cached_input_tokens),
        );
    }
    if usage.cache_creation_input_tokens != 0 {
        map.insert(
            "cache_creation_input_tokens".to_owned(),
            json!(usage.cache_creation_input_tokens),
        );
    }
    Value::Object(map)
}

fn request_extra(extra: Option<&Extra>, cx: &CodecContext) -> Result<Map<String, Value>> {
    extra_map(extra, cx, "request.extra")
}

fn response_extra(extra: Option<&Extra>, cx: &CodecContext) -> Result<Map<String, Value>> {
    extra_map(extra, cx, "response.extra")
}

fn extra_map(extra: Option<&Extra>, cx: &CodecContext, path: &str) -> Result<Map<String, Value>> {
    Extra::fields_for(extra, Protocol::Messages, cx, path)
}

pub(crate) fn with_extra(
    value: Value,
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Value> {
    Extra::merge_into(value, extra, Protocol::Messages, cx, path)
}

fn fallback_error_message(status: StatusCode) -> String {
    format!("upstream returned HTTP {}", status.as_u16())
}

fn error_kind(status: StatusCode, code: Option<&str>) -> ApiErrorKind {
    match code {
        Some("authentication_error") => ApiErrorKind::Authentication,
        Some("permission_error") => ApiErrorKind::Permission,
        Some("not_found_error") => ApiErrorKind::NotFound,
        Some("rate_limit_error") => ApiErrorKind::RateLimit,
        Some("overloaded_error") => ApiErrorKind::Overloaded,
        _ => ApiErrorKind::from_status(status),
    }
}

fn encode_call_id(
    id: &ToolCallId,
    seal: Option<&ReasoningSeal>,
    cx: &CodecContext,
) -> Result<String> {
    let valid = !id.as_str().is_empty()
        && id
            .as_str()
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');
    let seal = seal.filter(|seal| seal.protocol != Protocol::Messages);
    if valid && seal.is_none() {
        return Ok(id.as_str().to_owned());
    }
    let seal = match seal {
        Some(seal) if cx.options.seal_reasoning_in_tool_ids => Some(seal),
        Some(_) => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Messages,
                "content.tool_use.id",
                "reasoning seal could not be sealed into the tool-call ID",
            )?;
            None
        }
        None => None,
    };
    let payload = json!({"v":1,"id":id.as_str(),"seal":seal});
    let encoded = format!("mp_{}", URL_SAFE_NO_PAD.encode(payload.to_string()));
    if encoded.len() > cx.options.max_sealed_id_bytes {
        cx.diagnose(
            DiagnosticKind::Dropped,
            Protocol::Messages,
            "content.tool_use.id",
            "sealed tool-call ID exceeded max_sealed_id_bytes",
        )?;
        if valid {
            return Ok(id.as_str().to_owned());
        }
        let id_only = json!({"v":1,"id":id.as_str()});
        return Ok(format!(
            "mp_{}",
            URL_SAFE_NO_PAD.encode(id_only.to_string())
        ));
    }
    Ok(encoded)
}

fn decode_call_id(id: ToolCallId) -> (ToolCallId, Option<ReasoningSeal>) {
    let Some(encoded) = id.as_str().strip_prefix("mp_") else {
        return (id, None);
    };
    let Ok(bytes) = URL_SAFE_NO_PAD.decode(encoded) else {
        return (id, None);
    };
    let Ok(Value::Object(mut map)) = serde_json::from_slice::<Value>(&bytes) else {
        return (id, None);
    };
    if map.remove("v").and_then(|value| value.as_u64()) != Some(1) {
        return (id, None);
    }
    let Some(original) = wire::take_string(&mut map, "id") else {
        return (id, None);
    };
    let seal = map
        .remove("seal")
        .and_then(|value| serde_json::from_value::<ReasoningSeal>(value).ok());
    (ToolCallId::new(original), seal)
}
