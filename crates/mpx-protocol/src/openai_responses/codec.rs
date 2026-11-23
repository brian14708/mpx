use bytes::Bytes;
use http::{StatusCode, header};
use serde_json::{Map, Value, json};

use crate::json::{insert_option, take_object};
use crate::{
    ApiError, ApiErrorKind, ClientContext, CodecContext, ContentBlock, Continuation,
    DiagnosticKind, Endpoint, Error, Extra, MediaSource, Message, OutputFormat, OutputFormatKind,
    Protocol, ProtocolCodec, ReasoningParams, ReasoningSeal, Request, Response, ResponseId, Result,
    Role, SamplingParams, SearchResult, StopReason, StreamDecoder, StreamEncoder, Tool,
    ToolArguments, ToolCallId, ToolChoice, Usage,
    codec::json_response,
    openai_responses::{
        stream::{ResponsesStreamDecoder, ResponsesStreamEncoder},
        wire::{self, ResponsesRequest, ResponsesUsage},
    },
    request::{session_id_field, take_session_id},
    response::normalize_tool_stop_reason,
    tool::synthesized_tool_call_id,
};

/// Sans-IO codec for `OpenAI` Responses.
#[derive(Debug, Clone)]
pub(crate) struct ResponsesCodec;

impl ProtocolCodec for ResponsesCodec {
    fn protocol(&self) -> Protocol {
        Protocol::Responses
    }
    fn decode_request(
        &self,
        _endpoint: &Endpoint,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Request> {
        let wire: ResponsesRequest = serde_json::from_slice(body).map_err(wire::json_error)?;
        let mut instructions = match wire.instructions {
            Some(value) => decode_instruction_content(value, cx)?,
            None => Vec::new(),
        };
        let mut messages = match wire.input {
            Value::String(text) => vec![Message::text(Role::User, text)],
            Value::Array(items) => decode_input_items(items, cx)?,
            Value::Null => Vec::new(),
            value => vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Native {
                    protocol: Protocol::Responses,
                    json: value,
                }],
                extra: None,
            }],
        };
        while messages
            .first()
            .is_some_and(|message| matches!(message.role, Role::System | Role::Developer))
        {
            if let Some(message) = messages.first().cloned() {
                instructions.extend(message.content);
                messages.remove(0);
            }
        }
        let tools = wire
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(decode_tool)
            .collect();
        let reasoning = wire.reasoning.and_then(decode_reasoning);
        let (output_format, text_extra) = wire.text.map_or_else(|| (None, Map::new()), decode_text);
        let mut extra = wire.extra;
        if wire.previous_response_id.is_none() {
            insert_option(&mut extra, "store", wire.store.map(Value::Bool));
        }
        if !text_extra.is_empty() {
            extra.insert("text".to_owned(), Value::Object(text_extra));
        }
        let mut metadata = wire.metadata.unwrap_or_default();
        let session_id = take_session_id(&mut metadata).or_else(|| {
            extra
                .get("client_metadata")
                .and_then(Value::as_object)
                .and_then(session_id_field)
        });
        Ok(Request {
            model: wire.model,
            session_id,
            instructions,
            messages,
            tools,
            tool_choice: wire.tool_choice.map(decode_tool_choice),
            sampling: SamplingParams {
                max_output_tokens: wire.max_output_tokens,
                temperature: wire.temperature,
                top_p: wire.top_p,
                ..SamplingParams::default()
            },
            reasoning,
            output_format,
            parallel_tool_calls: wire.parallel_tool_calls,
            stream: wire.stream.unwrap_or(false),
            metadata,
            continuation: wire.previous_response_id.map(|id| Continuation {
                previous_response_id: ResponseId::new(id),
                store: wire.store,
            }),
            extra: wire::extra_fields(extra),
            client_include_usage: false,
            messages_max_tokens: false,
        })
    }

    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>> {
        let mut object = extra_map(req.extra.as_ref(), cx, "request.extra")?;
        object.insert("model".to_owned(), Value::String(req.model.clone()));
        object.insert("stream".to_owned(), Value::Bool(req.stream));
        if !req.instructions.is_empty() {
            object.insert(
                "instructions".to_owned(),
                Value::String(blocks_text(&req.instructions)),
            );
        }
        let mut input = Vec::new();
        for message in &req.messages {
            input.extend(encode_message_items(message, cx)?);
        }
        object.insert("input".to_owned(), Value::Array(input));
        if !req.tools.is_empty() {
            let mut tools = Vec::new();
            for tool in &req.tools {
                if let Some(tool) = encode_tool(tool, cx)? {
                    tools.push(tool);
                }
            }
            object.insert("tools".to_owned(), Value::Array(tools));
        }
        if let Some(choice) = &req.tool_choice {
            object.insert("tool_choice".to_owned(), encode_tool_choice(choice, cx)?);
        }
        diagnose_unsupported_sampling(req, cx)?;
        insert_option(
            &mut object,
            "max_output_tokens",
            req.sampling.max_output_tokens.map(Value::from),
        );
        insert_option(
            &mut object,
            "temperature",
            req.sampling.temperature.map(Value::from),
        );
        insert_option(&mut object, "top_p", req.sampling.top_p.map(Value::from));
        insert_option(
            &mut object,
            "parallel_tool_calls",
            req.parallel_tool_calls.map(Value::Bool),
        );
        insert_option(
            &mut object,
            "metadata",
            req.openai_metadata().map(Value::Object),
        );
        if let Some(continuation) = &req.continuation {
            object.insert(
                "previous_response_id".to_owned(),
                Value::String(continuation.previous_response_id.as_str().to_owned()),
            );
            insert_option(&mut object, "store", continuation.store.map(Value::Bool));
        }
        if let Some(reasoning) = &req.reasoning {
            object.insert("reasoning".to_owned(), encode_reasoning(reasoning, cx)?);
        }
        if let Some(format) = &req.output_format {
            let mut text = take_object(&mut object, "text");
            text.insert("format".to_owned(), encode_output_format(format));
            object.insert("text".to_owned(), Value::Object(text));
        }
        let accept = if req.stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        http::Request::builder()
            .method(http::Method::POST)
            .uri("/v1/responses")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, accept)
            .body(Bytes::from(Value::Object(object).to_string()))
            .map_err(Error::http)
    }

    fn decode_response(
        &self,
        _parts: &http::response::Parts,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Response> {
        let Value::Object(mut map) =
            serde_json::from_slice::<Value>(body).map_err(wire::json_error)?
        else {
            return Err(Error::Protocol {
                protocol: Protocol::Responses,
                message: "response must be a JSON object".to_owned(),
            });
        };
        if map.get("status").and_then(Value::as_str) == Some("failed") {
            let message = map
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Responses response.failed")
                .to_owned();
            return Err(Error::Protocol {
                protocol: Protocol::Responses,
                message,
            });
        }
        let id = wire::take_string(&mut map, "id").unwrap_or_default();
        let model = wire::take_string(&mut map, "model").unwrap_or_default();
        let status = wire::take_string(&mut map, "status");
        let incomplete_reason = map
            .get("incomplete_details")
            .and_then(|value| value.get("reason"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let output = map
            .remove("output")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        let mut content = Vec::new();
        for item in output {
            content.extend(decode_item(item, cx)?.1);
        }
        let usage = map
            .remove("usage")
            .map(|value| serde_json::from_value::<ResponsesUsage>(value).map_err(wire::json_error))
            .transpose()?
            .map(decode_usage)
            .unwrap_or_default();
        let mut stop_reason = if status.as_deref() == Some("incomplete") {
            if incomplete_reason.as_deref() == Some("max_output_tokens") {
                StopReason::MaxTokens
            } else {
                StopReason::Incomplete
            }
        } else {
            StopReason::EndTurn
        };
        stop_reason = normalize_tool_stop_reason(
            stop_reason,
            content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall { .. })),
        );
        Ok(Response {
            id,
            model,
            content,
            stop_reason,
            usage,
            safety: Vec::new(),
            extra: wire::extra_fields(map),
        })
    }

    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError {
        decode_openai_error(parts.status, body)
    }

    fn encode_response(
        &self,
        response: &Response,
        client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>> {
        let object = response_object(response, client, cx)?;
        Ok(json_response(
            StatusCode::OK,
            Bytes::from(Value::Object(object).to_string()),
        ))
    }

    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes> {
        json_response(
            error.status(),
            Bytes::from(json!({"error":{"message":error.message,"type":error.kind.as_str(),"param":null,"code":error.code}}).to_string()),
        )
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(ResponsesStreamDecoder::new())
    }
    fn stream_encoder(&self, client: &ClientContext) -> Box<dyn StreamEncoder> {
        Box::new(ResponsesStreamEncoder::new(client.clone()))
    }
}

#[derive(Debug)]
pub(crate) struct DecodedItem(
    pub(crate) Role,
    pub(crate) Vec<ContentBlock>,
    pub(crate) Option<Extra>,
);

fn decode_input_items(items: Vec<Value>, cx: &CodecContext) -> Result<Vec<Message>> {
    let mut messages: Vec<Message> = Vec::new();
    for item in items {
        let DecodedItem(role, blocks, extra) = decode_item(item, cx)?;
        if matches!(role, Role::Assistant | Role::Tool)
            && messages.last().is_some_and(|last| last.role == role)
            && let Some(last) = messages.last_mut()
        {
            last.content.extend(blocks);
            continue;
        }
        messages.push(Message {
            role,
            content: blocks,
            extra,
        });
    }
    Ok(messages)
}

fn decode_instruction_content(value: Value, cx: &CodecContext) -> Result<Vec<ContentBlock>> {
    match value {
        Value::String(text) => Ok(vec![ContentBlock::text(text)]),
        Value::Array(parts) => parts
            .into_iter()
            .map(|part| decode_content_part(part, cx))
            .collect(),
        other => Ok(vec![ContentBlock::Native {
            protocol: Protocol::Responses,
            json: other,
        }]),
    }
}

pub(crate) fn decode_item(value: Value, cx: &CodecContext) -> Result<DecodedItem> {
    let Value::Object(mut map) = value else {
        return Ok(DecodedItem(
            Role::Assistant,
            vec![ContentBlock::Native {
                protocol: Protocol::Responses,
                json: value,
            }],
            None,
        ));
    };
    let kind = wire::take_string(&mut map, "type").unwrap_or_else(|| "message".to_owned());
    match kind.as_str() {
        "message" => {
            let role = match wire::take_string(&mut map, "role").as_deref() {
                Some("system") => Role::System,
                Some("developer") => Role::Developer,
                Some("assistant") => Role::Assistant,
                _ => Role::User,
            };
            let content = match map.remove("content") {
                Some(Value::String(text)) => vec![ContentBlock::text(text)],
                Some(Value::Array(parts)) => parts
                    .into_iter()
                    .map(|part| decode_content_part(part, cx))
                    .collect::<Result<Vec<_>>>()?,
                _ => Vec::new(),
            };
            Ok(DecodedItem(role, content, wire::extra_fields(map)))
        }
        "function_call" | "custom_tool_call" => {
            let call_id = wire::take_string(&mut map, "call_id")
                .or_else(|| wire::take_string(&mut map, "id"))
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| synthesized_tool_call_id(0));
            let arguments = if kind == "custom_tool_call" {
                wire::take_string(&mut map, "input").unwrap_or_default()
            } else {
                wire::take_string(&mut map, "arguments").unwrap_or_default()
            };
            let mut extra = map;
            if kind == "custom_tool_call" {
                extra.insert("type".to_owned(), Value::String(kind));
            }
            Ok(DecodedItem(
                Role::Assistant,
                vec![ContentBlock::ToolCall {
                    id: ToolCallId::new(call_id),
                    name: wire::take_string(&mut extra, "name").unwrap_or_default(),
                    arguments: ToolArguments::new(arguments),
                    seal: None,
                    extra: wire::extra_fields(extra),
                }],
                None,
            ))
        }
        "function_call_output" => {
            let call_id = wire::take_string(&mut map, "call_id")
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| synthesized_tool_call_id(0));
            let output = map.remove("output").unwrap_or(Value::Null);
            let content = match output {
                Value::String(text) => vec![ContentBlock::text(text)],
                Value::Array(parts) => parts
                    .into_iter()
                    .map(|part| decode_content_part(part, cx))
                    .collect::<Result<Vec<_>>>()?,
                other => vec![ContentBlock::Native {
                    protocol: Protocol::Responses,
                    json: other,
                }],
            };
            Ok(DecodedItem(
                Role::Tool,
                vec![ContentBlock::ToolResult {
                    call_id: ToolCallId::new(call_id),
                    name: wire::take_string(&mut map, "name"),
                    content,
                    is_error: wire::take_bool(&mut map, "is_error").unwrap_or(false),
                    extra: wire::extra_fields(map),
                }],
                None,
            ))
        }
        "reasoning" => {
            let summary = map
                .remove("summary")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default()
                .into_iter()
                .filter_map(|value| value.get("text").and_then(Value::as_str).map(str::to_owned))
                .collect();
            let text = map
                .remove("content")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default()
                .into_iter()
                .filter_map(|value| value.get("text").and_then(Value::as_str).map(str::to_owned))
                .collect::<String>();
            let seal = wire::take_string(&mut map, "encrypted_content")
                .map(|value| ReasoningSeal::new(Protocol::Responses, value));
            Ok(DecodedItem(
                Role::Assistant,
                vec![ContentBlock::Reasoning {
                    text,
                    summary,
                    seal,
                    redacted: false,
                    extra: wire::extra_fields(map),
                }],
                None,
            ))
        }
        "web_search_call" => {
            let action = map.remove("action").unwrap_or(Value::Null);
            let query = action
                .get("query")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let results = action
                .get("sources")
                .and_then(Value::as_array)
                .map(|sources| {
                    sources
                        .iter()
                        .map(|source| SearchResult {
                            title: source
                                .get("title")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            url: source.get("url").and_then(Value::as_str).map(str::to_owned),
                            text: None,
                            extra: None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(DecodedItem(
                Role::Assistant,
                vec![ContentBlock::WebSearchResult {
                    query,
                    results,
                    extra: wire::extra_fields(map),
                }],
                None,
            ))
        }
        _ => {
            map.insert("type".to_owned(), Value::String(kind));
            Ok(DecodedItem(
                Role::Assistant,
                vec![ContentBlock::Native {
                    protocol: Protocol::Responses,
                    json: Value::Object(map),
                }],
                None,
            ))
        }
    }
}

pub(crate) fn decode_content_part(value: Value, _cx: &CodecContext) -> Result<ContentBlock> {
    let Value::Object(mut map) = value else {
        return Ok(ContentBlock::Native {
            protocol: Protocol::Responses,
            json: value,
        });
    };
    let kind = wire::take_string(&mut map, "type").unwrap_or_default();
    match kind.as_str() {
        "input_text" | "output_text" | "text" => Ok(ContentBlock::Text {
            text: wire::take_string(&mut map, "text").unwrap_or_default(),
            extra: wire::extra_fields(map),
        }),
        "refusal" => Ok(ContentBlock::Refusal {
            text: wire::take_string(&mut map, "refusal")
                .or_else(|| wire::take_string(&mut map, "text"))
                .unwrap_or_default(),
            extra: wire::extra_fields(map),
        }),
        "input_image" | "image" => {
            let source = if let Some(id) = wire::take_string(&mut map, "file_id") {
                MediaSource::File {
                    file_id: id,
                    media_type: None,
                }
            } else {
                let url = wire::take_string(&mut map, "image_url")
                    .or_else(|| wire::take_string(&mut map, "url"))
                    .unwrap_or_default();
                decode_url_source(url)
            };
            Ok(ContentBlock::Image {
                source,
                extra: wire::extra_fields(map),
            })
        }
        "input_file" | "file" => {
            let filename = wire::take_string(&mut map, "filename");
            let source = if let Some(id) = wire::take_string(&mut map, "file_id") {
                MediaSource::File {
                    file_id: id,
                    media_type: None,
                }
            } else if let Some(data) = wire::take_string(&mut map, "file_data") {
                MediaSource::Base64 {
                    media_type: "application/octet-stream".to_owned(),
                    data,
                }
            } else {
                MediaSource::Base64 {
                    media_type: "application/octet-stream".to_owned(),
                    data: String::new(),
                }
            };
            Ok(ContentBlock::File {
                source,
                filename,
                extra: wire::extra_fields(map),
            })
        }
        "input_audio" | "audio" => {
            let data = wire::take_string(&mut map, "data").unwrap_or_default();
            let format = wire::take_string(&mut map, "format").unwrap_or_else(|| "wav".to_owned());
            Ok(ContentBlock::Audio {
                source: MediaSource::Base64 {
                    media_type: format!("audio/{format}"),
                    data,
                },
                extra: wire::extra_fields(map),
            })
        }
        _ => {
            map.insert("type".to_owned(), Value::String(kind));
            Ok(ContentBlock::Native {
                protocol: Protocol::Responses,
                json: Value::Object(map),
            })
        }
    }
}

fn decode_url_source(url: String) -> MediaSource {
    if let Some(rest) = url.strip_prefix("data:")
        && let Some((media_type, data)) = rest.split_once(";base64,")
    {
        return MediaSource::Base64 {
            media_type: media_type.to_owned(),
            data: data.to_owned(),
        };
    }
    MediaSource::Url {
        url,
        media_type: None,
    }
}

pub(crate) fn encode_message_items(message: &Message, cx: &CodecContext) -> Result<Vec<Value>> {
    let mut items = Vec::new();
    let mut ordinary = Vec::new();
    let flush =
        |items: &mut Vec<Value>, ordinary: &mut Vec<Value>, cx: &CodecContext| -> Result<()> {
            if !ordinary.is_empty() {
                let role = match message.role {
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Developer => "developer",
                    Role::User | Role::Tool => "user",
                };
                let mut value =
                    json!({"type":"message","role":role,"content":std::mem::take(ordinary)});
                if let Some(extra) = &message.extra {
                    value = with_extra(value, Some(extra), cx, "input.message")?;
                }
                items.push(value);
            }
            Ok(())
        };
    for block in &message.content {
        match block {
            ContentBlock::ToolCall { .. }
            | ContentBlock::Reasoning { .. }
            | ContentBlock::WebSearchResult { .. }
            | ContentBlock::Native { .. } => {
                flush(&mut items, &mut ordinary, cx)?;
                if let Some(item) = encode_standalone_item(block, cx)? {
                    items.push(item);
                }
            }
            ContentBlock::ToolResult {
                call_id,
                content,
                is_error,
                extra,
                ..
            } => {
                flush(&mut items, &mut ordinary, cx)?;
                let output = if content.len() == 1 {
                    match &content[0] {
                        ContentBlock::Text { text, .. } => Value::String(text.clone()),
                        _ => Value::Array(encode_content_parts(content, cx)?),
                    }
                } else {
                    Value::Array(encode_content_parts(content, cx)?)
                };
                let mut value = json!({"type":"function_call_output","call_id":call_id.as_str(),"output":output});
                if *is_error {
                    value["is_error"] = Value::Bool(true);
                }
                items.push(with_extra(
                    value,
                    extra.as_ref(),
                    cx,
                    "input.function_call_output",
                )?);
            }
            _ => {
                if let Some(part) = encode_content_part(block, cx)? {
                    ordinary.push(part);
                }
            }
        }
    }
    flush(&mut items, &mut ordinary, cx)?;
    Ok(items)
}

fn encode_content_parts(blocks: &[ContentBlock], cx: &CodecContext) -> Result<Vec<Value>> {
    let mut output = Vec::new();
    for block in blocks {
        if let Some(part) = encode_content_part(block, cx)? {
            output.push(part);
        }
    }
    Ok(output)
}

fn encode_content_part(block: &ContentBlock, cx: &CodecContext) -> Result<Option<Value>> {
    match block {
        ContentBlock::Text { text, extra } => Ok(Some(with_extra(
            json!({"type":"input_text","text":text}),
            extra.as_ref(),
            cx,
            "content.text",
        )?)),
        ContentBlock::Refusal { text, extra } => Ok(Some(with_extra(
            json!({"type":"refusal","refusal":text}),
            extra.as_ref(),
            cx,
            "content.refusal",
        )?)),
        ContentBlock::Image { source, extra } => {
            let mut value = json!({"type":"input_image"});
            match source {
                MediaSource::Url { url, .. } => value["image_url"] = Value::String(url.clone()),
                MediaSource::Base64 { media_type, data } => {
                    value["image_url"] = Value::String(format!("data:{media_type};base64,{data}"));
                }
                MediaSource::File { file_id, .. } => {
                    value["file_id"] = Value::String(file_id.clone());
                }
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "content.image",
            )?))
        }
        ContentBlock::File {
            source,
            filename,
            extra,
        } => {
            let mut value = json!({"type":"input_file"});
            match source {
                MediaSource::Url { url, .. } => value["file_url"] = Value::String(url.clone()),
                MediaSource::Base64 { data, .. } => {
                    value["file_data"] = Value::String(data.clone());
                }
                MediaSource::File { file_id, .. } => {
                    value["file_id"] = Value::String(file_id.clone());
                }
            }
            if let Some(filename) = filename {
                value["filename"] = Value::String(filename.clone());
            }
            Ok(Some(with_extra(value, extra.as_ref(), cx, "content.file")?))
        }
        ContentBlock::Audio { source, extra } => {
            let MediaSource::Base64 { media_type, data } = source else {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    Protocol::Responses,
                    "content.audio",
                    "only inline Responses audio can be encoded",
                )?;
                return Ok(None);
            };
            let format = media_type.split('/').next_back().unwrap_or("wav");
            Ok(Some(with_extra(
                json!({"type":"input_audio","data":data,"format":format}),
                extra.as_ref(),
                cx,
                "content.audio",
            )?))
        }
        ContentBlock::Native { protocol, json } if *protocol == Protocol::Responses => {
            Ok(Some(json.clone()))
        }
        ContentBlock::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                "content.native",
                "native content belongs to another protocol",
            )?;
            Ok(None)
        }
        _ => Ok(None),
    }
}

pub(crate) fn encode_standalone_item(
    block: &ContentBlock,
    cx: &CodecContext,
) -> Result<Option<Value>> {
    match block {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            extra,
            ..
        } => {
            let custom = extra.as_ref().is_some_and(|extra| {
                extra.protocol == Protocol::Responses
                    && extra.fields.get("type").and_then(Value::as_str) == Some("custom_tool_call")
            });
            let value = if custom {
                json!({"type":"custom_tool_call","call_id":id.as_str(),"name":name,"input":arguments.as_str()})
            } else {
                json!({"type":"function_call","call_id":id.as_str(),"name":name,"arguments":arguments.as_str()})
            };
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "output.function_call",
            )?))
        }
        ContentBlock::Reasoning {
            text,
            summary,
            seal,
            extra,
            ..
        } => {
            if let Some(seal) = seal
                .as_ref()
                .filter(|seal| seal.protocol != Protocol::Responses)
            {
                let _ = seal;
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    Protocol::Responses,
                    "output.reasoning.encrypted_content",
                    "reasoning seal belongs to another protocol",
                )?;
            }
            let mut value = json!({"type":"reasoning","summary":summary.iter().map(|text| json!({"type":"summary_text","text":text})).collect::<Vec<_>>()});
            if !text.is_empty() {
                value["content"] = json!([{"type":"reasoning_text","text":text}]);
            }
            if let Some(seal) = seal
                .as_ref()
                .filter(|seal| seal.protocol == Protocol::Responses)
            {
                value["encrypted_content"] = Value::String(seal.value.clone());
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "output.reasoning",
            )?))
        }
        ContentBlock::WebSearchResult {
            query,
            results,
            extra,
        } => {
            let sources = results
                .iter()
                .map(|result| json!({"title":result.title,"url":result.url}))
                .collect::<Vec<_>>();
            Ok(Some(with_extra(
                json!({"type":"web_search_call","status":"completed","action":{"type":"search","query":query,"sources":sources}}),
                extra.as_ref(),
                cx,
                "output.web_search_call",
            )?))
        }
        ContentBlock::Native { protocol, json } if *protocol == Protocol::Responses => {
            Ok(Some(json.clone()))
        }
        ContentBlock::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                "output.native",
                "native item belongs to another protocol",
            )?;
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn decode_tool(value: Value) -> Tool {
    let Value::Object(mut map) = value else {
        return Tool::Hosted {
            protocol: Protocol::Responses,
            json: value,
        };
    };
    match wire::take_string(&mut map, "type").as_deref() {
        Some("function") => Tool::Function {
            name: wire::take_string(&mut map, "name").unwrap_or_default(),
            description: wire::take_string(&mut map, "description"),
            parameters: map.remove("parameters").unwrap_or_else(|| json!({})),
            strict: wire::take_bool(&mut map, "strict"),
            defer_loading: wire::take_bool(&mut map, "defer_loading"),
            extra: wire::extra_fields(map),
        },
        Some("custom") => Tool::Custom {
            name: wire::take_string(&mut map, "name").unwrap_or_default(),
            description: wire::take_string(&mut map, "description"),
            format: map.remove("format"),
            extra: wire::extra_fields(map),
        },
        _ => Tool::Hosted {
            protocol: Protocol::Responses,
            json: Value::Object(map),
        },
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
            let mut value = json!({"type":"function","name":name,"parameters":parameters});
            if let Some(description) = description {
                value["description"] = Value::String(description.clone());
            }
            if let Some(strict) = strict {
                value["strict"] = Value::Bool(*strict);
            }
            if let Some(defer_loading) = defer_loading {
                value["defer_loading"] = Value::Bool(*defer_loading);
            }
            Ok(Some(with_extra(
                value,
                extra.as_ref(),
                cx,
                "tools.function",
            )?))
        }
        Tool::Custom {
            name,
            description,
            format,
            extra,
        } => {
            let mut value = json!({"type":"custom","name":name});
            if let Some(description) = description {
                value["description"] = Value::String(description.clone());
            }
            if let Some(format) = format {
                value["format"] = format.clone();
            }
            Ok(Some(with_extra(value, extra.as_ref(), cx, "tools.custom")?))
        }
        Tool::Hosted { protocol, json } if *protocol == Protocol::Responses => {
            Ok(Some(json.clone()))
        }
        Tool::Hosted { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                "tools.hosted",
                "hosted tool belongs to another protocol",
            )?;
            Ok(None)
        }
    }
}

fn decode_tool_choice(value: Value) -> ToolChoice {
    match value {
        Value::String(choice) => match choice.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Value::Object(map) => {
            if let Some(name) = map.get("name").and_then(Value::as_str) {
                ToolChoice::Named(name.to_owned())
            } else {
                ToolChoice::Native {
                    protocol: Protocol::Responses,
                    json: Value::Object(map),
                }
            }
        }
        other => ToolChoice::Native {
            protocol: Protocol::Responses,
            json: other,
        },
    }
}

fn encode_tool_choice(choice: &ToolChoice, cx: &CodecContext) -> Result<Value> {
    match choice {
        ToolChoice::Auto => Ok(Value::String("auto".to_owned())),
        ToolChoice::None => Ok(Value::String("none".to_owned())),
        ToolChoice::Required => Ok(Value::String("required".to_owned())),
        ToolChoice::Named(name) => Ok(json!({"type":"function","name":name})),
        ToolChoice::Native { protocol, json } if *protocol == Protocol::Responses => {
            Ok(json.clone())
        }
        ToolChoice::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                "tool_choice",
                "native tool choice belongs to another protocol",
            )?;
            Ok(Value::String("auto".to_owned()))
        }
    }
}

fn decode_reasoning(value: Value) -> Option<ReasoningParams> {
    let Value::Object(mut map) = value else {
        return None;
    };
    Some(ReasoningParams {
        effort: wire::take_string(&mut map, "effort"),
        budget_tokens: None,
        summary: wire::take_string(&mut map, "summary"),
        extra: wire::extra_fields(map),
    })
}

fn encode_reasoning(reasoning: &ReasoningParams, cx: &CodecContext) -> Result<Value> {
    let mut map = match reasoning.extra.as_ref() {
        Some(extra) if extra.protocol == Protocol::Responses => extra.fields.clone(),
        Some(extra) if !extra.fields.is_empty() => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                "reasoning",
                "reasoning extension belongs to another protocol",
            )?;
            Map::new()
        }
        _ => Map::new(),
    };
    if let Some(effort) = &reasoning.effort {
        map.insert("effort".to_owned(), Value::String(effort.clone()));
    }
    if let Some(summary) = &reasoning.summary {
        map.insert("summary".to_owned(), Value::String(summary.clone()));
    }
    if reasoning.budget_tokens.is_some() {
        cx.diagnose(
            DiagnosticKind::Approximated,
            Protocol::Responses,
            "reasoning.budget_tokens",
            "fixed reasoning budgets are not supported; effort was retained",
        )?;
    }
    Ok(Value::Object(map))
}

fn decode_text(value: Value) -> (Option<OutputFormat>, Map<String, Value>) {
    let Value::Object(mut map) = value else {
        return (None, Map::new());
    };
    let format = map.remove("format").and_then(|value| {
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
    });
    (format, map)
}

fn encode_output_format(format: &OutputFormat) -> Value {
    let mut map = format
        .extra
        .as_ref()
        .filter(|extra| extra.protocol == Protocol::Responses)
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

pub(crate) fn decode_usage(usage: ResponsesUsage) -> Usage {
    let cached = usage
        .input_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .output_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .unwrap_or(0);
    let mut extra = usage.extra;
    if let Some(details) = usage.input_tokens_details
        && !details.extra.is_empty()
    {
        extra.insert(
            "input_tokens_details".to_owned(),
            Value::Object(details.extra),
        );
    }
    if let Some(details) = usage.output_tokens_details
        && !details.extra.is_empty()
    {
        extra.insert(
            "output_tokens_details".to_owned(),
            Value::Object(details.extra),
        );
    }
    Usage {
        input_tokens: usage.input_tokens.unwrap_or(0).saturating_sub(cached),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cached_input_tokens: cached,
        cache_creation_input_tokens: 0,
        reasoning_tokens: reasoning,
        extra: wire::extra_fields(extra),
    }
}

pub(crate) fn encode_usage(usage: &Usage) -> Value {
    let mut map = usage
        .extra
        .as_ref()
        .filter(|extra| extra.protocol == Protocol::Responses)
        .map_or_else(Map::new, |extra| extra.fields.clone());
    map.insert(
        "input_tokens".to_owned(),
        json!(usage.input_tokens.saturating_add(usage.cached_input_tokens)),
    );
    map.insert("output_tokens".to_owned(), json!(usage.output_tokens));
    map.insert(
        "total_tokens".to_owned(),
        json!(
            usage
                .input_tokens
                .saturating_add(usage.cached_input_tokens)
                .saturating_add(usage.output_tokens)
        ),
    );
    map.insert(
        "input_tokens_details".to_owned(),
        json!({"cached_tokens":usage.cached_input_tokens}),
    );
    map.insert(
        "output_tokens_details".to_owned(),
        json!({"reasoning_tokens":usage.reasoning_tokens}),
    );
    Value::Object(map)
}

pub(crate) fn response_object(
    response: &Response,
    client: &ClientContext,
    cx: &CodecContext,
) -> Result<Map<String, Value>> {
    let mut object = response_header(response, client, cx)?;
    let message = Message {
        role: Role::Assistant,
        content: response.content.clone(),
        extra: None,
    };
    let mut output = Vec::new();
    for item in encode_message_items(&message, cx)? {
        let mut item = item;
        if item.get("type").and_then(Value::as_str) == Some("message") {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                let converted = content
                    .iter()
                    .map(|part| {
                        if part.get("type").and_then(Value::as_str) == Some("input_text") {
                            json!({"type":"output_text","text":part.get("text").and_then(Value::as_str).unwrap_or_default(),"annotations":[]})
                        } else {
                            part.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                item["content"] = Value::Array(converted);
            }
            item["status"] = Value::String("completed".to_owned());
        }
        output.push(item);
    }
    object.insert("output".to_owned(), Value::Array(output));
    object.insert("error".to_owned(), Value::Null);
    object.insert("usage".to_owned(), encode_usage(&response.usage));
    if response.stop_reason == StopReason::MaxTokens {
        object.insert(
            "incomplete_details".to_owned(),
            json!({"reason":"max_output_tokens"}),
        );
    }
    Ok(object)
}

pub(crate) fn response_header(
    response: &Response,
    client: &ClientContext,
    cx: &CodecContext,
) -> Result<Map<String, Value>> {
    let mut object = client.responses_echo();
    for (key, value) in extra_map(response.extra.as_ref(), cx, "response.extra")? {
        object.insert(key, value);
    }
    object.insert("id".to_owned(), Value::String(response.id.clone()));
    object.insert("object".to_owned(), Value::String("response".to_owned()));
    object.entry("created_at".to_owned()).or_insert(json!(0));
    object.insert(
        "status".to_owned(),
        Value::String(
            if response.stop_reason == StopReason::Incomplete
                || response.stop_reason == StopReason::MaxTokens
            {
                "incomplete"
            } else {
                "completed"
            }
            .to_owned(),
        ),
    );
    object.insert("model".to_owned(), Value::String(response.model.clone()));
    Ok(object)
}

pub(crate) fn extra_map(
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Map<String, Value>> {
    Extra::fields_for(extra, Protocol::Responses, cx, path)
}

pub(crate) fn with_extra(
    value: Value,
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Value> {
    Extra::merge_into(value, extra, Protocol::Responses, cx, path)
}

fn blocks_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text, .. } | ContentBlock::Refusal { text, .. } => text.clone(),
            ContentBlock::Reasoning { text, .. } => text.clone(),
            other => format!("[{} content]", other.kind()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn diagnose_unsupported_sampling(req: &Request, cx: &CodecContext) -> Result<()> {
    let sampling = &req.sampling;
    for (path, supplied) in [
        (
            "sampling.stop_sequences",
            !sampling.stop_sequences.is_empty(),
        ),
        ("sampling.top_k", sampling.top_k.is_some()),
        ("sampling.seed", sampling.seed.is_some()),
        (
            "sampling.presence_penalty",
            sampling.presence_penalty.is_some(),
        ),
        (
            "sampling.frequency_penalty",
            sampling.frequency_penalty.is_some(),
        ),
    ] {
        if supplied {
            cx.diagnose(
                DiagnosticKind::Dropped,
                Protocol::Responses,
                path,
                "Responses does not support this sampling control",
            )?;
        }
    }
    Ok(())
}

pub(crate) fn decode_openai_error(status: StatusCode, body: &[u8]) -> ApiError {
    let value: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let error = value.get("error").unwrap_or(&value);
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| error.get("type").and_then(Value::as_str))
        .map(str::to_owned);
    ApiError {
        kind: ApiErrorKind::from_status(status),
        message: error.get("message").and_then(Value::as_str).map_or_else(
            || format!("upstream returned HTTP {}", status.as_u16()),
            str::to_owned,
        ),
        code,
    }
}
