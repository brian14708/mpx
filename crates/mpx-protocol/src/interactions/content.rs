use serde_json::{Map, Value, json};

use crate::{
    CodecContext, ContentBlock, DiagnosticKind, Error, Extra, MediaSource, ReasoningSeal, Result,
    ToolArguments, json::insert_option,
};

use super::{PROTOCOL, invalid};

pub(super) fn items(value: Value) -> Vec<Value> {
    match value {
        Value::Null => Vec::new(),
        Value::Array(items) => items,
        other => vec![other],
    }
}

pub(super) fn decode_content(value: Value) -> Result<ContentBlock> {
    let mut fields = match value {
        Value::String(text) => return Ok(ContentBlock::text(text)),
        Value::Object(fields) => fields,
        _ => return Err(invalid("expected a content object or string")),
    };
    let kind = string(&mut fields, "type").unwrap_or_default();
    let block = match kind.as_str() {
        "text" => ContentBlock::Text {
            text: required_string(&mut fields, "text")?,
            extra: extra(fields),
        },
        "image" | "audio" | "document" | "video" => {
            let mime_type = string(&mut fields, "mime_type");
            if kind == "video" && mime_type.is_none() {
                fields.insert("type".to_owned(), json!("video"));
            }
            let source = if let Some(data) = string(&mut fields, "data") {
                let media_type = mime_type.unwrap_or_else(|| default_mime(&kind).to_owned());
                MediaSource::Base64 { media_type, data }
            } else {
                MediaSource::Url {
                    url: required_string(&mut fields, "uri")?,
                    media_type: mime_type,
                }
            };
            match kind.as_str() {
                "image" => ContentBlock::Image {
                    source,
                    extra: extra(fields),
                },
                "audio" => ContentBlock::Audio {
                    source,
                    extra: extra(fields),
                },
                _ => ContentBlock::File {
                    source,
                    filename: None,
                    extra: extra(fields),
                },
            }
        }
        _ => return Ok(native(kind, fields)),
    };
    Ok(block)
}

pub(super) fn decode_step(value: Value) -> Result<Vec<ContentBlock>> {
    let Value::Object(mut fields) = value else {
        return Err(invalid("expected an interaction step object"));
    };
    let kind = required_string(&mut fields, "type")?;
    if matches!(kind.as_str(), "model_output" | "user_input") {
        return items(fields.remove("content").unwrap_or_default())
            .into_iter()
            .map(decode_content)
            .collect();
    }
    let block = match kind.as_str() {
        "function_call" => {
            let id = required_string(&mut fields, "id")?.into();
            let name = required_string(&mut fields, "name")?;
            let arguments = fields
                .remove("arguments")
                .map_or_else(ToolArguments::default, |args| {
                    ToolArguments::new(args.to_string())
                });
            let seal =
                string(&mut fields, "signature").map(|value| ReasoningSeal::new(PROTOCOL, value));
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                seal,
                extra: extra(fields),
            }
        }
        "function_result" => {
            let call_id = required_string(&mut fields, "call_id")?.into();
            let name = string(&mut fields, "name");
            let is_error = fields
                .remove("is_error")
                .and_then(|value| value.as_bool())
                .unwrap_or_default();
            let content = match fields.remove("result").unwrap_or_default() {
                Value::Object(map)
                    if !map.get("type").and_then(Value::as_str).is_some_and(|kind| {
                        matches!(kind, "text" | "image" | "audio" | "document" | "video")
                    }) =>
                {
                    vec![ContentBlock::text(Value::Object(map).to_string())]
                }
                result => items(result)
                    .into_iter()
                    .map(decode_content)
                    .collect::<Result<_>>()?,
            };
            ContentBlock::ToolResult {
                call_id,
                name,
                content,
                is_error,
                extra: extra(fields),
            }
        }
        "thought" => {
            let seal =
                string(&mut fields, "signature").map(|value| ReasoningSeal::new(PROTOCOL, value));
            let mut summary = Vec::new();
            let mut other = Vec::new();
            for mut item in items(fields.remove("summary").unwrap_or_default()) {
                if item["type"] == "text"
                    && item.as_object().is_some_and(|map| map.len() == 2)
                    && let Some(text) = item
                        .get_mut("text")
                        .and_then(|value| value.take().as_str().map(str::to_owned))
                {
                    summary.push(text);
                } else {
                    other.push(item);
                }
            }
            if !other.is_empty() {
                fields.insert("summary".to_owned(), Value::Array(other));
            }
            ContentBlock::Reasoning {
                text: String::new(),
                summary,
                seal,
                redacted: false,
                extra: extra(fields),
            }
        }
        _ => native(kind, fields),
    };
    Ok(vec![block])
}

pub(super) fn encode_content(block: &ContentBlock, cx: &CodecContext) -> Result<Option<Value>> {
    let (kind, source, extension) = match block {
        ContentBlock::Text { text, extra } | ContentBlock::Refusal { text, extra } => {
            return merge(
                json!({"type": "text", "text": text}),
                extra.as_ref(),
                cx,
                "content",
            )
            .map(Some);
        }
        ContentBlock::Image { source, extra } => ("image", source, extra),
        ContentBlock::Audio { source, extra } => ("audio", source, extra),
        ContentBlock::File { source, extra, .. } => {
            let mime = match source {
                MediaSource::Url { media_type, .. } | MediaSource::File { media_type, .. } => {
                    media_type.as_deref()
                }
                MediaSource::Base64 { media_type, .. } => Some(media_type.as_str()),
            };
            (
                if mime.is_some_and(|mime| mime.starts_with("video/"))
                    || extra.as_ref().is_some_and(|extra| {
                        extra.protocol == PROTOCOL
                            && extra.fields.get("type").and_then(Value::as_str) == Some("video")
                    })
                {
                    "video"
                } else {
                    "document"
                },
                source,
                extra,
            )
        }
        ContentBlock::Native {
            protocol: PROTOCOL,
            json,
        } => return Ok(Some(json.clone())),
        _ => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "content",
                format!("unsupported {} content", block.kind()),
            )?;
            return Ok(None);
        }
    };
    let mut value = json!({"type": kind});
    match source {
        MediaSource::Base64 { media_type, data } => {
            value["mime_type"] = json!(media_type);
            value["data"] = json!(data);
        }
        MediaSource::Url { url, media_type }
        | MediaSource::File {
            file_id: url,
            media_type,
        } => {
            value["uri"] = json!(url);
            if let Some(mime) = media_type {
                value["mime_type"] = json!(mime);
            }
        }
    }
    merge(value, extension.as_ref(), cx, "content").map(Some)
}

pub(super) fn encode_step(block: &ContentBlock, cx: &CodecContext) -> Result<Option<Value>> {
    let (mut value, extension, seal) = match block {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            seal,
            extra,
        } => {
            let arguments = if arguments.as_str().is_empty() {
                json!({})
            } else {
                arguments
                    .value()
                    .map_err(|error| Error::json(PROTOCOL, error))?
            };
            if !arguments.is_object() {
                return Err(invalid("function call arguments must be a JSON object"));
            }
            (
                json!({"type": "function_call", "id": id, "name": name, "arguments": arguments}),
                extra,
                seal.as_ref(),
            )
        }
        ContentBlock::ToolResult {
            call_id,
            name,
            content,
            is_error,
            extra,
        } => {
            let result = encode_contents(content, cx)?;
            let mut value =
                json!({"type": "function_result", "call_id": call_id, "result": result});
            if let Some(name) = name {
                value["name"] = json!(name);
            }
            if *is_error {
                value["is_error"] = Value::Bool(true);
            }
            (value, extra, None)
        }
        ContentBlock::Reasoning {
            text,
            summary,
            seal,
            extra,
            ..
        } => {
            let mut content: Vec<_> = summary
                .iter()
                .map(|text| json!({"type": "text", "text": text}))
                .collect();
            if !text.is_empty() {
                content.push(json!({"type": "text", "text": text}));
            }
            if let Some(extra) = extra.as_ref().filter(|extra| extra.protocol == PROTOCOL)
                && let Some(other) = extra.fields.get("summary").and_then(Value::as_array)
            {
                content.extend(other.iter().cloned());
            }
            let mut value = json!({"type": "thought"});
            if !content.is_empty() {
                value["summary"] = json!(content);
            }
            (value, extra, seal.as_ref())
        }
        ContentBlock::Native {
            protocol: PROTOCOL,
            json,
        } => return Ok(Some(json.clone())),
        _ => return Ok(None),
    };
    if let Some(seal) = seal {
        if seal.protocol == PROTOCOL {
            value["signature"] = json!(seal.value);
        } else {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "step.signature",
                "reasoning signature belongs to another protocol",
            )?;
        }
    }
    merge(value, extension.as_ref(), cx, "step").map(Some)
}

pub(super) fn is_step(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::ToolCall { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::Reasoning { .. } => true,
        ContentBlock::Native {
            protocol: PROTOCOL,
            json,
        } => json["type"].as_str().is_some_and(|kind| {
            kind.ends_with("_call")
                || kind.ends_with("_result")
                || kind.ends_with("_output")
                || kind == "user_input"
        }),
        _ => false,
    }
}

pub(super) fn encode_contents(blocks: &[ContentBlock], cx: &CodecContext) -> Result<Vec<Value>> {
    blocks
        .iter()
        .filter_map(|block| encode_content(block, cx).transpose())
        .collect()
}

pub(super) fn extra(fields: Map<String, Value>) -> Option<Extra> {
    Extra::from_unknown(PROTOCOL, fields)
}

pub(super) fn fields(
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Map<String, Value>> {
    Extra::fields_for(extra, PROTOCOL, cx, path)
}

pub(super) fn merge(
    value: Value,
    extra: Option<&Extra>,
    cx: &CodecContext,
    path: &str,
) -> Result<Value> {
    Extra::merge_into(value, extra, PROTOCOL, cx, path)
}

pub(super) fn string(fields: &mut Map<String, Value>, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

fn required_string(fields: &mut Map<String, Value>, key: &str) -> Result<String> {
    string(fields, key).ok_or_else(|| invalid(format!("expected string field `{key}`")))
}

fn native(kind: String, mut fields: Map<String, Value>) -> ContentBlock {
    insert_option(&mut fields, "type", Some(Value::String(kind)));
    ContentBlock::Native {
        protocol: PROTOCOL,
        json: Value::Object(fields),
    }
}

const fn default_mime(kind: &str) -> &'static str {
    match kind.as_bytes() {
        b"image" => "image/png",
        b"audio" => "audio/wav",
        b"video" => "video/mp4",
        _ => "application/pdf",
    }
}
