use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    CodecContext, ContentBlock, Continuation, DiagnosticKind, Endpoint, Error, Message,
    OutputFormat, OutputFormatKind, ReasoningParams, Request, Result, Role, SamplingParams, Tool,
    ToolChoice, json::insert_option, request::take_session_id,
};

use super::{
    PROTOCOL,
    content::{
        decode_content, decode_step, encode_content, encode_step, extra, fields, is_step, items,
        merge, string,
    },
    invalid, parse,
};

#[derive(Default, Deserialize, Serialize)]
struct GenerationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking_summaries: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

pub(super) fn decode(endpoint: &Endpoint, body: &[u8]) -> Result<Request> {
    let mut fields = parse(body)?;
    let model = string(&mut fields, "model")
        .or_else(|| endpoint.model_from_path.clone())
        .or_else(|| {
            fields
                .get("agent")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|model| !model.is_empty())
        .ok_or_else(|| invalid("expected a model or agent"))?;
    let messages = decode_input(fields.remove("input").unwrap_or_default())?;
    let instructions = string(&mut fields, "system_instruction")
        .map(ContentBlock::text)
        .into_iter()
        .collect();
    let tools = items(fields.remove("tools").unwrap_or_default())
        .into_iter()
        .map(decode_tool)
        .collect();
    let config = fields
        .remove("generation_config")
        .filter(|value| !value.is_null())
        .map(serde_json::from_value::<GenerationConfig>)
        .transpose()
        .map_err(|error| Error::json(PROTOCOL, error))?
        .unwrap_or_default();
    let reasoning = if config.thinking_level.is_some() || config.thinking_summaries.is_some() {
        Some(ReasoningParams {
            effort: config.thinking_level,
            summary: config.thinking_summaries,
            ..ReasoningParams::default()
        })
    } else {
        None
    };
    let output_format = fields
        .get("response_format")
        .cloned()
        .and_then(decode_format);
    if output_format.is_some() {
        fields.remove("response_format");
    }
    let continuation = string(&mut fields, "previous_interaction_id").map(|id| Continuation {
        previous_response_id: id.into(),
        store: fields.remove("store").and_then(|value| value.as_bool()),
    });
    let stream = fields
        .remove("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or_default();
    let metadata = crate::json::take_object(&mut fields, "user_metadata");
    let session_id = fields
        .get_mut("webhook_config")
        .and_then(Value::as_object_mut)
        .and_then(|config| config.get_mut("user_metadata"))
        .and_then(Value::as_object_mut)
        .and_then(take_session_id);
    Ok(Request {
        model,
        session_id,
        instructions,
        messages,
        tools,
        tool_choice: config.tool_choice.map(decode_choice),
        sampling: SamplingParams {
            max_output_tokens: config.max_output_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            stop_sequences: config.stop_sequences,
            seed: config.seed,
            extra: extra(config.extra),
            ..SamplingParams::default()
        },
        reasoning,
        output_format,
        stream,
        continuation,
        metadata,
        extra: extra(fields),
        ..Request::default()
    })
}

fn decode_input(input: Value) -> Result<Vec<Message>> {
    let mut messages: Vec<Message> = Vec::new();
    let mut loose_content = Vec::new();
    for item in items(input) {
        let kind = item["type"].as_str().unwrap_or_default();
        let role = match kind {
            "user_input" => Role::User,
            "function_result" => Role::Tool,
            "model_output" | "thought" | "function_call" => Role::Assistant,
            _ if kind.ends_with("_call") || kind.ends_with("_result") => Role::Assistant,
            _ => {
                loose_content.push(decode_content(item)?);
                continue;
            }
        };
        flush_loose_content(&mut messages, &mut loose_content);
        let extension = matches!(kind, "user_input" | "model_output")
            .then(|| {
                let mut fields = item.as_object().cloned().unwrap_or_default();
                fields.remove("type");
                fields.remove("content");
                extra(fields)
            })
            .flatten();
        messages.push(Message {
            role,
            content: decode_step(item)?,
            extra: extension,
        });
    }
    flush_loose_content(&mut messages, &mut loose_content);
    Ok(messages)
}

fn flush_loose_content(messages: &mut Vec<Message>, content: &mut Vec<ContentBlock>) {
    if !content.is_empty() {
        messages.push(Message {
            role: Role::User,
            content: core::mem::take(content),
            extra: None,
        });
    }
}

fn decode_tool(value: Value) -> Tool {
    if value["type"] == "function"
        && let Value::Object(mut fields) = value.clone()
    {
        fields.remove("type");
        return Tool::Function {
            name: string(&mut fields, "name").unwrap_or_default(),
            description: string(&mut fields, "description"),
            parameters: fields.remove("parameters").unwrap_or(Value::Null),
            strict: None,
            defer_loading: None,
            extra: extra(fields),
        };
    }
    Tool::Hosted {
        protocol: PROTOCOL,
        json: value,
    }
}

fn decode_choice(value: Value) -> ToolChoice {
    if let Some(allowed) = value.get("allowed_tools").and_then(Value::as_object)
        && value.as_object().is_some_and(|fields| fields.len() == 1)
        && allowed.len() == 2
        && allowed.get("mode").and_then(Value::as_str) == Some("any")
        && let Some([tool]) = allowed
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
        && let Some(name) = tool.as_str()
    {
        return ToolChoice::Named(name.to_owned());
    }
    let mode = value.as_str().or_else(|| value["mode"].as_str());
    match mode {
        Some("auto") if value.is_string() => ToolChoice::Auto,
        Some("none") if value.is_string() => ToolChoice::None,
        Some("any") if value.is_string() => ToolChoice::Required,
        _ => ToolChoice::Native {
            protocol: PROTOCOL,
            json: value,
        },
    }
}

fn decode_format(mut value: Value) -> Option<OutputFormat> {
    if let Value::Array(formats) = &mut value
        && formats.len() == 1
    {
        value = formats.pop()?;
    }
    let Value::Object(mut fields) = value else {
        return None;
    };
    if fields.remove("type")?.as_str()? != "text" {
        return None;
    }
    let mime = string(&mut fields, "mime_type");
    let schema = fields.remove("schema");
    let kind = match (mime.as_deref(), &schema) {
        (Some("application/json") | None, Some(_)) => OutputFormatKind::JsonSchema,
        (Some("application/json"), None) => OutputFormatKind::JsonObject,
        (Some("text/plain") | None, None) => OutputFormatKind::Text,
        _ => return None,
    };
    Some(OutputFormat {
        kind,
        schema,
        name: None,
        strict: None,
        extra: extra(fields),
    })
}

pub(super) fn encode(req: &Request, cx: &CodecContext) -> Result<Map<String, Value>> {
    if req.model.is_empty() {
        return Err(invalid("expected a model or agent"));
    }
    let mut wire = fields(req.extra.as_ref(), cx, "request")?;
    let model_key = if wire.contains_key("agent") {
        "agent"
    } else {
        "model"
    };
    wire.insert(model_key.to_owned(), json!(req.model));
    if req.stream {
        wire.insert("stream".to_owned(), Value::Bool(true));
    }
    if let Some(continuation) = &req.continuation {
        wire.insert(
            "previous_interaction_id".to_owned(),
            json!(continuation.previous_response_id),
        );
        insert_option(&mut wire, "store", continuation.store.map(Value::Bool));
    }
    if !req.metadata.is_empty() {
        wire.insert("user_metadata".to_owned(), json!(req.metadata));
    }
    if let Some(session_id) = &req.session_id
        && !session_id.is_empty()
        && let Some(config) = wire
            .get_mut("webhook_config")
            .and_then(Value::as_object_mut)
        && let Some(metadata) = config
            .entry("user_metadata")
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
    {
        metadata.insert("session_id".to_owned(), Value::String(session_id.clone()));
    }
    let mut instructions = Vec::new();
    append_instructions(&req.instructions, &mut instructions, cx)?;
    let input = encode_input(&req.messages, &mut instructions, cx)?;
    insert_option(&mut wire, "input", input);
    if !instructions.is_empty() {
        wire.insert(
            "system_instruction".to_owned(),
            json!(instructions.join("\n\n")),
        );
    }
    let tools: Vec<_> = req
        .tools
        .iter()
        .filter_map(|tool| encode_tool(tool, cx).transpose())
        .collect::<Result<_>>()?;
    if !tools.is_empty() {
        wire.insert("tools".to_owned(), json!(tools));
    }
    if let Some(format) = &req.output_format {
        wire.insert("response_format".to_owned(), encode_format(format, cx)?);
    }
    insert_option(&mut wire, "generation_config", encode_config(req, cx)?);
    Ok(wire)
}

fn append_instructions(
    blocks: &[ContentBlock],
    instructions: &mut Vec<String>,
    cx: &CodecContext,
) -> Result<()> {
    for block in blocks {
        if let ContentBlock::Text { text, extra } = block {
            if extra.as_ref().is_some_and(|extra| !extra.is_empty()) {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    "system_instruction",
                    "system instructions do not support content metadata",
                )?;
            }
            instructions.push(text.clone());
        } else {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "system_instruction",
                "system instructions only support text",
            )?;
        }
    }
    Ok(())
}

fn encode_input(
    messages: &[Message],
    instructions: &mut Vec<String>,
    cx: &CodecContext,
) -> Result<Option<Value>> {
    let mut steps = Vec::new();
    for message in messages {
        if matches!(message.role, Role::System | Role::Developer) {
            append_instructions(&message.content, instructions, cx)?;
            continue;
        }
        let kind = if message.role == Role::Assistant {
            "model_output"
        } else {
            "user_input"
        };
        let mut content = Vec::new();
        for block in &message.content {
            if is_step(block) {
                flush_step(&mut steps, &mut content, kind, message, cx)?;
                if let Some(step) = encode_step(block, cx)? {
                    steps.push(step);
                }
            } else if let Some(value) = encode_content(block, cx)? {
                content.push(value);
            }
        }
        flush_step(&mut steps, &mut content, kind, message, cx)?;
    }
    if let [step] = steps.as_slice()
        && step["type"] == "user_input"
        && step.as_object().is_some_and(|map| map.len() == 2)
    {
        let content = &step["content"];
        if let Some([single]) = content.as_array().map(Vec::as_slice)
            && single["type"] == "text"
            && single.as_object().is_some_and(|map| map.len() == 2)
        {
            return Ok(Some(single["text"].clone()));
        }
        return Ok(Some(content.clone()));
    }
    Ok((!steps.is_empty()).then_some(Value::Array(steps)))
}

fn flush_step(
    steps: &mut Vec<Value>,
    content: &mut Vec<Value>,
    kind: &str,
    message: &Message,
    cx: &CodecContext,
) -> Result<()> {
    if !content.is_empty() {
        steps.push(merge(
            json!({"type": kind, "content": core::mem::take(content)}),
            message.extra.as_ref(),
            cx,
            "input",
        )?);
    }
    Ok(())
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
            if strict.is_some() || defer_loading.is_some() {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    PROTOCOL,
                    "tools",
                    "function tools do not support strict or defer_loading",
                )?;
            }
            let mut value = json!({"type": "function", "name": name});
            if let Some(description) = description {
                value["description"] = json!(description);
            }
            if !parameters.is_null() {
                value["parameters"] = parameters.clone();
            }
            merge(value, extra.as_ref(), cx, "tools").map(Some)
        }
        Tool::Hosted {
            protocol: PROTOCOL,
            json,
        } => Ok(Some(json.clone())),
        _ => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "tools",
                "unsupported tool declaration",
            )?;
            Ok(None)
        }
    }
}

fn encode_choice(choice: &ToolChoice, cx: &CodecContext) -> Result<Option<Value>> {
    let value = match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("any"),
        ToolChoice::Named(name) => json!({"allowed_tools": {"mode": "any", "tools": [name]}}),
        ToolChoice::Native {
            protocol: PROTOCOL,
            json,
        } => json.clone(),
        ToolChoice::Native { .. } => {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                "tool_choice",
                "tool choice belongs to another protocol",
            )?;
            return Ok(None);
        }
    };
    Ok(Some(value))
}

fn encode_format(format: &OutputFormat, cx: &CodecContext) -> Result<Value> {
    if format.name.is_some() || format.strict.is_some() {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "response_format",
            "format names and strict flags are unavailable",
        )?;
    }
    let mime = match format.kind {
        OutputFormatKind::Text => "text/plain",
        OutputFormatKind::JsonObject | OutputFormatKind::JsonSchema => "application/json",
    };
    let mut value = json!({"type": "text", "mime_type": mime});
    if let Some(schema) = &format.schema {
        value["schema"] = schema.clone();
    }
    merge(value, format.extra.as_ref(), cx, "response_format")
}

fn encode_config(req: &Request, cx: &CodecContext) -> Result<Option<Value>> {
    let sampling = &req.sampling;
    for (present, name) in [
        (sampling.top_k.is_some(), "top_k"),
        (sampling.frequency_penalty.is_some(), "frequency_penalty"),
        (sampling.presence_penalty.is_some(), "presence_penalty"),
        (req.parallel_tool_calls.is_some(), "parallel_tool_calls"),
        (
            req.reasoning
                .as_ref()
                .is_some_and(|reasoning| reasoning.budget_tokens.is_some()),
            "thinking_budget",
        ),
    ] {
        if present {
            cx.diagnose(
                DiagnosticKind::Dropped,
                PROTOCOL,
                name,
                "generation control is unavailable",
            )?;
        }
    }
    let mut extension = fields(sampling.extra.as_ref(), cx, "generation_config")?;
    if let Some(reasoning) = &req.reasoning {
        crate::json::merge_missing(
            &mut extension,
            fields(reasoning.extra.as_ref(), cx, "generation_config.reasoning")?,
        );
    }
    let config = GenerationConfig {
        max_output_tokens: sampling.max_output_tokens,
        seed: sampling.seed,
        stop_sequences: sampling.stop_sequences.clone(),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        thinking_level: req
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_ref())
            .map(|effort| effort.to_ascii_lowercase()),
        thinking_summaries: req
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.clone()),
        tool_choice: req
            .tool_choice
            .as_ref()
            .map(|choice| encode_choice(choice, cx))
            .transpose()?
            .flatten(),
        extra: extension,
    };
    let value = serde_json::to_value(config).map_err(|error| Error::json(PROTOCOL, error))?;
    Ok(value
        .as_object()
        .is_some_and(|fields| !fields.is_empty())
        .then_some(value))
}
