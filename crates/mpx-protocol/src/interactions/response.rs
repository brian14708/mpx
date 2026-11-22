use http::StatusCode;
use serde_json::{Map, Value, json};

use crate::{
    ApiError, ApiErrorKind, CodecContext, ContentBlock, DiagnosticKind, Response, Result,
    StopReason, Usage,
};

use super::{
    PROTOCOL,
    content::{decode_step, encode_content, encode_step, extra, fields, is_step, items, string},
    invalid,
};

pub(super) fn decode(mut wire: Map<String, Value>) -> Result<Response> {
    let id = string(&mut wire, "id").ok_or_else(|| invalid("expected an interaction id"))?;
    let model = string(&mut wire, "model")
        .or_else(|| wire.get("agent").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default();
    let status =
        string(&mut wire, "status").ok_or_else(|| invalid("expected an interaction status"))?;
    let mut content = Vec::new();
    for step in items(wire.remove("steps").unwrap_or_default()) {
        // Full interaction resources may include echoed input as well as model output.
        if step["type"] != "user_input" {
            content.extend(decode_step(step)?);
        }
    }
    let usage = decode_usage(crate::json::take_object(&mut wire, "usage"));
    Ok(Response {
        id,
        model,
        stop_reason: stop_from_status(&status),
        content,
        usage,
        extra: extra(wire),
        ..Response::default()
    })
}

pub(super) fn encode(response: &Response, request_model: &str, cx: &CodecContext) -> Result<Value> {
    let mut wire = fields(response.extra.as_ref(), cx, "response")?;
    wire.entry("object").or_insert_with(|| json!("interaction"));
    wire.insert("id".to_owned(), json!(response.id));
    let model = if response.model.is_empty() {
        request_model
    } else {
        &response.model
    };
    if !model.is_empty() && !wire.contains_key("agent") {
        wire.insert("model".to_owned(), json!(model));
    }
    wire.insert(
        "status".to_owned(),
        json!(status_from_stop(&response.stop_reason, cx)?),
    );
    wire.insert(
        "steps".to_owned(),
        json!(encode_steps(&response.content, cx)?),
    );
    wire.insert("usage".to_owned(), encode_usage(&response.usage, cx)?);
    if !response.safety.is_empty() {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "response.safety",
            "interactions do not expose per-candidate safety ratings",
        )?;
    }
    Ok(Value::Object(wire))
}

fn encode_steps(content: &[ContentBlock], cx: &CodecContext) -> Result<Vec<Value>> {
    let mut steps = Vec::new();
    let mut output = Vec::new();
    for block in content {
        if is_step(block) {
            if !output.is_empty() {
                steps
                    .push(json!({"type": "model_output", "content": core::mem::take(&mut output)}));
            }
            if let Some(step) = encode_step(block, cx)? {
                steps.push(step);
            }
        } else if let Some(block) = encode_content(block, cx)? {
            output.push(block);
        }
    }
    if !output.is_empty() {
        steps.push(json!({"type": "model_output", "content": output}));
    }
    Ok(steps)
}

pub(super) fn stop_from_status(status: &str) -> StopReason {
    match status {
        "completed" => StopReason::EndTurn,
        "requires_action" => StopReason::ToolUse,
        "incomplete" => StopReason::Incomplete,
        other => StopReason::Other(other.to_owned()),
    }
}

pub(super) fn status_from_stop(reason: &StopReason, cx: &CodecContext) -> Result<String> {
    let status = match reason {
        StopReason::EndTurn | StopReason::StopSequence => "completed",
        StopReason::ToolUse | StopReason::PauseTurn => "requires_action",
        StopReason::Incomplete
        | StopReason::MaxTokens
        | StopReason::Refusal
        | StopReason::ContentFilter => "incomplete",
        StopReason::Other(status)
            if matches!(
                status.as_str(),
                "in_progress"
                    | "requires_continuation"
                    | "failed"
                    | "cancelled"
                    | "queued"
                    | "idle"
                    | "budget_exceeded"
            ) =>
        {
            status
        }
        StopReason::Other(_) => {
            cx.diagnose(
                DiagnosticKind::Approximated,
                PROTOCOL,
                "response.status",
                "unknown stop reason mapped to incomplete",
            )?;
            "incomplete"
        }
    };
    Ok(status.to_owned())
}

pub(super) fn decode_usage(mut wire: Map<String, Value>) -> Usage {
    let input = take_tokens(&mut wire, "total_input_tokens");
    let cached = take_tokens(&mut wire, "total_cached_tokens");
    Usage {
        input_tokens: input.saturating_sub(cached),
        cached_input_tokens: cached,
        output_tokens: take_tokens(&mut wire, "total_output_tokens"),
        reasoning_tokens: take_tokens(&mut wire, "total_thought_tokens"),
        extra: extra(wire),
        ..Usage::default()
    }
}

pub(super) fn encode_usage(usage: &Usage, cx: &CodecContext) -> Result<Value> {
    let mut wire = fields(usage.extra.as_ref(), cx, "usage")?;
    let input = usage.input_tokens.saturating_add(usage.cached_input_tokens);
    for (key, count) in [
        ("total_input_tokens", input),
        ("total_cached_tokens", usage.cached_input_tokens),
        ("total_output_tokens", usage.output_tokens),
        ("total_thought_tokens", usage.reasoning_tokens),
    ] {
        wire.insert(key.to_owned(), json!(count));
    }
    wire.entry("total_tokens").or_insert_with(|| {
        json!(
            input
                .saturating_add(usage.output_tokens)
                .saturating_add(usage.reasoning_tokens)
        )
    });
    if usage.cache_creation_input_tokens != 0 {
        cx.diagnose(
            DiagnosticKind::Dropped,
            PROTOCOL,
            "usage",
            "cache-write tokens are unavailable",
        )?;
    }
    Ok(Value::Object(wire))
}

fn take_tokens(wire: &mut Map<String, Value>, key: &str) -> u64 {
    wire.remove(key)
        .and_then(|value| value.as_u64())
        .unwrap_or_default()
}

pub(super) fn api_error(value: &Value, status: StatusCode) -> ApiError {
    let code = value["status"].as_str().or_else(|| value["code"].as_str());
    let kind = match code {
        Some("invalid_request" | "invalid_argument" | "INVALID_ARGUMENT") => {
            ApiErrorKind::InvalidRequest
        }
        Some("unauthenticated" | "UNAUTHENTICATED") => ApiErrorKind::Authentication,
        Some("permission_denied" | "PERMISSION_DENIED") => ApiErrorKind::Permission,
        Some("not_found" | "NOT_FOUND") => ApiErrorKind::NotFound,
        Some("rate_limit_exceeded" | "resource_exhausted" | "RESOURCE_EXHAUSTED") => {
            ApiErrorKind::RateLimit
        }
        Some("deadline_exceeded" | "DEADLINE_EXCEEDED") => ApiErrorKind::Timeout,
        Some("unavailable" | "UNAVAILABLE") => ApiErrorKind::Overloaded,
        _ => ApiErrorKind::from_status(status),
    };
    ApiError {
        kind,
        code: code.map(str::to_owned),
        message: value["message"]
            .as_str()
            .unwrap_or("Gemini interaction failed")
            .to_owned(),
    }
}
