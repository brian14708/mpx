use std::collections::{HashMap, HashSet};

use crate::{
    ContentBlock, DiagnosticKind, Diagnostics, Error, Message, Protocol, Request, Result, Role,
    ToolCallId,
};

/// Validates structural tool-call invariants before encoding.
///
/// Oddities accepted by an unchanged protocol are warnings. Structural issues
/// that make the selected target impossible to encode return an error.
pub(crate) fn validate_request(
    request: &Request,
    target: Protocol,
    source: Protocol,
    diagnostics: &Diagnostics,
) -> Result<()> {
    let mut calls = HashSet::new();
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolCall { id, .. } => {
                    if !calls.insert(id.clone()) {
                        return Err(Error::InvalidRequest {
                            protocol: target,
                            message: format!("duplicate tool-call identifier `{id}`"),
                        });
                    }
                }
                ContentBlock::ToolResult { .. } => {}
                _ => {}
            }
        }
    }
    if request.messages.is_empty() {
        diagnostics.push(crate::Diagnostic {
            kind: DiagnosticKind::Approximated,
            protocol: target,
            path: "messages".to_owned(),
            message: "request contains no messages".to_owned(),
        });
    }
    if source == target {
        return Ok(());
    }
    for message in &request.messages {
        for block in &message.content {
            if let ContentBlock::ToolResult { call_id, .. } = block
                && !calls.contains(call_id)
            {
                return Err(Error::InvalidRequest {
                    protocol: target,
                    message: format!("tool result `{call_id}` has no matching call"),
                });
            }
        }
    }
    Ok(())
}

/// Repairs translated tool history while preserving exact call arguments.
pub(crate) fn repair_history(request: &mut Request, target: Protocol, diagnostics: &Diagnostics) {
    merge_split_assistant_messages(request);

    let mut call_names = HashMap::<ToolCallId, String>::new();
    let mut answered = HashSet::<ToolCallId>::new();
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolCall { id, name, .. } => {
                    call_names.insert(id.clone(), name.clone());
                }
                ContentBlock::ToolResult { call_id, .. } => {
                    answered.insert(call_id.clone());
                }
                _ => {}
            }
        }
    }

    let mut repaired = Vec::with_capacity(request.messages.len());
    for mut message in request.messages.drain(..) {
        let mut ordinary = Vec::new();
        let mut had_orphan = false;
        for mut block in message.content.drain(..) {
            if let ContentBlock::ToolResult { call_id, name, .. } = &mut block {
                if name.is_none() {
                    *name = call_names.get(call_id).cloned();
                }
                if !call_names.contains_key(call_id) {
                    diagnostics.push(crate::Diagnostic {
                        kind: DiagnosticKind::Fallback,
                        protocol: target,
                        path: "messages.content.tool_result".to_owned(),
                        message: format!("orphan tool result `{call_id}` became user text"),
                    });
                    ordinary.extend(block_text_content(&block));
                    had_orphan = true;
                    continue;
                }
            }
            ordinary.push(block);
        }
        message.content = ordinary;
        if had_orphan && message.role == Role::Tool {
            message.role = Role::User;
        }
        let missing: Vec<(ToolCallId, String)> = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall { id, name, .. } if !answered.contains(id) => {
                    Some((id.clone(), name.clone()))
                }
                _ => None,
            })
            .collect();
        repaired.push(message);
        if !missing.is_empty() {
            let content = missing
                .into_iter()
                .map(|(call_id, name)| {
                    diagnostics.push(crate::Diagnostic {
                        kind: DiagnosticKind::Synthesized,
                        protocol: target,
                        path: "messages.content.tool_result".to_owned(),
                        message: format!(
                            "unanswered call `{call_id}` received an interrupted result"
                        ),
                    });
                    ContentBlock::ToolResult {
                        call_id,
                        name: Some(name),
                        content: vec![ContentBlock::text(
                            "Tool call interrupted before a result was available.",
                        )],
                        is_error: true,
                        extra: None,
                    }
                })
                .collect();
            repaired.push(Message {
                role: Role::Tool,
                content,
                extra: None,
            });
        }
    }
    request.messages = repaired;
}

fn block_text_content(block: &ContentBlock) -> Vec<ContentBlock> {
    match block {
        ContentBlock::ToolResult { content, .. } => content.clone(),
        other => vec![other.clone()],
    }
}

fn merge_split_assistant_messages(request: &mut Request) {
    let mut merged: Vec<Message> = Vec::with_capacity(request.messages.len());
    for message in request.messages.drain(..) {
        if message.role == Role::Assistant
            && let Some(previous) = merged.last_mut()
            && previous.role == Role::Assistant
        {
            previous.content.extend(message.content);
            if previous.extra.is_none() {
                previous.extra = message.extra;
            }
            continue;
        }
        merged.push(message);
    }
    request.messages = merged;
}
