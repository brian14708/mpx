use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{ApiError, ContentBlock, Extra, Message, Role, StreamEvent, Usage};

/// Why model generation stopped.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model completed its turn.
    #[default]
    EndTurn,
    /// The configured output-token limit was reached.
    MaxTokens,
    /// At least one client tool call requires a result.
    ToolUse,
    /// A provider-hosted tool paused the turn.
    PauseTurn,
    /// A configured stop sequence ended generation.
    StopSequence,
    /// The model refused the request.
    Refusal,
    /// A provider safety filter stopped generation.
    ContentFilter,
    /// A Responses turn ended incomplete for another reason.
    Incomplete,
    /// A provider value not yet modeled explicitly.
    Other(String),
}

pub(crate) fn normalize_tool_stop_reason(reason: StopReason, has_tool_calls: bool) -> StopReason {
    if has_tool_calls && reason == StopReason::EndTurn {
        StopReason::ToolUse
    } else {
        reason
    }
}

/// One provider safety rating associated with a response.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Provider safety category.
    #[builder(into)]
    pub category: String,
    /// Provider probability or severity label.
    #[builder(into)]
    pub probability: Option<String>,
    /// Whether this rating blocked output.
    #[builder(default)]
    pub blocked: bool,
    /// Unmodeled rating fields.
    pub extra: Option<Extra>,
}

/// A complete, normalized model response.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct Response {
    /// Provider response identifier.
    #[builder(into)]
    pub id: String,
    /// Model that generated the response.
    #[builder(into)]
    pub model: String,
    /// Ordered model output blocks.
    #[builder(default)]
    pub content: Vec<ContentBlock>,
    /// Terminal generation reason.
    #[builder(default)]
    pub stop_reason: StopReason,
    /// Token accounting.
    #[builder(default)]
    pub usage: Usage,
    /// Provider safety ratings.
    #[builder(default)]
    pub safety: Vec<SafetyRating>,
    /// Unmodeled response fields.
    pub extra: Option<Extra>,
}

impl Response {
    /// Converts the response to an assistant message without losing blocks.
    #[must_use]
    pub fn to_message(&self) -> Message {
        // Interaction resource fields such as timestamps describe the response,
        // while replaying its steps only needs the content blocks.
        let extra = self
            .extra
            .as_ref()
            .filter(|extra| extra.protocol != crate::Protocol::Interactions);
        Message {
            role: Role::Assistant,
            content: self.content.clone(),
            extra: extra.cloned(),
        }
    }

    /// Expands a complete response into the canonical event lifecycle.
    ///
    /// Block indices are local to this response. Encoders renumber them for
    /// the client and emit usage and stop only after every block is closed.
    #[must_use]
    pub fn to_stream_events(&self) -> Vec<StreamEvent> {
        let mut events = Vec::with_capacity(self.content.len() * 3 + 4);
        events.push(StreamEvent::MessageStart {
            id: self.id.clone(),
            model: self.model.clone(),
            extra: self.extra.clone(),
        });
        for (position, block) in self.content.iter().enumerate() {
            let index = u32::try_from(position).unwrap_or(u32::MAX);
            let (start, delta) = stream_block(block);
            events.push(StreamEvent::BlockStart {
                index,
                block: start,
            });
            if let Some(delta) = delta {
                events.push(StreamEvent::BlockDelta { index, delta });
            }
            events.push(StreamEvent::BlockStop { index });
        }
        events.push(StreamEvent::Usage {
            usage: self.usage.clone(),
        });
        if !self.safety.is_empty() {
            events.push(StreamEvent::Safety {
                ratings: self.safety.clone(),
            });
        }
        events.push(StreamEvent::MessageStop {
            stop_reason: self.stop_reason.clone(),
        });
        events
    }
}

fn stream_block(block: &ContentBlock) -> (ContentBlock, Option<crate::BlockDelta>) {
    match block {
        ContentBlock::Text { text, extra } => (
            ContentBlock::Text {
                text: String::new(),
                extra: extra.clone(),
            },
            Some(crate::BlockDelta::Text(text.clone())),
        ),
        ContentBlock::Reasoning {
            text,
            summary,
            seal,
            redacted,
            extra,
        } => (
            ContentBlock::Reasoning {
                text: String::new(),
                summary: summary.clone(),
                seal: seal.clone(),
                redacted: *redacted,
                extra: extra.clone(),
            },
            (!text.is_empty()).then(|| crate::BlockDelta::Reasoning(text.clone())),
        ),
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            seal,
            extra,
        } => (
            ContentBlock::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: crate::ToolArguments::default(),
                seal: seal.clone(),
                extra: extra.clone(),
            },
            Some(crate::BlockDelta::ToolArguments(
                arguments.as_str().to_owned(),
            )),
        ),
        ContentBlock::Refusal { text, extra } => (
            ContentBlock::Refusal {
                text: String::new(),
                extra: extra.clone(),
            },
            Some(crate::BlockDelta::Refusal(text.clone())),
        ),
        other => (other.clone(), None),
    }
}

/// The body of a decoded upstream response.
#[derive(Debug)]
#[non_exhaustive]
pub enum ResponseBody {
    /// A complete JSON response.
    Complete(Response),
    /// A normalized streaming response.
    Stream(crate::EventStream),
    /// A native response from an HTTP passthrough route.
    Passthrough(crate::Body),
    /// A non-success API response, preserved as data for retry middleware.
    Error(ApiError),
}
