use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    ContentBlock, Extra, Message, OutputFormat, ReasoningParams, Response, SamplingParams, Tool,
    ToolChoice, json::nonempty,
};

pub(crate) fn session_id_field(fields: &Map<String, Value>) -> Option<String> {
    nonempty(fields.get("session_id")?.as_str()?.trim())
}

pub(crate) fn take_session_id(fields: &mut Map<String, Value>) -> Option<String> {
    let session_id = session_id_field(fields)?;
    fields.remove("session_id");
    Some(session_id)
}

/// A provider response identifier used for server-side continuation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResponseId(String);

impl ResponseId {
    /// Creates an identifier from provider text.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the newtype and returns the identifier text.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for ResponseId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ResponseId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl AsRef<str> for ResponseId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for ResponseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Server-side continuation state supplied by a model API client.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct Continuation {
    /// Previous provider response to continue.
    pub previous_response_id: ResponseId,
    /// Provider-side storage preference.
    pub store: Option<bool>,
}

/// An editable, protocol-independent model request.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct Request {
    /// Provider model identifier.
    #[builder(into)]
    pub model: String,
    /// Client session identifier, extracted from recognized headers or provider fields.
    /// `x-session-id` takes precedence over native session headers and body fields.
    /// `thread-id` is a fallback on Responses requests.
    #[builder(into)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Leading system/developer content in canonical order.
    #[builder(default)]
    pub instructions: Vec<ContentBlock>,
    /// Conversation history and current input.
    #[builder(default)]
    pub messages: Vec<Message>,
    /// Tools available to the model.
    #[builder(default)]
    pub tools: Vec<Tool>,
    /// Tool-selection policy, if explicitly supplied.
    pub tool_choice: Option<ToolChoice>,
    /// Generation controls.
    #[builder(default)]
    pub sampling: SamplingParams,
    /// Reasoning controls.
    pub reasoning: Option<ReasoningParams>,
    /// Structured-output requirements.
    pub output_format: Option<OutputFormat>,
    /// Whether parallel calls are permitted, if explicitly supplied.
    pub parallel_tool_calls: Option<bool>,
    /// Whether the client requested a stream.
    #[builder(default)]
    pub stream: bool,
    /// Client metadata retained for the origin protocol.
    #[builder(default)]
    pub metadata: Map<String, Value>,
    /// Optional server-side continuation.
    pub continuation: Option<Continuation>,
    /// Unmodeled request fields.
    pub extra: Option<Extra>,
    #[builder(skip)]
    #[serde(skip)]
    pub(crate) client_include_usage: bool,
    #[builder(skip)]
    #[serde(skip)]
    pub(crate) messages_max_tokens: bool,
}

impl Request {
    pub(crate) fn openai_metadata(&self) -> Option<Map<String, Value>> {
        let mut metadata = self.metadata.clone();
        // OpenAI allows at most 16 entries and 512 characters per value.
        if let Some(session_id) = &self.session_id
            && !session_id.is_empty()
            && session_id.chars().take(513).count() <= 512
            && (metadata.len() < 16 || metadata.contains_key("session_id"))
        {
            metadata.insert("session_id".to_owned(), Value::String(session_id.clone()));
        }
        (!metadata.is_empty()).then_some(metadata)
    }

    /// Appends a complete model response as one assistant history message.
    ///
    /// Blocks are cloned without altering seals, extension data, or exact tool
    /// argument text, making same-protocol continuation cache-friendly.
    pub fn append_response(&mut self, response: &Response) {
        self.messages.push(response.to_message());
    }

    /// Visits every content block, including nested tool-result content.
    pub fn visit_blocks_mut(&mut self, visitor: &mut impl FnMut(&mut ContentBlock)) {
        visit_slice(&mut self.instructions, visitor);
        for message in &mut self.messages {
            visit_slice(&mut message.content, visitor);
        }
    }
}

pub(crate) fn visit_slice(
    blocks: &mut [ContentBlock],
    visitor: &mut impl FnMut(&mut ContentBlock),
) {
    for block in blocks {
        visitor(block);
        if let ContentBlock::ToolResult { content, .. } = block {
            visit_slice(content, visitor);
        }
    }
}
