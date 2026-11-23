use core::fmt;

use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{Extra, Protocol, ReasoningSeal};

/// A stable identifier joining a tool call to its result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Creates an identifier from provider or synthesized text.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the newtype and returns its text.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolCallId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolCallId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl AsRef<str> for ToolCallId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Exact JSON text of a tool call's arguments.
///
/// Keeping the original text avoids re-escaping arguments during replay and
/// continuation. Use [`ToolArguments::value`] when structured access is needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolArguments(String);

impl ToolArguments {
    /// Creates arguments from their JSON representation.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// Returns the original JSON text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parses the arguments as JSON.
    pub fn value(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::from_str(&self.0)
    }

    /// Replaces the arguments with a JSON value's compact representation.
    pub fn set(&mut self, value: &serde_json::Value) {
        self.0 = value.to_string();
    }

    pub(crate) const fn as_mut_string(&mut self) -> &mut String {
        &mut self.0
    }

    /// Consumes the newtype and returns the exact argument text.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl Default for ToolArguments {
    fn default() -> Self {
        Self("{}".to_owned())
    }
}

impl From<String> for ToolArguments {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolArguments {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl AsRef<str> for ToolArguments {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ToolArguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Conversation role in the canonical request history.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A system instruction that remains at its conversation position.
    System,
    /// `OpenAI` developer instructions.
    Developer,
    /// User-provided input.
    User,
    /// Model-generated output.
    Assistant,
    /// Output from a previously requested tool.
    Tool,
}

/// Location and encoding of media content.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    /// A remote URL.
    Url {
        /// Media location.
        url: String,
        /// MIME type when supplied by the client.
        media_type: Option<String>,
    },
    /// Base64-encoded inline bytes.
    Base64 {
        /// MIME type of the decoded bytes.
        media_type: String,
        /// Base64 payload without a data-URL prefix.
        data: String,
    },
    /// A provider-hosted file reference.
    File {
        /// Provider file identifier.
        file_id: String,
        /// MIME type when known.
        media_type: Option<String>,
    },
}

/// One item in normalized web-search output.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title.
    #[builder(into)]
    pub title: Option<String>,
    /// Result URL.
    #[builder(into)]
    pub url: Option<String>,
    /// Snippet or page text.
    #[builder(into)]
    pub text: Option<String>,
    /// Unmodeled origin-protocol fields.
    pub extra: Option<Extra>,
}

/// A canonical content block shared by requests, responses, and streams.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Visible text.
    Text {
        /// Text contents.
        text: String,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// An image input or output.
    Image {
        /// Image location or bytes.
        source: MediaSource,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// A file or document.
    File {
        /// File location or bytes.
        source: MediaSource,
        /// Original filename, when supplied.
        filename: Option<String>,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// Audio content.
    Audio {
        /// Audio location or bytes.
        source: MediaSource,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// Visible reasoning, a summary, or redacted reasoning.
    Reasoning {
        /// Reasoning text, empty for summary-only or redacted blocks.
        text: String,
        /// Provider-generated reasoning summaries.
        summary: Vec<String>,
        /// Opaque provider signature or encrypted content.
        seal: Option<ReasoningSeal>,
        /// Whether the provider intentionally redacted this reasoning.
        redacted: bool,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// A model request to invoke a client tool.
    ToolCall {
        /// Identifier used by the subsequent result.
        id: ToolCallId,
        /// Tool name.
        name: String,
        /// Exact JSON argument text.
        arguments: ToolArguments,
        /// Reasoning seal attached to this call by Gemini.
        seal: Option<ReasoningSeal>,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// A client tool's result.
    ToolResult {
        /// Identifier of the matching call.
        call_id: ToolCallId,
        /// Function name, required by Gemini and recovered from history elsewhere.
        name: Option<String>,
        /// Result content.
        content: Vec<Self>,
        /// Whether execution failed.
        is_error: bool,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// Results of a provider-hosted web search.
    WebSearchResult {
        /// Search query when exposed by the provider.
        query: Option<String>,
        /// Normalized search results.
        results: Vec<SearchResult>,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// Model refusal text.
    Refusal {
        /// Refusal explanation.
        text: String,
        /// Unmodeled fields on the original block.
        extra: Option<Extra>,
    },
    /// A block meaningful only to its origin protocol.
    Native {
        /// Origin protocol.
        protocol: Protocol,
        /// Complete original wire block.
        json: serde_json::Value,
    },
}

impl ContentBlock {
    /// Visits this block and every nested tool-result block.
    pub fn visit_mut(&mut self, visitor: &mut impl FnMut(&mut Self)) {
        crate::request::visit_slice(std::slice::from_mut(self), visitor);
    }

    /// Creates a text block with no extension fields.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            extra: None,
        }
    }

    /// Creates an empty reasoning block.
    #[must_use]
    pub const fn reasoning() -> Self {
        Self::Reasoning {
            text: String::new(),
            summary: Vec::new(),
            seal: None,
            redacted: false,
            extra: None,
        }
    }

    /// Creates a tool call with no seal or extension fields.
    pub fn tool_call(
        id: impl Into<ToolCallId>,
        name: impl Into<String>,
        arguments: impl Into<ToolArguments>,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            seal: None,
            extra: None,
        }
    }

    /// Returns a short content-kind name useful in diagnostics.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Image { .. } => "image",
            Self::File { .. } => "file",
            Self::Audio { .. } => "audio",
            Self::Reasoning { .. } => "reasoning",
            Self::ToolCall { .. } => "tool_call",
            Self::ToolResult { .. } => "tool_result",
            Self::WebSearchResult { .. } => "web_search_result",
            Self::Refusal { .. } => "refusal",
            Self::Native { .. } => "native",
        }
    }
}

/// One canonical conversation message.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Builder, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// Ordered content blocks.
    #[builder(default)]
    pub content: Vec<ContentBlock>,
    /// Unmodeled message-level fields.
    pub extra: Option<Extra>,
}

impl Message {
    /// Creates a plain-text message with no extension fields.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::text(text)],
            extra: None,
        }
    }

    pub(crate) fn split_tool_results(self, output: &mut Vec<Self>) {
        if self.role != Role::User {
            output.push(self);
            return;
        }
        let mut role = Role::User;
        let mut content = Vec::new();
        for block in self.content {
            let next_role = if matches!(block, ContentBlock::ToolResult { .. }) {
                Role::Tool
            } else {
                Role::User
            };
            if next_role != role && !content.is_empty() {
                output.push(Self {
                    role,
                    content: std::mem::take(&mut content),
                    extra: None,
                });
            }
            role = next_role;
            content.push(block);
        }
        if !content.is_empty() {
            output.push(Self {
                role,
                content,
                extra: self.extra,
            });
        }
    }
}
