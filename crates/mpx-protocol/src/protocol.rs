use core::fmt;

use serde::{Deserialize, Serialize};

/// A supported model API family.
///
/// Replacement codecs reuse one of these values while changing the wire
/// dialect. New families may be added without a breaking change.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// `OpenAI` Chat Completions.
    #[default]
    ChatCompletions,
    /// `OpenAI` Responses.
    Responses,
    /// Anthropic Messages.
    Messages,
    /// Gemini `generateContent` and `streamGenerateContent`.
    GenerateContent,
    /// Gemini Interactions.
    Interactions,
}

impl Protocol {
    /// Returns the capabilities of the built-in codec for this family.
    #[must_use]
    pub const fn capabilities(self) -> crate::adapter::Capabilities {
        crate::adapter::Capabilities {
            assistant_prefill: matches!(
                self,
                Self::ChatCompletions | Self::Messages | Self::GenerateContent
            ),
            server_continuation: matches!(self, Self::Responses | Self::Interactions),
            pause_turn: matches!(self, Self::Messages),
            streaming: true,
            tool_calls: true,
            reasoning: true,
        }
    }

    /// Returns the stable, lowercase protocol name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::Responses => "responses",
            Self::Messages => "messages",
            Self::GenerateContent => "generate_content",
            Self::Interactions => "interactions",
        }
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for Protocol {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
