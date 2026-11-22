use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WireRole {
    User,
    Assistant,
    System,
    Developer,
    Unknown(String),
}

impl WireRole {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Developer => "developer",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for WireRole {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for WireRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for WireRole {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for WireRole {
    fn from(value: String) -> Self {
        match value.as_str() {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "developer" => Self::Developer,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    MaxTokens,
    ToolUse,
    PauseTurn,
    StopSequence,
    Refusal,
    EndTurn,
    Unknown(String),
}

impl StopReason {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::ToolUse => "tool_use",
            Self::PauseTurn => "pause_turn",
            Self::StopSequence => "stop_sequence",
            Self::Refusal => "refusal",
            Self::EndTurn => "end_turn",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for StopReason {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for StopReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for StopReason {
    fn from(value: String) -> Self {
        match value.as_str() {
            "max_tokens" => Self::MaxTokens,
            "tool_use" => Self::ToolUse,
            "pause_turn" => Self::PauseTurn,
            "stop_sequence" => Self::StopSequence,
            "refusal" => Self::Refusal,
            "end_turn" => Self::EndTurn,
            _ => Self::Unknown(value),
        }
    }
}

impl From<&crate::StopReason> for StopReason {
    fn from(reason: &crate::StopReason) -> Self {
        match reason {
            crate::StopReason::MaxTokens => Self::MaxTokens,
            crate::StopReason::ToolUse => Self::ToolUse,
            crate::StopReason::PauseTurn => Self::PauseTurn,
            crate::StopReason::StopSequence => Self::StopSequence,
            crate::StopReason::Refusal => Self::Refusal,
            crate::StopReason::EndTurn
            | crate::StopReason::ContentFilter
            | crate::StopReason::Incomplete => Self::EndTurn,
            crate::StopReason::Other(value) => Self::Unknown(value.clone()),
        }
    }
}

impl From<&StopReason> for crate::StopReason {
    fn from(reason: &StopReason) -> Self {
        match reason {
            StopReason::MaxTokens => Self::MaxTokens,
            StopReason::ToolUse => Self::ToolUse,
            StopReason::PauseTurn => Self::PauseTurn,
            StopReason::StopSequence => Self::StopSequence,
            StopReason::Refusal => Self::Refusal,
            StopReason::EndTurn => Self::EndTurn,
            StopReason::Unknown(value) => Self::Other(value.clone()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MessagesRequest {
    pub(crate) model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    pub(crate) messages: Vec<WireMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) system: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<Value>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WireMessage {
    pub(crate) role: WireRole,
    pub(crate) content: Value,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MessagesResponse {
    #[serde(default)]
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) content: Vec<Value>,
    #[serde(default)]
    pub(crate) stop_reason: Option<StopReason>,
    #[serde(default)]
    pub(crate) usage: Option<WireUsage>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct WireUsage {
    #[serde(default)]
    pub(crate) input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: Option<u64>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

pub(crate) use crate::json::{take_bool, take_string};

pub(crate) fn extra_fields(map: Map<String, Value>) -> Option<crate::Extra> {
    crate::Extra::from_unknown(crate::Protocol::Messages, map)
}

pub(crate) fn json_error(error: serde_json::Error) -> crate::Error {
    crate::Error::json(crate::Protocol::Messages, error)
}
