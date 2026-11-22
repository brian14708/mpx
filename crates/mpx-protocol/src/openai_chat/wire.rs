use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum WireRole {
    System,
    Developer,
    #[default]
    User,
    Assistant,
    Tool,
    Function,
    Unknown(String),
}

impl WireRole {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Function => "function",
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
            "system" => Self::System,
            "developer" => Self::Developer,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            "function" => Self::Function,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ContentPartKind {
    #[default]
    Text,
    OutputText,
    ImageUrl,
    InputImage,
    InputAudio,
    File,
    Refusal,
    Reasoning,
    ReasoningContent,
    Unknown(String),
}

impl ContentPartKind {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::OutputText => "output_text",
            Self::ImageUrl => "image_url",
            Self::InputImage => "input_image",
            Self::InputAudio => "input_audio",
            Self::File => "file",
            Self::Refusal => "refusal",
            Self::Reasoning => "reasoning",
            Self::ReasoningContent => "reasoning_content",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for ContentPartKind {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for ContentPartKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for ContentPartKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for ContentPartKind {
    fn from(value: String) -> Self {
        match value.as_str() {
            "text" => Self::Text,
            "output_text" => Self::OutputText,
            "image_url" => Self::ImageUrl,
            "input_image" => Self::InputImage,
            "input_audio" => Self::InputAudio,
            "file" => Self::File,
            "refusal" => Self::Refusal,
            "reasoning" => Self::Reasoning,
            "reasoning_content" => Self::ReasoningContent,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolCallKind {
    Function,
    Unknown(String),
}

impl ToolCallKind {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Function => "function",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for ToolCallKind {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for ToolCallKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for ToolCallKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for ToolCallKind {
    fn from(value: String) -> Self {
        if value == "function" {
            Self::Function
        } else {
            Self::Unknown(value)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    FunctionCall,
    ContentFilter,
    Refusal,
    Pause,
    Unknown(String),
}

impl FinishReason {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::FunctionCall => "function_call",
            Self::ContentFilter => "content_filter",
            Self::Refusal => "refusal",
            Self::Pause => "pause",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for FinishReason {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for FinishReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for FinishReason {
    fn from(value: String) -> Self {
        match value.as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "function_call" => Self::FunctionCall,
            "content_filter" => Self::ContentFilter,
            "refusal" => Self::Refusal,
            "pause" => Self::Pause,
            _ => Self::Unknown(value),
        }
    }
}

impl From<&crate::StopReason> for FinishReason {
    fn from(reason: &crate::StopReason) -> Self {
        match reason {
            crate::StopReason::EndTurn
            | crate::StopReason::StopSequence
            | crate::StopReason::Incomplete => Self::Stop,
            crate::StopReason::MaxTokens => Self::Length,
            crate::StopReason::ToolUse => Self::ToolCalls,
            crate::StopReason::PauseTurn => Self::Pause,
            crate::StopReason::Refusal => Self::Refusal,
            crate::StopReason::ContentFilter => Self::ContentFilter,
            crate::StopReason::Other(value) => Self::Unknown(value.clone()),
        }
    }
}

impl From<&FinishReason> for crate::StopReason {
    fn from(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Stop => Self::EndTurn,
            FinishReason::Length => Self::MaxTokens,
            FinishReason::ToolCalls | FinishReason::FunctionCall => Self::ToolUse,
            FinishReason::ContentFilter => Self::ContentFilter,
            FinishReason::Refusal => Self::Refusal,
            FinishReason::Pause => Self::PauseTurn,
            FinishReason::Unknown(value) => Self::Other(value.clone()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RequestWire {
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) messages: Vec<MessageWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<StreamOptionsWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<StopWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct StreamOptionsWire {
    #[serde(default)]
    pub(crate) include_usage: bool,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopWire {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct MessageWire {
    pub(crate) role: WireRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<MessageContentWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<ToolCallWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) audio: Option<AudioOutputWire>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContentWire {
    Text(String),
    Parts(Vec<ContentPartWire>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ContentPartWire {
    #[serde(rename = "type")]
    pub(crate) kind: ContentPartKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) image_url: Option<ImageUrlWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) input_audio: Option<InputAudioWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) file: Option<FileWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Value>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum ImageUrlWire {
    Url(String),
    Object(ImageUrlObjectWire),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ImageUrlObjectWire {
    pub(crate) url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct InputAudioWire {
    pub(crate) data: String,
    #[serde(default)]
    pub(crate) format: String,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct FileWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) file_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) filename: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct AudioOutputWire {
    #[serde(default)]
    pub(crate) data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) transcript: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) format: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ToolCallWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<ToolCallKind>,
    #[serde(default)]
    pub(crate) function: FunctionCallWire,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct FunctionCallWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) arguments: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ResponseWire {
    #[serde(default)]
    pub(crate) id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) object: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) created: Option<u64>,
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) choices: Vec<ChoiceWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<UsageWire>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ChoiceWire {
    #[serde(default)]
    pub(crate) index: u32,
    #[serde(default)]
    pub(crate) message: MessageWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<FinishReason>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ChunkWire {
    #[serde(default)]
    pub(crate) id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) object: Option<String>,
    #[serde(default)]
    pub(crate) created: u64,
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) choices: Vec<StreamChoiceWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<UsageWire>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct StreamChoiceWire {
    #[serde(default)]
    pub(crate) index: u32,
    #[serde(default)]
    pub(crate) delta: DeltaWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<FinishReason>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct DeltaWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<WireRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<DeltaToolCallWire>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct DeltaToolCallWire {
    #[serde(default)]
    pub(crate) index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<ToolCallKind>,
    #[serde(default)]
    pub(crate) function: FunctionCallWire,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct UsageWire {
    #[serde(default)]
    pub(crate) prompt_tokens: u64,
    #[serde(default)]
    pub(crate) completion_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_tokens_details: Option<PromptTokensDetailsWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) completion_tokens_details: Option<CompletionTokensDetailsWire>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct PromptTokensDetailsWire {
    #[serde(default)]
    pub(crate) cached_tokens: u64,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct CompletionTokensDetailsWire {
    #[serde(default)]
    pub(crate) reasoning_tokens: u64,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ErrorEnvelopeWire {
    pub(crate) error: ErrorBodyWire,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ErrorBodyWire {
    #[serde(default)]
    pub(crate) message: String,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) code: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) param: Option<Value>,
}
