use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WireRole {
    User,
    Model,
    Function,
    Unknown(String),
}

impl WireRole {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Model => "model",
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
            "user" => Self::User,
            "model" => Self::Model,
            "function" => Self::Function,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FunctionCallingMode {
    Auto,
    Any,
    None,
    Unknown(String),
}

impl FunctionCallingMode {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Auto => "AUTO",
            Self::Any => "ANY",
            Self::None => "NONE",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for FunctionCallingMode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for FunctionCallingMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for FunctionCallingMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for FunctionCallingMode {
    fn from(value: String) -> Self {
        match value.as_str() {
            "AUTO" => Self::Auto,
            "ANY" => Self::Any,
            "NONE" => Self::None,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Blocklist,
    ProhibitedContent,
    Spii,
    ImageSafety,
    ImageProhibitedContent,
    ImageRecitation,
    MalformedFunctionCall,
    NoImage,
    UnexpectedToolCall,
    Unknown(String),
}

impl FinishReason {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Stop => "STOP",
            Self::MaxTokens => "MAX_TOKENS",
            Self::Safety => "SAFETY",
            Self::Recitation => "RECITATION",
            Self::Language => "LANGUAGE",
            Self::Blocklist => "BLOCKLIST",
            Self::ProhibitedContent => "PROHIBITED_CONTENT",
            Self::Spii => "SPII",
            Self::ImageSafety => "IMAGE_SAFETY",
            Self::ImageProhibitedContent => "IMAGE_PROHIBITED_CONTENT",
            Self::ImageRecitation => "IMAGE_RECITATION",
            Self::MalformedFunctionCall => "MALFORMED_FUNCTION_CALL",
            Self::NoImage => "NO_IMAGE",
            Self::UnexpectedToolCall => "UNEXPECTED_TOOL_CALL",
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
            "STOP" => Self::Stop,
            "MAX_TOKENS" => Self::MaxTokens,
            "SAFETY" => Self::Safety,
            "RECITATION" => Self::Recitation,
            "LANGUAGE" => Self::Language,
            "BLOCKLIST" => Self::Blocklist,
            "PROHIBITED_CONTENT" => Self::ProhibitedContent,
            "SPII" => Self::Spii,
            "IMAGE_SAFETY" => Self::ImageSafety,
            "IMAGE_PROHIBITED_CONTENT" => Self::ImageProhibitedContent,
            "IMAGE_RECITATION" => Self::ImageRecitation,
            "MALFORMED_FUNCTION_CALL" => Self::MalformedFunctionCall,
            "NO_IMAGE" => Self::NoImage,
            "UNEXPECTED_TOOL_CALL" => Self::UnexpectedToolCall,
            _ => Self::Unknown(value),
        }
    }
}

impl From<&crate::StopReason> for FinishReason {
    fn from(reason: &crate::StopReason) -> Self {
        match reason {
            crate::StopReason::EndTurn
            | crate::StopReason::StopSequence
            | crate::StopReason::ToolUse => Self::Stop,
            crate::StopReason::MaxTokens => Self::MaxTokens,
            crate::StopReason::ContentFilter | crate::StopReason::Refusal => Self::Safety,
            crate::StopReason::PauseTurn | crate::StopReason::Incomplete => {
                Self::Unknown("OTHER".to_owned())
            }
            crate::StopReason::Other(value) => Self::Unknown(value.clone()),
        }
    }
}

impl From<&FinishReason> for crate::StopReason {
    fn from(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Stop => Self::EndTurn,
            FinishReason::MaxTokens => Self::MaxTokens,
            FinishReason::Safety
            | FinishReason::Recitation
            | FinishReason::Language
            | FinishReason::Blocklist
            | FinishReason::ProhibitedContent
            | FinishReason::Spii
            | FinishReason::ImageSafety
            | FinishReason::ImageProhibitedContent
            | FinishReason::ImageRecitation => Self::ContentFilter,
            FinishReason::MalformedFunctionCall
            | FinishReason::NoImage
            | FinishReason::UnexpectedToolCall => Self::Incomplete,
            FinishReason::Unknown(value) => Self::Other(value.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ResponseMimeType {
    Text,
    Json,
    Unknown(String),
}

impl ResponseMimeType {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Text => "text/plain",
            Self::Json => "application/json",
            Self::Unknown(value) => value,
        }
    }
}

impl AsRef<str> for ResponseMimeType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for ResponseMimeType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

impl<'de> Deserialize<'de> for ResponseMimeType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl From<String> for ResponseMimeType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "text/plain" => Self::Text,
            "application/json" => Self::Json,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerateContentRequest {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) contents: Vec<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) system_instruction: Option<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) tool_config: Option<ToolConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) generation_config: Option<GenerationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) model: Option<String>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Content {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) role: Option<WireRole>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) parts: Vec<Part>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Part {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "inline_data")]
    pub(super) inline_data: Option<Blob>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "file_data")]
    pub(super) file_data: Option<FileData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "function_call")]
    pub(super) function_call: Option<FunctionCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "function_response")]
    pub(super) function_response: Option<FunctionResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thought: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(alias = "thought_signature")]
    pub(super) thought_signature: Option<String>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Blob {
    #[serde(default)]
    #[serde(alias = "mime_type")]
    pub(super) mime_type: String,
    #[serde(default)]
    pub(super) data: String,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FileData {
    #[serde(default)]
    #[serde(alias = "mime_type")]
    pub(super) mime_type: String,
    #[serde(default)]
    #[serde(alias = "file_uri")]
    pub(super) file_uri: String,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) id: Option<String>,
    pub(super) name: String,
    #[serde(default = "empty_object")]
    pub(super) args: Value,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FunctionResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) id: Option<String>,
    pub(super) name: Option<String>,
    #[serde(default = "empty_object")]
    pub(super) response: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) parts: Vec<Part>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Tool {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) function_declarations: Vec<FunctionDeclaration>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FunctionDeclaration {
    pub(super) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) parameters_json_schema: Option<Value>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) function_calling_config: Option<FunctionCallingConfig>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FunctionCallingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mode: Option<FunctionCallingMode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) allowed_function_names: Vec<String>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) response_mime_type: Option<ResponseMimeType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) response_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) response_json_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thinking_config: Option<ThinkingConfig>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ThinkingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thinking_budget: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thinking_level: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) include_thoughts: Option<bool>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerateContentResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) candidates: Vec<Candidate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prompt_feedback: Option<PromptFeedback>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) usage_metadata: Option<UsageMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) model_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) response_id: Option<String>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Candidate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) safety_ratings: Vec<SafetyRating>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) index: Option<u32>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PromptFeedback {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) block_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) safety_ratings: Vec<SafetyRating>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SafetyRating {
    #[serde(default)]
    pub(super) category: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) probability: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) blocked: Option<bool>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct UsageMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prompt_token_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) candidates_token_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cached_content_token_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) thoughts_token_count: Option<u64>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ErrorEnvelope {
    pub(super) error: ProviderError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ProviderError {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) code: Option<u16>,
    #[serde(default)]
    pub(super) message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) status: Option<String>,
    #[serde(default, flatten)]
    pub(super) extra: Map<String, Value>,
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}
