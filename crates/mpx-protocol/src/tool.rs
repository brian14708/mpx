use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Extra, Protocol};

pub(crate) fn synthesized_tool_call_id(index: impl std::fmt::Display) -> String {
    format!("call_{index}")
}

/// A client- or provider-hosted tool declaration.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    /// A JSON-schema function tool.
    Function {
        /// Function name.
        name: String,
        /// Human- and model-readable purpose.
        description: Option<String>,
        /// JSON Schema for the arguments.
        parameters: Value,
        /// Whether strict schema adherence was requested.
        strict: Option<bool>,
        /// Whether a tool-search provider may defer loading this definition.
        defer_loading: Option<bool>,
        /// Unmodeled tool fields.
        extra: Option<Extra>,
    },
    /// A free-form custom tool.
    Custom {
        /// Tool name.
        name: String,
        /// Human- and model-readable purpose.
        description: Option<String>,
        /// Origin-specific input format declaration.
        format: Option<Value>,
        /// Unmodeled tool fields.
        extra: Option<Extra>,
    },
    /// A provider-hosted tool preserved as opaque wire data.
    Hosted {
        /// Protocol that understands this tool.
        protocol: Protocol,
        /// Complete original tool declaration.
        json: Value,
    },
}

impl Tool {
    /// Creates a function declaration with no optional fields.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: Some(description.into()),
            parameters,
            strict: None,
            defer_loading: None,
            extra: None,
        }
    }
}

/// Canonical tool-selection policy.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model may call any offered tool or answer directly.
    Auto,
    /// Tool calls are disabled.
    None,
    /// At least one tool call is required.
    Required,
    /// One named function must be called.
    Named(String),
    /// An origin-specific selection policy.
    Native {
        /// Protocol that understands this value.
        protocol: Protocol,
        /// Complete original value.
        json: Value,
    },
}

/// Normalized sampling controls.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Maximum generated tokens.
    pub max_output_tokens: Option<u32>,
    /// Randomness control.
    pub temperature: Option<f64>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f64>,
    /// Number of highest-probability tokens considered.
    pub top_k: Option<u32>,
    /// Sequences that end generation.
    #[builder(default)]
    pub stop_sequences: Vec<String>,
    /// Deterministic generation seed where supported.
    pub seed: Option<i64>,
    /// `OpenAI` presence penalty.
    pub presence_penalty: Option<f64>,
    /// `OpenAI` frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Unmodeled sampling controls.
    pub extra: Option<Extra>,
    // Retain the wire spelling for Chat-compatible routes that use the legacy field.
    #[builder(skip)]
    #[serde(skip)]
    pub(crate) chat_legacy_max_tokens: bool,
}

/// Reasoning controls shared by the model APIs.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Builder, Serialize, Deserialize)]
pub struct ReasoningParams {
    /// Provider-neutral effort label such as `low`, `medium`, or `high`.
    #[builder(into)]
    pub effort: Option<String>,
    /// Legacy fixed reasoning-token budget.
    pub budget_tokens: Option<u32>,
    /// Requested summary mode.
    #[builder(into)]
    pub summary: Option<String>,
    /// Unmodeled reasoning controls.
    pub extra: Option<Extra>,
}

/// The broad shape of constrained model output.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormatKind {
    /// Unconstrained text.
    Text,
    /// A JSON object without a supplied schema.
    JsonObject,
    /// JSON constrained by a schema.
    JsonSchema,
}

/// Structured-output requirements.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Builder, Serialize, Deserialize)]
pub struct OutputFormat {
    /// Broad output shape.
    pub kind: OutputFormatKind,
    /// Schema for [`OutputFormatKind::JsonSchema`].
    pub schema: Option<Value>,
    /// Schema name required by some `OpenAI` routes.
    #[builder(into)]
    pub name: Option<String>,
    /// Whether strict schema adherence was requested.
    pub strict: Option<bool>,
    /// Unmodeled format fields.
    pub extra: Option<Extra>,
}
