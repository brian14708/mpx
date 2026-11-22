use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ResponsesRequest {
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) instructions: Option<Value>,
    #[serde(default)]
    pub(crate) input: Value,
    #[serde(default)]
    pub(crate) tools: Option<Vec<Value>>,
    #[serde(default)]
    pub(crate) tool_choice: Option<Value>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) temperature: Option<f64>,
    #[serde(default)]
    pub(crate) top_p: Option<f64>,
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub(crate) metadata: Option<Map<String, Value>>,
    #[serde(default)]
    pub(crate) previous_response_id: Option<String>,
    #[serde(default)]
    pub(crate) store: Option<bool>,
    #[serde(default)]
    pub(crate) reasoning: Option<Value>,
    #[serde(default)]
    pub(crate) text: Option<Value>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ResponsesUsage {
    #[serde(default)]
    pub(crate) input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) input_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    pub(crate) output_tokens_details: Option<TokenDetails>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct TokenDetails {
    #[serde(default)]
    pub(crate) cached_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) reasoning_tokens: Option<u64>,
    #[serde(flatten)]
    pub(crate) extra: Map<String, Value>,
}

pub(crate) use crate::json::{take_bool, take_string};

pub(crate) fn extra_fields(map: Map<String, Value>) -> Option<crate::Extra> {
    crate::Extra::from_unknown(crate::Protocol::Responses, map)
}

pub(crate) fn json_error(error: serde_json::Error) -> crate::Error {
    crate::Error::json(crate::Protocol::Responses, error)
}
