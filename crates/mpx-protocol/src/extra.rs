use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{CodecContext, DiagnosticKind, Protocol, Result, json};

/// Unmodeled fields retained for a lossless return to their origin protocol.
///
/// Encoders emit these fields only when the target protocol matches
/// [`Extra::protocol`]. Cross-protocol encoders report their removal instead.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct Extra {
    /// Protocol whose wire object owns these fields.
    pub protocol: Protocol,
    /// Original fields not modeled explicitly by the IR.
    #[builder(default)]
    pub fields: Map<String, Value>,
    // Preserve Gemini's schema field on replay without treating it as an unmodeled field.
    #[builder(skip)]
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(crate) gemini_legacy_schema: bool,
}

impl Extra {
    /// Creates an extension bag for `protocol`.
    #[must_use]
    pub fn new(protocol: Protocol) -> Self {
        Self::with_fields(protocol, Map::new())
    }

    /// Creates an extension bag from original wire fields.
    #[must_use]
    pub const fn with_fields(protocol: Protocol, fields: Map<String, Value>) -> Self {
        Self {
            protocol,
            fields,
            gemini_legacy_schema: false,
        }
    }

    /// Returns whether no unmodeled fields are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Wraps unmodeled wire fields, or returns `None` when there are none.
    pub(crate) fn from_unknown(protocol: Protocol, fields: Map<String, Value>) -> Option<Self> {
        (!fields.is_empty()).then(|| Self::with_fields(protocol, fields))
    }

    /// Returns the fields `extra` may emit into `target`'s wire object.
    ///
    /// Non-empty fields owned by another protocol are dropped with a
    /// diagnostic.
    pub(crate) fn fields_for(
        extra: Option<&Self>,
        target: Protocol,
        cx: &CodecContext,
        path: &str,
    ) -> Result<Map<String, Value>> {
        match extra {
            Some(extra) if extra.protocol == target => Ok(extra.fields.clone()),
            Some(extra) if !extra.fields.is_empty() => {
                cx.diagnose(
                    DiagnosticKind::Dropped,
                    target,
                    path,
                    format!("extension fields belong to {}", extra.protocol),
                )?;
                Ok(Map::new())
            }
            _ => Ok(Map::new()),
        }
    }

    /// Merges the fields `extra` may emit into `target` into a wire object,
    /// without overwriting modeled fields.
    pub(crate) fn merge_into(
        value: Value,
        extra: Option<&Self>,
        target: Protocol,
        cx: &CodecContext,
        path: &str,
    ) -> Result<Value> {
        let Value::Object(mut map) = value else {
            return Ok(value);
        };
        json::merge_missing(&mut map, Self::fields_for(extra, target, cx, path)?);
        Ok(Value::Object(map))
    }
}

/// A reasoning signature or encrypted reasoning payload.
///
/// The value is opaque. Its origin protocol controls whether it can be emitted
/// directly or must be sealed into a tool-call identifier.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct ReasoningSeal {
    /// Protocol that created the opaque value.
    pub protocol: Protocol,
    /// Signature, `thoughtSignature`, or encrypted reasoning text.
    #[builder(into)]
    pub value: String,
}

impl ReasoningSeal {
    /// Creates an opaque seal for `protocol`.
    pub fn new(protocol: Protocol, value: impl Into<String>) -> Self {
        Self {
            protocol,
            value: value.into(),
        }
    }
}
