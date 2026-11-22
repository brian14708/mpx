use std::sync::{Arc, Mutex};

use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::Protocol;

/// How strictly a translation treats information loss.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossPolicy {
    /// Record loss and continue with the closest valid representation.
    #[default]
    Warn,
    /// Turn every loss diagnostic into a translation error.
    Reject,
}

/// The kind of change made while translating between protocols.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
    /// Data could not be represented and was omitted.
    Dropped,
    /// Data was retained with changed or approximate semantics.
    Approximated,
    /// Data required by the target was deterministically synthesized.
    Synthesized,
    /// A less expressive fallback representation was used.
    Fallback,
}

/// One observable translation warning.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Kind of translation change.
    pub kind: DiagnosticKind,
    /// Source or target protocol associated with the change.
    pub protocol: Protocol,
    /// JSON-style path to the affected value.
    #[builder(into)]
    pub path: String,
    /// Explanation of the change.
    #[builder(into)]
    pub message: String,
}

/// A cheap, shared append-only diagnostics handle.
///
/// A handle is placed in request and response extensions. Stream segments and
/// continuation layers clone it so all warnings remain in one invocation log.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    entries: Arc<Mutex<Vec<Diagnostic>>>,
}

impl Diagnostics {
    /// Creates an empty diagnostics log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a diagnostic to the shared log.
    pub fn push(&self, diagnostic: Diagnostic) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.push(diagnostic);
    }

    /// Returns an owned snapshot of all diagnostics recorded so far.
    #[must_use]
    pub fn entries(&self) -> Vec<Diagnostic> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Returns whether no diagnostics have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// Returns the number of recorded diagnostics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}
