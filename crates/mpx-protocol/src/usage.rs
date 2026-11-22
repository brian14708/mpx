use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::Extra;

/// Normalized token accounting.
///
/// `input_tokens` excludes cached input tokens. Cache-read, cache-write, and
/// reasoning counts are separate so providers need not agree on a total.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input tokens.
    #[builder(default)]
    pub input_tokens: u64,
    /// Generated output tokens.
    #[builder(default)]
    pub output_tokens: u64,
    /// Input tokens served from a provider cache.
    #[builder(default)]
    pub cached_input_tokens: u64,
    /// Input tokens written to a provider cache.
    #[builder(default)]
    pub cache_creation_input_tokens: u64,
    /// Output tokens attributed to reasoning.
    #[builder(default)]
    pub reasoning_tokens: u64,
    /// Provider-specific accounting not modeled above.
    pub extra: Option<Extra>,
}

impl Usage {
    /// Merges reports within one segment using non-zero-overwrites semantics.
    pub fn merge_nonzero(&mut self, other: &Self) {
        merge_field(&mut self.input_tokens, other.input_tokens);
        merge_field(&mut self.output_tokens, other.output_tokens);
        merge_field(&mut self.cached_input_tokens, other.cached_input_tokens);
        merge_field(
            &mut self.cache_creation_input_tokens,
            other.cache_creation_input_tokens,
        );
        merge_field(&mut self.reasoning_tokens, other.reasoning_tokens);
        if other.extra.is_some() {
            self.extra.clone_from(&other.extra);
        }
    }

    /// Adds another segment's usage to continuation totals.
    pub fn add_segment(&mut self, other: &Self) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(other.cache_creation_input_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        if other.extra.is_some() {
            self.extra.clone_from(&other.extra);
        }
    }

    /// Returns whether every normalized count is zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cached_input_tokens == 0
            && self.cache_creation_input_tokens == 0
            && self.reasoning_tokens == 0
    }
}

const fn merge_field(target: &mut u64, value: u64) {
    if value != 0 {
        *target = value;
    }
}
