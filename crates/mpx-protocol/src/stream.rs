use core::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::{Stream, TryStreamExt, stream};
use serde::{Deserialize, Serialize};
use std::iter::FromIterator;

use crate::{ApiError, ContentBlock, Extra, ReasoningSeal, Result, StopReason, Usage};

/// An incremental change to an open content block.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum BlockDelta {
    /// Append visible text.
    Text(String),
    /// Append reasoning text.
    Reasoning(String),
    /// Append a reasoning summary.
    ReasoningSummary(String),
    /// Append exact tool-argument JSON text.
    ToolArguments(String),
    /// Replace or attach a reasoning seal.
    Seal(ReasoningSeal),
    /// Mark reasoning as redacted.
    Redacted,
    /// Append refusal text.
    Refusal(String),
    /// An origin-specific delta not modeled above.
    Native {
        /// Origin protocol.
        protocol: crate::Protocol,
        /// Original delta object.
        json: serde_json::Value,
    },
}

/// One protocol-independent streaming event.
///
/// Indices identify blocks within one upstream segment. Client-facing encoders
/// assign their own indices and identifiers rather than copying these values.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Start of a model response.
    MessageStart {
        /// Upstream response identifier.
        id: String,
        /// Upstream model identifier.
        model: String,
        /// Unmodeled initial response fields.
        extra: Option<Extra>,
    },
    /// Opens a content block at a segment-local index.
    BlockStart {
        /// Segment-local block index.
        index: u32,
        /// Initial block state.
        block: ContentBlock,
    },
    /// Changes an open content block.
    BlockDelta {
        /// Segment-local block index.
        index: u32,
        /// Incremental content.
        delta: BlockDelta,
    },
    /// Closes a content block.
    BlockStop {
        /// Segment-local block index.
        index: u32,
    },
    /// Reports usage at stream end.
    Usage {
        /// Latest usage report.
        usage: Usage,
    },
    /// Supplements response metadata learned after the stream began.
    ResponseMetadata {
        /// Provider-specific response fields.
        extra: Extra,
    },
    /// Provider safety ratings associated with the response.
    Safety {
        /// Ratings as reported by the provider.
        ratings: Vec<crate::SafetyRating>,
    },
    /// Ends the response after all blocks and usage.
    MessageStop {
        /// Terminal generation reason.
        stop_reason: StopReason,
    },
    /// A provider or transport error represented in the stream.
    Error {
        /// Normalized error data.
        error: ApiError,
    },
    /// A provider keep-alive with no model semantics.
    Ping,
}

/// A named, boxed `Send` stream of normalized events.
///
/// Middleware can map or collect this stream without depending on an async
/// runtime. Use [`EventStream::from_stream`] to wrap any compatible stream.
pub struct EventStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 'static>>,
}

impl EventStream {
    /// Wraps any `Send` stream of normalized events.
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Creates a stream from an owned event sequence.
    #[must_use]
    pub fn from_events(events: Vec<StreamEvent>) -> Self {
        Self::from_stream(stream::iter(events.into_iter().map(Ok)))
    }

    /// Creates an empty event stream.
    #[must_use]
    pub fn empty() -> Self {
        Self::from_stream(stream::empty())
    }

    /// Maps events while preserving asynchronous backpressure and errors.
    #[must_use]
    pub fn map<F>(self, map: F) -> Self
    where
        F: FnMut(StreamEvent) -> StreamEvent + Send + 'static,
    {
        Self::from_stream(self.inner.map_ok(map))
    }

    /// Collects all events into a vector.
    pub async fn into_events(self) -> Result<Vec<StreamEvent>> {
        self.inner.try_collect().await
    }
}

impl Default for EventStream {
    fn default() -> Self {
        Self::empty()
    }
}

impl FromIterator<StreamEvent> for EventStream {
    fn from_iter<T: IntoIterator<Item = StreamEvent>>(iter: T) -> Self {
        Self::from_events(iter.into_iter().collect())
    }
}

impl Stream for EventStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl core::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventStream").finish_non_exhaustive()
    }
}
