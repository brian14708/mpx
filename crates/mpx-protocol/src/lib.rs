//! Editable protocol IR and sans-IO codecs for LLM gateways.
//!
//! The crate translates `OpenAI` Chat Completions, `OpenAI` Responses,
//! Anthropic Messages, Gemini `generateContent`, and Gemini Interactions without choosing an async
//! runtime.
//! [`Codec`] adds endpoint detection and `http` body glue; middleware can edit
//! the canonical [`Request`] and [`Response`] values between those operations.
//!
//! ```
//! use bytes::Bytes;
//! use mpx_protocol::{Codec, Protocol};
//!
//! let codec = Codec::default();
//! let raw = http::Request::post("/v1/chat/completions")
//!     .header("content-type", "application/json")
//!     .body(Bytes::from(r#"{"model":"gpt-5","messages":[{"role":"user","content":"Hello"}]}"#))?;
//! let request = codec.decode_request(raw)?;
//! let upstream = codec.encode_request(&request, Protocol::Messages)?;
//! assert_eq!(upstream.uri().path(), "/v1/messages");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Provider wire adapters intentionally use broad dispatch tables and
// open-ended extension fields. Keep those translations readable while the
// workspace's Rust and API lints remain enabled.
#![allow(
    clippy::large_enum_variant,
    clippy::match_same_arms,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]

mod accumulator;
mod anthropic_messages;
mod codec;
mod diagnostic;
mod error;
mod extra;
mod gemini;
mod interactions;
mod json;
mod message;
mod openai_chat;
mod openai_responses;
mod protocol;
mod protocol_codec;
mod request;
mod response;
mod route;
mod sse;
mod stream;
mod tool;
mod usage;
mod validate;

/// Traits and context for replacing a built-in codec through [`crate::CodecBuilder::codec`].
pub mod adapter {
    pub use crate::protocol_codec::{Capabilities, CodecContext, CodecOptions, ProtocolCodec};
    pub use crate::sse::{SseEvent, StreamDecoder, StreamEncoder};
}

pub use accumulator::ResponseAccumulator;
pub use codec::{Body, Codec, CodecBuilder};
pub use diagnostic::{Diagnostic, DiagnosticKind, Diagnostics, LossPolicy};
pub use error::{ApiError, ApiErrorKind, BoxError, Error, Result};
pub use extra::{Extra, ReasoningSeal};
pub use message::{
    ContentBlock, MediaSource, Message, Role, SearchResult, ToolArguments, ToolCallId,
};
pub use protocol::Protocol;
pub use protocol_codec::ClientContext;
pub use request::{Continuation, Request, ResponseId};
pub use response::{Response, ResponseBody, SafetyRating, StopReason};
pub use route::{Endpoint, Route};
pub use stream::{BlockDelta, EventStream, StreamEvent};
pub use tool::{OutputFormat, OutputFormatKind, ReasoningParams, SamplingParams, Tool, ToolChoice};
pub use usage::Usage;

pub(crate) use adapter::{CodecContext, ProtocolCodec, SseEvent, StreamDecoder, StreamEncoder};
