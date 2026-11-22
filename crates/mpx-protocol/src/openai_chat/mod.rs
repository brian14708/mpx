//! `OpenAI` Chat Completions wire protocol support.

mod codec;
mod stream;
mod wire;

use crate::Protocol;

pub(crate) use codec::detect;

/// A synchronous, sans-IO codec for `OpenAI` Chat Completions.
///
/// The codec accepts both the conventional `/v1/chat/completions` endpoint and
/// Azure deployment URLs ending in `/chat/completions`. It supports complete
/// JSON requests and responses as well as incremental SSE translation.
#[derive(Debug, Clone)]
pub(crate) struct ChatCompletionsCodec;

impl ChatCompletionsCodec {
    pub(crate) const fn protocol() -> Protocol {
        Protocol::ChatCompletions
    }
}
