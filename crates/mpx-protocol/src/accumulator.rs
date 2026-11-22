use std::collections::HashMap;

use crate::{ApiError, BlockDelta, ContentBlock, Response, StopReason, StreamEvent, Usage};

/// Builds a complete response from normalized stream events.
///
/// The accumulator also supports continuation segments. Usage sums across
/// segments, non-zero reports overwrite within a segment, and the first text
/// block after a `MaxTokens` cut can continue the preceding block of that kind.
#[derive(Debug, Default, Clone)]
pub struct ResponseAccumulator {
    response: Response,
    completed_usage: Usage,
    blocks: HashMap<u32, BlockPosition>,
    continue_last_block: bool,
    error: Option<ApiError>,
}

#[derive(Debug, Clone, Copy)]
struct BlockPosition {
    position: usize,
    complete_arguments: bool,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one event.
    pub fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::MessageStart { id, model, extra } => {
                if !id.is_empty() {
                    self.response.id = id;
                }
                if !model.is_empty() {
                    self.response.model = model;
                }
                if extra.is_some() {
                    self.response.extra = extra;
                }
            }
            StreamEvent::BlockStart { index, block } => self.start_block(index, block),
            StreamEvent::BlockDelta { index, delta } => self.apply_delta(index, delta),
            StreamEvent::BlockStop { .. } | StreamEvent::Ping => {}
            StreamEvent::Usage { usage } => self.response.usage.merge_nonzero(&usage),
            StreamEvent::ResponseMetadata { extra } => {
                if let Some(current) = self.response.extra.as_mut()
                    && current.protocol == extra.protocol
                {
                    current.fields.extend(extra.fields);
                } else {
                    self.response.extra = Some(extra);
                }
            }
            StreamEvent::Safety { ratings } => self.response.safety.extend(ratings),
            StreamEvent::MessageStop { stop_reason } => {
                self.response.stop_reason = stop_reason;
            }
            StreamEvent::Error { error } => self.error = Some(error),
        }
    }

    /// Extends the accumulator with an event iterator.
    pub fn extend(&mut self, events: impl IntoIterator<Item = StreamEvent>) {
        for event in events {
            self.push(event);
        }
    }

    /// Starts another upstream segment after continuation.
    ///
    /// Existing content is retained. The next compatible block can merge into
    /// the final block if the previous segment stopped at `MaxTokens`.
    pub fn next_segment(&mut self) {
        self.completed_usage.add_segment(&self.response.usage);
        self.response.usage = Usage::default();
        self.blocks.clear();
        self.continue_last_block = self.response.stop_reason == StopReason::MaxTokens;
    }

    /// Returns the first stream error, if one occurred.
    #[must_use]
    pub const fn error(&self) -> Option<&ApiError> {
        self.error.as_ref()
    }

    /// Finishes accumulation and returns the complete response.
    #[must_use]
    pub fn finish(mut self) -> Response {
        let mut usage = self.completed_usage;
        usage.add_segment(&self.response.usage);
        self.response.usage = usage;
        self.response
    }

    fn start_block(&mut self, index: u32, block: ContentBlock) {
        if self.continue_last_block
            && self.blocks.is_empty()
            && let Some((last_index, last)) = self.response.content.iter_mut().enumerate().last()
            && can_continue(last, &block)
        {
            append_initial(last, &block);
            self.blocks.insert(
                index,
                BlockPosition {
                    position: last_index,
                    complete_arguments: false,
                },
            );
            self.continue_last_block = false;
            return;
        }
        self.continue_last_block = false;
        let position = self.response.content.len();
        let complete_arguments = matches!(
            &block,
            ContentBlock::ToolCall { arguments, .. }
                if !arguments.as_str().is_empty() && arguments.as_str() != "{}"
        );
        self.blocks.insert(
            index,
            BlockPosition {
                position,
                complete_arguments,
            },
        );
        self.response.content.push(block);
    }

    fn apply_delta(&mut self, index: u32, delta: BlockDelta) {
        if !self.blocks.contains_key(&index) {
            self.start_block(index, block_for_delta(&delta));
        }
        let Some(BlockPosition {
            position,
            complete_arguments,
        }) = self.blocks.get(&index).copied()
        else {
            return;
        };
        let Some(block) = self.response.content.get_mut(position) else {
            return;
        };
        match (block, delta) {
            (ContentBlock::Text { text, .. }, BlockDelta::Text(value))
            | (ContentBlock::Refusal { text, .. }, BlockDelta::Refusal(value))
            | (ContentBlock::Reasoning { text, .. }, BlockDelta::Reasoning(value)) => {
                text.push_str(&value);
            }
            (ContentBlock::Reasoning { summary, .. }, BlockDelta::ReasoningSummary(value)) => {
                if let Some(last) = summary.last_mut() {
                    last.push_str(&value);
                } else {
                    summary.push(value);
                }
            }
            (ContentBlock::ToolCall { arguments, .. }, BlockDelta::ToolArguments(value))
                if !complete_arguments =>
            {
                let arguments = arguments.as_mut_string();
                if arguments == "{}" {
                    arguments.clear();
                }
                arguments.push_str(&value);
            }
            (
                ContentBlock::Reasoning { seal, .. } | ContentBlock::ToolCall { seal, .. },
                BlockDelta::Seal(value),
            ) => {
                *seal = Some(value);
            }
            (ContentBlock::Reasoning { redacted, .. }, BlockDelta::Redacted) => *redacted = true,
            (_, BlockDelta::Native { .. }) => {}
            _ => {}
        }
    }
}

const fn can_continue(last: &ContentBlock, next: &ContentBlock) -> bool {
    matches!(
        (last, next),
        (ContentBlock::Text { .. }, ContentBlock::Text { .. })
            | (
                ContentBlock::Reasoning { .. },
                ContentBlock::Reasoning { .. }
            )
            | (ContentBlock::Refusal { .. }, ContentBlock::Refusal { .. })
    )
}

fn append_initial(last: &mut ContentBlock, next: &ContentBlock) {
    match (last, next) {
        (ContentBlock::Text { text, .. }, ContentBlock::Text { text: value, .. })
        | (ContentBlock::Reasoning { text, .. }, ContentBlock::Reasoning { text: value, .. })
        | (ContentBlock::Refusal { text, .. }, ContentBlock::Refusal { text: value, .. }) => {
            text.push_str(value);
        }
        _ => {}
    }
}

fn block_for_delta(delta: &BlockDelta) -> ContentBlock {
    match delta {
        BlockDelta::Text(_) => ContentBlock::text(""),
        BlockDelta::Reasoning(_)
        | BlockDelta::ReasoningSummary(_)
        | BlockDelta::Seal(_)
        | BlockDelta::Redacted => ContentBlock::reasoning(),
        BlockDelta::ToolArguments(_) => ContentBlock::tool_call("", "", ""),
        BlockDelta::Refusal(_) => ContentBlock::Refusal {
            text: String::new(),
            extra: None,
        },
        BlockDelta::Native { protocol, json } => ContentBlock::Native {
            protocol: *protocol,
            json: json.clone(),
        },
    }
}
