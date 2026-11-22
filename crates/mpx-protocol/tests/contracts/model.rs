use mpx_protocol::{BlockDelta, ContentBlock, ResponseAccumulator, StopReason, StreamEvent, Usage};
use mpx_protocol::{Codec, Protocol, ResponseBody};
use serde_json::json;

use super::body;

#[test]
fn usage_nonzero_merge_is_idempotent() {
    for (first, second, expected) in [
        (0, 0, 0),
        (0, 1, 1),
        (u64::MAX, 0, u64::MAX),
        (u64::MAX, 1, 1),
    ] {
        let mut merged = Usage::builder().input_tokens(first).build();
        let second = Usage::builder().input_tokens(second).build();
        merged.merge_nonzero(&second);
        assert_eq!(merged.input_tokens, expected);
        merged.merge_nonzero(&second);
        assert_eq!(merged.input_tokens, expected);
        merged.merge_nonzero(&Usage::default());
        assert_eq!(merged.input_tokens, expected);
    }
}

#[test]
fn accumulator_merges_segments_and_renumbers_independent_indices() {
    let mut accumulator = ResponseAccumulator::new();
    accumulator.extend([
        StreamEvent::MessageStart {
            id: "r".to_owned(),
            model: "m".to_owned(),
            extra: None,
        },
        StreamEvent::BlockStart {
            index: 9,
            block: ContentBlock::text("one"),
        },
        StreamEvent::BlockDelta {
            index: 9,
            delta: BlockDelta::Text(" one".to_owned()),
        },
        StreamEvent::BlockStop { index: 9 },
        StreamEvent::Usage {
            usage: Usage::builder().input_tokens(2).output_tokens(3).build(),
        },
        StreamEvent::MessageStop {
            stop_reason: StopReason::MaxTokens,
        },
    ]);
    accumulator.next_segment();
    accumulator.extend([
        StreamEvent::MessageStart {
            id: "r2".to_owned(),
            model: "m".to_owned(),
            extra: None,
        },
        StreamEvent::BlockStart {
            index: 77,
            block: ContentBlock::text(" continued"),
        },
        StreamEvent::BlockDelta {
            index: 77,
            delta: BlockDelta::Text("!".to_owned()),
        },
        StreamEvent::BlockStop { index: 77 },
        StreamEvent::Usage {
            usage: Usage::builder().input_tokens(4).output_tokens(5).build(),
        },
        StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
        },
    ]);
    let response = accumulator.finish();
    assert_eq!(
        response.content,
        vec![ContentBlock::text("one one continued!")]
    );
    assert_eq!(response.usage.input_tokens, 6);
    assert_eq!(response.usage.output_tokens, 8);
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn cached_usage_keeps_reads_writes_and_generated_tokens_distinct() {
    let cases = [
        (
            Protocol::ChatCompletions,
            json!({
                "id":"chat_1","model":"m",
                "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1000,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":800}}
            }),
            (200, 800, 0, 5),
        ),
        (
            Protocol::Responses,
            json!({
                "id":"resp_1","model":"m","status":"completed","output":[],
                "usage":{"input_tokens":1000,"output_tokens":5,"input_tokens_details":{"cached_tokens":800}}
            }),
            (200, 800, 0, 5),
        ),
        (
            Protocol::Messages,
            json!({
                "id":"msg_1","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
                "usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":800,"cache_creation_input_tokens":100}
            }),
            (10, 800, 100, 5),
        ),
        (
            Protocol::GenerateContent,
            json!({
                "candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"ok"}]}}],
                "usageMetadata":{"promptTokenCount":1000,"cachedContentTokenCount":800,"candidatesTokenCount":5}
            }),
            (200, 800, 0, 5),
        ),
    ];
    let codec = Codec::default();
    for (protocol, fixture, expected) in cases {
        let response = codec
            .decode_response(http::Response::new(body(fixture.to_string())), protocol)
            .await
            .expect("fixture decodes");
        let ResponseBody::Complete(response) = response.into_body() else {
            panic!("expected a complete response");
        };
        let usage = response.usage;
        assert_eq!(
            (
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                usage.output_tokens,
            ),
            expected,
            "{protocol}"
        );
    }
}
