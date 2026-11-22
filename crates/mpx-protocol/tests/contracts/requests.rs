use bytes::Bytes;
use http::header;
use mpx_protocol::{
    Codec, ContentBlock, Diagnostics, Endpoint, Error, LossPolicy, Message, Protocol, Role, Route,
    Tool,
};
use serde_json::{Value, json};

use super::{collect_body, request};

#[test]
fn detects_all_builtin_endpoints_and_configured_routes() {
    let codec = Codec::default();
    let cases = [
        ("/v1/chat/completions", Protocol::ChatCompletions),
        ("/proxy/v1/responses", Protocol::Responses),
        ("/v1/messages", Protocol::Messages),
        ("/v1/interactions", Protocol::Interactions),
        ("/proxy/v1beta/interactions", Protocol::Interactions),
        (
            "/v1beta/models/gemini-2:generateContent",
            Protocol::GenerateContent,
        ),
        (
            "/v1/projects/p/locations/l/publishers/google/models/gemini-2:streamGenerateContent",
            Protocol::GenerateContent,
        ),
    ];
    for (path, protocol) in cases {
        let parts = request(path, &json!({})).into_parts().0;
        let endpoint = codec.detect(&parts).expect("endpoint is detected");
        assert_eq!(endpoint.protocol, protocol);
    }

    let codec = Codec::builder()
        .route(Route::alias("/gateway/{*rest}/chat", Protocol::Messages))
        .build();
    let parts = request("/gateway/a/b/chat", &json!({})).into_parts().0;
    let endpoint = codec.detect(&parts).expect("configured route is detected");
    assert_eq!(endpoint.protocol, Protocol::Messages);
}

#[test]
fn explicit_endpoints_override_detection_and_routes_do_not_depend_on_headers() {
    let codec = Codec::default();
    let mut request = request("/unknown", &json!({}));
    request
        .extensions_mut()
        .insert(Endpoint::from(Protocol::GenerateContent));
    let endpoint = codec
        .detect(&request.into_parts().0)
        .expect("override wins");
    assert_eq!(endpoint.protocol, Protocol::GenerateContent);

    let request = http::Request::post("/ambiguous")
        .header("anthropic-version", "2023-06-01")
        .body(())
        .expect("test request is valid");
    assert!(codec.detect(&request.into_parts().0).is_err());

    let request = http::Request::post("/v1/messages")
        .header("anthropic-version", "2023-06-01")
        .body(())
        .expect("test request is valid");
    let endpoint = codec.detect(&request.into_parts().0).expect("path wins");
    assert_eq!(endpoint.protocol, Protocol::Messages);
}

#[tokio::test]
async fn diagnostics_reject_loss_and_preserve_warnings() {
    let request = http::Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Bytes::from(
            json!({"model":"claude","messages":[{"role":"user","content":"hello"}]}).to_string(),
        ))
        .expect("request is valid");
    let diagnostics = Diagnostics::new();
    let codec = Codec::builder().default_max_tokens(64).build();
    let mut request = codec.decode_request(request).expect("request decodes");
    request.extensions_mut().insert(diagnostics.clone());
    let encoded = codec
        .encode_request(&request, Protocol::Messages)
        .expect("missing max_tokens is synthesized");
    assert!(!diagnostics.is_empty());
    let bytes = collect_body(encoded.into_body()).await;
    let value: Value = serde_json::from_slice(&bytes).expect("body is JSON");
    assert_eq!(value["max_tokens"], 64);

    let codec = Codec::builder().loss_policy(LossPolicy::Reject).build();
    let request = codec
        .decode_request(
            http::Request::post("/v1/chat/completions")
                .body(Bytes::from_static(br#"{"model":"gpt","messages":[]}"#))
                .expect("request is valid"),
        )
        .expect("request decodes");
    let error = codec
        .encode_request(&request, Protocol::Messages)
        .expect_err("loss is rejected");
    assert!(matches!(error, Error::LossPolicy { .. }));
}

#[tokio::test]
async fn gemini_function_response_preserves_multimodal_parts_and_snake_case() {
    let codec = Codec::default();
    let decoded = codec
        .decode_request(request(
            "/v1beta/models/gemini-2:generateContent",
            &json!({
                "contents": [
                    {"role":"model","parts":[{"functionCall":{"id":"call_1","name":"view_file","args":{}}}]},
                    {"role":"user","parts":[{"function_response":{
                        "id":"call_1",
                        "name":"view_file",
                        "response":{"output":"image attached"},
                        "parts":[{"inline_data":{"mime_type":"image/jpeg","data":"/9j/4AAQ"}}]
                    }}]}
                ]
            }),
        ))
        .expect("Gemini function response decodes");
    let result = decoded
        .body()
        .messages
        .get(1)
        .and_then(|message| message.content.first())
        .expect("tool result is present");
    let ContentBlock::ToolResult { content, .. } = result else {
        panic!("expected tool result");
    };
    assert!(content.iter().any(|block| matches!(
        block,
        ContentBlock::Image { source: mpx_protocol::MediaSource::Base64 { media_type, data }, .. }
            if media_type == "image/jpeg" && data == "/9j/4AAQ"
    )));

    let encoded = codec
        .encode_request(&decoded, Protocol::GenerateContent)
        .expect("Gemini function response encodes");
    let body = collect_body(encoded.into_body()).await;
    let value: Value = serde_json::from_slice(&body).expect("encoded body is JSON");
    assert_eq!(
        value["contents"][1]["parts"][0]["function_response"]["name"],
        "view_file"
    );
    assert_eq!(
        value["contents"][1]["parts"][0]["function_response"]["parts"]
            .as_array()
            .and_then(|parts| parts.iter().find(|part| part.get("inline_data").is_some()))
            .and_then(|part| part["inline_data"]["data"].as_str()),
        Some("/9j/4AAQ")
    );
}

#[test]
fn chat_empty_tool_ids_are_stable_and_pair_results() {
    let codec = Codec::default();
    let decoded = codec
        .decode_request(request(
            "/v1/chat/completions",
            &json!({
                "model":"gpt-5",
                "messages":[
                    {"role":"assistant","tool_calls":[{"id":"","type":"function","function":{"name":"bash","arguments":"{}"}}]},
                    {"role":"tool","tool_call_id":"","content":"done"}
                ]
            }),
        ))
        .expect("Chat request decodes");
    let call_id = match &decoded.body().messages[0].content[0] {
        ContentBlock::ToolCall { id, .. } => id.as_str().to_owned(),
        block => panic!("expected tool call, got {}", block.kind()),
    };
    let result_id = match &decoded.body().messages[1].content[0] {
        ContentBlock::ToolResult { call_id, .. } => call_id.as_str(),
        block => panic!("expected tool result, got {}", block.kind()),
    };
    assert_eq!(call_id, result_id);
    assert_eq!(call_id, "call_0");
}

#[tokio::test]
async fn responses_preserves_storage_preference_with_or_without_continuation() {
    let codec = Codec::default();
    for previous_response_id in [None, Some("resp_previous")] {
        for store in [false, true] {
            let mut input = json!({"model":"gpt-5", "input":"hello", "store":store});
            if let Some(id) = previous_response_id {
                input["previous_response_id"] = json!(id);
            }
            let decoded = codec
                .decode_request(request("/v1/responses", &input))
                .expect("Responses request decodes");
            let encoded = codec
                .encode_request(&decoded, Protocol::Responses)
                .expect("Responses request encodes");
            let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
                .expect("encoded request is JSON");
            assert_eq!(wire["store"], store);
            assert_eq!(wire["previous_response_id"].as_str(), previous_response_id);
        }
    }
}

#[tokio::test]
async fn responses_accepts_string_and_array_message_content() {
    let codec = Codec::default();
    let decoded = codec
        .decode_request(request(
            "/v1/responses",
            &json!({
                "model":"gpt-5",
                "input":[
                    {"role":"developer","content":"Be concise"},
                    {"role":"user","content":"First question"},
                    {"role":"user","content":[{"type":"input_text","text":"Second question"}]}
                ]
            }),
        ))
        .expect("Responses request decodes");
    let encoded = codec
        .encode_request(&decoded, Protocol::Responses)
        .expect("Responses request encodes");
    let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
        .expect("encoded request is JSON");
    assert_eq!(
        decoded.body().instructions,
        [ContentBlock::text("Be concise")]
    );
    assert_eq!(
        decoded.body().messages,
        [
            Message::text(Role::User, "First question"),
            Message::text(Role::User, "Second question"),
        ]
    );
    assert_eq!(wire["input"][0]["content"], "Be concise");
    assert_eq!(wire["input"][1]["content"], "First question");
    assert_eq!(wire["input"][2]["content"][0]["text"], "Second question");
}

#[tokio::test]
async fn chat_retains_legacy_or_modern_token_limit_field() {
    let codec = Codec::default();
    for (legacy, modern) in [
        (None, Some(1000)),
        (Some(800), None),
        (Some(800), Some(1000)),
    ] {
        let mut input = json!({
            "model":"o3",
            "messages":[{"role":"user","content":"Hello"}]
        });
        if let Some(limit) = legacy {
            input["max_tokens"] = json!(limit);
        }
        if let Some(limit) = modern {
            input["max_completion_tokens"] = json!(limit);
        }
        let decoded = codec
            .decode_request(request("/v1/chat/completions", &input))
            .expect("Chat request decodes");
        assert_eq!(decoded.body().sampling.max_output_tokens, modern.or(legacy));
        let encoded = codec
            .encode_request(&decoded, Protocol::ChatCompletions)
            .expect("Chat request encodes");
        let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("encoded request is JSON");
        assert_eq!(wire, input);
    }
}

#[tokio::test]
async fn gemini_pairs_idless_results_by_name_and_order_without_overriding_explicit_ids() {
    let codec = Codec::default();
    let decoded = codec
        .decode_request(request(
            "/v1beta/models/gemini:generateContent",
            &json!({
                "contents":[
                    {"role":"user","parts":[{"text":"Look up the weather"}]},
                    {"role":"model","parts":[
                        {"functionCall":{"id":"named_id","name":"weather","args":{}}},
                        {"functionCall":{"name":"weather","args":{}}},
                        {"functionCall":{"name":"traffic","args":{}}},
                        {"functionCall":{"name":"weather","args":{}}}
                    ]},
                    {"role":"user","parts":[
                        {"functionResponse":{"name":"traffic","response":{"output":"light"}}},
                        {"functionResponse":{"name":"weather","response":{"output":"sunny"}}},
                        {"functionResponse":{"id":"named_id","response":{"output":"cloudy"}}},
                        {"functionResponse":{"name":"weather","response":{"output":"rain"}}},
                        {"text":"Summarize the results"}
                    ]}
                ]
            }),
        ))
        .expect("Gemini request decodes");
    let encoded = codec
        .encode_request(&decoded, Protocol::ChatCompletions)
        .expect("tool history translates to Chat");
    let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
        .expect("encoded request is JSON");
    let messages = wire["messages"].as_array().expect("messages is an array");
    let calls = &messages[1]["tool_calls"];
    for (message, (call, expected)) in
        messages[2..6]
            .iter()
            .zip([(2, "light"), (1, "sunny"), (0, "cloudy"), (3, "rain")])
    {
        assert_eq!(message["role"], "tool");
        assert_eq!(message["tool_call_id"], calls[call]["id"]);
        assert_eq!(message["content"], expected);
    }
    assert_eq!(messages.len(), 7);
    assert_eq!(messages[6]["role"], "user");
    assert_eq!(messages[6]["content"], "Summarize the results");
}

#[tokio::test]
async fn gemini_round_trips_dynamic_disabled_and_fixed_thinking_budgets() {
    let codec = Codec::default();
    for budget in [-1, 0, 1024] {
        let decoded = codec
            .decode_request(request(
                "/v1beta/models/gemini:generateContent",
                &json!({
                    "contents":[{"role":"user","parts":[{"text":"hello"}]}],
                    "generationConfig":{"thinkingConfig":{"thinkingBudget":budget}}
                }),
            ))
            .expect("Gemini request decodes");
        let encoded = codec
            .encode_request(&decoded, Protocol::GenerateContent)
            .expect("Gemini request encodes");
        let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("encoded request is JSON");
        assert_eq!(
            wire["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            budget
        );
    }
}

#[tokio::test]
async fn gemini_passes_translated_json_schemas_through() {
    let codec = Codec::builder().loss_policy(LossPolicy::Reject).build();
    let schema = json!({
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "additionalProperties": false
    });
    let decoded = codec
        .decode_request(request(
            "/v1/messages",
            &json!({
                "model": "m", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"name": "lookup", "input_schema": schema},
                    {"name": "ping", "input_schema": {}}
                ],
                "output_config": {"format": {"type": "json_schema", "schema": schema}}
            }),
        ))
        .expect("request decodes");
    let encoded = codec
        .encode_request(&decoded, Protocol::GenerateContent)
        .expect("request encodes without loss");
    let actual: Value =
        serde_json::from_slice(&collect_body(encoded.into_body()).await).expect("request is JSON");
    assert_eq!(
        actual["tools"][0]["functionDeclarations"],
        json!([{"name": "lookup", "parametersJsonSchema": schema}, {"name": "ping"}])
    );
    assert_eq!(
        actual["generationConfig"],
        json!({"maxOutputTokens": 64, "responseMimeType": "application/json", "responseJsonSchema": schema})
    );
}

#[tokio::test]
async fn gemini_keeps_native_schema_fields_when_schemas_are_edited() {
    let codec = Codec::default();
    for (parameters_key, response_key, original, updated) in [
        (
            "parameters",
            "responseSchema",
            json!({"type": "OBJECT", "properties": {"value": {"type": "STRING", "nullable": true}}}),
            json!({"type": "OBJECT", "properties": {"value": {"type": "STRING", "minLength": 1}}}),
        ),
        (
            "parametersJsonSchema",
            "responseJsonSchema",
            json!({"type": "object", "properties": {"value": {"type": ["string", "null"]}}}),
            json!({"type": "object", "properties": {"value": {"type": "string", "minLength": 1}}}),
        ),
    ] {
        let mut decoded = codec
            .decode_request(request(
                "/v1beta/models/m:generateContent",
                &json!({
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    "tools": [{"functionDeclarations": [{"name": "lookup", (parameters_key): original}]}],
                    "generationConfig": {"responseMimeType": "application/json", (response_key): original}
                }),
            ))
            .expect("request decodes");
        let Tool::Function { parameters, .. } = &mut decoded.body_mut().tools[0] else {
            panic!("expected a function tool");
        };
        *parameters = updated.clone();
        decoded
            .body_mut()
            .output_format
            .as_mut()
            .expect("output format decodes")
            .schema = Some(updated.clone());

        let encoded = codec
            .encode_request(&decoded, Protocol::GenerateContent)
            .expect("edited request encodes");
        let actual: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("request is JSON");
        assert_eq!(
            actual["tools"][0]["functionDeclarations"],
            json!([{"name": "lookup", (parameters_key): updated}])
        );
        assert_eq!(
            actual["generationConfig"],
            json!({"responseMimeType": "application/json", (response_key): updated})
        );
    }
}

#[test]
fn responses_reports_unsupported_sampling_controls_in_warn_and_reject_modes() {
    let codec = Codec::default();
    let strict = Codec::builder().loss_policy(LossPolicy::Reject).build();
    for (field, value, path) in [
        ("stop", json!("STOP"), "sampling.stop_sequences"),
        ("top_k", json!(5), "sampling.top_k"),
        ("seed", json!(123), "sampling.seed"),
        ("presence_penalty", json!(0.2), "sampling.presence_penalty"),
        (
            "frequency_penalty",
            json!(0.3),
            "sampling.frequency_penalty",
        ),
    ] {
        let mut input = json!({"model":"m","messages":[{"role":"user","content":"Hello"}]});
        input[field] = value;
        let decoded = codec
            .decode_request(request("/v1/chat/completions", &input))
            .expect("Chat request decodes");
        codec
            .encode_request(&decoded, Protocol::Responses)
            .expect("warn mode permits translation");
        let diagnostics = decoded
            .extensions()
            .get::<Diagnostics>()
            .expect("diagnostics attached");
        assert!(
            diagnostics
                .entries()
                .iter()
                .any(|diagnostic| diagnostic.path == path)
        );
        assert!(matches!(
            strict.encode_request(&decoded, Protocol::Responses),
            Err(Error::LossPolicy { message }) if message.contains(path)
        ));
    }
}

fn request_fixtures() -> [(Protocol, &'static str, Value); 5] {
    [
        (
            Protocol::ChatCompletions,
            "/v1/chat/completions",
            json!({
                "model":"m","stream":true,"max_completion_tokens":128,"temperature":0.25,
                "messages":[{"role":"system","content":"Be concise"},
                            {"role":"user","content":"First question"},
                            {"role":"assistant","content":"First answer"},
                            {"role":"user","content":"Second question"}]
            }),
        ),
        (
            Protocol::Responses,
            "/v1/responses",
            json!({
                "model":"m","stream":true,"max_output_tokens":128,"temperature":0.25,
                "instructions":"Be concise",
                "input":[{"role":"user","content":"First question"},
                         {"role":"assistant","content":"First answer"},
                         {"role":"user","content":"Second question"}]
            }),
        ),
        (
            Protocol::Messages,
            "/v1/messages",
            json!({
                "model":"m","stream":true,"max_tokens":128,"temperature":0.25,
                "system":"Be concise",
                "messages":[{"role":"user","content":"First question"},
                            {"role":"assistant","content":"First answer"},
                            {"role":"user","content":"Second question"}]
            }),
        ),
        (
            Protocol::GenerateContent,
            "/v1beta/models/m:streamGenerateContent",
            json!({
                "systemInstruction":{"parts":[{"text":"Be concise"}]},
                "generationConfig":{"maxOutputTokens":128,"temperature":0.25},
                "contents":[{"role":"user","parts":[{"text":"First question"}]},
                            {"role":"model","parts":[{"text":"First answer"}]},
                            {"role":"user","parts":[{"text":"Second question"}]}]
            }),
        ),
        (
            Protocol::Interactions,
            "/v1/interactions",
            json!({
                "model":"m","stream":true,"system_instruction":"Be concise",
                "generation_config":{"max_output_tokens":128,"temperature":0.25},
                "input":[{"type":"user_input","content":"First question"},
                         {"type":"model_output","content":[{"type":"text","text":"First answer"}]},
                         {"type":"user_input","content":"Second question"}]
            }),
        ),
    ]
}

#[tokio::test]
async fn supported_request_fields_survive_every_protocol_pair() {
    let fixtures = request_fixtures();
    let codec = Codec::builder().loss_policy(LossPolicy::Reject).build();
    for (origin, path, fixture) in &fixtures {
        let decoded = codec
            .decode_request(request(path, fixture))
            .expect("fixture decodes");
        let canonical = decoded.body();
        assert_eq!(canonical.model, "m");
        assert_eq!(canonical.instructions, [ContentBlock::text("Be concise")]);
        assert_eq!(
            canonical.messages,
            [
                Message::text(Role::User, "First question"),
                Message::text(Role::Assistant, "First answer"),
                Message::text(Role::User, "Second question"),
            ]
        );
        assert!(canonical.stream);
        assert_eq!(canonical.sampling.max_output_tokens, Some(128));

        for (target, _, _) in &fixtures {
            let encoded = codec
                .encode_request(&decoded, *target)
                .unwrap_or_else(|error| panic!("{origin} to {target}: {error}"));
            let (parts, body) = encoded.into_parts();
            let wire: Value =
                serde_json::from_slice(&collect_body(body).await).expect("translated body is JSON");
            if origin == target {
                assert_eq!(&wire, fixture);
                continue;
            }
            let (messages, instruction, budget, temperature) = match target {
                Protocol::ChatCompletions => (
                    &wire["messages"],
                    &wire["messages"][0]["content"],
                    &wire["max_completion_tokens"],
                    &wire["temperature"],
                ),
                Protocol::Responses => (
                    &wire["input"],
                    &wire["instructions"],
                    &wire["max_output_tokens"],
                    &wire["temperature"],
                ),
                Protocol::Messages => (
                    &wire["messages"],
                    &wire["system"][0]["text"],
                    &wire["max_tokens"],
                    &wire["temperature"],
                ),
                Protocol::GenerateContent => (
                    &wire["contents"],
                    &wire["systemInstruction"]["parts"][0]["text"],
                    &wire["generationConfig"]["maxOutputTokens"],
                    &wire["generationConfig"]["temperature"],
                ),
                Protocol::Interactions => (
                    &wire["input"],
                    &wire["system_instruction"],
                    &wire["generation_config"]["max_output_tokens"],
                    &wire["generation_config"]["temperature"],
                ),
                _ => unreachable!("matrix enumerates the built-in codecs"),
            };
            assert_eq!(instruction, "Be concise", "{origin} to {target}");
            assert_eq!(budget, 128, "{origin} to {target}");
            assert_eq!(temperature, 0.25, "{origin} to {target}");
            if *target == Protocol::GenerateContent {
                assert!(
                    parts
                        .uri
                        .path()
                        .ends_with("/models/m:streamGenerateContent")
                );
            } else {
                assert_eq!(wire["model"], "m", "{origin} to {target}");
                assert_eq!(wire["stream"], true, "{origin} to {target}");
            }

            assert_dialogue(*origin, *target, messages);
        }
    }
}

fn assert_dialogue(origin: Protocol, target: Protocol, messages: &Value) {
    let messages = messages.as_array().expect("messages are an array");
    let dialogue = &messages[usize::from(target == Protocol::ChatCompletions)..];
    assert_eq!(dialogue.len(), 3, "{origin} to {target}");
    for (index, (message, expected)) in dialogue
        .iter()
        .zip(["First question", "First answer", "Second question"])
        .enumerate()
    {
        if target == Protocol::Interactions {
            assert_eq!(
                message["type"],
                if index == 1 {
                    "model_output"
                } else {
                    "user_input"
                }
            );
            assert_eq!(message["content"][0]["text"], expected);
            continue;
        }
        let role = match (target, index) {
            (Protocol::GenerateContent, 1) => "model",
            (_, 1) => "assistant",
            _ => "user",
        };
        let text = match target {
            Protocol::ChatCompletions => &message["content"],
            Protocol::Messages | Protocol::Responses => &message["content"][0]["text"],
            Protocol::GenerateContent => &message["parts"][0]["text"],
            _ => unreachable!("matrix enumerates the built-in codecs"),
        };
        assert_eq!(message["role"], role, "{origin} to {target}");
        assert_eq!(text, expected, "{origin} to {target}");
    }
}

// Magpie's native replay scenarios exercise opaque provider-issued state through request history.
#[tokio::test]
async fn native_reasoning_state_survives_request_replay() {
    let codec = Codec::default();
    for (protocol, path, fixture, pointer) in [
        (
            Protocol::Messages,
            "/v1/messages",
            json!({"model":"m","max_tokens":128,"messages":[{"role":"assistant","content":[
             {"type":"thinking","thinking":"earlier thought","signature":"opaque-claude-state"},
             {"type":"text","text":"answer"}]},{"role":"user","content":"Continue"}]}),
            "/messages/0/content/0/signature",
        ),
        (
            Protocol::Responses,
            "/v1/responses",
            json!({"model":"m","store":false,"input":[{"type":"reasoning","id":"rs_original",
             "summary":[],"encrypted_content":"opaque-openai-state"},
             {"role":"user","content":"Continue"}]}),
            "/input/0/encrypted_content",
        ),
        (
            Protocol::GenerateContent,
            "/v1beta/models/m:generateContent",
            json!({"contents":[{"role":"model","parts":[{"functionCall":{"name":"lookup","id":"call_1","args":{}},
             "thoughtSignature":"opaque-gemini-state"}]},{"role":"user","parts":[{"functionResponse":{
             "name":"lookup","id":"call_1","response":{"output":"found"}}}]}]}),
            "/contents/0/parts/0/thoughtSignature",
        ),
    ] {
        let expected = fixture.pointer(pointer).expect("fixture contains a seal");
        let decoded = codec
            .decode_request(request(path, &fixture))
            .expect("history decodes");
        let encoded = codec
            .encode_request(&decoded, protocol)
            .expect("history replays");
        let wire: Value = serde_json::from_slice(&collect_body(encoded.into_body()).await)
            .expect("replayed request is JSON");
        assert_eq!(wire.pointer(pointer), Some(expected), "{protocol}");
    }
}

#[tokio::test]
async fn interactions_continuation_uses_native_id_and_preserves_tool_results() {
    let codec = Codec::default();
    let native = json!({
        "model": "gemini-3.8-flash",
        "previous_interaction_id": "int_1",
        "store": true,
        "input": { "type": "function_result", "call_id": "fc_1", "result": "52°F" }
    });
    let mut decoded = codec
        .decode_request(request("/v1/interactions", &native))
        .expect("tool continuation decodes");
    let continuation = decoded.body().continuation.as_ref().expect("continuation");
    assert_eq!(continuation.previous_response_id.as_str(), "int_1");
    assert_eq!(continuation.store, Some(true));
    assert!(matches!(
        &decoded.body().messages[0].content[0],
        ContentBlock::ToolResult { call_id, .. } if call_id.as_str() == "fc_1"
    ));
    assert!(codec.encode_request(&decoded, Protocol::Responses).is_err());
    decoded.body_mut().stream = true;
    let encoded = codec
        .encode_request(&decoded, Protocol::Interactions)
        .expect("native continuation encodes");
    assert_eq!(encoded.headers()[header::ACCEPT], "text/event-stream");
    let bytes = collect_body(encoded.into_body()).await;
    let actual: Value = serde_json::from_slice(&bytes).expect("body is JSON");
    let mut expected = native;
    expected["stream"] = json!(true);
    assert_eq!(actual, expected);
}
