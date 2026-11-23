//! Release-mode redaction benchmarks with synthetic inputs and an in-process upstream.

use std::{fmt::Write as _, hint::black_box};

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use http::{Request, Response, Uri, header};
use http_body_util::{BodyExt, Full};
use mpx_core::{
    IngressLayer, RedactionLayer, Upstream, UpstreamService,
    mpx_protocol::{Body, BoxError, Codec, Protocol},
    mpx_redact::{
        Digits, GeneratorBinding, Hex, Mask, Matcher, PlaceholderTemplate, RedactionSession, Rule,
        Rules, WordList,
    },
};
use serde_json::{Value, json};
use tower::{ServiceBuilder, ServiceExt, service_fn};

fn marker_cases(criterion: &mut Criterion) {
    let tags = Rules::new([Rule::new(
        Matcher::regex(r"\bprivate-[0-9]{6}\b").unwrap(),
        Mask::tag(),
    )]);
    let mut group = criterion.benchmark_group("redaction/markers");
    for size in [4096_usize, 65_536, 1_048_576] {
        let clean = "Ordinary text. ".repeat(size.div_ceil(15));
        group.throughput(Throughput::Bytes(clean.len() as u64));
        group.bench_function(BenchmarkId::new("mask_clean", size), |bencher| {
            bencher.iter(|| {
                RedactionSession::new()
                    .mask(black_box(&clean), &tags)
                    .unwrap()
            });
        });
    }

    for count in [64, 2048, 16384] {
        let text = (0..count)
            .map(|index| format!("private-{index:06}"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut session = RedactionSession::new();
        let masked = session.mask(&text, &tags).unwrap();
        assert_eq!(session.restore(&masked), text);
        assert_eq!(session.replacement_map().entries().len(), count);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function(BenchmarkId::new("mask_unique", count), |bencher| {
            bencher.iter(|| {
                RedactionSession::new()
                    .mask(black_box(&text), &tags)
                    .unwrap()
            });
        });
        group.throughput(Throughput::Bytes(masked.len() as u64));
        group.bench_function(BenchmarkId::new("restore_all", count), |bencher| {
            bencher.iter(|| session.restore(black_box(&masked)));
        });
        let first = masked.split_once(' ').unwrap().0;
        group.throughput(Throughput::Bytes(first.len() as u64));
        group.bench_function(BenchmarkId::new("restore_one", count), |bencher| {
            bencher.iter(|| session.restore(black_box(first)));
        });
    }

    let template = PlaceholderTemplate::new(
        "<{{digits}}>",
        &[GeneratorBinding::new("digits", Digits(4))],
    )
    .unwrap();
    let crowded = (0..4000)
        .map(|index| format!("private-{index:06}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mask_crowded = || {
        let mut session = RedactionSession::new().with_placeholder_template(template.clone());
        let masked = session.mask(black_box(&crowded), &tags).unwrap();
        (session, masked)
    };
    let (session, masked) = mask_crowded();
    assert_eq!(session.replacement_map().entries().len(), 4000);
    assert_eq!(session.restore(&masked), crowded);
    group.throughput(Throughput::Bytes(crowded.len() as u64));
    group.bench_function("mask_4000_of_10000_markers", |bencher| {
        bencher.iter(mask_crowded);
    });
    group.finish();
}

fn alias_cases(criterion: &mut Criterion) {
    let aliases = Rules::new([Rule::new(
        Matcher::Email,
        Mask::alias(
            "email",
            "example-{{hex}}@example.invalid",
            &[GeneratorBinding::new("hex", Hex(12))],
        )
        .unwrap(),
    )]);
    let mut group = criterion.benchmark_group("redaction/aliases");
    let text = "contact person@example.org. ".repeat(2048);
    let mut session = RedactionSession::new();
    let masked = session.mask(&text, &aliases).unwrap();
    let chunks: Vec<_> = masked
        .as_bytes()
        .chunks(32)
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect();
    let restore_stream = || {
        let mut restorer = session.restorer();
        let mut output = String::new();
        for chunk in &chunks {
            output.push_str(&restorer.push(black_box(chunk)));
        }
        output.push_str(&restorer.finish());
        output
    };
    assert_eq!(restore_stream(), text);
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("mask_2048_repeated_emails", |bencher| {
        bencher.iter(|| {
            RedactionSession::new()
                .mask(black_box(&text), &aliases)
                .unwrap()
        });
    });
    group.throughput(Throughput::Bytes(masked.len() as u64));
    group.bench_function("restore_aliases_in_32_byte_chunks", |bencher| {
        bencher.iter(restore_stream);
    });
    group.finish();
}

fn gateway_cases(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let codec = Codec::default();
    let text = "ordinary text private-000001. ".repeat(136);
    let rules = Rules::new([Rule::new(
        Matcher::Literal("private-000001".into()),
        Mask::tag(),
    )]);
    let mut group = criterion.benchmark_group("redaction/gateway");
    for (name, streaming) in [("json", false), ("sse", true)] {
        let transport = service_fn(move |request: Request<Body>| async move {
            let bytes = request.into_body().collect().await.unwrap().to_bytes();
            let prompt: Value = serde_json::from_slice(&bytes).unwrap();
            let content = prompt["messages"][0]["content"].as_str().unwrap();
            let (content_type, payload) = if streaming {
                let mut payload = String::new();
                for chunk in content.as_bytes().chunks(64) {
                    let chunk = std::str::from_utf8(chunk).unwrap();
                    let value = json!({"id":"mock","model":"model","choices":[{"index":0,"delta":{"content":chunk}}]});
                    writeln!(&mut payload, "data: {value}\n").unwrap();
                }
                payload.push_str("data: [DONE]\n\n");
                ("text/event-stream", payload)
            } else {
                ("application/json", json!({"id":"mock","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}]}).to_string())
            };
            Ok::<_, BoxError>(
                Response::builder()
                    .header(header::CONTENT_TYPE, content_type)
                    .body(Full::new(Bytes::from(payload)))
                    .unwrap(),
            )
        });
        let service = ServiceBuilder::new()
            .layer(IngressLayer::new(codec.clone()))
            .layer(RedactionLayer::new(rules.clone()))
            .service(
                UpstreamService::new(transport, codec.clone()).default_upstream(Upstream::new(
                    Protocol::ChatCompletions,
                    Uri::from_static("https://mock.invalid"),
                )),
            );
        let input = Bytes::from(
            json!({"model":"model","stream":streaming,"messages":[{"role":"user","content":text}]})
                .to_string(),
        );
        let run = || async {
            let request = Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Full::new(input.clone()))
                .unwrap();
            let response = service.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), 200);
            response.into_body().collect().await.unwrap().to_bytes()
        };
        let output = runtime.block_on(run());
        assert!(
            !std::str::from_utf8(&output)
                .unwrap()
                .contains("<|mpx-redact-")
        );
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_function(name, |bencher| {
            bencher.to_async(&runtime).iter(run);
        });
    }
    group.finish();
}

fn word_list_cases(criterion: &mut Criterion) {
    let words: Vec<_> = (0..65_536).map(|index| format!("name{index}")).collect();
    let entries: Vec<_> = words.iter().map(String::as_str).collect();
    let mut group = criterion.benchmark_group("redaction/word_list");
    group.throughput(Throughput::Bytes(
        words.iter().map(|word| word.len() as u64).sum(),
    ));
    group.bench_function("compile_65536_words", |bencher| {
        bencher.iter(|| WordList::new(black_box(&entries)).unwrap());
    });
    let words = WordList::new(&entries).unwrap();
    let rules = Rules::new([Rule::new(
        Matcher::Email,
        Mask::alias(
            "email",
            "{{word}}-{{word}}@example.invalid",
            &[GeneratorBinding::new("word", words)],
        )
        .unwrap(),
    )]);
    let text = (0..2048)
        .map(|index| format!("person{index}@example.org"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut session = RedactionSession::new();
    let masked = session.mask(&text, &rules).unwrap();
    assert_eq!(session.restore(&masked), text);
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("mask_2048_emails", |bencher| {
        bencher.iter(|| {
            RedactionSession::new()
                .mask(black_box(&text), &rules)
                .unwrap()
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    marker_cases,
    alias_cases,
    gateway_cases,
    word_list_cases
);
criterion_main!(benches);
