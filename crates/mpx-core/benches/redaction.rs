//! Release-mode redaction benchmarks with synthetic inputs and an in-process upstream.

use std::{fmt::Write as _, hint::black_box};

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use http::{Request, Response, Uri, header};
use http_body_util::{BodyExt, Full};
use mpx_core::{
    IngressLayer, RedactionLayer, Upstream, UpstreamService,
    mpx_protocol::{Body, BoxError, Codec, Protocol},
    mpx_redact::{Matcher, RedactionSession, Replacement, Rules, Slot},
};
use serde_json::{Value, json};
use tower::{ServiceBuilder, ServiceExt, service_fn};

fn marker_replacement() -> Replacement {
    Replacement::kind("SECRET").unwrap()
}

fn email_matcher() -> Matcher {
    Matcher::regex(
        r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}",
    )
    .unwrap()
}

fn marker_cases(criterion: &mut Criterion) {
    let tags = Rules::new([(
        Matcher::regex(r"\bprivate-[0-9]{6}\b").unwrap(),
        marker_replacement(),
    )]);
    let mut group = criterion.benchmark_group("redaction/markers");
    let size = 65_536_usize;
    let clean = "Ordinary text. ".repeat(size.div_ceil(15));
    group.throughput(Throughput::Bytes(clean.len() as u64));
    group.bench_function(BenchmarkId::new("mask_clean", size), |bencher| {
        bencher.iter(|| {
            RedactionSession::new()
                .mask(black_box(&clean), &tags)
                .unwrap()
        });
    });

    for count in [2048, 16_384] {
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
        if count == 16_384 {
            let with_marker = format!("{{{{SECRET_aaaaaaaa}}}} {text}");
            let mut reserved = RedactionSession::new();
            let masked_with_marker = reserved.mask(&with_marker, &tags).unwrap();
            assert_eq!(reserved.restore(&masked_with_marker), with_marker);
            group.throughput(Throughput::Bytes(with_marker.len() as u64));
            group.bench_function(
                BenchmarkId::new("mask_with_existing_marker", count),
                |bencher| {
                    bencher.iter(|| {
                        RedactionSession::new()
                            .mask(black_box(&with_marker), &tags)
                            .unwrap()
                    });
                },
            );
            group.throughput(Throughput::Bytes(masked.len() as u64));
            group.bench_function(BenchmarkId::new("restore_all", count), |bencher| {
                bencher.iter(|| session.restore(black_box(&masked)));
            });
        }
    }
    group.finish();
}

fn alias_cases(criterion: &mut Criterion) {
    let aliases = Rules::new([(
        email_matcher(),
        Replacement::template("example-{}@example.invalid", &[Slot::base16(12)]).unwrap(),
    )]);
    let mut group = criterion.benchmark_group("redaction/aliases");
    let small = "contact person@example.org.";
    group.throughput(Throughput::Bytes(small.len() as u64));
    group.bench_function("mask_one_email", |bencher| {
        bencher.iter(|| {
            RedactionSession::new()
                .mask(black_box(small), &aliases)
                .unwrap()
        });
    });
    let text = "contact person@example.org. ".repeat(2048);
    let mut session = RedactionSession::new();
    let masked = session.mask(&text, &aliases).unwrap();
    assert_eq!(session.restore(&masked), text);
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("mask_2048_repeated_emails", |bencher| {
        bencher.iter(|| {
            RedactionSession::new()
                .mask(black_box(&text), &aliases)
                .unwrap()
        });
    });
    let unique = (0..48)
        .map(|index| format!("person{index}@example.org"))
        .collect::<Vec<_>>()
        .join(" ");
    let colliding_alias = RedactionSession::new()
        .mask("person0@example.org", &aliases)
        .unwrap();
    let mut history = format!("{colliding_alias}\n");
    for index in 0..65_536 {
        writeln!(history, "item-{index:06}-ready").unwrap();
    }
    let reserve = || {
        let mut session = RedactionSession::new().with_fallback(marker_replacement());
        session.reserve_replacements_in(&history, &aliases);
        session.reserve_replacements_in(&unique, &aliases);
        session
    };
    let mut session = reserve();
    let masked = session.mask_reserved(&unique, &aliases).unwrap();
    assert!(masked.starts_with("{{SECRET_"));
    assert_eq!(session.restore(&masked), unique);
    group.throughput(Throughput::Bytes((history.len() + unique.len()) as u64));
    group.bench_function("mask_48_emails_with_reserved_collision", |bencher| {
        bencher.iter_batched(
            reserve,
            |mut session| session.mask_reserved(black_box(&unique), &aliases).unwrap(),
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn restorer_cases(criterion: &mut Criterion) {
    let rules = Rules::new([
        (Matcher::literal("private-000001"), marker_replacement()),
        (
            Matcher::literal("private-000002"),
            Replacement::kind("TOKEN").unwrap(),
        ),
    ]);
    let mut group = criterion.benchmark_group("redaction/restore");
    let prose = "The response contains ordinary words and short sentences. ".repeat(16);
    for (name, text, rules, benchmark_complete) in [
        (
            "sparse_markers",
            format!("{prose} private-000001 private-000002. ").repeat(64),
            rules,
            false,
        ),
        (
            "sparse_aliases",
            format!("{prose} person@example.org. ").repeat(64),
            Rules::new([(
                email_matcher(),
                Replacement::template("example-{}@example.invalid", &[Slot::base16(12)]).unwrap(),
            )]),
            true,
        ),
    ] {
        let mut session = RedactionSession::new();
        let masked = session.mask(&text, &rules).unwrap();
        let chunks: Vec<_> = masked
            .as_bytes()
            .chunks(32)
            .map(|chunk| std::str::from_utf8(chunk).unwrap())
            .collect();
        let restore_stream = || {
            let mut restorer = session.restorer();
            let mut output = String::new();
            for &chunk in &chunks {
                output.push_str(&restorer.push(black_box(chunk)));
            }
            output.push_str(&restorer.finish());
            output
        };
        assert_eq!(session.restore(&masked), text);
        assert_eq!(restore_stream(), text);
        group.throughput(Throughput::Bytes(masked.len() as u64));
        if benchmark_complete {
            group.bench_function(BenchmarkId::new(name, "complete"), |bencher| {
                bencher.iter(|| session.restore(black_box(&masked)));
            });
        }
        group.bench_function(BenchmarkId::new(name, "32_byte_chunks"), |bencher| {
            bencher.iter(restore_stream);
        });
    }
    group.finish();
}

fn gateway_cases(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let codec = Codec::default();
    let text = "ordinary text private-000001. ".repeat(136);
    let rules = Rules::new([(Matcher::literal("private-000001"), marker_replacement())]);
    let mut group = criterion.benchmark_group("redaction/gateway");
    let transport = service_fn(|request: Request<Body>| async move {
        let bytes = request.into_body().collect().await.unwrap().to_bytes();
        let prompt: Value = serde_json::from_slice(&bytes).unwrap();
        let content = prompt["messages"][0]["content"].as_str().unwrap();
        let mut payload = String::new();
        for chunk in content.as_bytes().chunks(64) {
            let chunk = std::str::from_utf8(chunk).unwrap();
            let value = json!({
                "id": "mock",
                "model": "model",
                "choices": [{"index": 0, "delta": {"content": chunk}}],
            });
            writeln!(&mut payload, "data: {value}\n").unwrap();
        }
        payload.push_str("data: [DONE]\n\n");
        Ok::<_, BoxError>(
            Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Full::new(Bytes::from(payload)))
                .unwrap(),
        )
    });
    let service = ServiceBuilder::new()
        .layer(IngressLayer::new(codec.clone()))
        .layer(RedactionLayer::new(rules))
        .service(
            UpstreamService::new(transport, codec).default_upstream(Upstream::new(
                Protocol::ChatCompletions,
                Uri::from_static("https://mock.invalid"),
            )),
        );
    let input = Bytes::from(
        json!({
            "model": "model",
            "stream": true,
            "messages": [{"role": "user", "content": text}],
        })
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
    assert!(!std::str::from_utf8(&output).unwrap().contains("{{SECRET_"));
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("sse", |bencher| {
        bencher.to_async(&runtime).iter(run);
    });
    group.finish();
}

fn word_list_cases(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("redaction/word_list");
    let words = Slot::wordlist((0..65_536).map(|index| format!("name{index}"))).unwrap();
    let rules = Rules::new([(
        email_matcher(),
        Replacement::template("{}-{}@example.invalid", &[words.clone(), words]).unwrap(),
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
    restorer_cases,
    gateway_cases,
    word_list_cases
);
criterion_main!(benches);
