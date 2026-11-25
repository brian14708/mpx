use std::{
    borrow::Cow,
    fmt,
    sync::{Arc, OnceLock},
};

use async_stream::{stream as async_stream, try_stream};
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use bytes::{Buf, Bytes, BytesMut};
use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::{Stream, StreamExt, TryStreamExt, stream};
use http_body::{Body as HttpBody, Frame};
use http_body_util::{BodyDataStream, BodyExt, BodyStream, StreamBody};
use sync_wrapper::SyncStream;

use crate::{
    ApiError, Body, BoxError, CodecContext, Error, EventStream, Protocol, Result, StreamEvent,
    codec::box_body_error,
};

pub use eventsource_stream::Event as SseEvent;

/// Incremental decoder for one protocol's SSE events.
pub trait StreamDecoder: fmt::Debug + Send {
    /// Decodes one wire event into zero or more normalized events.
    fn decode(&mut self, event: SseEvent, cx: &CodecContext) -> Result<Vec<StreamEvent>>;

    /// Finishes the stream and emits any required terminal events.
    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<StreamEvent>> {
        Ok(Vec::new())
    }
}

/// Incremental encoder for one client protocol's SSE events.
pub trait StreamEncoder: fmt::Debug + Send {
    /// Encodes one normalized event into zero or more SSE events.
    fn encode(&mut self, event: StreamEvent, cx: &CodecContext) -> Result<Vec<SseEvent>>;

    /// Finishes the client stream, including any protocol terminal marker.
    fn finish(&mut self, _cx: &CodecContext) -> Result<Vec<SseEvent>> {
        Ok(Vec::new())
    }
}

pub(crate) fn named(event: &str, data: impl Into<String>) -> SseEvent {
    SseEvent {
        event: event.to_owned(),
        data: data.into(),
        ..SseEvent::default()
    }
}

pub(crate) fn event_name(event: &SseEvent) -> Option<&str> {
    let name = event.event.as_str();
    (!matches!(name, "" | "message")).then_some(name)
}

#[derive(Clone)]
pub(crate) struct BufferedResponse {
    pub(crate) protocol: Protocol,
    pub(crate) body: Arc<OnceLock<Bytes>>,
}

impl BufferedResponse {
    pub(crate) fn new(protocol: Protocol) -> Self {
        Self {
            protocol,
            body: Arc::new(OnceLock::new()),
        }
    }
}

const BUFFERED_RESPONSE_COMMENT: &str = ": mpx-buffered-response-v1 ";

pub(crate) fn decode_event_stream<B>(
    body: B,
    mut decoder: Box<dyn StreamDecoder>,
    cx: CodecContext,
    buffered: BufferedResponse,
) -> EventStream
where
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError>,
{
    let protocol = buffered.protocol;
    EventStream::from_stream(async_stream! {
        let mut source = std::pin::pin!(decode_sse(body, Some(buffered)));
        loop {
            let next = source.next().await;
            let eof = next.is_none();
            let events = match next {
                Some(Ok(event)) => decoder.decode(event, &cx),
                Some(Err(error)) => Err(Error::Body { message: error.to_string() }),
                None => decoder.finish(&cx),
            };
            match events {
                Ok(events) => {
                    let mut finished = eof;
                    for event in events {
                        finished |= matches!(event, StreamEvent::MessageStop { .. } | StreamEvent::Error { .. });
                        yield Ok(event);
                    }
                    if finished {
                        break;
                    }
                }
                Err(error) => {
                    yield Ok(StreamEvent::Error {
                        error: ApiError::new(
                            crate::ApiErrorKind::BadResponse,
                            format!("{protocol}: {error}"),
                        ),
                    });
                    break;
                }
            }
        }
    })
}

pub(crate) fn encode_event_body(
    mut events: EventStream,
    mut encoder: Box<dyn StreamEncoder>,
    cx: CodecContext,
) -> Body {
    let frames = try_stream! {
        while let Some(event) = events.try_next().await? {
            for frame in encoder.encode(event, &cx)? {
                yield Frame::data(encode_sse(&frame));
            }
        }
        for frame in encoder.finish(&cx)? {
            yield Frame::data(encode_sse(&frame));
        }
    };

    BodyExt::boxed(StreamBody::new(SyncStream::new(frames)))
}

/// Encodes known events up front so encoding failures surface before a response starts.
pub(crate) fn encode_events(
    events: Vec<StreamEvent>,
    mut encoder: Box<dyn StreamEncoder>,
    cx: &CodecContext,
) -> Result<Bytes> {
    let mut output = String::new();
    for event in events {
        for frame in encoder.encode(event, cx)? {
            encode_sse_into(&mut output, &frame);
        }
    }
    for frame in encoder.finish(cx)? {
        encode_sse_into(&mut output, &frame);
    }
    Ok(Bytes::from(output))
}

pub(crate) fn prepend_buffered_response(body: Body, protocol: Protocol, bytes: &[u8]) -> Body {
    // A comment carries buffered-only fields through the wire without changing native events.
    let comment = format!(
        "{BUFFERED_RESPONSE_COMMENT}{protocol} {}\n",
        STANDARD_NO_PAD.encode(bytes)
    );
    let leading = stream::iter([Ok(Frame::data(Bytes::from(comment)))]);
    BodyExt::boxed(StreamBody::new(leading.chain(BodyStream::new(body))))
}

// EventSource 0.2 slices its leading BOM at a byte offset of one, which panics
// on the three-byte UTF-8 BOM, and needs a following byte to disambiguate a
// final CR from CRLF. The spec's single BOM is stripped here; a leading blank
// line, which dispatches nothing, keeps any further BOM away from that slice.
fn decode_sse<B>(
    body: B,
    buffered: Option<BufferedResponse>,
) -> impl Stream<Item = std::result::Result<SseEvent, EventStreamError<Error>>>
where
    B: HttpBody<Data = Bytes>,
    B::Error: Into<BoxError>,
{
    let input = try_stream! {
        const BOM: &[u8] = b"\xef\xbb\xbf";
        let mut body = std::pin::pin!(BodyDataStream::new(body));
        let mut prefix = Some(BytesMut::new());
        let mut trailing_cr = false;
        let mut started = false;
        let mut bom_checked = false;
        while let Some(data) = body.try_next().await.map_err(box_body_error)? {
            if data.is_empty() {
                continue;
            }
            let data = if let Some(mut first) = prefix.take() {
                first.extend_from_slice(&data);
                if !bom_checked {
                    if first.len() < BOM.len() && BOM.starts_with(&first) {
                        prefix = Some(first);
                        continue;
                    }
                    bom_checked = true;
                    if first.starts_with(BOM) {
                        first.advance(BOM.len());
                    }
                }
                if let Some(buffered) = &buffered
                    && !take_buffered_comment(&mut first, buffered.protocol, &buffered.body)
                {
                    prefix = Some(first);
                    continue;
                }
                first.freeze()
            } else {
                data
            };
            if !data.is_empty() {
                if !started {
                    started = true;
                    yield Bytes::from_static(b"\n");
                }
                trailing_cr = data.ends_with(b"\r");
                yield data;
            }
        }
        if let Some(prefix) = prefix.filter(|prefix| !prefix.is_empty()) {
            if !started {
                yield Bytes::from_static(b"\n");
            }
            yield prefix.freeze();
        } else if trailing_cr {
            yield Bytes::from_static(b"\n");
        }
    };
    input.eventsource()
}

fn take_buffered_comment(
    first: &mut BytesMut,
    protocol: Protocol,
    buffered: &OnceLock<Bytes>,
) -> bool {
    let prefix = BUFFERED_RESPONSE_COMMENT.as_bytes();
    if prefix.starts_with(first) {
        return false;
    }
    if !first.starts_with(prefix) {
        return true;
    }
    let Some(end) = first.iter().position(|byte| *byte == b'\n') else {
        return false;
    };
    let line = first.split_to(end + 1);
    let value = &line[prefix.len()..end];
    if let Some(separator) = value.iter().position(|byte| *byte == b' ')
        && &value[..separator] == protocol.as_str().as_bytes()
        && let Ok(body) = STANDARD_NO_PAD.decode(&value[separator + 1..])
    {
        let _ = buffered.set(Bytes::from(body));
    }
    true
}

fn encode_sse(frame: &SseEvent) -> Bytes {
    let mut output = String::with_capacity(frame.data.len() + 32);
    encode_sse_into(&mut output, frame);
    Bytes::from(output)
}

fn encode_sse_into(output: &mut String, frame: &SseEvent) {
    if !frame.event.is_empty() && frame.event != "message" && !frame.event.contains(['\r', '\n']) {
        encode_field(output, "event", &frame.event);
    }
    if !frame.id.is_empty() && !frame.id.contains(['\0', '\r', '\n']) {
        encode_field(output, "id", &frame.id);
    }
    if let Some(retry) = frame.retry {
        encode_field(output, "retry", &retry.as_millis().to_string());
    }
    let data = if frame.data.contains('\r') {
        Cow::Owned(frame.data.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(frame.data.as_str())
    };
    for line in data.split('\n') {
        encode_field(output, "data", line);
    }
    output.push('\n');
}

fn encode_field(output: &mut String, field: &str, value: &str) {
    output.push_str(field);
    output.push_str(": ");
    output.push_str(value);
    output.push('\n');
}
