//! Gemini `generateContent` and `streamGenerateContent` protocol support.

mod codec;
mod stream;
mod wire;

use bytes::Bytes;
use http::{Method, StatusCode, header};
use percent_encoding::{AsciiSet, CONTROLS, percent_decode_str, utf8_percent_encode};

use crate::{
    ApiError, ClientContext, CodecContext, Endpoint, Error, Protocol, ProtocolCodec, Request,
    Response, Result, StreamDecoder, StreamEncoder,
};

const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b':')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

/// A synchronous, sans-IO codec for Gemini's `generateContent` API.
///
/// The codec recognizes both the standard `/models/{model}` routes and Vertex
/// AI's longer publisher-model routes. It translates requests, complete
/// responses, errors, and Gemini JSON SSE chunks to and from the canonical IR.
#[derive(Debug, Clone)]
pub(crate) struct GenerateContentCodec;

impl ProtocolCodec for GenerateContentCodec {
    fn protocol(&self) -> Protocol {
        Protocol::GenerateContent
    }

    fn decode_request(
        &self,
        endpoint: &Endpoint,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Request> {
        codec::decode_request(endpoint, body, cx)
    }

    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>> {
        if req.model.is_empty() {
            return Err(Error::InvalidRequest {
                protocol: Protocol::GenerateContent,
                message: "a model is required to construct the Gemini request path".to_owned(),
            });
        }
        let body = codec::encode_request(req, cx)?;
        let body = serde_json::to_vec(&body)
            .map_err(|error| Error::json(Protocol::GenerateContent, error))?;
        let model = utf8_percent_encode(&req.model, PATH_SEGMENT);
        let operation = if req.stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        let uri = format!("/v1beta/models/{model}:{operation}");
        let accept = if req.stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        http::Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, accept)
            .body(Bytes::from(body))
            .map_err(Error::http)
    }

    fn decode_response(
        &self,
        _parts: &http::response::Parts,
        body: &[u8],
        cx: &CodecContext,
    ) -> Result<Response> {
        codec::decode_response(body, cx)
    }

    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError {
        codec::decode_error(parts, body)
    }

    fn encode_response(
        &self,
        response: &Response,
        _client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>> {
        let body = codec::encode_response(response, cx)?;
        let body = serde_json::to_vec(&body)
            .map_err(|error| Error::json(Protocol::GenerateContent, error))?;
        Ok(crate::codec::json_response(
            StatusCode::OK,
            Bytes::from(body),
        ))
    }

    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes> {
        let body = codec::encode_error_value(error);
        let body = serde_json::to_string(&body).unwrap_or_else(|_| {
            "{\"error\":{\"code\":500,\"message\":\"internal error\",\"status\":\"INTERNAL\"}}"
                .to_owned()
        });
        crate::codec::json_response(error.status(), Bytes::from(body))
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::GeminiStreamDecoder::default())
    }

    fn stream_encoder(&self, _client: &ClientContext) -> Box<dyn StreamEncoder> {
        Box::new(stream::GeminiStreamEncoder::default())
    }
}

pub(crate) fn detect(parts: &http::request::Parts) -> Option<Endpoint> {
    if parts.method != Method::POST {
        return None;
    }
    let path = parts.uri.path();
    let (stream, operation) = if path.ends_with(":streamGenerateContent") {
        (true, ":streamGenerateContent")
    } else if path.ends_with(":generateContent") {
        (false, ":generateContent")
    } else {
        return None;
    };
    let prefix = path.strip_suffix(operation)?;
    let (_, encoded_model) = prefix.rsplit_once("/models/")?;
    if encoded_model.is_empty() || encoded_model.split('/').any(str::is_empty) {
        return None;
    }
    let model = percent_decode_str(encoded_model)
        .decode_utf8()
        .ok()?
        .into_owned();
    Some(Endpoint {
        protocol: Protocol::GenerateContent,
        model_from_path: Some(model),
        stream,
        passthrough: None,
    })
}
