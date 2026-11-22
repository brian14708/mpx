//! Gemini Interactions model generation.

mod content;
mod request;
mod resource;
mod response;
mod stream;

use bytes::Bytes;
use http::{StatusCode, header};

pub(crate) use resource::detect_resource;
use serde_json::{Map, Value};

use crate::{
    ApiError, ClientContext, CodecContext, Endpoint, Error, Protocol, ProtocolCodec, Request,
    Response, Result, StreamDecoder, StreamEncoder, gemini::GenerateContentCodec,
};

const PROTOCOL: Protocol = Protocol::Interactions;

/// A synchronous codec for creating Gemini Interactions and streaming their steps.
///
/// Detects the v1 and v1beta creation routes and uses stable v1 upstream.
#[derive(Debug, Clone)]
pub(crate) struct InteractionsCodec;

impl ProtocolCodec for InteractionsCodec {
    fn protocol(&self) -> Protocol {
        PROTOCOL
    }

    fn decode_request(
        &self,
        endpoint: &Endpoint,
        body: &[u8],
        _cx: &CodecContext,
    ) -> Result<Request> {
        request::decode(endpoint, body)
    }

    fn encode_request(&self, req: &Request, cx: &CodecContext) -> Result<http::Request<Bytes>> {
        http::Request::post("/v1/interactions")
            .header(header::CONTENT_TYPE, "application/json")
            .header(
                header::ACCEPT,
                if req.stream {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .body(serialize(&request::encode(req, cx)?)?)
            .map_err(Error::http)
    }

    fn decode_response(
        &self,
        _parts: &http::response::Parts,
        body: &[u8],
        _cx: &CodecContext,
    ) -> Result<Response> {
        response::decode(parse(body)?)
    }

    fn decode_error(&self, parts: &http::response::Parts, body: &[u8]) -> ApiError {
        if let Ok(value) = serde_json::from_slice::<Value>(body)
            && value["error"]["code"].is_string()
        {
            return response::api_error(&value["error"], parts.status);
        }
        GenerateContentCodec.decode_error(parts, body)
    }

    fn encode_response(
        &self,
        response: &Response,
        client: &ClientContext,
        cx: &CodecContext,
    ) -> Result<http::Response<Bytes>> {
        let body = serialize(&response::encode(response, client.request_model(), cx)?)?;
        Ok(crate::codec::json_response(StatusCode::OK, body))
    }

    fn encode_error(&self, error: &ApiError) -> http::Response<Bytes> {
        GenerateContentCodec.encode_error(error)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::Decoder::default())
    }

    fn stream_encoder(&self, client: &ClientContext) -> Box<dyn StreamEncoder> {
        Box::new(stream::Encoder::new(client.request_model().to_owned()))
    }
}

fn parse(body: &[u8]) -> Result<Map<String, Value>> {
    serde_json::from_slice(body).map_err(|error| Error::json(PROTOCOL, error))
}

fn serialize(value: &impl serde::Serialize) -> Result<Bytes> {
    serde_json::to_vec(value)
        .map(Bytes::from)
        .map_err(|error| Error::json(PROTOCOL, error))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Protocol {
        protocol: PROTOCOL,
        message: message.into(),
    }
}
