use http::Method;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};

use crate::{Endpoint, Error, Result};

use super::PROTOCOL;

const ESCAPED: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

pub(crate) fn detect_resource(parts: &http::request::Parts) -> Result<Option<Endpoint>> {
    let Some((_, suffix)) = parts.uri.path().rsplit_once("/interactions/") else {
        return Ok(None);
    };
    let (id, cancel) = suffix
        .strip_suffix("/cancel")
        .map_or((suffix, false), |id| (id, true));
    if id.is_empty()
        || id.contains('/')
        || !matches!(
            (&parts.method, cancel),
            (&Method::GET | &Method::DELETE, false) | (&Method::POST, true)
        )
    {
        return Ok(None);
    }
    let Ok(id) = percent_decode_str(id).decode_utf8() else {
        return Ok(None);
    };
    if matches!(id.as_ref(), "." | "..") {
        return Err(invalid_resource("expected an interaction id"));
    }
    let id = utf8_percent_encode(&id, ESCAPED);
    let mut path = format!("/v1/interactions/{id}");
    let mut stream = false;
    match (&parts.method, cancel) {
        (&Method::GET, false) => {
            let query = parts.uri.query().unwrap_or_default();
            let mut parameters = Vec::new();
            stream = query_value(query, "stream").as_deref() == Some("true");
            if stream {
                parameters.push("stream=true".to_owned());
            }
            if let Some(include_input) =
                query_value(query, "include_input").and_then(|value| value.parse::<bool>().ok())
            {
                parameters.push(format!("include_input={include_input}"));
            }
            if let Some(cursor) = query_value(query, "last_event_id") {
                if !stream {
                    return Err(invalid_resource(
                        "last_event_id requires streaming retrieval",
                    ));
                }
                let cursor = utf8_percent_encode(&cursor, ESCAPED);
                parameters.push(format!("last_event_id={cursor}"));
            }
            if !parameters.is_empty() {
                path.push('?');
                path.push_str(&parameters.join("&"));
            }
        }
        (&Method::POST, true) => path.push_str("/cancel"),
        (&Method::DELETE, false) => {}
        _ => return Ok(None),
    }
    let mut endpoint = Endpoint::passthrough(
        PROTOCOL,
        parts.method.clone(),
        path.parse()
            .map_err(|_| invalid_resource("invalid interaction resource path"))?,
    );
    endpoint.stream = stream;
    if let Some(route) = &mut endpoint.passthrough {
        route.allow_body = false;
    }
    Ok(Some(endpoint))
}

fn invalid_resource(message: &str) -> Error {
    Error::InvalidRequest {
        protocol: PROTOCOL,
        message: message.to_owned(),
    }
}

fn query_value(query: &str, key: &str) -> Option<String> {
    let value = query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find_map(|(name, value)| (name == key).then_some(value))?;
    percent_decode_str(value)
        .decode_utf8()
        .ok()
        .map(std::borrow::Cow::into_owned)
}
