use std::{fmt, sync::Arc};

use bon::Builder;
use http::{Method, request::Parts, uri::PathAndQuery};
use serde::{Deserialize, Serialize};

use crate::{Protocol, Result, gemini, interactions, openai_chat};

/// The detected endpoint and any model identity carried by its path.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct Endpoint {
    /// Selected protocol family.
    #[builder(start_fn)]
    pub protocol: Protocol,
    /// Model selected by the route.
    #[builder(into)]
    pub model_from_path: Option<String>,
    /// Whether the route explicitly selects streaming.
    #[builder(default)]
    pub stream: bool,
    #[builder(skip)]
    #[serde(skip)]
    pub(crate) passthrough: Option<Passthrough>,
}

impl From<Protocol> for Endpoint {
    fn from(protocol: Protocol) -> Self {
        Self::builder(protocol).build()
    }
}

impl Endpoint {
    /// Creates a native route to an origin-relative upstream path and query.
    ///
    /// The gateway forwards the body and the allowed headers to an upstream
    /// using the same protocol. Its successful response body is forwarded as
    /// received, including any server-sent events.
    #[must_use]
    pub fn passthrough(protocol: Protocol, method: Method, path: PathAndQuery) -> Self {
        Self {
            passthrough: Some(Passthrough {
                method,
                path,
                allow_body: true,
            }),
            ..Self::from(protocol)
        }
    }

    /// Returns whether the route forwards the native HTTP payloads.
    #[must_use]
    pub const fn is_passthrough(&self) -> bool {
        self.passthrough.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Passthrough {
    pub(crate) method: Method,
    pub(crate) path: PathAndQuery,
    pub(crate) allow_body: bool,
}

type Matcher = dyn Fn(&Parts) -> Option<Endpoint> + Send + Sync;

/// A matcher that selects an endpoint from an incoming HTTP request.
///
/// Custom routes are tried in registration order before the built-in routes.
/// An explicit [`Endpoint`] request extension takes precedence over both.
#[derive(Clone)]
pub struct Route(Arc<Matcher>);

impl Route {
    /// Creates a matcher that can inspect the method, URI, headers, and extensions.
    pub fn new(matcher: impl Fn(&Parts) -> Option<Endpoint> + Send + Sync + 'static) -> Self {
        Self(Arc::new(matcher))
    }

    /// Creates an alias for a model API using a path pattern.
    ///
    /// A segment in braces matches one segment and a segment prefixed with
    /// an asterisk inside braces matches zero or more segments.
    pub fn alias(pattern: impl Into<String>, protocol: Protocol) -> Self {
        let pattern = pattern.into();
        Self::new(move |parts| {
            route_matches(&pattern, parts.uri.path()).then(|| Endpoint::from(protocol))
        })
    }

    pub(crate) fn detect(&self, parts: &Parts) -> Option<Endpoint> {
        (self.0)(parts)
    }
}

impl fmt::Debug for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Route").finish_non_exhaustive()
    }
}

pub(crate) fn detect_builtin(parts: &Parts) -> Result<Option<Endpoint>> {
    if let Some(endpoint) = interactions::detect_resource(parts)? {
        return Ok(Some(endpoint));
    }
    if parts.method != Method::POST {
        return Ok(None);
    }
    let path = parts.uri.path();
    let endpoint = if path.ends_with("/interactions") {
        Some(Protocol::Interactions.into())
    } else if path.ends_with("/messages") {
        Some(Protocol::Messages.into())
    } else if path.ends_with("/responses") {
        Some(Protocol::Responses.into())
    } else {
        openai_chat::detect(parts).or_else(|| gemini::detect(parts))
    };
    Ok(endpoint)
}

fn route_matches(pattern: &str, path: &str) -> bool {
    let pattern = split_segments(pattern).collect::<Vec<_>>();
    let path = split_segments(path).collect::<Vec<_>>();
    route_segments_match(&pattern, &path)
}

fn route_segments_match(pattern: &[&str], path: &[&str]) -> bool {
    let Some((head, tail)) = pattern.split_first() else {
        return path.is_empty();
    };
    if head.starts_with("{*") {
        return (0..=path.len()).any(|count| route_segments_match(tail, &path[count..]));
    }
    let Some((value, rest)) = path.split_first() else {
        return false;
    };
    ((is_capture(head) && !value.is_empty()) || *head == *value) && route_segments_match(tail, rest)
}

fn split_segments(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|segment| !segment.is_empty())
}

fn is_capture(segment: &str) -> bool {
    segment.starts_with('{') && segment.ends_with('}') && !segment.starts_with("{*")
}
