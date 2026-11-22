use bon::Builder;
use http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::Protocol;

/// The result type used by protocol operations.
pub type Result<T> = core::result::Result<T, Error>;

/// An owned, thread-safe error used at asynchronous body boundaries.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Broad API error categories shared by all supported providers.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorKind {
    /// The request is malformed or cannot be represented by the target API.
    InvalidRequest,
    /// Authentication failed.
    Authentication,
    /// The authenticated caller cannot perform this operation.
    Permission,
    /// No endpoint, model, or resource matched.
    NotFound,
    /// The request exceeded an admission or provider rate limit.
    RateLimit,
    /// The upstream is temporarily overloaded.
    Overloaded,
    /// An operation exceeded its deadline.
    Timeout,
    /// A requested protocol capability is unavailable.
    Unsupported,
    /// A request body exceeded its configured byte limit.
    BodyTooLarge,
    /// The upstream returned an invalid or incomplete response.
    BadResponse,
    /// An upstream transport or provider failure occurred.
    Upstream,
    /// An internal gateway operation failed.
    #[default]
    Internal,
}

impl ApiErrorKind {
    /// Returns the conventional HTTP status for this category.
    #[must_use]
    pub const fn status(self) -> StatusCode {
        match self {
            Self::InvalidRequest | Self::Unsupported => StatusCode::BAD_REQUEST,
            Self::Authentication => StatusCode::UNAUTHORIZED,
            Self::Permission => StatusCode::FORBIDDEN,
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::RateLimit => StatusCode::TOO_MANY_REQUESTS,
            Self::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Timeout => StatusCode::GATEWAY_TIMEOUT,
            Self::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            Self::BadResponse | Self::Upstream => StatusCode::BAD_GATEWAY,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Classifies an upstream HTTP status; the inverse of [`Self::status`].
    #[must_use]
    pub fn from_status(status: StatusCode) -> Self {
        match status {
            StatusCode::UNAUTHORIZED => Self::Authentication,
            StatusCode::FORBIDDEN => Self::Permission,
            StatusCode::NOT_FOUND => Self::NotFound,
            StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => Self::Timeout,
            StatusCode::PAYLOAD_TOO_LARGE => Self::BodyTooLarge,
            StatusCode::TOO_MANY_REQUESTS => Self::RateLimit,
            StatusCode::SERVICE_UNAVAILABLE => Self::Overloaded,
            status if status.is_server_error() => Self::Upstream,
            status if status.is_client_error() => Self::InvalidRequest,
            _ => Self::Internal,
        }
    }

    /// Returns the stable machine-readable category name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Authentication => "authentication",
            Self::Permission => "permission",
            Self::NotFound => "not_found",
            Self::RateLimit => "rate_limit",
            Self::Overloaded => "overloaded",
            Self::Timeout => "timeout",
            Self::Unsupported => "unsupported",
            Self::BodyTooLarge => "body_too_large",
            Self::BadResponse => "bad_response",
            Self::Upstream => "upstream",
            Self::Internal => "internal",
        }
    }
}

/// Protocol-independent API error data.
///
/// Unlike [`Error`], this is response content: upstream non-success responses
/// remain values so middleware can inspect, retry, or translate them.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize)]
pub struct ApiError {
    /// Broad error category.
    pub kind: ApiErrorKind,
    /// Human-readable detail safe to return to a client.
    #[builder(into)]
    pub message: String,
    /// Provider-specific or gateway-specific stable code, when available.
    #[builder(into)]
    pub code: Option<String>,
}

impl ApiError {
    /// Creates an error with no provider-specific code.
    pub fn new(kind: ApiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            code: None,
        }
    }

    /// Returns the status used when no upstream status must be preserved.
    #[must_use]
    pub const fn status(&self) -> StatusCode {
        self.kind.status()
    }
}

/// A failure to decode, translate, encode, or transport protocol data.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Wire data does not conform to the selected protocol.
    #[error("invalid {protocol} payload: {message}")]
    Protocol {
        /// Protocol being decoded or encoded.
        protocol: Protocol,
        /// Validation or serde detail.
        message: String,
    },
    /// An IR value cannot be represented by the requested protocol.
    #[error("invalid request for {protocol}: {message}")]
    InvalidRequest {
        /// Target protocol.
        protocol: Protocol,
        /// Validation detail.
        message: String,
    },
    /// No endpoint could be selected without guessing semantics.
    #[error("no model API endpoint matches {path}")]
    NotFound {
        /// Request path that failed detection.
        path: String,
    },
    /// A request or response body exceeded its limit.
    #[error("body exceeds the {limit}-byte limit")]
    BodyTooLarge {
        /// Configured maximum size.
        limit: usize,
    },
    /// Lossy translation was rejected by policy.
    #[error("translation rejected by loss policy: {message}")]
    LossPolicy {
        /// Diagnostic that triggered rejection.
        message: String,
    },
    /// An HTTP body failed while being collected or streamed.
    #[error("HTTP body error: {message}")]
    Body {
        /// Transport detail.
        message: String,
    },
    /// An HTTP value could not be constructed.
    #[error("HTTP construction error: {message}")]
    Http {
        /// Construction detail.
        message: String,
    },
    /// An internal operation failed without a more specific category.
    #[error("internal protocol error: {message}")]
    Internal {
        /// Failure detail.
        message: String,
    },
}

impl Error {
    /// Converts this Rust error into data suitable for a protocol response.
    #[must_use]
    pub fn to_api_error(&self) -> ApiError {
        let (kind, code) = match self {
            Self::Protocol { .. } | Self::InvalidRequest { .. } => {
                (ApiErrorKind::InvalidRequest, "invalid_request")
            }
            Self::NotFound { .. } => (ApiErrorKind::NotFound, "not_found"),
            Self::BodyTooLarge { .. } => (ApiErrorKind::BodyTooLarge, "body_too_large"),
            Self::LossPolicy { .. } => (ApiErrorKind::Unsupported, "loss_policy"),
            Self::Body { .. } => (ApiErrorKind::BadResponse, "body_error"),
            Self::Http { .. } | Self::Internal { .. } => (ApiErrorKind::Internal, "internal"),
        };
        ApiError {
            kind,
            message: self.to_string(),
            code: Some(code.to_owned()),
        }
    }

    pub(crate) fn json(protocol: Protocol, error: serde_json::Error) -> Self {
        Self::Protocol {
            protocol,
            message: error.to_string(),
        }
    }

    pub(crate) fn http(error: http::Error) -> Self {
        Self::Http {
            message: error.to_string(),
        }
    }
}
