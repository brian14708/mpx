/// Errors produced by the Tower transport boundary.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Protocol decoding, translation, or encoding failed.
    #[error(transparent)]
    Protocol(#[from] mpx_protocol::Error),
    /// No `Upstream` extension or configured default was available.
    #[error("request has no upstream target")]
    MissingUpstream,
    /// The configured upstream URI could not be constructed.
    #[error("invalid upstream URI: {0}")]
    InvalidUpstreamUri(String),
    /// The injected transport failed.
    #[error("upstream transport failed: {0}")]
    Transport(#[source] mpx_protocol::BoxError),
    /// The upstream response could not be decoded or collected.
    #[error("invalid upstream response: {0}")]
    UpstreamResponse(#[source] mpx_protocol::Error),
}
