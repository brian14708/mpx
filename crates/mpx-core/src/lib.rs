//! Tower middleware for the `mpx-protocol` editable IR.
//!
//! [`IngressLayer`] adapts collected client HTTP to normalized requests and
//! encodes the response for the original client. [`UpstreamService`] sends those
//! requests to an injected transport selected by [`Upstream`].

mod client;
mod error;
mod ingress;
mod redaction;
mod service;

/// Compatibility name for [`UpstreamService`].
pub use client::UpstreamService as Client;
pub use client::{Upstream, UpstreamService};
pub use error::Error;
pub use ingress::{Ingress, IngressLayer};
pub use redaction::{Redaction, RedactionLayer};
pub use service::ModelService;

pub use mpx_protocol;
pub use mpx_redact;
