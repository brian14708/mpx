use mpx_protocol::{BoxError, Request, ResponseBody};
use tower_service::Service;

/// A cloneable Tower service that accepts and returns normalized protocol IR.
///
/// Timeout, concurrency-limit, retry, and continuation layers can use this
/// blanket alias without depending on a concrete gateway error type.
pub trait ModelService:
    Service<http::Request<Request>, Response = http::Response<ResponseBody>, Error: Into<BoxError>>
    + Clone
{
}

impl<S> ModelService for S where
    S: Service<
            http::Request<Request>,
            Response = http::Response<ResponseBody>,
            Error: Into<BoxError>,
        > + Clone
{
}
