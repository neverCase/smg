//! Native Cohere chat router.
//!
//! This router only extracts gateway routing metadata from `/v1/chat` and
//! `/v2/chat` requests. The original JSON bytes are forwarded unchanged.

use std::{any::Any, fmt, sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::StreamExt;
use serde_json::Value;

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::{
                apply_provider_headers, extract_auth_header, should_forward_request_header,
            },
            retry::{is_retryable_status, RetryExecutor},
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        error, RouterTrait,
    },
    worker::{ConnectionMode, ProviderType, WorkerRegistry},
};

/// Cohere chat endpoint version accepted by SMG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohereChatEndpoint {
    V1,
    V2,
}

impl CohereChatEndpoint {
    /// Return the upstream path that should be preserved for this endpoint.
    pub fn path(self) -> &'static str {
        match self {
            Self::V1 => "/v1/chat",
            Self::V2 => "/v2/chat",
        }
    }

    /// Return the metrics endpoint label for this Cohere route.
    fn metrics_label(self) -> &'static str {
        match self {
            Self::V1 => metrics_labels::ENDPOINT_COHERE_CHAT_V1,
            Self::V2 => metrics_labels::ENDPOINT_COHERE_CHAT_V2,
        }
    }
}

/// Routing metadata extracted from a native Cohere chat request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohereChatMetadata {
    model: String,
    stream: bool,
}

/// Raw Cohere chat request plus the minimal metadata SMG needs for routing.
#[derive(Debug, Clone)]
pub struct CohereChatRequest {
    endpoint: CohereChatEndpoint,
    body: Bytes,
    metadata: CohereChatMetadata,
}

impl CohereChatRequest {
    /// Parse only routing metadata while retaining the original request bytes.
    pub fn from_bytes(
        endpoint: CohereChatEndpoint,
        body: Bytes,
    ) -> Result<Self, CohereChatRequestError> {
        let value: Value = serde_json::from_slice(&body)
            .map_err(|e| CohereChatRequestError::new(format!("Invalid Cohere JSON body: {e}")))?;

        let model = value
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| {
                CohereChatRequestError::new(
                    "Cohere chat routing requires a non-empty 'model' field".to_string(),
                )
            })?
            .to_string();

        let stream = value
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        Ok(Self {
            endpoint,
            body,
            metadata: CohereChatMetadata { model, stream },
        })
    }

    /// Return the endpoint version and path for this request.
    pub fn endpoint(&self) -> CohereChatEndpoint {
        self.endpoint
    }

    /// Return the original raw JSON request bytes.
    pub fn body(&self) -> &Bytes {
        &self.body
    }

    /// Return the model id used for worker selection.
    pub fn model(&self) -> &str {
        &self.metadata.model
    }

    /// Return whether the request asks for streaming.
    pub fn stream(&self) -> bool {
        self.metadata.stream
    }
}

/// Error returned when Cohere routing metadata cannot be extracted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohereChatRequestError {
    message: String,
}

impl CohereChatRequestError {
    /// Create a request metadata extraction error.
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for CohereChatRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CohereChatRequestError {}

/// Router for native Cohere chat APIs.
pub struct CohereRouter {
    client: reqwest::Client,
    worker_registry: Arc<WorkerRegistry>,
    retry_config: RetryConfig,
}

impl fmt::Debug for CohereRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohereRouter").finish()
    }
}

impl CohereRouter {
    /// Create a Cohere router from the shared application context.
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self {
            client: ctx.client.clone(),
            worker_registry: ctx.worker_registry.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        }
    }
}

#[async_trait]
impl RouterTrait for CohereRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn route_cohere_chat(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &CohereChatRequest,
        model_id: &str,
    ) -> Response {
        route_cohere_chat(self, headers, body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        "cohere"
    }
}

/// Route one native Cohere chat request to a Cohere-capable worker.
async fn route_cohere_chat(
    router: &CohereRouter,
    headers: Option<&HeaderMap>,
    body: &CohereChatRequest,
    model_id: &str,
) -> Response {
    let start = Instant::now();
    let model = model_id;
    let endpoint = body.endpoint();
    let endpoint_label = endpoint.metrics_label();
    let streaming = body.stream();

    Metrics::record_router_request(
        metrics_labels::ROUTER_COHERE,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        endpoint_label,
        bool_to_static_str(streaming),
    );

    let selector = WorkerSelector::new(&router.worker_registry, &router.client);
    let worker = match selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers,
            provider: Some(ProviderType::Cohere),
            connection_mode: Some(ConnectionMode::Http),
            ..Default::default()
        })
        .await
    {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_COHERE,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint_label,
                metrics_labels::ERROR_NO_WORKERS,
            );
            return response;
        }
    };

    let upstream_url = format!("{}{}", worker.url().trim_end_matches('/'), endpoint.path());
    let request_body = Arc::new(body.body().clone());
    let headers_cloned = Arc::new(headers.cloned());
    let worker_api_key = Arc::new(worker.api_key().cloned());
    let client = router.client.clone();
    let worker = Arc::clone(&worker);

    let response = RetryExecutor::execute_response_with_retry(
        &router.retry_config,
        |_attempt| {
            let client = client.clone();
            let upstream_url = upstream_url.clone();
            let request_body = Arc::clone(&request_body);
            let headers = Arc::clone(&headers_cloned);
            let worker_api_key = Arc::clone(&worker_api_key);
            let worker = Arc::clone(&worker);

            async move {
                let mut req = client
                    .post(&upstream_url)
                    .body((*request_body).clone())
                    .header(
                        CONTENT_TYPE,
                        original_content_type(headers.as_ref().as_ref()),
                    );

                if let Some(headers) = headers.as_ref().as_ref() {
                    for (name, value) in headers {
                        if should_forward_request_header(name.as_str())
                            && !name.as_str().eq_ignore_ascii_case("authorization")
                        {
                            req = req.header(name.clone(), value.clone());
                        }
                    }
                }

                let auth_header =
                    extract_auth_header(headers.as_ref().as_ref(), (*worker_api_key).as_ref());
                req = apply_provider_headers(req, &upstream_url, auth_header.as_ref());

                if streaming {
                    req = req.header("Accept", "text/event-stream");
                }

                let resp = match req.send().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        worker.record_outcome(503);
                        return error::service_unavailable(
                            "upstream_error",
                            format!("Failed to contact upstream: {e}"),
                        );
                    }
                };

                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                worker.record_outcome(status.as_u16());
                let content_type = resp.headers().get(CONTENT_TYPE).cloned();

                if streaming {
                    let stream = resp
                        .bytes_stream()
                        .map(|chunk| chunk.map_err(std::io::Error::other));
                    let mut response = Response::new(Body::from_stream(stream));
                    *response.status_mut() = status;
                    if let Some(ct) = content_type {
                        response.headers_mut().insert(CONTENT_TYPE, ct);
                    }
                    response
                } else {
                    match resp.bytes().await {
                        Ok(bytes) => {
                            let mut response = Response::new(Body::from(bytes));
                            *response.status_mut() = status;
                            if let Some(ct) = content_type {
                                response.headers_mut().insert(CONTENT_TYPE, ct);
                            }
                            response
                        }
                        Err(e) => error::internal_error(
                            "upstream_error",
                            format!("Failed to read response: {e}"),
                        ),
                    }
                }
            }
        },
        |res, _attempt| is_retryable_status(res.status()),
        |delay, attempt| {
            Metrics::record_worker_retry(metrics_labels::BACKEND_EXTERNAL, endpoint_label);
            Metrics::record_worker_retry_backoff(attempt, delay);
        },
        || {
            Metrics::record_worker_retries_exhausted(
                metrics_labels::BACKEND_EXTERNAL,
                endpoint_label,
            );
        },
    )
    .await;

    if response.status().is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_COHERE,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint_label,
            start.elapsed(),
        );
    } else {
        Metrics::record_router_error(
            metrics_labels::ROUTER_COHERE,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint_label,
            metrics_labels::ERROR_BACKEND,
        );
    }

    response
}

/// Resolve the client-supplied content type or use Cohere's JSON default.
fn original_content_type(headers: Option<&HeaderMap>) -> HeaderValue {
    headers
        .and_then(|headers| headers.get(CONTENT_TYPE))
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"))
}
