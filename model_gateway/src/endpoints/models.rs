//! `GET /v1/models` and `GET /v1/models/{id}`: the gateway's model inventory.
//!
//! Self-hosted models are read from the worker registry. A caller presenting
//! its own provider credential (bring your own key) is answered from the
//! external upstreams instead, but never when that credential is one of the
//! gateway's own keys: a tenant-scoped key must not be forwarded to a third
//! party as if it were the caller's.
//! When remote auth is configured, its token is never treated as BYOK: the
//! endpoint returns only healthy local models approved by `/allowed-models`.
//!
//! The list and the single model come in the shape of the API family the
//! caller speaks: a request carrying the `anthropic-version` header gets the
//! Anthropic objects (`type`, `display_name`, `created_at`, page markers),
//! every other request the OpenAI ones (`object`, `created`, `owned_by`).

use std::collections::HashSet;

use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::future::select_all;
use openai_protocol::{messages, model_card::ModelCard, models::ListModelsResponse};
use serde_json::Value;
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    middleware::{into_messages_envelope, AuthConfig, RemoteAuthClient},
    routers::{common::header_utils::apply_provider_headers, error},
    worker::{ProviderType, RuntimeType, WorkerRegistry},
};

// Return all health model cards
pub fn get_models_all(context: &AppContext) -> Vec<ModelCard> {
    context
        .worker_registry
        .get_all()
        .iter()
        .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
        .flat_map(|w| w.models())
        .collect()
}

/// The API family a models request speaks, read from its headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelShape {
    OpenAi,
    Anthropic,
}

impl ModelShape {
    /// Anthropic clients send `anthropic-version` on every request; no
    /// OpenAI-compatible client does.
    fn from_headers(headers: &HeaderMap) -> Self {
        if headers.contains_key("anthropic-version") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }

    fn list(self, cards: Vec<ModelCard>) -> Response {
        match self {
            Self::OpenAi => Json(ListModelsResponse::from_model_cards(cards)).into_response(),
            Self::Anthropic => {
                Json(messages::ListModelsResponse::from_model_cards(cards)).into_response()
            }
        }
    }

    fn single(self, card: ModelCard) -> Response {
        match self {
            Self::OpenAi => Json(card.into_model_object()).into_response(),
            Self::Anthropic => Json(messages::ModelInfo::from_model_card(&card)).into_response(),
        }
    }

    /// The 404 of an unknown id: the same OpenAI-shaped envelope an unknown
    /// model gets on every other route, re-shaped into the Messages envelope
    /// (`not_found_error`, top-level `type`, `request_id`) for a Messages
    /// client, as `/v1/messages` answers it.
    async fn not_found(self, model_id: &str, request_id: String) -> Response {
        self.error(error::model_not_found(model_id), request_id)
            .await
    }

    /// The 503 of an inventory with no model in it (no worker registered, or
    /// every worker gone): the gateway's error envelope with a typed code, as
    /// every other refusal answers, in the caller's shape.

    async fn no_models(self, request_id: String) -> Response {
        self.error(
            error::service_unavailable("no_models_available", "No models available"),
            request_id,
        )
        .await
    }

    /// An OpenAI-shaped error response in the caller's shape: as it is for an
    /// OpenAI client, re-shaped into the Messages envelope for a Messages
    /// client.

    async fn error(self, response: Response, request_id: String) -> Response {
        match self {
            Self::OpenAi => response,
            Self::Anthropic => into_messages_envelope(response, request_id).await,
        }
    }
}

/// Answer `GET /v1/models` for the caller identified by `headers`
/// (`request_id` names the request in the Messages envelope of a refusal).
pub async fn list_models(
    context: &AppContext,
    headers: &HeaderMap,
    request_id: String,
) -> Response {
    if let Some(remote_auth) = context.remote_auth_client.as_ref() {
        return list_remote_authorized_models(
            &context.worker_registry,
            remote_auth,
            headers,
            request_id,
        )
        .await;
    }

    list_models_with(
        &context.worker_registry,
        &context.client,
        &context.gateway_auth,
        headers,
        request_id,
    )
    .await
}

/// Return the healthy local models authorized by the remote auth service.
///
/// Remote-auth tokens are platform credentials, so this path intentionally
/// bypasses BYOK discovery and never forwards the token to an external worker.
async fn list_remote_authorized_models(
    registry: &WorkerRegistry,
    remote_auth: &RemoteAuthClient,
    headers: &HeaderMap,
    request_id: String,
) -> Response {
    match remote_authorized_cards(registry, remote_auth, headers, request_id).await {
        Ok(cards) => ModelShape::from_headers(headers).list(cards),
        Err(response) => response,
    }
}

async fn remote_authorized_cards(
    registry: &WorkerRegistry,
    remote_auth: &RemoteAuthClient,
    headers: &HeaderMap,
    request_id: String,
) -> Result<Vec<ModelCard>, Response> {
    let shape = ModelShape::from_headers(headers);
    let Some(token) = remote_auth.extract_token(headers) else {
        let response = (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": "Missing bearer token",
                    "type": "authentication_error",
                }
            })),
        )
            .into_response();
        return Err(shape.error(response, request_id).await);
    };

    let Some(authorization) = remote_auth.allowed_models(&token).await else {
        return Err(shape.no_models(request_id).await);
    };

    let allowed_models = (!authorization.allow_all)
        .then(|| authorization.models.into_iter().collect::<HashSet<_>>());

    Ok(authorized_self_hosted_cards(
        registry,
        allowed_models.as_ref(),
    ))
}

/// [`list_models`] over explicit collaborators.
pub(crate) async fn list_models_with(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    gateway_auth: &AuthConfig,
    headers: &HeaderMap,
    request_id: String,
) -> Response {
    let bearer_token = bearer_token(headers);
    let shape = ModelShape::from_headers(headers);

    // Short-circuit: a token that is one of the gateway's own credentials
    // is answered from the registry. It must never reach the BYOK fan-out
    // below, which would forward it externally.
    if let Some(ref token) = bearer_token {
        if gateway_auth.contains_token(token) {
            return registry_models_response(registry, shape, request_id).await;
        }
    }

    // A foreign token is tried against the external upstreams first (bring
    // your own key). On total failure fall through to the registry.
    if let Some(ref token) = bearer_token {
        let upstream_cards = fetch_upstream_models(registry, client, token).await;
        if !upstream_cards.is_empty() {
            return shape.list(upstream_cards);
        }
    }

    registry_models_response(registry, shape, request_id).await
}

/// Answer `GET /v1/models/{id}` from the self-hosted inventory: the card of
/// the model `model_id` names (its id, or an alias the registry resolves),
/// in the caller's API shape, or the 404 an unknown model gets on every other
/// route, in the caller's error envelope (`request_id` names the request in
/// the Messages envelope).
pub async fn retrieve_model(
    context: &AppContext,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    if let Some(remote_auth) = context.remote_auth_client.as_ref() {
        return retrieve_remote_authorized_model(
            &context.worker_registry,
            remote_auth,
            headers,
            model_id,
            request_id,
        )
        .await;
    }
    retrieve_model_with(&context.worker_registry, headers, model_id, request_id).await
}

async fn retrieve_remote_authorized_model(
    registry: &WorkerRegistry,
    remote_auth: &RemoteAuthClient,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    let cards =
        match remote_authorized_cards(registry, remote_auth, headers, request_id.clone()).await {
            Ok(cards) => cards,
            Err(response) => return response,
        };
    let canonical = registry.resolve_model_alias(model_id);
    let wanted = canonical.as_deref().unwrap_or(model_id);
    let shape = ModelShape::from_headers(headers);
    match cards.into_iter().find(|card| card.id == wanted) {
        Some(card) => shape.single(card),
        None => shape.not_found(model_id, request_id).await,
    }
}

/// [`retrieve_model`] over an explicit registry. The lookup is by the
/// resolved canonical id alone: a canonical id that is also another card's
/// alias names its own card, as it does on every other route.
pub(crate) async fn retrieve_model_with(
    registry: &WorkerRegistry,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    let canonical = registry.resolve_model_alias(model_id);
    let wanted = canonical.as_deref().unwrap_or(model_id);
    let card = self_hosted_cards(registry)
        .into_iter()
        .find(|card| card.id == wanted);
    let shape = ModelShape::from_headers(headers);
    match card {
        Some(card) => shape.single(card),
        None => shape.not_found(model_id, request_id).await,
    }
}

/// The caller's credential: a `Bearer` token (case-insensitive scheme per
/// RFC 7235) or an Anthropic-style `x-api-key` header.
fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| {
            let lower = h.to_ascii_lowercase();
            lower.starts_with("bearer ").then(|| h[7..].to_string())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|h| h.to_str().ok())
                .map(String::from)
        })
}

/// The self-hosted inventory: every model card of every non-external worker.
fn self_hosted_cards(registry: &WorkerRegistry) -> Vec<ModelCard> {
    registry
        .get_all()
        .iter()
        .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
        .flat_map(|w| w.models())
        .collect()
}

/// The remotely-authorized inventory contains only healthy local workers.
fn authorized_self_hosted_cards(
    registry: &WorkerRegistry,
    allowed_models: Option<&HashSet<String>>,
) -> Vec<ModelCard> {
    registry
        .get_workers_filtered(None, None, None, None, true)
        .iter()
        .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
        .flat_map(|w| w.models())
        .filter(|card| allowed_models.is_none_or(|allowed| allowed.contains(&card.id)))
        .collect()
}

/// The self-hosted inventory as a list in the caller's shape, or the 503 of
/// an empty one in the caller's error envelope.
async fn registry_models_response(
    registry: &WorkerRegistry,
    shape: ModelShape,
    request_id: String,
) -> Response {
    let cards = self_hosted_cards(registry);
    if cards.is_empty() {
        shape.no_models(request_id).await
    } else {
        shape.list(cards)
    }
}
/// Fan out to all healthy external upstreams concurrently with the caller's
/// bearer token and return the first successful model inventory. Returns an
/// empty vec on total failure.
async fn fetch_upstream_models(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    bearer_token: &str,
) -> Vec<ModelCard> {
    let unique_urls: Vec<_> = registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .iter()
        .map(|w| w.url().to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if unique_urls.is_empty() {
        return Vec::new();
    }

    debug!(
        "Trying {} upstream(s) for model discovery",
        unique_urls.len()
    );

    let auth = match HeaderValue::from_str(&format!("Bearer {bearer_token}")) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!("Bearer token contains invalid header characters: {e}");
            return Vec::new();
        }
    };

    // Fan out concurrently; return the first non-empty result.
    let mut pending: Vec<_> = unique_urls
        .into_iter()
        .map(|url| Box::pin(fetch_models_from(client.clone(), url, auth.clone())))
        .collect();

    while !pending.is_empty() {
        let (cards, _index, remaining) = select_all(pending).await;
        if !cards.is_empty() {
            return cards;
        }
        pending = remaining;
    }

    Vec::new()
}

/// Fetch models from a single upstream endpoint.
async fn fetch_models_from(
    client: reqwest::Client,
    base_url: String,
    auth: Option<HeaderValue>,
) -> Vec<ModelCard> {
    let url = format!("{base_url}/v1/models");
    let req = apply_provider_headers(client.get(&url), &url, auth.as_ref());

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            debug!("Failed to reach upstream {url}: {e}");
            return Vec::new();
        }
    };

    if !resp.status().is_success() {
        debug!(
            "Upstream {url} returned {} for model discovery",
            resp.status()
        );
        return Vec::new();
    }

    match resp.json::<Value>().await {
        Ok(json) => ListModelsResponse::parse_upstream(&json, ProviderType::from_url(&url)),
        Err(e) => {
            warn!("Failed to parse upstream models from {url}: {e}");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::{
        http::{Request, StatusCode},
        routing::post,
    };
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        config::types::{RemoteAuthConfig, TenantApiKeyEntry},
        worker::{BasicWorkerBuilder, Worker, WorkerType},
    };

    #[expect(
        clippy::disallowed_methods,
        clippy::unwrap_used,
        reason = "test infrastructure"
    )]
    async fn spawn_remote_auth(
        response: serde_json::Value,
    ) -> (RemoteAuthClient, Arc<AtomicUsize>) {
        let hit_count = Arc::new(AtomicUsize::new(0));
        let handler_hit_count = hit_count.clone();
        let app = axum::Router::new().route(
            "/allowed-models",
            post(move |Json(body): Json<serde_json::Value>| {
                let response = response.clone();
                let hit_count = handler_hit_count.clone();
                async move {
                    hit_count.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(body["token"], "test-token");
                    Json(response)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = RemoteAuthClient::new(
            reqwest::Client::new(),
            RemoteAuthConfig {
                url: format!("http://{addr}"),
                timeout_secs: 1,
                fail_closed: true,
                token_header: "authorization".to_string(),
            },
        );
        (client, hit_count)
    }

    fn register_local_model(registry: &WorkerRegistry, url: &str, model: &str, healthy: bool) {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::Sglang)
                .models(vec![ModelCard::new(model)])
                .health_config(HealthCheckConfig {
                    disable_health_check: healthy,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker);
    }

    #[expect(clippy::unwrap_used, reason = "test response parsing")]
    async fn response_model_ids(response: Response) -> HashSet<String> {
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response: ListModelsResponse = serde_json::from_slice(&body).unwrap();
        response.data.into_iter().map(|model| model.id).collect()
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test request construction")]
    async fn remote_auth_returns_only_allowed_available_local_models() {
        let registry = WorkerRegistry::new();
        register_local_model(&registry, "http://allowed.invalid", "allowed", true);
        register_local_model(&registry, "http://denied.invalid", "denied", true);
        register_local_model(&registry, "http://unhealthy.invalid", "unhealthy", false);
        let external_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://external.invalid")
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(vec![ModelCard::new("external")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(external_worker);

        let (remote_auth, hit_count) = spawn_remote_auth(serde_json::json!({
            "allow_all": false,
            "models": ["allowed", "unhealthy", "external", "not-registered"]
        }))
        .await;
        let request = Request::builder()
            .header(header::AUTHORIZATION, "bearer test-token")
            .body(())
            .unwrap();

        let response =
            list_remote_authorized_models(&registry, &remote_auth, request.headers(), rid()).await;

        assert_eq!(
            response_model_ids(response).await,
            HashSet::from(["allowed".into()])
        );
        assert_eq!(hit_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test request construction")]
    async fn remote_auth_allow_all_returns_every_available_local_model() {
        let registry = WorkerRegistry::new();
        register_local_model(&registry, "http://one.invalid", "model-one", true);
        register_local_model(&registry, "http://two.invalid", "model-two", true);

        // `models` may be omitted when `allow_all` is true.
        let (remote_auth, _) = spawn_remote_auth(serde_json::json!({ "allow_all": true })).await;
        let request = Request::builder()
            .header(header::AUTHORIZATION, "Bearer test-token")
            .body(())
            .unwrap();

        let response =
            list_remote_authorized_models(&registry, &remote_auth, request.headers(), rid()).await;

        assert_eq!(
            response_model_ids(response).await,
            HashSet::from(["model-one".into(), "model-two".into()])
        );
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test request construction")]
    async fn remote_auth_returns_empty_list_when_no_allowed_model_is_available() {
        let registry = WorkerRegistry::new();
        register_local_model(&registry, "http://local.invalid", "local-model", true);

        let (remote_auth, _) = spawn_remote_auth(serde_json::json!({
            "allow_all": false,
            "models": ["not-available"]
        }))
        .await;
        let request = Request::builder()
            .header(header::AUTHORIZATION, "Bearer test-token")
            .body(())
            .unwrap();

        let response =
            list_remote_authorized_models(&registry, &remote_auth, request.headers(), rid()).await;

        assert!(response_model_ids(response).await.is_empty());
    }

    #[tokio::test]
    async fn remote_auth_requires_token_before_fetching_allowed_models() {
        let registry = WorkerRegistry::new();
        let (remote_auth, hit_count) =
            spawn_remote_auth(serde_json::json!({ "allow_all": true })).await;

        let response =
            list_remote_authorized_models(&registry, &remote_auth, &HeaderMap::new(), rid()).await;

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let mut empty_token_headers = HeaderMap::new();
        empty_token_headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer "));
        let response =
            list_remote_authorized_models(&registry, &remote_auth, &empty_token_headers, rid())
                .await;

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(hit_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn remote_auth_list_preserves_the_anthropic_shape() {
        let registry = WorkerRegistry::new();
        register_local_model(&registry, "http://allowed.invalid", "allowed", true);
        register_local_model(&registry, "http://denied.invalid", "denied", true);
        let (remote_auth, hits) = spawn_remote_auth(serde_json::json!({
            "allow_all": false,
            "models": ["allowed"]
        }))
        .await;
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("test-token"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let response =
            list_remote_authorized_models(&registry, &remote_auth, &headers, rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["data"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["data"][0]["id"], "allowed");
        assert_eq!(body["data"][0]["type"], "model");
        assert_eq!(body["has_more"], false);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn remote_auth_single_model_requires_an_allowed_healthy_local_card() {
        let registry = WorkerRegistry::new();
        let allowed: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://allowed.invalid")
                .runtime_type(RuntimeType::Sglang)
                .models(vec![ModelCard::new("allowed").with_alias("alias")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(allowed);
        register_local_model(&registry, "http://denied.invalid", "denied", true);
        register_local_model(&registry, "http://unhealthy.invalid", "unhealthy", false);
        let external: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://external.invalid")
                .runtime_type(RuntimeType::External)
                .models(vec![ModelCard::new("external")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(external);
        let (remote_auth, _) = spawn_remote_auth(serde_json::json!({
            "allow_all": false,
            "models": ["allowed", "unhealthy", "external", "absent"]
        }))
        .await;
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-token"),
        );

        for anthropic in [false, true] {
            if anthropic {
                headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            }
            for model in [
                "allowed",
                "alias",
                "denied",
                "unhealthy",
                "external",
                "absent",
            ] {
                let response = retrieve_remote_authorized_model(
                    &registry,
                    &remote_auth,
                    &headers,
                    model,
                    rid(),
                )
                .await;
                let expected_status = if matches!(model, "allowed" | "alias") {
                    StatusCode::OK
                } else {
                    StatusCode::NOT_FOUND
                };
                assert_eq!(response.status(), expected_status, "{model}");
                let body = json_body(response).await;
                if expected_status == StatusCode::OK {
                    assert_eq!(body["id"], "allowed");
                    if anthropic {
                        assert_eq!(body["type"], "model");
                    } else {
                        assert_eq!(body["object"], "model");
                    }
                } else if anthropic {
                    assert_eq!(body["error"]["type"], "not_found_error");
                    assert_eq!(body["request_id"], rid());
                } else {
                    assert_eq!(body["error"]["code"], "model_not_found");
                }
            }
        }
    }

    #[tokio::test]
    async fn remote_auth_single_model_rejects_missing_token_before_discovery() {
        let registry = WorkerRegistry::new();
        let (remote_auth, hits) = spawn_remote_auth(serde_json::json!({"allow_all": true})).await;
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let response =
            retrieve_remote_authorized_model(&registry, &remote_auth, &headers, "allowed", rid())
                .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = json_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "authentication_error");
        assert_eq!(body["request_id"], rid());
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn remote_auth_discovery_failure_uses_the_callers_error_envelope() {
        let registry = WorkerRegistry::new();
        let (remote_auth, _) = spawn_remote_auth(serde_json::json!("invalid response")).await;
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-token"),
        );

        for anthropic in [false, true] {
            if anthropic {
                headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            }
            let list =
                list_remote_authorized_models(&registry, &remote_auth, &headers, rid()).await;
            let single = retrieve_remote_authorized_model(
                &registry,
                &remote_auth,
                &headers,
                "allowed",
                rid(),
            )
            .await;
            for response in [list, single] {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                let body = json_body(response).await;
                assert_eq!(body["error"]["message"], "No models available");
                if anthropic {
                    assert_eq!(body["type"], "error");
                    assert_eq!(body["request_id"], rid());
                } else {
                    assert_eq!(body["error"]["code"], "no_models_available");
                }
            }
        }
    }

    /// An inventory with no model in it answers the gateway's JSON error
    /// envelope with a typed code, as every other refusal does, in the
    /// caller's shape: the OpenAI envelope, or the Messages envelope with its
    /// `request_id` for a Messages client.
    #[tokio::test]
    async fn an_empty_inventory_answers_the_error_envelope_in_the_callers_shape() {
        let registry = WorkerRegistry::new();

        let response =
            registry_models_response(&registry, ModelShape::OpenAi, "req_1".to_string()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "no_models_available", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert!(body["error"]["type"].is_string(), "{body}");

        let response =
            registry_models_response(&registry, ModelShape::Anthropic, "req_2".to_string()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(response).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert_eq!(body["request_id"], "req_2", "{body}");
    }

    /// A tenant-scoped credential must never reach the upstream fan-out:
    /// that would forward the gateway's own secret to an external provider
    /// as if it were the caller's BYOK token. Verified against a real mock
    /// upstream that counts hits, since a stubbed HTTP client can't
    /// distinguish "short-circuited" from "fell through and failed" by
    /// response alone: both end up returning registry models on failure.
    #[tokio::test]
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn a_gateway_credential_never_fans_out_upstream() {
        let hit_count = Arc::new(AtomicUsize::new(0));
        let hit_count_clone = hit_count.clone();
        let mock_app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move || {
                let hit_count = hit_count_clone.clone();
                async move {
                    hit_count.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({"object": "list", "data": []}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let registry = WorkerRegistry::new();
        let external_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{addr}"))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(external_worker);
        // A non-external worker so the short-circuit path has something to
        // return besides 503.
        let internal_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("internal-model")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(internal_worker);

        let client = crate::http_client::client();
        let gateway_auth = AuthConfig::with_tenant_keys(
            Some("shared-secret".to_string()),
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let req = Request::builder()
            .header(header::AUTHORIZATION, "Bearer team-red-secret")
            .body(())
            .unwrap();
        let response = list_models_with(
            &registry,
            &client,
            &gateway_auth,
            req.headers(),
            "req_test".to_string(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            0,
            "a tenant-scoped key must never trigger upstream BYOK fan-out"
        );

        // Control: a genuinely unrecognized token still fans out, which
        // confirms the short-circuit is keyed on the gateway's own
        // credentials rather than disabled entirely.
        let req = Request::builder()
            .header(header::AUTHORIZATION, "Bearer not-a-gateway-key")
            .body(())
            .unwrap();
        let _ = list_models_with(
            &registry,
            &client,
            &gateway_auth,
            req.headers(),
            "req_test".to_string(),
        )
        .await;
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            1,
            "an unrecognized token should still fan out to upstream providers"
        );
    }

    fn registry_with(cards: Vec<ModelCard>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(cards)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker);
        registry
    }

    /// A request id for the tests that do not read it back.
    fn rid() -> String {
        "req_test".to_owned()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The same inventory answers both API families, each in its own shape:
    /// the OpenAI list by default, the Anthropic list (page markers, `type`,
    /// `display_name`, `created_at`) when the request carries the header
    /// every Anthropic client sends.
    #[tokio::test]
    async fn the_list_takes_the_shape_of_the_callers_api_family() {
        let mut card = ModelCard::new("m1");
        card.display_name = Some("Model One".to_owned());
        let registry = registry_with(vec![card]);
        let client = crate::http_client::client();
        let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);

        let openai = json_body(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &HeaderMap::new(),
                "req_test".to_string(),
            )
            .await,
        )
        .await;
        assert_eq!(openai["object"], "list");
        assert_eq!(openai["data"][0]["object"], "model");
        assert_eq!(openai["data"][0]["id"], "m1");
        assert!(openai.get("has_more").is_none());

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let anthropic = json_body(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &headers,
                "req_test".to_string(),
            )
            .await,
        )
        .await;
        assert_eq!(anthropic["has_more"], false);
        assert_eq!(anthropic["first_id"], "m1");
        assert_eq!(anthropic["last_id"], "m1");
        assert_eq!(
            anthropic["data"][0],
            serde_json::json!({
                "type": "model",
                "id": "m1",
                "display_name": "Model One",
                "created_at": "1970-01-01T00:00:00Z"
            })
        );
    }

    /// A single model is found by its id or by an alias and answered in the
    /// caller's shape; an unknown id is the 404 every other route gives.
    #[tokio::test]
    async fn a_single_model_is_retrieved_by_id_or_alias() {
        let mut card = ModelCard::new("m1");
        card.aliases = vec!["alias-1".to_owned()];
        let registry = registry_with(vec![card]);

        let by_id = retrieve_model_with(&registry, &HeaderMap::new(), "m1", rid()).await;
        assert_eq!(by_id.status(), StatusCode::OK);
        let body = json_body(by_id).await;
        assert_eq!(body["object"], "model");
        assert_eq!(body["id"], "m1");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let by_alias = retrieve_model_with(&registry, &headers, "alias-1", rid()).await;
        assert_eq!(by_alias.status(), StatusCode::OK);
        let body = json_body(by_alias).await;
        assert_eq!(body["type"], "model");
        assert_eq!(body["id"], "m1");

        let unknown = retrieve_model_with(&registry, &headers, "nope", rid()).await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    /// An unknown id is refused in the envelope of the API the caller speaks:
    /// the OpenAI-shaped one by default, the Messages envelope (top-level
    /// `type`, `error.type` `not_found_error`, `request_id`) under the header
    /// every Anthropic client sends, as `/v1/messages` refuses the same model.
    #[tokio::test]
    async fn an_unknown_model_is_refused_in_the_callers_envelope() {
        let registry = registry_with(vec![ModelCard::new("m1")]);

        let openai = retrieve_model_with(&registry, &HeaderMap::new(), "nope", rid()).await;
        assert_eq!(openai.status(), StatusCode::NOT_FOUND);
        let body = json_body(openai).await;
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert!(body.get("type").is_none(), "{body}");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let messages =
            retrieve_model_with(&registry, &headers, "nope", "req_models_1".to_owned()).await;
        assert_eq!(messages.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            messages.headers()["content-type"],
            "application/json",
            "{:?}",
            messages.headers()
        );
        let body = json_body(messages).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["type"], "not_found_error", "{body}");
        assert_eq!(
            body["error"]["message"], "No worker available for model 'nope'",
            "{body}"
        );
        assert_eq!(body["request_id"], "req_models_1", "{body}");
        assert!(body["error"].get("code").is_none(), "{body}");
        assert!(body["error"].get("param").is_none(), "{body}");
    }

    /// A canonical id that is also another card's alias names its own card:
    /// the lookup goes by the resolved canonical id, never by a raw alias.
    #[tokio::test]
    async fn a_canonical_id_that_is_another_cards_alias_answers_its_own_card() {
        let mut aliased = ModelCard::new("org/first");
        aliased.aliases = vec!["org/second".to_owned()];
        let registry = registry_with(vec![aliased, ModelCard::new("org/second")]);

        let response = retrieve_model_with(&registry, &HeaderMap::new(), "org/second", rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["id"], "org/second");
    }
}
