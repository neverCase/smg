//! Remote authentication client for data-plane requests.
//!
//! Delegates token + model authorization to an external HTTP service.
//! Two endpoints are used:
//!
//! - `POST {url}/verify`   — accepts `{ "token": "...", "model_id": "..." }`,
//!   returns 200 on success / 403 on deny.
//! - `POST {url}/allowed-models` — accepts `{ "token": "..." }`,
//!   returns `{ "models": ["m1", "m2", ...] }`.
//!
//! The client is read-only; it only calls *out* to the auth service and never
//! mutates local state.  Timeout and fail-closed/-open are configurable via
//! [`RemoteAuthConfig`].

use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::config::RemoteAuthConfig;

/// Thin wrapper around [`reqwest::Client`] that calls a remote auth service.
///
/// Created once at startup and reused for every request (the inner `Client`
/// manages a connection pool).  Wrapped in `Arc` so it can be shared across
/// handler tasks cheaply.
#[derive(Clone)]
pub struct RemoteAuthClient {
    client: Client,
    config: RemoteAuthConfig,
}

impl std::fmt::Debug for RemoteAuthClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteAuthClient")
            .field("url", &self.config.url)
            .field("timeout_secs", &self.config.timeout_secs)
            .field("fail_closed", &self.config.fail_closed)
            .finish()
    }
}

impl RemoteAuthClient {
    /// Build a new client from the global `reqwest::Client` (connection pool)
    /// and the auth-specific configuration.
    pub fn new(client: Client, config: RemoteAuthConfig) -> Self {
        Self { client, config }
    }

    // ── token extraction helper ──────────────────────────────────────────

    /// Extract the bearer token from request headers.
    ///
    /// Checks (in order):
    /// 1. `Authorization: Bearer <token>`
    /// 2. `x-api-key: <token>` (Anthropic-style)
    pub fn extract_token(
        &self,
        headers: &axum::http::HeaderMap,
    ) -> Option<String> {
        // 1. Authorization: Bearer <token>
        if let Some(token) = headers
            .get(&self.config.token_header)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| {
                let lower = h.to_ascii_lowercase();
                lower
                    .starts_with("bearer ")
                    .then(|| h[7..].to_string())
            })
        {
            return Some(token);
        }

        // 2. x-api-key (Anthropic-style, header name is canonicalised by hyper)
        headers
            .get("x-api-key")
            .and_then(|h| h.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(String::from)
    }

    // ── verify ───────────────────────────────────────────────────────────

    /// Verify that a token is authorized for the given model.
    ///
    /// Calls `POST {url}/verify` with JSON body `{ "token", "model_id" }`.
    ///
    /// Returns:
    /// - `Ok(())` when the auth service responds 2xx.
    /// - `Err(status, message)` when the auth service denies (4xx/5xx) or is
    ///   unreachable (and `fail_closed` is true).
    ///
    /// When `fail_closed` is `false` and the service is unreachable, the call
    /// returns `Ok(())` (fail-open) with a warning log.
    pub async fn verify(
        &self,
        token: &str,
        model_id: &str,
    ) -> Result<(), (StatusCode, String)> {
        let url = format!("{}/verify", self.config.url);
        let timeout = Duration::from_secs(self.config.timeout_secs);

        info!(
            url = %url,
            model_id = %model_id,
            "RemoteAuth: verifying token for model"
        );

        let result = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "token": token,
                "model_id": model_id,
            }))
            .timeout(timeout)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                info!("RemoteAuth: token verified successfully");
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    status = %status,
                    body = %body,
                    model_id = %model_id,
                    "RemoteAuth: verification denied by auth service"
                );
                Err((
                    StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::FORBIDDEN),
                    format!("Authentication failed: {body}"),
                ))
            }
            Err(e) => {
                if self.config.fail_closed {
                    warn!(
                        error = %e,
                        url = %url,
                        "RemoteAuth: auth service unreachable, failing closed"
                    );
                    Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Authentication service unavailable".to_string(),
                    ))
                } else {
                    warn!(
                        error = %e,
                        url = %url,
                        "RemoteAuth: auth service unreachable, failing open (allowing request)"
                    );
                    Ok(())
                }
            }
        }
    }

    // ── allowed-models ───────────────────────────────────────────────────

    /// Return the list of model IDs that `token` is authorized to access.
    ///
    /// Calls `POST {url}/allowed-models` with JSON body `{ "token" }`.
    ///
    /// Returns an empty `Vec` on any error (auth service unreachable, non-2xx,
    /// or malformed JSON) — the caller should treat this as "no models
    /// available" for the given token.
    pub async fn allowed_models(&self, token: &str) -> Vec<String> {
        let url = format!("{}/allowed-models", self.config.url);
        let timeout = Duration::from_secs(self.config.timeout_secs);

        debug!(url = %url, "RemoteAuth: fetching allowed models");

        let result = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "token": token }))
            .timeout(timeout)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<AllowedModelsResponse>().await {
                    Ok(body) => {
                        debug!(
                            count = body.models.len(),
                            "RemoteAuth: received allowed models"
                        );
                        body.models
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            url = %url,
                            "RemoteAuth: failed to parse allowed-models response"
                        );
                        Vec::new()
                    }
                }
            }
            Ok(resp) => {
                warn!(
                    status = %resp.status(),
                    url = %url,
                    "RemoteAuth: non-success status from allowed-models"
                );
                Vec::new()
            }
            Err(e) => {
                warn!(
                    error = %e,
                    url = %url,
                    "RemoteAuth: auth service unreachable for model listing"
                );
                Vec::new()
            }
        }
    }
}

// ── internal types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AllowedModelsResponse {
    models: Vec<String>,
}

// ── convenience helpers ─────────────────────────────────────────────────

/// Build an `Arc<RemoteAuthClient>` when the config has a non-empty `url`.
/// Returns `None` when the URL is empty (feature disabled).
pub fn create_remote_auth_client(
    client: Client,
    config: &RemoteAuthConfig,
) -> Option<Arc<RemoteAuthClient>> {
    if config.url.is_empty() {
        None
    } else {
        Some(Arc::new(RemoteAuthClient::new(client, config.clone())))
    }
}
