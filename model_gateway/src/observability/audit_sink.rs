//! Async audit-log sink for forwarding per-request observability data to an
//! external Python service (which can then relay to Langfuse / Loki / etc.).
//!
//! Design constraints (see workspace memory rules):
//! - Hot path MUST be non-blocking. We use [`tokio::sync::mpsc::Sender::try_send`]
//!   and drop the log when the queue is full. Never `.await` on the send path.
//! - Hot path MUST avoid string copies of request / response bodies. We carry
//!   payloads as [`bytes::Bytes`] (Arc-counted) and only encode to JSON inside
//!   the background flush task.
//! - The remote `reqwest::Client` MUST have an explicit timeout and is reused
//!   inside the background task.
//! - Sampling is enforced before constructing the [`AuditLog`] so dropped
//!   requests pay zero cost.
//! - Only an explicit allow-list of HTTP paths is audited. Health probes,
//!   metrics scrapes and other high-frequency noise are rejected at the door.
//!
//! Wire-format sent to the Python sidecar:
//! ```json
//! POST {endpoint}
//! Content-Type: application/json
//! { "logs": [ AuditLog, AuditLog, ... ] }
//! ```
//! Each `AuditLog` matches the schema produced by the reference wasm
//! authz filter (`api_key`, `user_id`, `request_id`, `raw_request`,
//! `raw_response`, timing fields, `is_streaming`).

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::{Bytes, BytesMut};
use futures_util::Stream;
use metrics::counter;
use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::rerank::RerankRequest;
use openai_protocol::embedding::EmbeddingRequest;
use reqwest::Client;
use serde::Serialize;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// Configuration for [`AuditSink`].
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Remote endpoint, e.g. `http://audit-sidecar:8080/audit/log`.
    pub endpoint: String,
    /// Max in-flight queue depth. Above this, new logs are dropped.
    pub queue_capacity: usize,
    /// Flush when this many logs accumulated.
    pub batch_size: usize,
    /// Flush at least this often even if batch is not full.
    pub flush_interval: Duration,
    /// HTTP request timeout for the upstream POST.
    pub request_timeout: Duration,
    /// 0.0 ~ 1.0. Requests are sampled uniformly; non-business endpoints
    /// (health/metrics/...) are still hard-filtered regardless of this value.
    pub sample_rate: f32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            endpoint: String::from("http://audit-service.kubesphere-controls-system.svc.cluster.local:12346/audit/log/v2"),
            queue_capacity: 10_000,
            batch_size: 64,
            flush_interval: Duration::from_secs(1),
            request_timeout: Duration::from_secs(3),
            sample_rate: 1.0,
        }
    }
}

impl AuditConfig {
    /// Construct from environment variables, falling back to [`Default`] values
    /// for any variable that is not set. The feature is disabled **only** when
    /// `SMG_AUDIT_ENDPOINT` is explicitly set to the literal `"disabled"`.
    ///
    /// Recognized variables and their defaults:
    ///
    /// | Variable | Default |
    /// |---|---|
    /// | `SMG_AUDIT_ENDPOINT` | `http://audit-service.kubesphere-controls-system.svc.cluster.local:12346/audit/log/v2` |
    /// | `SMG_AUDIT_SAMPLE_RATE` | `0.1` |
    /// | `SMG_AUDIT_QUEUE_CAPACITY` | `10000` |
    /// | `SMG_AUDIT_BATCH_SIZE` | `64` |
    /// | `SMG_AUDIT_FLUSH_MS` | `1000` |
    /// | `SMG_AUDIT_TIMEOUT_MS` | `3000` |
    pub fn from_env() -> Option<Self> {
        let defaults = Self::default();

        let endpoint = std::env::var("SMG_AUDIT_ENDPOINT")
            .unwrap_or(defaults.endpoint);
        if endpoint.eq_ignore_ascii_case("disabled") {
            return None;
        }

        let mut cfg = Self {
            endpoint,
            ..defaults
        };

        cfg.sample_rate = std::env::var("SMG_AUDIT_SAMPLE_RATE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|r| r.clamp(0.0, 1.0))
            .unwrap_or(defaults.sample_rate);

        cfg.queue_capacity = std::env::var("SMG_AUDIT_QUEUE_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.max(1))
            .unwrap_or(defaults.queue_capacity);

        cfg.batch_size = std::env::var("SMG_AUDIT_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.max(1))
            .unwrap_or(defaults.batch_size);

        cfg.flush_interval = std::env::var("SMG_AUDIT_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|n| Duration::from_millis(n.max(50)))
            .unwrap_or(defaults.flush_interval);

        cfg.request_timeout = std::env::var("SMG_AUDIT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|n| Duration::from_millis(n.max(100)))
            .unwrap_or(defaults.request_timeout);

        Some(cfg)
    }
}

/// One audit record. Designed to be cheap to construct on the hot path:
/// bodies are kept as [`Bytes`] (zero-copy slices over the original buffers).
#[derive(Debug, Clone, Serialize)]
pub struct AuditLog {
    pub request_id: String,
    pub api_key: String,
    pub user_id: String,
    pub model: String,
    pub endpoint: String,
    pub is_streaming: bool,
    /// Raw request body (typically the original JSON sent by the client).
    /// Serialized as a UTF-8 string when shipped to the sidecar.
    #[serde(serialize_with = "serialize_bytes_as_str")]
    pub raw_request: Bytes,
    /// Raw response body. May be empty for streaming responses if the caller
    /// did not buffer chunks (configurable per call site).
    #[serde(serialize_with = "serialize_bytes_as_str")]
    pub raw_response: Bytes,
    pub status_code: u16,
    pub request_start_ms: u64,
    pub response_end_ms: u64,
    pub time_to_first_byte_ms: Option<u64>,
    /// Set when the upstream / client terminated the stream prematurely.
    pub error: Option<String>,
}

fn serialize_bytes_as_str<S: serde::Serializer>(
    bytes: &Bytes,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match std::str::from_utf8(bytes) {
        Ok(s) => serializer.serialize_str(s),
        // Fallback: base64. Keeps the JSON valid for binary bodies.
        Err(_) => serializer.serialize_str(&base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            bytes,
        )),
    }
}

/// HTTP paths eligible for auditing. Phase 1: only chat-completions.
#[inline]
fn is_audited_path(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/rerank" | "/v1/embeddings")
}

/// Lock-free atomic counters exposed for tests and ops dashboards.
#[derive(Debug, Default)]
struct Stats {
    accepted: AtomicU64,
    sampled_out: AtomicU64,
    path_filtered: AtomicU64,
    dropped_full: AtomicU64,
}

/// Async, non-blocking audit log sink.
///
/// Clone is cheap (`Arc`-wrapped sender).
#[derive(Clone)]
pub struct AuditSink {
    tx: mpsc::Sender<AuditLog>,
    sample_rate: f32,
    stats: Arc<Stats>,
}

impl AuditSink {
    /// Spawn the background flush task and return a cloneable handle.
    pub fn spawn(cfg: AuditConfig) -> Self {
        let (tx, rx) = mpsc::channel::<AuditLog>(cfg.queue_capacity);
        let stats = Arc::new(Stats::default());

        let client = Client::builder()
            .timeout(cfg.request_timeout)
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .build()
            .unwrap_or_else(|e| {
                warn!(error = %e, "audit_sink: failed to build reqwest client, using default");
                Client::new()
            });

        tokio::spawn(run_flush_loop(rx, client, cfg.clone(), stats.clone()));

        Self {
            tx,
            sample_rate: cfg.sample_rate,
            stats,
        }
    }

    /// Hot-path entry point. Returns immediately. Guarantees:
    /// - Never blocks the caller.
    /// - Allocates only a [`AuditLog`] struct (bodies are `Bytes` clones, ~32B).
    /// - Drops the record silently when the queue is full (counted).
    pub fn try_send(&self, log: AuditLog) {
        match self.tx.try_send(log) {
            Ok(_) => {
                self.stats.accepted.fetch_add(1, Ordering::Relaxed);
                counter!("smg_audit_accepted_total").increment(1);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.stats.dropped_full.fetch_add(1, Ordering::Relaxed);
                counter!("smg_audit_dropped_full_total").increment(1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                // Background task gone (process shutting down); ignore.
            }
        }
    }

    /// Returns `true` when the request should be audited given current
    /// sampling and path policy. Path filtering is hard (cannot be sampled in).
    pub fn should_audit(&self, path: &str) -> bool {
        if !is_audited_path(path) {
            self.stats.path_filtered.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // Cheap PRNG. `fastrand` is not in deps, so we use a tiny thread-local
        // xorshift via [`std::time`] seeded once per call. For a sink that
        // already drops on backpressure the bias is acceptable.
        let r: f32 = rand_unit();
        if r >= self.sample_rate {
            self.stats.sampled_out.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Extract user/api-key/model metadata from a ChatCompletionRequest plus
    /// the incoming headers. Returns `None` when sampling / path policy says
    /// "skip".
    pub fn prepare_chat(
        &self,
        path: &str,
        headers: &http::HeaderMap,
        body: &ChatCompletionRequest,
        raw_request: Bytes,
    ) -> Option<PendingAudit> {
        if !self.should_audit(path) {
            return None;
        }
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let api_key = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_start_matches("Bearer ").to_string())
            .unwrap_or_default();
        let user_id = body.safety_identifier.clone().unwrap_or_default();
        Some(PendingAudit {
            sink: self.clone(),
            request_id,
            api_key,
            user_id,
            model: body.model.clone(),
            endpoint: path.to_string(),
            is_streaming: body.stream,
            raw_request,
            start_ms: now_ms(),
            raw_response: BytesMut::new(),
            ttft_ms: None,
            status_code: 0,
        })
    }

    /// Extract user/api-key/model metadata from a RerankRequest plus
    /// the incoming headers. Returns `None` when sampling / path policy says
    /// "skip".
    pub fn prepare_rerank(
        &self,
        path: &str,
        headers: &http::HeaderMap,
        body: &RerankRequest,
        raw_request: Bytes,
    ) -> Option<PendingAudit> {
        if !self.should_audit(path) {
            return None;
        }
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let api_key = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_start_matches("Bearer ").to_string())
            .unwrap_or_default();
        let user_id = body.user.clone().unwrap_or_default();
        Some(PendingAudit {
            sink: self.clone(),
            request_id,
            api_key,
            user_id,
            model: body.model.clone(),
            endpoint: path.to_string(),
            is_streaming: false,
            raw_request,
            start_ms: now_ms(),
            raw_response: BytesMut::new(),
            ttft_ms: None,
            status_code: 0,
        })
    }

    /// Extract user/api-key/model metadata from a EmbeddingRequest plus
    /// the incoming headers. Returns `None` when sampling / path policy says
    /// "skip".
    pub fn prepare_embeddings(
        &self,
        path: &str,
        headers: &http::HeaderMap,
        body: &EmbeddingRequest,
        raw_request: Bytes,
    ) -> Option<PendingAudit> {
        if !self.should_audit(path) {
            return None;
        }
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let api_key = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_start_matches("Bearer ").to_string())
            .unwrap_or_default();
        let user_id = body.user.clone().unwrap_or_default();
        Some(PendingAudit {
            sink: self.clone(),
            request_id,
            api_key,
            user_id,
            model: body.model.clone(),
            endpoint: path.to_string(),
            is_streaming: false,
            raw_request,
            start_ms: now_ms(),
            raw_response: BytesMut::new(),
            ttft_ms: None,
            status_code: 0,
        })
    }

    /// Snapshot of internal counters (mostly for tests / debug endpoints).
    pub fn stats(&self) -> AuditStats {
        AuditStats {
            accepted: self.stats.accepted.load(Ordering::Relaxed),
            sampled_out: self.stats.sampled_out.load(Ordering::Relaxed),
            path_filtered: self.stats.path_filtered.load(Ordering::Relaxed),
            dropped_full: self.stats.dropped_full.load(Ordering::Relaxed),
        }
    }
}

/// Public stats snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditStats {
    pub accepted: u64,
    pub sampled_out: u64,
    pub path_filtered: u64,
    pub dropped_full: u64,
}

/// Intermediate handle returned by [`AuditSink::prepare_chat`]. Holds the
/// pre-computed request-side metadata until the response is available, then
/// finalizes into an [`AuditLog`] and enqueues it.
///
/// Use [`PendingAudit::finalize`] explicitly, or rely on `Drop` for the
/// "client disconnected mid-stream" safety net (the Drop path enqueues with
/// `error = "dropped_without_finalize"` to make the omission visible).
pub struct PendingAudit {
    sink: AuditSink,
    request_id: String,
    api_key: String,
    user_id: String,
    model: String,
    endpoint: String,
    is_streaming: bool,
    raw_request: Bytes,
    start_ms: u64,
    /// Response body accumulator. Filled incrementally via
    /// [`PendingAudit::append_response_chunk`] (streaming) or in one shot via
    /// [`PendingAudit::set_raw_response`] (non-streaming). Zero-cost when empty.
    raw_response: BytesMut,
    /// Time-to-first-byte in milliseconds, measured from [`Self::start_ms`].
    /// Recorded on the first call to [`PendingAudit::record_first_byte`].
    ttft_ms: Option<u64>,
    /// HTTP status code of the upstream response. `0` until set by
    /// [`PendingAudit::set_status_code`].
    status_code: u16,
}

impl PendingAudit {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Set the HTTP status code observed on the outgoing response. Typically
    /// called once, right after the upstream returns its headers.
    pub fn set_status_code(&mut self, code: u16) {
        self.status_code = code;
    }

    /// Record TTFT (time-to-first-byte / time-to-first-token) on the very
    /// first invocation. Subsequent calls are no-ops, so it is safe to call
    /// from every chunk arrival.
    pub fn record_first_byte(&mut self) {
        if self.ttft_ms.is_none() {
            self.ttft_ms = Some(now_ms().saturating_sub(self.start_ms));
        }
    }

    /// Append a response chunk to the accumulating buffer. Cheap: a `BytesMut`
    /// extend, no allocation when the buffer has spare capacity.
    pub fn append_response_chunk(&mut self, chunk: &[u8]) {
        self.raw_response.extend_from_slice(chunk);
    }

    /// Overwrite the accumulated response with a fully-buffered payload.
    /// Use this in non-streaming code paths where the complete body is
    /// already materialized as [`Bytes`].
    pub fn set_raw_response(&mut self, raw: Bytes) {
        self.raw_response.clear();
        self.raw_response.extend_from_slice(&raw);
    }

    /// Submit the audit record. Consumes the handle so Drop won't double-fire.
    /// All response-side data (`raw_response`, `ttft_ms`, `status_code`) must
    /// have been fed in via the setters above; this method just stamps the
    /// terminal timestamp and forwards.
    pub fn finalize(mut self, error: Option<String>) {
        let raw_response = std::mem::take(&mut self.raw_response).freeze();
        let log = AuditLog {
            request_id: std::mem::take(&mut self.request_id),
            api_key: std::mem::take(&mut self.api_key),
            user_id: std::mem::take(&mut self.user_id),
            model: std::mem::take(&mut self.model),
            endpoint: std::mem::take(&mut self.endpoint),
            is_streaming: self.is_streaming,
            raw_request: std::mem::take(&mut self.raw_request),
            raw_response,
            status_code: self.status_code,
            request_start_ms: self.start_ms,
            response_end_ms: now_ms(),
            time_to_first_byte_ms: self.ttft_ms,
            error,
        };
        // Mark as finalized by stealing the sink Sender out of `self`.
        let sink = std::mem::replace(
            &mut self.sink,
            // Cheap placeholder: a closed channel. Drop won't enqueue because
            // we'll have moved `request_id` already.
            AuditSink {
                tx: mpsc::channel(1).0,
                sample_rate: 0.0,
                stats: Arc::new(Stats::default()),
            },
        );
        sink.try_send(log);
    }
}

impl Drop for PendingAudit {
    fn drop(&mut self) {
        // If `finalize` ran, `request_id` was taken and is now empty; skip.
        if self.request_id.is_empty() {
            return;
        }
        // Safety-net path: emit whatever data was accumulated so far. This
        // keeps partial streaming responses observable when the client
        // disconnected before terminal events fired.
        let raw_response = std::mem::take(&mut self.raw_response).freeze();
        let log = AuditLog {
            request_id: std::mem::take(&mut self.request_id),
            api_key: std::mem::take(&mut self.api_key),
            user_id: std::mem::take(&mut self.user_id),
            model: std::mem::take(&mut self.model),
            endpoint: std::mem::take(&mut self.endpoint),
            is_streaming: self.is_streaming,
            raw_request: std::mem::take(&mut self.raw_request),
            raw_response,
            status_code: self.status_code,
            request_start_ms: self.start_ms,
            response_end_ms: now_ms(),
            time_to_first_byte_ms: self.ttft_ms,
            error: Some("dropped_without_finalize".to_string()),
        };
        self.sink.try_send(log);
    }
}

// ---------- audited body stream (TTFT + raw_response capture) ----------

/// Stream wrapper that:
/// 1. Passes through each response chunk unchanged (so SSE streaming keeps
///    its incremental delivery semantics).
/// 2. Records the wall-clock delta between request start and the *first*
///    chunk — i.e. Time To First Byte / Time To First Token.
/// 3. Accumulates all chunks into a [`BytesMut`] buffer so the audit log
///    can carry the full upstream payload.
/// 4. On stream end (normal, error, or drop-on-disconnect) finalizes the
///    underlying [`PendingAudit`] exactly once.
///
/// Designed to wrap an `axum::body::BodyDataStream` whose item type is
/// `Result<Bytes, axum::Error>`, but is generic over any compatible stream.
pub struct AuditedBodyStream<S> {
    inner: S,
    pending: Option<PendingAudit>,
}

impl<S> AuditedBodyStream<S> {
    pub fn new(inner: S, mut pending: PendingAudit, status_code: u16) -> Self {
        pending.set_status_code(status_code);
        Self {
            inner,
            pending: Some(pending),
        }
    }

    /// Finalize the audit log with the current accumulated state. Called from
    /// terminal stream events (`Ready(None)`, `Ready(Some(Err))`) and from
    /// `Drop` as a safety net.
    fn finalize(&mut self, error: Option<String>) {
        if let Some(pending) = self.pending.take() {
            pending.finalize(error);
        }
    }
}

impl<S, E> Stream for AuditedBodyStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    type Item = Result<Bytes, E>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        let this = self.as_mut().get_mut();
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(pending) = this.pending.as_mut() {
                    // Stamps TTFT only on the first call; cheap on subsequent.
                    pending.record_first_byte();
                    pending.append_response_chunk(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                let msg = format!("stream_error: {e}");
                this.finalize(Some(msg));
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                this.finalize(None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for AuditedBodyStream<S> {
    fn drop(&mut self) {
        // Stream dropped without reaching its terminal state — typically a
        // client disconnect mid-stream. Flush whatever we've accumulated.
        self.finalize(Some("client_disconnected".to_string()));
    }
}

// ---------- background flush loop ----------

async fn run_flush_loop(
    mut rx: mpsc::Receiver<AuditLog>,
    client: Client,
    cfg: AuditConfig,
    stats: Arc<Stats>,
) {
    let mut buf: Vec<AuditLog> = Vec::with_capacity(cfg.batch_size);
    let mut ticker = tokio::time::interval(cfg.flush_interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            maybe_log = rx.recv() => {
                match maybe_log {
                    Some(log) => {
                        buf.push(log);
                        if buf.len() >= cfg.batch_size {
                            flush(&client, &cfg.endpoint, &mut buf).await;
                        }
                    }
                    None => {
                        // Sender closed -> shutdown. Drain remaining.
                        if !buf.is_empty() {
                            flush(&client, &cfg.endpoint, &mut buf).await;
                        }
                        debug!("audit_sink: flush loop exiting");
                        return;
                    }
                }
            }
            _ = ticker.tick() => {
                if !buf.is_empty() {
                    flush(&client, &cfg.endpoint, &mut buf).await;
                }
            }
        }
    }
    // Keep `stats` alive even if compiler complains about unused.
    #[allow(unreachable_code)]
    {
        let _ = stats;
    }
}

#[derive(Serialize)]
struct AuditBatch<'a> {
    logs: &'a [AuditLog],
}

async fn flush(client: &Client, endpoint: &str, buf: &mut Vec<AuditLog>) {
    if buf.is_empty() {
        return;
    }
    let batch = AuditBatch { logs: buf };
    let res = client.post(endpoint).json(&batch).send().await;
    match res {
        Ok(resp) if resp.status().is_success() => {
            counter!("smg_audit_flushed_total").increment(buf.len() as u64);
        }
        Ok(resp) => {
            warn!(status = %resp.status(), count = buf.len(), "audit_sink: sidecar non-2xx");
            counter!("smg_audit_flush_failed_total").increment(buf.len() as u64);
        }
        Err(e) => {
            warn!(error = %e, count = buf.len(), "audit_sink: sidecar unreachable");
            counter!("smg_audit_flush_failed_total").increment(buf.len() as u64);
        }
    }
    buf.clear();
}

// ---------- tiny utils ----------

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Returns a pseudo-random `f32` in `[0, 1)`. Uses [`rand::random`] (already
/// in workspace dependencies).
fn rand_unit() -> f32 {
    rand::random::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_allow_list_only_chat() {
        assert!(is_audited_path("/v1/chat/completions"));
        assert!(!is_audited_path("/v1/completions"));
        assert!(!is_audited_path("/health"));
        assert!(!is_audited_path("/metrics"));
        assert!(!is_audited_path("/v1/models"));
    }

    #[tokio::test]
    async fn drop_when_queue_full() {
        let cfg = AuditConfig {
            endpoint: "http://127.0.0.1:1/never".into(),
            queue_capacity: 1,
            batch_size: 1024,
            flush_interval: Duration::from_secs(3600),
            request_timeout: Duration::from_millis(100),
            sample_rate: 1.0,
        };
        let sink = AuditSink::spawn(cfg);
        let mk = || AuditLog {
            request_id: "r".into(),
            api_key: String::new(),
            user_id: String::new(),
            model: "m".into(),
            endpoint: "/v1/chat/completions".into(),
            is_streaming: false,
            raw_request: Bytes::from_static(b"{}"),
            raw_response: Bytes::new(),
            status_code: 200,
            request_start_ms: 0,
            response_end_ms: 0,
            time_to_first_byte_ms: None,
            error: None,
        };
        // Fill the queue + overflow several times.
        for _ in 0..10 {
            sink.try_send(mk());
        }
        let s = sink.stats();
        assert!(s.dropped_full > 0, "expected drops, got {s:?}");
    }

    #[test]
    fn sample_zero_rejects_all() {
        let cfg = AuditConfig {
            endpoint: "http://x".into(),
            sample_rate: 0.0,
            ..Default::default()
        };
        let sink = AuditSink::spawn(cfg);
        for _ in 0..50 {
            assert!(!sink.should_audit("/v1/chat/completions"));
        }
    }

    #[test]
    fn sample_one_accepts_all_for_audited_path() {
        let cfg = AuditConfig {
            endpoint: "http://x".into(),
            sample_rate: 1.0,
            ..Default::default()
        };
        let sink = AuditSink::spawn(cfg);
        for _ in 0..50 {
            assert!(sink.should_audit("/v1/chat/completions"));
        }
        // Non-audited paths still rejected.
        assert!(!sink.should_audit("/health"));
    }
}
