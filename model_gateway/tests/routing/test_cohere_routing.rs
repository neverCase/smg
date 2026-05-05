//! Integration tests for native Cohere chat routing.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Request, State},
    http::{header::CONTENT_TYPE, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::{
        cohere::{CohereChatEndpoint, CohereChatRequest, CohereRouter},
        factory::router_ids,
        router_manager::RouterManager,
        RouterFactory, RouterTrait,
    },
    tenant::{RouteRequestMeta, TenantKey},
    worker::{
        BasicWorkerBuilder, ConnectionMode, ModelCard, ProviderType, RuntimeType, Worker,
        WorkerType,
    },
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

use crate::common::test_app::{
    create_test_app_context, create_test_app_with_context, register_external_worker_with_card,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedCohereRequest {
    path: String,
    content_type: Option<String>,
    body: Bytes,
}

#[derive(Clone, Default)]
struct CohereCapture {
    requests: Arc<Mutex<Vec<CapturedCohereRequest>>>,
}

/// Build tenant metadata used by router tests.
fn test_tenant_meta() -> smg::middleware::TenantRequestMeta {
    RouteRequestMeta::new(TenantKey::from("test-tenant"))
}

/// Start a fake Cohere-native worker that records raw requests.
#[expect(clippy::unwrap_used, reason = "test helper with known-valid setup")]
async fn start_fake_cohere_worker() -> (String, CohereCapture, JoinHandle<()>) {
    let capture = CohereCapture::default();
    let app = Router::new()
        .route("/v1/chat", post(fake_cohere_chat))
        .route("/v2/chat", post(fake_cohere_chat))
        .with_state(capture.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (base_url, capture, handle)
}

/// Record the raw request and return a Cohere-native response shape.
#[expect(clippy::unwrap_used, reason = "test helper with known-valid JSON")]
async fn fake_cohere_chat(State(capture): State<CohereCapture>, req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();
    let content_type = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let body = to_bytes(req.into_body(), usize::MAX).await.unwrap();
    capture
        .requests
        .lock()
        .unwrap()
        .push(CapturedCohereRequest {
            path,
            content_type,
            body: body.clone(),
        });

    let payload: Value = serde_json::from_slice(&body).unwrap();
    if payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (
            [(CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            "event: message-start\ndata: {\"type\":\"message-start\"}\n\nevent: content-delta\ndata: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hello\"}}}}\n\n",
        )
            .into_response();
    }

    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/json")],
        json!({
            "id": "cohere-test-response",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}]
            },
            "usage": {"billed_units": {"input_tokens": 2, "output_tokens": 1}},
            "logprobs": [{"token": "hello", "logprob": -0.1}],
            "citations": [{"start": 0, "end": 5, "text": "hello"}],
            "meta": {"api_version": {"version": "2"}}
        })
        .to_string(),
    )
        .into_response()
}

/// Register a local HTTP worker with Cohere provider capability.
fn register_local_cohere_worker(ctx: &Arc<AppContext>, url: &str, model: &str) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .runtime_type(RuntimeType::Vllm)
            .model(ModelCard::new(model).with_provider(ProviderType::Cohere))
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    ctx.worker_registry.register(worker);
}

#[test]
fn cohere_chat_metadata_extraction_keeps_original_bytes() {
    let raw = r#"{"model":"command-r-plus","stream":true,"message":"hi","cohere_only":{"citation_quality":"accurate"}}"#;
    let request =
        CohereChatRequest::from_bytes(CohereChatEndpoint::V2, Bytes::from_static(raw.as_bytes()))
            .expect("metadata should parse");

    assert_eq!(request.model(), "command-r-plus");
    assert!(request.stream());
    assert_eq!(request.endpoint().path(), "/v2/chat");
    assert_eq!(request.body(), raw.as_bytes());
}

#[test]
fn cohere_chat_metadata_requires_model_for_v1_and_v2() {
    for endpoint in [CohereChatEndpoint::V1, CohereChatEndpoint::V2] {
        let err =
            CohereChatRequest::from_bytes(endpoint, Bytes::from_static(br#"{"stream":false}"#))
                .expect_err("missing model should be rejected");
        assert!(err.to_string().contains("model"));
    }
}

#[tokio::test]
async fn cohere_router_forwards_v2_raw_body_and_native_response() {
    let (base_url, capture, server) = start_fake_cohere_worker().await;
    let ctx = create_test_app_context().await;
    register_local_cohere_worker(&ctx, &base_url, "command-r-plus");
    let router = CohereRouter::new(ctx);

    let raw = r#"{"model":"command-r-plus","message":"hi","documents":[{"title":"doc"}],"metadata":{"trace":"abc"}}"#;
    let request =
        CohereChatRequest::from_bytes(CohereChatEndpoint::V2, Bytes::from_static(raw.as_bytes()))
            .expect("metadata should parse");
    let response = router
        .route_cohere_chat(None, &test_tenant_meta(), &request, request.model())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["finish_reason"], "COMPLETE");
    assert_eq!(json["message"]["content"][0]["text"], "hello");
    assert!(json.get("logprobs").is_some());
    assert!(json.get("citations").is_some());

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].path, "/v2/chat");
    assert_eq!(captured[0].body, Bytes::from_static(raw.as_bytes()));
    server.abort();
}

#[tokio::test]
async fn cohere_router_forwards_v1_streaming_events_unchanged() {
    let (base_url, capture, server) = start_fake_cohere_worker().await;
    let ctx = create_test_app_context().await;
    register_local_cohere_worker(&ctx, &base_url, "command-r");
    let router = CohereRouter::new(ctx);

    let raw = r#"{"model":"command-r","message":"hi","stream":true,"chat_history":[{"role":"USER","message":"hello"}]}"#;
    let request =
        CohereChatRequest::from_bytes(CohereChatEndpoint::V1, Bytes::from_static(raw.as_bytes()))
            .expect("metadata should parse");
    let response = router
        .route_cohere_chat(None, &test_tenant_meta(), &request, request.model())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get(CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("event: message-start"));
    assert!(text.contains("event: content-delta"));

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].path, "/v1/chat");
    assert_eq!(captured[0].body, Bytes::from_static(raw.as_bytes()));
    server.abort();
}

#[derive(Debug)]
struct NeverCalledRouter;

#[async_trait]
impl RouterTrait for NeverCalledRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_cohere_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &smg::middleware::TenantRequestMeta,
        _body: &CohereChatRequest,
        _model_id: &str,
    ) -> Response {
        panic!("request validation should reject missing model before routing");
    }

    fn router_type(&self) -> &'static str {
        "never-called"
    }
}

#[tokio::test]
async fn server_rejects_cohere_chat_without_model() {
    let ctx = create_test_app_context().await;
    let app = create_test_app_with_context(Arc::new(NeverCalledRouter), ctx);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"message":"missing model"}"#))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("model"));
}

#[tokio::test]
async fn router_manager_selects_cohere_router_for_local_cohere_worker() {
    let ctx = create_test_app_context().await;
    register_local_cohere_worker(&ctx, "http://cohere-worker.local", "command-r-plus");

    let manager = RouterManager::new(ctx.worker_registry.clone(), ctx.client.clone());
    manager.register_router(router_ids::HTTP_REGULAR, Arc::new(NeverCalledRouter));
    manager.register_router(router_ids::HTTP_COHERE, Arc::new(CohereRouter::new(ctx)));

    let router = manager
        .get_router_for_model("command-r-plus")
        .expect("router should be selected");
    assert_eq!(router.router_type(), "cohere");
}

#[tokio::test]
async fn router_factory_creates_cohere_mode_router() {
    let config = RouterConfig::new(
        RoutingMode::Cohere {
            worker_urls: vec!["http://cohere-worker:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    let ctx = crate::common::create_test_context(config).await;

    let router = RouterFactory::create_router(&ctx)
        .await
        .expect("create cohere router");
    assert_eq!(router.router_type(), "cohere");
}

#[tokio::test]
async fn cohere_external_worker_card_routes_to_cohere_router() {
    let ctx = create_test_app_context().await;
    register_external_worker_with_card(
        &ctx,
        "https://api.cohere.ai",
        ModelCard::new("command-r-plus").with_provider(ProviderType::Cohere),
    );

    let manager = RouterManager::new(ctx.worker_registry.clone(), ctx.client.clone());
    manager.register_router(router_ids::HTTP_COHERE, Arc::new(NeverCalledRouter));

    let router = manager
        .get_router_for_model("command-r-plus")
        .expect("router should be selected");
    assert_eq!(router.router_type(), "never-called");
}
