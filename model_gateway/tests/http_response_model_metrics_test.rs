//! Requested models survive authorization and routing into HTTP response metrics.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

mod common;

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use common::test_app::{create_test_app_context, create_test_app_with_context};
use metrics_exporter_prometheus::PrometheusBuilder;
use serde_json::{json, Value};
use smg::{
    config::RemoteAuthConfig,
    middleware::RemoteAuthClient,
    routers::http::router::Router as HttpRouter,
    worker::{BasicWorkerBuilder, ModelCard, RuntimeType},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

const MODEL: &str = "http-metrics-model";
const ALIAS: &str = "http-metrics-alias";

struct Upstream {
    url: String,
    task: JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn chat(Json(body): Json<Value>) -> Response {
    if body["messages"][0]["content"] == "fail" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"code": "upstream_rejected", "message": "refused"}})),
        )
            .into_response();
    }
    if body["stream"] == true {
        return ([("content-type", "text/event-stream")], "data: [DONE]\n\n").into_response();
    }
    Json(json!({"model": MODEL, "choices": []})).into_response()
}

#[expect(clippy::disallowed_methods, reason = "test server is aborted on drop")]
async fn upstream() -> Upstream {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route(
            "/verify",
            post(|Json(body): Json<Value>| async move {
                if body["token"] == "denied" {
                    StatusCode::FORBIDDEN
                } else {
                    StatusCode::OK
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Upstream { url, task }
}

async fn app(upstream: &Upstream, remote_auth: bool) -> Router {
    let mut context = create_test_app_context().await;
    if remote_auth {
        let client = RemoteAuthClient::new(
            context.client.clone(),
            RemoteAuthConfig {
                url: upstream.url.clone(),
                ..Default::default()
            },
        );
        Arc::get_mut(&mut context).unwrap().remote_auth_client = Some(Arc::new(client));
    }
    context
        .worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&upstream.url)
                .runtime_type(RuntimeType::Sglang)
                .model(ModelCard::new(MODEL).with_alias(ALIAS))
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    let router = Arc::new(HttpRouter::new(&context).await.unwrap());
    create_test_app_with_context(router, context)
}

async fn send(
    app: &Router,
    model: &str,
    token: Option<&str>,
    content: &str,
    stream: bool,
) -> Response {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(token) = token {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(
            request
                .body(Body::from(
                    json!({
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "stream": stream
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
}

fn assert_sample(rendered: &str, path: &str, status: u16, model: &str, count: u64) {
    let labels = [
        format!("path=\"{path}\""),
        format!("status_code=\"{status}\""),
        format!("model_id=\"{model}\""),
    ];
    let matches: Vec<_> = rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_http_responses_total{")
                && labels.iter().all(|label| line.contains(label))
        })
        .collect();
    assert_eq!(matches.len(), 1, "{labels:?}:\n{rendered}");
    assert!(matches[0].ends_with(&format!(" {count}")), "{}", matches[0]);
}

#[tokio::test]
async fn http_response_metrics_keep_requested_models_on_success_and_refusal() {
    // This integration test owns its recorder in a separate test binary.
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder).unwrap();

    let upstream = upstream().await;
    let authorized = app(&upstream, true).await;
    for (model, token, content, expected) in [
        (MODEL, None, "hi", StatusCode::UNAUTHORIZED),
        (
            "missing-model",
            Some("allowed"),
            "hi",
            StatusCode::FORBIDDEN,
        ),
        (MODEL, Some("denied"), "hi", StatusCode::FORBIDDEN),
        (MODEL, Some("allowed"), "hi", StatusCode::OK),
        (MODEL, Some("allowed"), "fail", StatusCode::BAD_REQUEST),
    ] {
        let response = send(&authorized, model, token, content, false).await;
        assert_eq!(response.status(), expected);
    }

    // The response body and router know the canonical model; the edge metric
    // keeps the caller's alias, including when the response streams.
    let unauthenticated = app(&upstream, false).await;
    let response = send(&unauthenticated, ALIAS, None, "hi", true).await;
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();

    let response = unauthenticated
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response = unauthenticated
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let rendered = handle.render();
    for (status, model) in [
        (401, MODEL),
        (403, "missing-model"),
        (403, MODEL),
        (200, MODEL),
        (400, MODEL),
        (200, ALIAS),
        (400, ""),
    ] {
        assert_sample(&rendered, "/v1/chat/completions", status, model, 1);
    }
    assert_sample(&rendered, "/health", 200, "", 1);
}
