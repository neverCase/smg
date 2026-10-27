use std::collections::HashSet;
use std::sync::Arc;
use serde_json::json;
use std::time::Duration;
use futures::future::join_all;
use tokio::time::{interval, timeout, MissedTickBehavior};
use tracing::{info, warn};
use crate::worker::{ConnectionMode, WorkerMonitor};
use crate::app_context::AppContext;
use kube::api::{Api, ListParams, Patch, PatchParams};
use k8s_openapi::api::{
    apps::v1::{Deployment, StatefulSet},
    core::v1::Pod,
};
use openai_protocol::worker::WorkerStatus;
use crate::service_discovery::POD_UID_LABEL;

trait WorkloadSpec {
    fn match_labels(&self) -> Option<std::collections::BTreeMap<String, String>>;
    fn replicas(&self) -> Option<i32>;
}

impl WorkloadSpec for Deployment {
    fn match_labels(&self) -> Option<std::collections::BTreeMap<String, String>> {
        self.spec.as_ref()?.selector.match_labels.clone()
    }

    fn replicas(&self) -> Option<i32> {
        self.spec.as_ref().and_then(|s| s.replicas)
    }
}

impl WorkloadSpec for StatefulSet {
    fn match_labels(&self) -> Option<std::collections::BTreeMap<String, String>> {
        self.spec.as_ref()?.selector.match_labels.clone()
    }

    fn replicas(&self) -> Option<i32> {
        self.spec.as_ref().and_then(|s| s.replicas)
    }
}

async fn get_pods_for_workload<K>(
    client: kube::Client,
    namespace: &str,
    name: &str,
) -> Result<Vec<Pod>, Box<dyn std::error::Error>> where
    K: kube::Resource<DynamicType = (), Scope = kube::core::NamespaceResourceScope> + kube::core::ResourceExt + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
    K: WorkloadSpec,
{
    let deploys: Api<K> = Api::namespaced(client.clone(), namespace);
    let pods: Api<Pod> = Api::namespaced(client, namespace);

    let workload = deploys.get(name).await?;

    let selector_string = match workload.match_labels() {
        Some(labels) => {
            labels.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<String>>()
                .join(",")
        }
        None => return Ok(vec![]),
    };

    let lp = ListParams::default().labels(&selector_string);
    let pod_list = pods.list(&lp).await?;

    Ok(pod_list.items)
}

pub(crate) async fn patch_workload_replicas<K>(
    client: kube::Client,
    namespace: &str,
    name: &str,
    new_replicas: i32,
) -> Result<(), Box<dyn std::error::Error>> where
    K: kube::Resource<DynamicType = (), Scope = kube::core::NamespaceResourceScope> + kube::core::ResourceExt + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let api: Api<K> = Api::namespaced(client, namespace);

    let patch_data = json!({
        "spec": {
            "replicas": new_replicas
        }
    });

    let params = PatchParams::apply("workload-drain-controller");

    // todo catch Patch failed or error
    api.patch(name, &params, &Patch::Merge(&patch_data)).await?;

    Ok(())
}

pub(crate) async fn run_workload_drain(
    app_context: Arc<AppContext>,
    kind: &str,
    name: &str
) -> Result<(), String> {
    let namespace = app_context
        .kube_namespace
        .as_deref()
        .ok_or_else(|| {
            "Workload drain requires a configured namespace".to_string()
        })?;

    let kube_client = app_context.kube_client.clone();

    let pods = match kind {
        "deployment" => {
            get_pods_for_workload::<Deployment>(
                kube_client.clone(), namespace, name,
            )
                .await
        }
        "statefulset" => {
            get_pods_for_workload::<StatefulSet>(
                kube_client.clone(), namespace, name,
            )
                .await
        }
        _ => return Err(format!("Unsupported workload kind: {kind}")),
    }
    .map_err(|err| format!("Failed to query workload pods: {err}"))?;

    let pod_uids: HashSet<&str> = pods
        .iter()
        .filter_map(|pod| pod.metadata.uid.as_deref())
        .collect();

    let registry = &app_context.worker_registry;

    let target_workers: Vec<_> = registry
        .get_all_with_ids()
        .into_iter()
        .filter(|(_, worker) | {
            worker
                .metadata()
                .spec
                .labels
                .get(POD_UID_LABEL)
                .is_some_and(|uid| pod_uids.contains(uid.as_str()))
        })
        .collect();

    if target_workers.is_empty() {
        return Err(format!(
            "No registered workers matched workload {namespace}/{name}"
        ));
    }

    for (worker_id, worker) in &target_workers {
        registry
            .apply_if_revision(worker_id, worker.revision(), |_| {
                ((), Some(WorkerStatus::Draining))
            })
            .ok_or_else(|| {
                format!(
                    "Worker {} was removed or replaced while draining",
                    worker_id.as_str()
                )
            })?;
    }

    let drain_worker_timeout = app_context.router_config.drain_worker_timeout;
    let drain_timeout = Duration::from_secs(drain_worker_timeout);
    let drain_result = timeout(drain_timeout, async {
        let mut ticker = interval(Duration::from_secs(10));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;

            let reports = join_all(
                target_workers.iter().map(|(_, worker)| async move {
                    // 这个 5 秒超时保留：限制单个 worker 的查询时间。
                    timeout(Duration::from_secs(5), async {
                        match worker.connection_mode() {
                            ConnectionMode::Http => {
                                WorkerMonitor::fetch_http_load(worker, None).await
                            }
                            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                                WorkerMonitor::fetch_backend_load(worker).await
                            }
                        }
                    })
                    .await
                }),
            )
            .await;

            let mut all_idle = true;

            for ((worker_id, worker), result) in
                target_workers.iter().zip(reports)
            {
                let report = match result {
                    Ok(Some(report)) if !report.loads.is_empty() => report,
                    _ => {
                        all_idle = false;
                        warn!(
                            worker_id = %worker_id.as_str(),
                            "Unable to confirm worker load; retrying"
                        );
                        continue;
                    }
                };

                let in_flight = worker.load();

                info!(
                    worker_id = %worker_id.as_str(),
                    in_flight,
                    running = report.total_running_reqs(),
                    waiting = report.total_waiting_reqs(),
                    "Checking worker drain progress"
                );

                let engine_idle = report.loads.iter().all(|rank| {
                    rank.num_running_reqs == 0
                        && rank.num_waiting_reqs == 0
                });

                if in_flight != 0 || !engine_idle {
                    all_idle = false;
                }

            }

            if all_idle {
                info!("All target workers have drained");
                break;
            }

        }
    })
    .await;

    if drain_result.is_err() {
        warn!(
            namespace,
            kind, name, drain_worker_timeout, "Drain timed out; forcing replicas to zero"
        );
    }

    // todo patch workload
    match kind {
        "deployment" => {
            patch_workload_replicas::<Deployment>(
                kube_client, namespace, name, 0,
            )
            .await
        }
        "statefulset" => {
            patch_workload_replicas::<StatefulSet>(
                kube_client, namespace, name, 0,
            )
            .await
        }
        _ => return Err(format!("Unsupported workload kind: {kind}")),
    }
    .map_err(|err| format!("Failed to patch workload replicas: {err}"))?;

    Ok(())
}
