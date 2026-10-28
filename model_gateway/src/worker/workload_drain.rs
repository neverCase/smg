use std::collections::HashSet;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use serde_json::json;
use std::time::Duration;
use futures::future::join_all;
use tokio::time::{interval, timeout, MissedTickBehavior};
use tracing::{info, warn};
use crate::worker::{ConnectionMode, WorkerMonitor};
use crate::app_context::AppContext;
use kube::api::{Api, ListParams, Patch, PatchParams};
use k8s_openapi::api::{
    apps::v1::{Deployment, StatefulSet, ReplicaSet},
    core::v1::Pod,
};
use kube::runtime::reflector::Lookup;
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
    resource_version: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> where
    K: kube::Resource<DynamicType = (), Scope = kube::core::NamespaceResourceScope> + kube::core::ResourceExt + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let api: Api<K> = Api::namespaced(client, namespace);

    let mut patch_data = json!({
        "spec": {
            "replicas": new_replicas
        }
    });

    if let Some(rv) = resource_version {
        patch_data["metadata"] = json!({
            "resourceVersion": rv
        });
    }

    let params = PatchParams::apply("workload-drain-controller");

    // todo catch Patch failed or error
    api.patch(name, &params, &Patch::Merge(&patch_data)).await?;

    Ok(())
}

pub(crate) async fn patch_pod_deletion_cost(
    client: kube::Client,
    namespace: &str,
    name: &str,
    deletion_cost: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let api: Api<Pod> = Api::namespaced(client, namespace);

    let patch_data = json!({
        "metadata": {
            "annotations": {
                "controller.kubernetes.io/pod-deletion-cost": deletion_cost.to_string()
            }
        }
    });
    let params = PatchParams::apply("workload-drain-controller");
    api.patch(name, &params, &Patch::Merge(&patch_data)).await?;

    Ok(())
}

async fn get_deployment_for_pod(
    kube_client: kube::Client,
    namespace: &str,
    pod_name: &str,
) -> Result<Deployment, Box<dyn std::error::Error>> {
    let pods: Api<Pod> = Api::namespaced(kube_client.clone(), namespace);
    let replica_sets: Api<ReplicaSet> = Api::namespaced(kube_client.clone(), namespace);
    let deployments: Api<Deployment> = Api::namespaced(kube_client.clone(), namespace);

    let pod = pods.get(pod_name).await.map_err(|err| {
        format!("Failed to get pod {namespace}/{pod_name}: {err}")
    })?;
    let pod_owner = pod.metadata.owner_references.as_deref().unwrap_or_default();
    let target_rs_owner = pod_owner.iter()
        .find(|owner| {
            owner.controller == Some(true) && owner.kind == "ReplicaSet"
        })
        .ok_or_else(|| format!("Pod {namespace}/{pod_name} has no owner reference"))?;

    let replica_set = replica_sets.get(&target_rs_owner.name).await.map_err(|err| {
        format!("Failed to get replica set {namespace}/{pod_name}: {err}")
    })?;
    let rs_owner = replica_set.metadata.owner_references.as_deref().unwrap_or_default();
    let target_dp_owner = rs_owner.iter()
        .find(|owner| {
            owner.controller == Some(true) && owner.kind == "Deployment"
        })
        .ok_or_else(|| format!("Replica set {namespace}/pod_name/{pod_name}/replica_set/ has no owner reference"))?;

    let deployment = deployments.get(&target_dp_owner.name).await.map_err(|err| {
        format!("Failed to get deployment {namespace}/{pod_name}: {err}")
    })?;

    Ok(deployment)
}

pub(crate) async fn run_workload_drain(
    app_context: Arc<AppContext>,
    name: &str
) -> Result<(), String> {
    let namespace = app_context
        .kube_namespace
        .as_deref()
        .ok_or_else(|| {
            "Workload drain requires a configured namespace".to_string()
        })?;
    let kube_client = app_context.kube_client.clone();

    let pods: Api<Pod> = Api::namespaced(kube_client.clone(), namespace);
    let deployments: Api<Deployment> = Api::namespaced(kube_client.clone(), namespace);

    let pod = pods.get(name).await.map_err(|err| {
        format!("Failed to get pod {namespace}/{name}: {err}")
    })?;

    let deployment = get_deployment_for_pod(
        kube_client.clone(),
        namespace,
        name
    ).await.map_err(|err| {
        format!("Failed to get deployment {namespace}/{name}: {err}")
    })?;

    let deployment_name = deployment.metadata.name.as_deref()
        .ok_or_else(|| "Deployment has no name".to_string())?;

    let pod_uids: HashSet<&str> = pod.metadata.uid
        .as_deref()
        .into_iter()
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
            name, drain_worker_timeout, "Drain timed out; forcing replicas to sub one"
        );
    }

    // lock
    let _guard = app_context.workload_scale_lock.lock().await;

    // patch pod deletion-cost annotations
    // controller.kubernetes.io/pod-deletion-cost="-999"
    let max_patch_pod_attempts = 5;

    let old = app_context.drain_counter.fetch_add(1, Ordering::Relaxed);
    let offset = i32::try_from(old)
        .map_err(|_| "drain_counter exceeds i32 range".to_string())?;
    let delete_cost = i32::MIN + offset;

    for attempt in 1..=max_patch_pod_attempts {
        match patch_pod_deletion_cost(
            kube_client.clone(), namespace, name, delete_cost,
        )
        .await
        {
            Ok(()) => {
                info!("Successfully patched pod {namespace} {name} deletion cost");
                break;
            },
            Err(err) => {
                warn!("Patch pod failed, attempt {attempt}/{max_patch_pod_attempts}: {err}");

                if attempt == max_patch_pod_attempts {
                    return Err(format!(
                        "Failed to patch pod {namespace}/{name} deletion cost: {err}"
                    ));
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // patch deployment replica -1
    let max_attempts = 20;

    for attempt in 1..=max_attempts {
        let result: Result<(), String> = async {
            let latest = deployments.get(deployment_name)
                .await
                .map_err(|err| format!("Failed to get deployment: {err}"))?;

            let replicas = latest.replicas().unwrap_or(1);
            if replicas <= 0 {
                return Ok(());
            }

            patch_workload_replicas::<Deployment>(
                kube_client.clone(),
                namespace,
                deployment_name,
                replicas - 1,
                latest.metadata.resource_version.as_deref()
            )
            .await
            .map_err(|err| format!("Failed to patch deployment replicas: {err}"))
        }
        .await;

        match result {
            Ok(()) => {
                info!("Successfully patched deployment {namespace}/{deployment_name}");
                break;
            }
            Err(err) => {
                warn!("Attempt {attempt}/{max_attempts} failed: {err}");

                if attempt == max_attempts {
                    return Err(err);
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    Ok(())
}
