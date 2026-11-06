use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    collections::HashMap
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use rand::seq::IteratorRandom;

use crate::service_discovery::{POD_NAME_LABEL, POD_UID_LABEL};
use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// POST /query 请求体。
#[derive(Debug, Serialize)]
pub(crate) struct ConductorQueryRequest {
    /// 模型名称，必须与 Conductor 中注册的模型一致，不能为空。
    pub model: String,

    /// KV block 大小，必须大于 0。
    pub block_size: i64,

    /// 完整 prompt 的 token IDs。
    pub token_ids: Vec<i32>,

    /// 租户；省略时默认为 "default"。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,

    /// LoRA 名称；省略时默认为空字符串，表示基础模型。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,

    /// 缓存哈希 salt；必须与引擎使用的值一致。
    /// 省略或空字符串表示没有 salt。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    /// 限定查询某个实例。
    /// 省略时查询当前上下文的所有实例。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
}

/// POST /query 成功响应。
#[derive(Debug, Deserialize)]
pub struct ConductorQueryResponse {
    /// 注册时的 instance_id -> 实例缓存命中信息。
    pub instances: HashMap<String, ConductorInstanceMatch>,
}

/// 单个实例的缓存命中信息。
/// 所有长度的单位都是 token。
#[derive(Debug, Deserialize)]
pub struct ConductorInstanceMatch {
    /// 跨所有层级和 DP rank 的最长连续前缀，等于 disk。
    pub longest_matched: i64,

    /// 设备内存中的前缀命中长度。
    pub device: i64,

    /// 包含设备内存和本地 CPU 缓存的累计前缀命中长度。
    pub cpu_local: i64,

    /// 进一步包含共享 CPU 缓存的累计前缀命中长度。
    pub cpu_share: i64,

    /// 进一步包含磁盘缓存的累计前缀命中长度。
    pub disk: i64,

    /// DP rank -> 该 rank 的 device 命中长度。
    /// HTTP 响应中的 rank key 是 "0"、"1" 等字符串。
    pub dp: HashMap<String, i64>,

    /// DP rank -> 该 rank 各层的累计前缀命中长度。
    pub rank_matches: HashMap<String, ConductorRankMatch>,
}

/// 单个 DP rank 的缓存命中信息。
#[derive(Debug, Deserialize)]
pub struct ConductorRankMatch {
    pub device: i64,
    pub cpu_local: i64,
    pub cpu_share: i64,
    pub disk: i64,

}

#[derive(Debug, Default)]
pub struct KvcacheConductorPolicy {
    tenant_id: String,
    client: reqwest::Client,
    next_start: AtomicUsize,
    tick: AtomicU64,
}

impl KvcacheConductorPolicy {
    pub fn new(tenant_id: String) -> Self {

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("Failed to build client");

        Self {
            tenant_id,
            client,
            next_start: AtomicUsize::new(0),
            tick: AtomicU64::new(0),
        }
    }

    pub(crate) async fn query_conductor(
        &self,
        url: &str,
        request: &ConductorQueryRequest,
    ) -> Result<ConductorQueryResponse, String> {
        let body = rmp_serde::to_vec_named(request)
            .map_err(|e| format!("Encode Conductor request failed: {}", e))?;

        let response = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/msgpack")
            .body(body)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| format!("Conductor request failed: {}", e))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| format!("Read Conductor response failed: {e}"))?;

        match rmp_serde::from_slice::<serde_json::Value>(&bytes) {
            Ok(body) => {
                info!("Conductor query response: status={}, body={}", status, body);
            }
            Err(error) => {
                info!(
                    "Conductor response decode failed: status={}, error={}, raw={:?}",
                    status, error, bytes
                );
            }
        }

        if !status.is_success() {
            let detail = rmp_serde::from_slice::<serde_json::Value>(&bytes)
                .map(|value | value.to_string())
                .unwrap_or_else(|e| format!("Cannot decode error body: {e}"));

            return Err(format!("Conductor returned HTTP {status}: {detail}"));
        }

        rmp_serde::from_slice::<ConductorQueryResponse>(&bytes)
            .map_err(|e| format!("Decode Conductor response failed: {e}"))
    }

    pub async fn prepare_request(
        &self, workers: &[Arc<dyn Worker>], tokens: &[u32]
    ) -> Result<ConductorQueryResponse, String> {
        let tokens_i32: Vec<i32> = tokens
            .iter()
            .copied()
            .map(|token| {
                i32::try_from(token)
                    .map_err(|_| format!("Token ID {token} exceeds i32::MAX"))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // smg::policies::kvcache_conductor: model_gateway/src/policies/kvcache_conductor.rs:148: Conductor query response: status=200 OK, body={"instances":{"xhive-44bf960c-5977-4ea2-8e82-f72ab51c1861-0":{"longest_matched":2224,"dp":{"0":2224},"rank_matches":{"0":{"device":2224,"cpu_local":2224,"cpu_share":2224,"disk":2224}},"device":2224,"cpu_local":2224,"cpu_share":2224,"disk":2224},"xhive-8d5587aa-f2df-4193-b63c-a6a50f8eabc7-0":{"longest_matched":2096,"dp":{"0":2096},"rank_matches":{"0":{"device":2096,"cpu_local":2096,"cpu_share":2096,"disk":2096}},"device":2096,"cpu_local":2096,"cpu_share":2096,"disk":2096}}}
        // 2026-11-05T15:43:57.733192793+08:00 2026-11-05 07:43:57  INFO http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module="smg" request_id="e83a1fd3-2bcd-9ae6-9744-71370517f80c"}: smg::policies::kvcache_conductor: model_gateway/src/policies/kvcache_conductor.rs:228: Conductor query response: instance_count=2, instance_ids=["xhive-8d5587aa-f2df-4193-b63c-a6a50f8eabc7-0", "xhive-44bf960c-5977-4ea2-8e82-f72ab51c1861-0"]

        // scan blocks map
        let mut merged_instances = ConductorQueryResponse {
            instances: HashMap::new(),
        };

        let mut instance_with_block_size: HashMap<String, i32> = HashMap::new();

        for (_, worker) in workers.iter().enumerate() {
            let pod_name = worker
                .metadata()
                .spec
                .labels
                .get(POD_NAME_LABEL)
                .map(String::as_str)
                .unwrap_or("");
            let pod_uid = worker
                .metadata()
                .spec
                .labels
                .get(POD_UID_LABEL)
                .map(String::as_str)
                .unwrap_or("");

            let model_id = match worker.kv_engine_id() {
                Some(m) => m,
                None => {
                    warn!("Worker - pod name: {} uid: {} has no kv_engine_id", pod_name, pod_uid);
                    continue;
                }
            };
            let block_size = match worker.metadata().spec.kv_block_size {
                Some(b) => b,
                None => {
                    warn!("Worker - pod name: {} uid: {} has no kv_block_size", pod_name, pod_uid);
                    continue;
                }
            };
            let key = format!("model_id-{}-block_size-{}", model_id, block_size);
            if instance_with_block_size.contains_key(&key) {
                warn!("Worker - pod name: {} uid: {} has duplicate kv_block_size", pod_name, pod_uid);
                continue;
            }

            // replace url
            let connector_str = match worker.kv_connector() {
                Some(m) => m,
                None => {
                    warn!("Worker - pod name: {} uid: {} has no mooncake_url", pod_name, pod_uid);
                    continue;
                }
            };
            let (namespace, _) = match connector_str.split_once('/') {
                Some((s, ns)) => (s.to_string(), ns.to_string()),
                None => {
                    warn!("Worker - pod name: {} uid: {} has invalid mooncake_url format (missing '/')", pod_name, pod_uid);
                    continue;
                }
            };
            let url = format!("http://mooncake-conductor.{}.svc.cluster.local:13333/query", namespace);

            let request = ConductorQueryRequest {
                model: model_id,
                block_size: block_size as i64,
                token_ids: tokens_i32.clone(),
                tenant_id: Some(self.tenant_id.clone()),
                lora_name: None,
                cache_salt: None,
                instance_id: None,
            };

            let response = match self.query_conductor(&url, &request).await {
                Ok(response) => response,
                Err(error) => {
                    warn!(
                        url = %url,
                        model = %request.model,
                        block_size = request.block_size,
                        error = %error,
                        "Conductor query failed"
                    );
                    continue;
                }
            };

            info!(
                "Conductor query response: instance_count={}, instance_ids={:?}",
                response.instances.len(),
                response.instances.keys().collect::<Vec<_>>()
            );

            for (instance_id, hit) in response.instances {
                // instance_id：Conductor 注册的实例 ID
                // hit.device：设备缓存命中的 token 数
                // hit.cpu_share：包含共享 CPU 缓存的累计命中数
                info!("Instance {} hit: device {} / cpu_local {} /cpu_share {} / disk {}",
                instance_id, hit.device, hit.cpu_local, hit.cpu_share,hit.disk);

                merged_instances.instances.entry(instance_id)
                    .or_insert_with(|| hit);
            }
            instance_with_block_size.insert(key.clone(), 1);
        }

        Ok(merged_instances)
    }
}

impl LoadBalancingPolicy for KvcacheConductorPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        select_worker_info: &SelectWorkerInfo
    ) -> Option<usize> {
        if let Some(matches) = select_worker_info.conductor_matches {
            for (instance_id, hit) in &matches.instances {
                info!(
                    instance_id = %instance_id,
                    device = hit.device,
                    cpu_local = hit.cpu_local,
                    cpu_share = hit.cpu_share,
                    disk = hit.disk,
                    "Conductor result reached worker selection"
                );
            }
        }

        let mut rng = rand::rng();

        let Some(matches) = select_worker_info.conductor_matches else {
            return get_healthy_worker_indices(workers)
                .into_iter()
                .choose(&mut rng);
        };

        let mut candidates: Vec<(usize, i64)> = workers
            .iter()
            .enumerate()
            .filter_map(|(index, worker)| {
                if !worker.is_available() {
                    return None;
                }

                let pod_uid = worker.metadata().spec.labels.get(POD_UID_LABEL)?;
                let restart_count = worker.restart_count()?;
                let instance_id = format!("xhive-{}-{}", pod_uid, restart_count);
                let hit = matches.instances.get(&instance_id)?;

                Some((index, hit.longest_matched))
            })
            .collect();

        // longest_matched 从大到小排序。
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        if let Some((_, best_matched)) = candidates.first().copied() {
            return candidates
                .into_iter()
                .take_while(|(_, matched)| *matched == best_matched)
                .map(|(index, _)| index)
                .choose(&mut rng);
        }

        // 没有匹配到可用 worker，随机回退。
        warn!("No available worker matches Conductor instances");
        get_healthy_worker_indices(workers)
            .into_iter()
            .choose(&mut rng)
    }
    fn name(&self) -> &'static str {
        "kvcache_conductor"
    }

    fn reset(&self) {
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

