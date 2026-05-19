//! Runtime bundle for mesh-backed cross-region signal sync.
//!
//! Owns producers, the mesh subscriber, and stale-entry GC. Dropping the
//! bundle aborts its background tasks.

use std::{sync::Arc, time::Duration};

use parking_lot::RwLock;
use smg_mesh::{MeshKV, StreamConfig, StreamNamespace, StreamRouting};
use tokio::task::JoinHandle;

use super::{
    adapters::{CrossRegionProducers, ProducerCadences, ProducerHandles},
    decode_envelope, CrossRegionContext, CrossRegionResult, CrossRegionState,
    CrossRegionSyncService, RegionPeerRegistry, CROSS_REGION_NAMESPACE_PREFIX,
};
use crate::{server::ReadinessGate, worker::WorkerRegistry};

/// Required by `StreamConfig`; broadcast drain traffic does not use it.
const CROSS_REGION_STREAM_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Keep stale entries past the projection freshness window before GC.
const GC_AGE_MULTIPLIER: u64 = 4;

/// Materialized-state GC cadence.
const GC_INTERVAL: Duration = Duration::from_secs(30);

/// Cross-region sync plane handles owned by the gateway.
#[derive(Debug)]
pub struct CrossRegionSyncRuntime {
    producers: CrossRegionProducers,
    /// Configured peer registry for diagnostics/request-plane consumers.
    peers: RegionPeerRegistry,
    /// Keeps producer tasks alive; drop aborts them.
    _producer_handles: ProducerHandles,
    /// Applies inbound mesh stream entries to materialized state.
    _subscriber: SubscriberHandle,
    /// Evicts stale materialized entries.
    _gc: GcHandle,
}

impl CrossRegionSyncRuntime {
    /// Start producers, subscriber, and GC over a mesh namespace.
    pub fn start(
        context: &CrossRegionContext,
        namespace: Arc<StreamNamespace>,
        worker_registry: Arc<WorkerRegistry>,
        readiness_gate: ReadinessGate,
    ) -> CrossRegionResult<Self> {
        let producers = CrossRegionProducers::new(
            context.config.region_id.clone(),
            context.config.server_name.clone(),
            namespace.clone(),
            readiness_gate,
        )?;
        let handles = producers.start(worker_registry, ProducerCadences::default());
        let subscriber = spawn_subscriber(producers.sync.clone());
        let gc_max_age_ms = gc_max_age_ms(context.config.sync_plane.signal_stale_after_seconds);
        let gc = spawn_gc_loop(producers.sync.state(), GC_INTERVAL, gc_max_age_ms);

        Ok(Self {
            producers,
            peers: context.peers.clone(),
            _producer_handles: handles,
            _subscriber: subscriber,
            _gc: gc,
        })
    }

    /// Register the `cross_region:` stream namespace and start the runtime.
    pub fn start_with_mesh_kv(
        context: &CrossRegionContext,
        mesh_kv: &Arc<MeshKV>,
        worker_registry: Arc<WorkerRegistry>,
        readiness_gate: ReadinessGate,
    ) -> CrossRegionResult<Self> {
        let namespace = mesh_kv.configure_stream_prefix(
            CROSS_REGION_NAMESPACE_PREFIX,
            StreamConfig {
                max_buffer_bytes: CROSS_REGION_STREAM_BUFFER_BYTES,
                routing: StreamRouting::Broadcast,
            },
        );
        Self::start(context, namespace, worker_registry, readiness_gate)
    }

    /// Shared sync service handle for routing/projection consumers.
    pub fn sync(&self) -> Arc<CrossRegionSyncService> {
        self.producers.sync.clone()
    }

    /// Producer bundle, including adapters used by request-path hooks.
    pub fn producers(&self) -> &CrossRegionProducers {
        &self.producers
    }

    /// Configured region peers.
    pub fn peers(&self) -> &RegionPeerRegistry {
        &self.peers
    }
}

/// Abort-on-drop task handle.
#[derive(Debug)]
struct SubscriberHandle {
    task: JoinHandle<()>,
}

impl Drop for SubscriberHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Abort-on-drop task handle.
#[derive(Debug)]
struct GcHandle {
    task: JoinHandle<()>,
}

impl Drop for GcHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Convert freshness seconds to the GC eviction age in milliseconds.
fn gc_max_age_ms(signal_stale_after_seconds: u64) -> i64 {
    let millis = signal_stale_after_seconds
        .saturating_mul(GC_AGE_MULTIPLIER)
        .saturating_mul(1_000);
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[expect(
    clippy::disallowed_methods,
    reason = "GC task is bounded by CrossRegionSyncRuntime which aborts on drop"
)]
fn spawn_gc_loop(
    state: Arc<RwLock<CrossRegionState>>,
    interval: Duration,
    max_age_ms: i64,
) -> GcHandle {
    let task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let dropped = state.write().gc_stale(now_ms(), max_age_ms);
            if dropped > 0 {
                tracing::debug!(
                    dropped,
                    max_age_ms,
                    "cross-region GC swept stale materialized entries"
                );
            }
        }
    });
    GcHandle { task }
}

#[expect(
    clippy::disallowed_methods,
    reason = "subscriber task is bounded by CrossRegionSyncRuntime which aborts on drop"
)]
fn spawn_subscriber(sync: Arc<CrossRegionSyncService>) -> SubscriberHandle {
    let namespace = sync.namespace();
    let state = sync.state();
    let mut subscription = namespace.subscribe("");
    let task = tokio::spawn(async move {
        while let Some((key, value)) = subscription.receiver.recv().await {
            let signal_path = key
                .strip_prefix(CROSS_REGION_NAMESPACE_PREFIX)
                .unwrap_or(key.as_str());
            match value {
                Some(chunks) => match decode_envelope(&chunks) {
                    Ok(envelope) => {
                        crate::cross_region::apply_envelope_to_state(&mut state.write(), &envelope);
                    }
                    Err(error) => {
                        tracing::warn!(
                            key = %signal_path,
                            error = %error,
                            "dropping malformed cross-region envelope"
                        );
                    }
                },
                None => {
                    tracing::warn!(
                        key = %signal_path,
                        "unexpected tombstone delivery on cross-region stream namespace"
                    );
                }
            }
        }
    });
    SubscriberHandle { task }
}
