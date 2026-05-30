use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use rand::seq::{IndexedRandom, SliceRandom};
use tokio::sync::{mpsc, watch, Mutex};
use tonic::transport::Endpoint;
use tracing as log;
use tracing::{instrument, Instrument};

use super::{
    flow_control::RetryManager,
    mtls::MTLSManager,
    service::{
        broadcast_node_states, configure_mtls_endpoint_for_peer,
        gossip::{
            gossip_client::GossipClient, gossip_message, stream_message::Payload as StreamPayload,
            NodeState, NodeStatus, NodeUpdate, Ping, PingReq, StateSync, StreamMessage,
            StreamMessageType,
        },
        try_ping, ClusterState,
    },
    stores::StateStores,
    sync::MeshSyncManager,
};
use crate::{
    chunking::{
        build_stream_batches, chunk_value, dispatch_stream_batch, next_generation,
        DEFAULT_MAX_CHUNKS_PER_BATCH, MAX_STREAM_CHUNK_BYTES,
    },
    collector::{CentralCollector, PeerWatermark, RoundBatch},
    flow_control::{MessageSizeValidator, MAX_MESSAGE_SIZE},
    metrics,
    service::gossip::IncrementalUpdate,
};

pub struct MeshController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    stores: Arc<StateStores>,
    sync_manager: Arc<MeshSyncManager>,
    mtls_manager: Option<Arc<MTLSManager>>,
    // Track active sync_stream connections
    sync_connections: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    /// Central collector that runs once per gossip round.
    central_collector: Arc<CentralCollector>,
    /// Current round batch, updated once per round by the central collector.
    /// Per-peer senders read and apply their own watermark filtering.
    current_batch: Arc<parking_lot::RwLock<Arc<RoundBatch>>>,
    /// Current stream round batch, drained once per round from MeshKV.
    /// Per-peer senders read this and filter targeted entries to their
    /// own peer; drain_entries are broadcast to every peer.
    current_stream_batch: Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>,
    /// Node-wide MeshKV handle. Owns the stream buffers, subscriber
    /// registry, and chunk assembler shared with the server-side
    /// SyncStream handlers.
    mesh_kv: Option<Arc<crate::kv::MeshKV>>,
}

impl MeshController {
    /// Cadence for Down re-probes: one round in every N. The regular peer
    /// picker excludes Down nodes, so without this a node marked Down once
    /// is invisible to the cluster forever. Probing every 10 rounds (~10s)
    /// is rare enough to not noticeably load the network on a busy cluster
    /// and frequent enough that a genuinely-back peer recovers within seconds.
    const DOWN_PROBE_EVERY_N_ROUNDS: u64 = 10;

    /// Pick one random Down peer from the snapshot, excluding self. Returns
    /// None when the cluster has no Down peers (the common steady-state case).
    fn pick_down_probe_target(
        snapshot: &BTreeMap<String, NodeState>,
        self_name: &str,
    ) -> Option<NodeState> {
        let down_peers: Vec<&NodeState> = snapshot
            .iter()
            .filter(|(k, v)| k.as_str() != self_name && v.status == NodeStatus::Down as i32)
            .map(|(_, v)| v)
            .collect();
        down_peers.choose(&mut rand::rng()).map(|&v| v.clone())
    }

    /// Create a new MeshController with stores and sync manager
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
        stores: Arc<StateStores>,
        sync_manager: Arc<MeshSyncManager>,
        mtls_manager: Option<Arc<MTLSManager>>,
    ) -> Self {
        let central_collector =
            Arc::new(CentralCollector::new(stores.clone(), self_name.to_string()));
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            stores,
            sync_manager,
            mtls_manager,
            sync_connections: Arc::new(Mutex::new(HashMap::new())),
            central_collector,
            current_batch: Arc::new(parking_lot::RwLock::new(Arc::new(RoundBatch::default()))),
            current_stream_batch: Arc::new(parking_lot::RwLock::new(Arc::new(
                crate::kv::RoundBatch::default(),
            ))),
            mesh_kv: None,
        }
    }

    /// Attach the node-wide MeshKV handle. Plumbed from the server
    /// builder so stream buffers, subscribers, and the chunk assembler
    /// are shared between client-side (outbound) and server-side
    /// (inbound) SyncStream handlers.
    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    /// Get a handle to the shared RoundBatch. Used by GossipService to
    /// share the centrally collected batch with server-side sync_stream handlers.
    pub fn current_batch(&self) -> Arc<parking_lot::RwLock<Arc<RoundBatch>>> {
        self.current_batch.clone()
    }

    /// Get a handle to the shared stream RoundBatch. Used by GossipService
    /// so server-side sync_stream handlers see the same drained stream
    /// entries as client-side handlers.
    pub fn current_stream_batch(&self) -> Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>> {
        self.current_stream_batch.clone()
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: watch::Receiver<bool>) -> Result<()> {
        let init_state = self.state.clone();
        let read_state = self.state.clone();
        let mut cnt: u64 = 0;

        // Track retry managers for each peer
        use std::collections::HashMap;
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();

        loop {
            log::info!("Round {} Status:{:?}", cnt, read_state.read());

            // Clean up finished sync_stream connections
            {
                let mut connections = self.sync_connections.lock().await;
                connections.retain(|peer_name, handle| {
                    if handle.is_finished() {
                        log::info!(
                            "Sync stream connection to {} has finished, removing",
                            peer_name
                        );
                        false
                    } else {
                        true
                    }
                });
            }

            // Self-incarnation refresh — refute a stale Suspected/Down on the
            // local node's own entry. SWIM's standard recovery primitive:
            // when peers (or partition heuristics) mark us non-Alive, the
            // only way to break the sticky state is for us to broadcast a
            // higher-version Alive. Without this, once peers' Ping fails (a
            // transient blip during rollouts or cross-region routing
            // hiccups), they mark us Down → the peer-picker `retain` below
            // permanently excludes us → no peer ever Pings us again → we
            // stay Down forever despite being healthy.
            {
                let mut state = init_state.write();
                if let Some(self_state) = state.get(&self.self_name).cloned() {
                    if self_state.status != NodeStatus::Alive as i32 {
                        let new_version = self_state.version.saturating_add(1);
                        log::info!(
                            "Self-incarnation refresh: local node {} was marked {:?} (v{}); \
                             bumping to Alive v{}",
                            self.self_name,
                            self_state.status,
                            self_state.version,
                            new_version,
                        );
                        state.insert(
                            self.self_name.clone(),
                            NodeState {
                                name: self_state.name.clone(),
                                address: self_state.address.clone(),
                                status: NodeStatus::Alive as i32,
                                version: new_version,
                                metadata: self_state.metadata.clone(),
                            },
                        );
                    }
                }
            }

            // Get available peers from cluster state
            let snapshot = init_state.read().clone();
            let mut map = snapshot.clone();
            map.retain(|k, v| {
                k.ne(&self.self_name.to_string())
                    && v.status != NodeStatus::Down as i32
                    && v.status != NodeStatus::Leaving as i32
            });

            // Down re-probe target: every DOWN_PROBE_EVERY_N_ROUNDS, pick
            // one random Down peer to ping with our full StateSync. This
            // closes the sticky-Down trap where a node that's actually back
            // can never escape Down because (a) our regular picker excludes
            // Down nodes and (b) PingServer Ack only carries the receiver's
            // own status, so the remote node never learns it's been marked
            // Down and never self-refreshes. A successful probe revives the
            // peer locally via apply_peer_node_update; even on failure, the
            // remote node merges our state_sync (which includes its own
            // Down v_n) and triggers its self-incarnation refresh next round.
            let down_probe_target = if cnt.is_multiple_of(Self::DOWN_PROBE_EVERY_N_ROUNDS) {
                Self::pick_down_probe_target(&snapshot, &self.self_name)
            } else {
                None
            };

            let peer = if map.is_empty() {
                // No live peers in cluster state — keep retrying init_peer
                // every round until gossip or service discovery populates
                // membership. The previous behavior (cnt == 0 only) wedged
                // forever after a single failed dial when init_peer wasn't
                // yet up — a cold-start race that bit us repeatedly during
                // multi-region rollouts where regions boot at different
                // times.
                self.init_peer.map(|init_peer| NodeState {
                    name: "init_peer".to_string(),
                    address: init_peer.to_string(),
                    status: NodeStatus::Suspected as i32,
                    version: 1,
                    metadata: HashMap::new(),
                })
            } else {
                // Use nodes from cluster state (from service discovery or gossip)
                let random_nodes = get_random_values_refs(&map, 1);
                random_nodes.first().map(|&node| node.clone())
            };
            cnt += 1;

            // Checkpoint tree state every 10 rounds (~10s) by exporting
            // the live radix tree from CacheAwarePolicy into tree_configs.
            // This keeps the periodic structure snapshot fresh.
            if cnt.is_multiple_of(10) {
                self.sync_manager.checkpoint_tree_states();
            }

            // Chunk assembler GC: every 5 rounds (~5s), drop partial
            // assemblies older than 30s. Partial chunks the receiver has
            // been holding for a full assembly timeout are assumed lost;
            // the sender will re-publish on its own retry cycle with a
            // fresh generation.
            if cnt.is_multiple_of(5) {
                if let Some(mesh_kv) = &self.mesh_kv {
                    mesh_kv.chunk_assembler().gc(Duration::from_secs(30));
                }
            }

            // Periodic GC: clean up tombstoned CRDT metadata every 60 rounds (~60s)
            if cnt.is_multiple_of(60) {
                let removed = self.stores.gc_tombstones();
                if removed > 0 {
                    log::info!("GC: removed {removed} tombstoned CRDT metadata entries");
                }
                let tree_removed = self.stores.gc_stale_tree_entries();
                if tree_removed > 0 {
                    log::info!("GC: removed {tree_removed} stale tree_configs entries");
                }
                // Record store sizes for monitoring
                metrics::record_store_sizes(
                    self.stores.worker.len(),
                    self.stores.policy.len(),
                    self.stores.membership.len(),
                    self.stores.app.len(),
                );

                // Log all mesh data structure sizes for memory debugging.
                let tree_configs_bytes: usize = self
                    .stores
                    .tree_configs
                    .iter()
                    .map(|e| e.value().len())
                    .sum();
                let tenant_inserts: usize = self
                    .stores
                    .tenant_delta_inserts
                    .iter()
                    .map(|e| e.value().len())
                    .sum();
                let tenant_evictions: usize = self
                    .stores
                    .tenant_delta_evictions
                    .iter()
                    .map(|e| e.value().len())
                    .sum();
                let tree_ops_pending: usize = self
                    .stores
                    .tree_ops_pending
                    .iter()
                    .map(|e| e.value().len())
                    .sum();
                log::info!(
                    "Mesh memory: tree_configs={} entries ({} bytes), tree_versions={}, \
                     tenant_inserts={}, tenant_evictions={}, tree_ops_pending={}, \
                     policy_crdt={}, worker_crdt={}",
                    self.stores.tree_configs.len(),
                    tree_configs_bytes,
                    self.stores.tree_versions.len(),
                    tenant_inserts,
                    tenant_evictions,
                    tree_ops_pending,
                    self.stores.policy.len(),
                    self.stores.worker.len(),
                );

                // Log CRDT policy store operation log length for memory debugging
                let policy_oplog_len = self.stores.policy.get_operation_log().len();
                log::info!(
                    policy_oplog_len,
                    "GC: CRDT policy store operation log length"
                );

                // Clean up retry managers for peers no longer in cluster state
                retry_managers.retain(|peer_name, _| map.contains_key(peer_name));
            }

            // Central collection: run once per round. Drains tenant deltas
            // (destructive) and collects all store changes into one batch.
            // Per-peer senders read this batch and filter by their watermarks.
            {
                let batch = self.central_collector.collect();
                *self.current_batch.write() = Arc::new(batch);
                self.central_collector.advance_generations();
            }

            // Stream round collection: drain stream namespace buffers and
            // drain callbacks exactly once per round (destructive). Per-peer
            // senders filter targeted_entries by their own peer_id and
            // broadcast drain_entries to all peers. Empty batch if no
            // MeshKV is attached (legacy path pre-Step 3).
            if let Some(mesh_kv) = &self.mesh_kv {
                let stream_batch = mesh_kv.collect_round_batch();
                *self.current_stream_batch.write() = Arc::new(stream_batch);
            }

            tokio::select! {

                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                () = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Some(peer) = peer {
                        let peer_name = peer.name.clone();

                        // Get or create retry manager for this peer
                        let retry_manager = retry_managers
                            .entry(peer_name.clone())
                            .or_default();

                        // Check if we should retry based on backoff
                        if retry_manager.should_retry() {
                            match self.connect_to_peer(peer.clone()).await {
                                Ok(()) => {
                                    // Success - reset retry state
                                    retry_manager.reset();
                                    log::info!("Successfully connected to peer {}", peer_name);
                                }
                                Err(e) => {
                                    // Failure - record attempt and calculate next delay
                                    retry_manager.record_attempt();
                                    let next_delay = retry_manager.next_delay();
                                    let attempt = retry_manager.attempt_count();
                                    log::warn!(
                                        "Error connecting to peer {} (attempt {}): {}. Next retry in {:?}",
                                        peer_name,
                                        attempt,
                                        e,
                                        next_delay
                                    );
                                }
                            }
                        } else {
                            // Still in backoff period, skip this attempt
                            let next_delay = retry_manager.next_delay();
                            log::debug!(
                                "Skipping connection to peer {} (backoff: {:?} remaining)",
                                peer_name,
                                next_delay
                            );
                        }
                    } else {
                        log::info!("No peer address available to connect");
                    }

                    if let Some(down_peer) = down_probe_target {
                        if let Err(e) = self.probe_down_peer(down_peer).await {
                            // Probe failures are expected (the peer is Down,
                            // possibly genuinely unreachable). Log at debug
                            // so we don't spam warn-level on a known-down peer.
                            log::debug!("{}", e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn connect_to_peer(&self, peer: NodeState) -> Result<()> {
        log::info!("Connecting to peer {} at {}", peer.name, peer.address);

        let read_state = self.state.clone();

        // TODO: Maybe we don't need to send the whole state.
        let state_sync = StateSync {
            nodes: read_state.read().values().cloned().collect(),
        };
        let peer_addr = peer.address.parse::<SocketAddr>()?;
        let peer_name = peer.name.clone();
        match try_ping(
            &peer,
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            })),
            self.mtls_manager.clone(),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                self.apply_peer_node_update(node_update).await;
            }
            Err(e) => {
                log::info!("Failed to connect to peer: {}, now try ping-req", e);
                let mut map = read_state.read().clone();
                map.retain(|k, v| {
                    k.ne(&self.self_name)
                        && k.ne(&peer_name)
                        && v.status == NodeStatus::Alive as i32
                });
                let random_nodes = get_random_values_refs(&map, 3);
                let mut reachable = false;
                for node in random_nodes {
                    log::info!(
                        "Trying to ping-req node {}, req target: {}",
                        node.address,
                        peer_addr
                    );
                    if try_ping(
                        node,
                        Some(gossip_message::Payload::PingReq(PingReq {
                            node: Some(peer.clone()),
                        })),
                        self.mtls_manager.clone(),
                    )
                    .await
                    .is_ok()
                    {
                        reachable = true;
                        break;
                    }
                }
                if !reachable {
                    let mut target = read_state.read().clone();

                    // Broadcast only the unreachable node's status is enough.
                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        if unreachable_node.status == NodeStatus::Suspected as i32 {
                            unreachable_node.status = NodeStatus::Down as i32;
                        } else {
                            unreachable_node.status = NodeStatus::Suspected as i32;
                        }
                        unreachable_node.version += 1;

                        // Broadcast target nodes should include self.
                        let target_nodes: Vec<NodeState> = target
                            .values()
                            .filter(|v| {
                                v.name.ne(&peer_name)
                                    && v.status == NodeStatus::Alive as i32
                                    && v.status != NodeStatus::Leaving as i32
                            })
                            .cloned()
                            .collect();

                        log::info!(
                            "Broadcasting node status to {} alive nodes, new_state: {:?}",
                            target_nodes.len(),
                            unreachable_node
                        );

                        let (success_count, total_count) = broadcast_node_states(
                            vec![unreachable_node],
                            target_nodes,
                            None, // Use default timeout
                            self.mtls_manager.clone(),
                        )
                        .await;

                        log::info!(
                            "Broadcast node status: {}/{} successful",
                            success_count,
                            total_count
                        );
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to connect to peer {peer_name}: direct ping and ping-req both failed"
                    ));
                }
            }
        }

        log::info!("Successfully connected to peer {}", peer_addr);

        Ok(())
    }

    /// Apply a NodeUpdate received from a peer to local cluster_state, and
    /// — if the update is Alive — establish a sync_stream connection. The
    /// status write is unconditional (no version comparison) because a fresh
    /// NodeUpdate from the peer itself is authoritative for its own status:
    /// it's the strongest possible signal that the peer is up. This is the
    /// only revival path for a node that was previously marked Down locally.
    /// Shared by `connect_to_peer` (regular Ping) and `probe_down_peer`
    /// (Down re-probe).
    async fn apply_peer_node_update(&self, node_update: NodeUpdate) {
        if node_update.status != NodeStatus::Alive as i32
            && node_update.status != NodeStatus::Leaving as i32
        {
            return;
        }
        let updated_peer = {
            let mut s = self.state.write();
            let entry = s
                .entry(node_update.name.clone())
                .and_modify(|e| {
                    e.status = node_update.status;
                    e.address.clone_from(&node_update.address);
                })
                .or_insert_with(|| NodeState {
                    name: node_update.name.clone(),
                    address: node_update.address.clone(),
                    status: node_update.status,
                    version: 1,
                    metadata: HashMap::new(),
                });
            entry.clone()
        };
        if node_update.status == NodeStatus::Alive as i32 {
            if let Err(e) = self
                .start_sync_stream_connection(updated_peer.clone())
                .await
            {
                log::warn!(
                    "Failed to start sync_stream to {}: {}",
                    updated_peer.name,
                    e
                );
            }
        }
    }

    /// Send a Ping (with full local StateSync) to a Down peer to give the
    /// cluster a chance to revive it. On Ack-Alive, the peer is revived via
    /// `apply_peer_node_update`. On failure, status is left unchanged — Down
    /// re-probes are best-effort and must not bump the version (which would
    /// just churn Down→Suspected→Down indefinitely without learning anything).
    ///
    /// The Ping carries this node's full view of cluster_state. If the
    /// remote peer's local self-state is still Alive v1 (stale from boot),
    /// the incoming `Down v_n` entry for itself will trigger the remote's
    /// self-incarnation refresh next round, after which a fresh `Alive v(n+1)`
    /// floods back. This is the recovery loop the cluster needs when a node
    /// boots into an already-poisoned membership.
    async fn probe_down_peer(&self, peer: NodeState) -> Result<()> {
        log::info!("Down-probe: pinging {} at {}", peer.name, peer.address);

        let state_sync = StateSync {
            nodes: self.state.read().values().cloned().collect(),
        };
        let node_update = try_ping(
            &peer,
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            })),
            self.mtls_manager.clone(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Down-probe ping to {} failed: {}", peer.name, e))?;

        log::info!("Down-probe ack from {}: {:?}", peer.name, node_update);
        self.apply_peer_node_update(node_update).await;
        Ok(())
    }

    /// Determine if this node should initiate sync_stream connection
    /// Use lexicographic ordering to avoid duplicate connections
    fn should_initiate_connection(&self, peer_name: &str) -> bool {
        self.self_name.as_str() < peer_name
    }

    /// Spawn a task to handle sync_stream messages
    fn spawn_sync_stream_handler(
        &self,
        mut incoming_stream: tonic::Streaming<StreamMessage>,
        tx: mpsc::Sender<StreamMessage>,
        self_name: String,
        peer_name: String,
    ) -> tokio::task::JoinHandle<()> {
        let stores = self.stores.clone();
        let sync_manager = self.sync_manager.clone();
        let sync_connections = self.sync_connections.clone();
        let current_batch = self.current_batch.clone();
        let current_stream_batch = self.current_stream_batch.clone();
        let mesh_kv = self.mesh_kv.clone();

        // Log connection lifecycle: spawn
        log::debug!(
            peer = %peer_name,
            "spawn_sync_stream_handler called — spawning handler task"
        );

        // Create a span for the spawned task
        let span = tracing::info_span!(
            "sync_stream_handler",
            peer = %peer_name
        );

        #[expect(clippy::disallowed_methods, reason = "handle is returned to caller (spawn_sync_stream_handler) and stored in sync_connections map for lifecycle tracking")]
        tokio::spawn(
            async move {
                use tokio_stream::StreamExt;

                // Log active connection count at handler start
                let active_connections = sync_connections.lock().await.len();
                log::debug!(
                    peer = %peer_name,
                    active_connections,
                    "Sync stream handler started"
                );

                let sequence = Arc::new(AtomicU64::new(0));

                // Send initial heartbeat
                let heartbeat = StreamMessage {
                    message_type: StreamMessageType::Heartbeat as i32,
                    payload: None,
                    sequence: sequence.fetch_add(1, Ordering::Relaxed),
                    peer_id: self_name.clone(),
                };
                if tx.send(heartbeat).await.is_err() {
                    log::warn!("Failed to send initial heartbeat to {}", peer_name);
                    return;
                }

                // Spawn a task to periodically send incremental updates (client-side sender).
                // Uses PeerWatermark to filter the centrally collected batch.
                let incremental_sender_handle = {
                    let mut watermark = PeerWatermark::new(peer_name.clone());
                    log::debug!(
                        peer = %peer_name,
                        "PeerWatermark created for centralized gossip"
                    );
                    let tx_incremental = tx.clone();
                    let self_name_incremental = self_name.clone();
                    let peer_name_incremental = peer_name.clone();
                    let shared_sequence = sequence.clone();
                    let size_validator = MessageSizeValidator::default();
                    let batch_handle = current_batch.clone();
                    let stream_batch_handle = current_stream_batch.clone();

                    #[expect(clippy::disallowed_methods, reason = "incremental sender handle is stored and aborted when the parent sync_stream handler exits")]
                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(1));
                        // Skip re-emission of an unchanged stream batch (main
                        // loop hasn't collected a new one since last tick).
                        let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;

                        loop {
                            interval.tick().await;

                            let round_start = std::time::Instant::now();

                            // Read the centrally collected batch and filter by
                            // this peer's watermark. No collection happens here.
                            let batch = batch_handle.read().clone();
                            let all_updates = watermark.filter(&batch);

                            let collect_elapsed = round_start.elapsed();

                            if !all_updates.is_empty() {
                                for (store_type, updates) in &all_updates {
                                    let proto_store_type = store_type.to_proto();

                                    // Validate message size before sending
                                    let batch_size: usize = updates.iter().map(|u| u.value.len()).sum();

                                    log::debug!(
                                        peer = %peer_name_incremental,
                                        store = ?store_type,
                                        updates = updates.len(),
                                        batch_bytes = batch_size,
                                        "mesh sync store batch"
                                    );
                                    metrics::record_sync_batch_bytes(
                                        &peer_name_incremental,
                                        store_type.as_str(),
                                        batch_size,
                                    );

                                    if let Err(e) = size_validator.validate(batch_size) {
                                        log::warn!(
                                            "Incremental update too large, skipping store {:?}: {} (max: {} bytes)",
                                            store_type,
                                            e,
                                            size_validator.max_size()
                                        );
                                        // Mark non-tree stores as sent to prevent infinite
                                        // retry loops. Tree updates are retried next round.
                                        let is_tree_update =
                                            updates.iter().any(|u| u.key.starts_with("tree:"));
                                        if !is_tree_update {
                                            watermark.mark_sent(*store_type, updates);
                                        }
                                        continue;
                                    }

                                    let incremental_update = StreamMessage {
                                        message_type: StreamMessageType::IncrementalUpdate as i32,
                                        payload: Some(
                                            super::service::gossip::stream_message::Payload::Incremental(
                                                IncrementalUpdate {
                                                    store: proto_store_type,
                                                    updates: updates.clone(),
                                                    version: 0,
                                                },
                                            ),
                                        ),
                                        sequence: shared_sequence.fetch_add(1, Ordering::Relaxed),
                                        peer_id: self_name_incremental.clone(),
                                    };

                                    log::debug!(
                                        "Sending incremental update to {}: store={:?}, {} updates",
                                        peer_name_incremental,
                                        store_type,
                                        updates.len(),
                                    );

                                    match tx_incremental.try_send(incremental_update) {
                                        Ok(()) => {
                                            // Mark as sent after successful transmission
                                            watermark.mark_sent(*store_type, updates);
                                        }
                                        Err(mpsc::error::TrySendError::Full(_)) => {
                                            log::debug!(
                                                "Backpressure: channel full, skipping send (will retry next interval)"
                                            );
                                            continue;
                                        }
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            log::warn!(
                                                "Channel closed, stopping incremental update sender"
                                            );
                                            break;
                                        }
                                    }
                                }
                            }

                            // Stream batches: drain-portion (broadcast) +
                            // targeted entries addressed to this peer. Each
                            // entry is chunked if oversized. On channel
                            // full, the round's stream traffic for this
                            // peer is dropped — no retry (at-most-once).
                            // Application regenerates on its own retry cycle.
                            let stream_batch = stream_batch_handle.read().clone();
                            let fresh_batch = last_stream_batch
                                .as_ref()
                                .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                            if fresh_batch {
                                last_stream_batch = Some(stream_batch.clone());
                                let mut entries = Vec::new();
                                // Drain entries are broadcast: every peer emits.
                                // Generation is per-value so concurrent publishes
                                // to the same key get distinct tags.
                                for (key, value) in &stream_batch.drain_entries {
                                    entries.extend(chunk_value(
                                        key.clone(),
                                        next_generation(),
                                        value.clone(),
                                        MAX_STREAM_CHUNK_BYTES,
                                    ));
                                }
                                // Targeted entries: only those addressed to this peer.
                                for (target, key, value) in &stream_batch.targeted_entries {
                                    if target == &peer_name_incremental {
                                        entries.extend(chunk_value(
                                            key.clone(),
                                            next_generation(),
                                            value.clone(),
                                            MAX_STREAM_CHUNK_BYTES,
                                        ));
                                    }
                                }
                                if !entries.is_empty() {
                                    for batch in build_stream_batches(
                                        entries,
                                        DEFAULT_MAX_CHUNKS_PER_BATCH,
                                        MAX_STREAM_CHUNK_BYTES,
                                    ) {
                                        let msg = StreamMessage {
                                            message_type: StreamMessageType::StreamBatch as i32,
                                            payload: Some(StreamPayload::StreamBatch(batch)),
                                            sequence: shared_sequence
                                                .fetch_add(1, Ordering::Relaxed),
                                            peer_id: self_name_incremental.clone(),
                                        };
                                        match tx_incremental.try_send(msg) {
                                            Ok(()) => {}
                                            Err(mpsc::error::TrySendError::Full(_)) => {
                                                log::debug!(
                                                    peer = %peer_name_incremental,
                                                    "stream batch dropped on backpressure"
                                                );
                                                // TODO(metrics): bump
                                                // stream_dropped_on_backpressure
                                                break;
                                            }
                                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                                log::warn!(
                                                    peer = %peer_name_incremental,
                                                    "stream sender: channel closed, stopping"
                                                );
                                                return;
                                            }
                                        }
                                    }
                                }
                            }

                            let round_elapsed = round_start.elapsed();
                            metrics::record_sync_round_duration(
                                &peer_name_incremental,
                                round_elapsed,
                            );
                            if round_elapsed.as_millis() > 10 || !all_updates.is_empty() {
                                log::info!(
                                    peer = %peer_name_incremental,
                                    round_ms = round_elapsed.as_millis(),
                                    collect_ms = collect_elapsed.as_millis(),
                                    stores_with_updates = all_updates.len(),
                                    "mesh sync round"
                                );
                            }
                        }
                    })
                };

                // Handle incoming messages
                const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
                loop {
                    match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming_stream.next()).await {
                        Ok(Some(Ok(msg))) => {
                            sequence.fetch_add(1, Ordering::Relaxed);

                            match msg.message_type() {
                                StreamMessageType::IncrementalUpdate => {
                                    log::info!(
                                        "[CLIENT] Received incremental update from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );

                                    // Apply incremental updates to local stores
                                    if let Some(
                                        super::service::gossip::stream_message::Payload::Incremental(
                                            update,
                                        ),
                                    ) = &msg.payload
                                    {
                                        use super::stores::StoreType as LocalStoreType;

                                        let store_type = LocalStoreType::from_proto(update.store);
                                        log::info!(
                                            "[CLIENT] Applying incremental update from {}: store={:?}, {} updates",
                                            peer_name,
                                            store_type,
                                            update.updates.len()
                                        );

                                        // Apply updates based on store type
                                        for state_update in &update.updates {
                                            match store_type {
                                                LocalStoreType::App => {
                                                    // Deserialize and apply app state
                                                    if let Ok(app_state) = bincode::deserialize::<
                                                        super::stores::AppState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        let dominated = stores.app.get(&app_state.key)
                                                            .is_some_and(|existing| existing.version >= app_state.version);
                                                        if !dominated {
                                                            // Mirror into the v2 `config:` CRDT
                                                            // namespace so v2-only readers can
                                                            // reach the same value during a
                                                            // rolling upgrade, even when the
                                                            // source is a v1 node still writing
                                                            // to AppStore.
                                                            if let Some(ref kv) = mesh_kv {
                                                                kv.configs().put(
                                                                    &format!(
                                                                        "config:{}",
                                                                        app_state.key
                                                                    ),
                                                                    app_state.value.clone(),
                                                                );
                                                            }
                                                            if let Err(err) = stores.app.insert(
                                                                app_state.key.clone(),
                                                                app_state,
                                                            ) {
                                                                log::warn!(error = %err, "Failed to apply app state update");
                                                            }
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Membership => {
                                                    // Deserialize and apply membership state
                                                    if let Ok(membership_state) = bincode::deserialize::<
                                                        super::stores::MembershipState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let Err(err) = stores.membership.insert(
                                                            membership_state.name.clone(),
                                                            membership_state,
                                                        ) {
                                                            log::warn!(error = %err, "Failed to apply membership state update");
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Worker => {
                                                    // Deserialize and apply worker state
                                                    if let Ok(worker_state) = bincode::deserialize::<
                                                        super::stores::WorkerState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        let actor = Some(state_update.actor.clone());
                                                        sync_manager.apply_remote_worker_state(
                                                            worker_state,
                                                            actor,
                                                        );
                                                    }
                                                }
                                                LocalStoreType::Policy => {
                                                    // Deserialize and apply policy state
                                                    if let Ok(policy_state) = bincode::deserialize::<
                                                        super::stores::PolicyState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        let actor = Some(state_update.actor.clone());

                                                        if policy_state.policy_type
                                                            == "tenant_delta"
                                                        {
                                                            // Lightweight tenant delta — no tree structure, no prompt text
                                                            match super::tree_ops::TenantDelta::from_bytes(
                                                                &policy_state.config,
                                                            ) {
                                                                Ok(delta) => {
                                                                    sync_manager
                                                                        .apply_remote_tenant_delta(
                                                                            delta, actor,
                                                                        );
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to deserialize tenant delta for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state_lz4"
                                                        {
                                                            // LZ4-compressed snapshot (TreeState or TreeSnapshot bytes)
                                                            match super::tree_ops::lz4_decompress(
                                                                &policy_state.config,
                                                            ) {
                                                                Ok(decompressed) => {
                                                                    // Try TreeState first (backward compat)
                                                                    if let Ok(tree_state) =
                                                                        super::tree_ops::TreeState::from_bytes(
                                                                            &decompressed,
                                                                        )
                                                                    {
                                                                        sync_manager
                                                                            .apply_remote_tree_operation(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                tree_state,
                                                                                actor,
                                                                            );
                                                                    } else if let Ok(snap) =
                                                                        kv_index::snapshot::TreeSnapshot::from_bytes(
                                                                            &decompressed,
                                                                        )
                                                                    {
                                                                        let tree_state =
                                                                            super::tree_ops::TreeState::from_snapshot(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                &snap,
                                                                                policy_state.version,
                                                                            );
                                                                        sync_manager
                                                                            .apply_remote_tree_operation(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                tree_state,
                                                                                actor,
                                                                            );
                                                                    } else {
                                                                        log::warn!(
                                                                            "Failed to deserialize tree_state_lz4 payload for model {}",
                                                                            policy_state.model_id
                                                                        );
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to LZ4-decompress tree state for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state_delta"
                                                        {
                                                            // Delta: apply only the new operations
                                                            match super::tree_ops::TreeStateDelta::from_bytes(
                                                                    &policy_state.config,
                                                                )
                                                            {
                                                                Ok(delta) => {
                                                                    sync_manager
                                                                        .apply_remote_tree_delta(
                                                                            delta, actor,
                                                                        );
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to deserialize tree state delta for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state"
                                                        {
                                                            // Full state: replace (backward compatible)
                                                            match super::tree_ops::TreeState::from_bytes(
                                                                    &policy_state.config,
                                                                )
                                                            {
                                                                Ok(tree_state) => {
                                                                    sync_manager
                                                                        .apply_remote_tree_operation(
                                                                            policy_state
                                                                                .model_id
                                                                                .clone(),
                                                                            tree_state,
                                                                            actor,
                                                                        );
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to deserialize tree state for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else {
                                                            // Regular policy state update
                                                            sync_manager.apply_remote_policy_state(
                                                                policy_state,
                                                                actor,
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::RateLimit => {
                                                    // Backward-compatible rate-limit decoding:
                                                    // old payloads may send OperationLog, newer ones send raw i64.
                                                    if let Ok(log) = bincode::deserialize::<
                                                        super::crdt_kv::OperationLog,
                                                    >(&state_update.value)
                                                    {
                                                        sync_manager
                                                            .apply_remote_rate_limit_counter(&log);
                                                    } else if let Ok(counter_value) =
                                                        bincode::deserialize::<i64>(
                                                            &state_update.value,
                                                        )
                                                    {
                                                        sync_manager
                                                            .apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
                                                                state_update.key.clone(),
                                                                state_update.actor.clone(),
                                                                counter_value,
                                                                state_update.timestamp,
                                                            );
                                                    } else {
                                                        log::warn!(
                                                            key = %state_update.key,
                                                            "Failed to decode rate-limit update as OperationLog or i64"
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(StreamPayload::Ack(
                                            super::service::gossip::StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence: sequence.fetch_add(1, Ordering::Relaxed),
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(ack).await.is_err() {
                                        log::warn!("Failed to send ACK to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::SnapshotChunk => {
                                    log::info!(
                                        "Received snapshot chunk from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                    // Server side handles snapshot assembly
                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(StreamPayload::Ack(
                                            super::service::gossip::StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence: sequence.fetch_add(1, Ordering::Relaxed),
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(ack).await.is_err() {
                                        log::warn!("Failed to send ACK to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::Heartbeat => {
                                    log::trace!("Received heartbeat from {}", peer_name);
                                    // Send heartbeat back
                                    let heartbeat = StreamMessage {
                                        message_type: StreamMessageType::Heartbeat as i32,
                                        payload: None,
                                        sequence: sequence.fetch_add(1, Ordering::Relaxed),
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(heartbeat).await.is_err() {
                                        log::warn!("Failed to send heartbeat to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::SnapshotRequest => {
                                    log::info!("Received snapshot request from {}", peer_name);
                                    // Handle snapshot request - generate and send snapshot using GossipService
                                    if let Some(StreamPayload::SnapshotRequest(req)) = &msg.payload {
                                        use std::net::SocketAddr;

                                        use super::{
                                            ping_server::GossipService,
                                            stores::StoreType as LocalStoreType,
                                        };

                                        let store_type = LocalStoreType::from_proto(req.store);
                                        log::info!(
                                            "Generating snapshot for store {:?}",
                                            store_type
                                        );

                                        // Create a temporary GossipService to generate snapshot chunks
                                        let service = GossipService::new(
                                            Arc::new(parking_lot::RwLock::new(BTreeMap::new())),
                                            SocketAddr::from(([0, 0, 0, 0], 0)),
                                            SocketAddr::from(([0, 0, 0, 0], 0)),
                                            &self_name,
                                        )
                                        .with_stores(stores.clone())
                                        .with_sync_manager(sync_manager.clone());

                                        let chunks =
                                            service.create_snapshot_chunks(store_type, 100);
                                        let total_chunks = chunks.len() as u64;

                                        log::info!(
                                            "Sending {} snapshot chunks for store {:?}",
                                            total_chunks,
                                            store_type
                                        );

                                        let mut sent_chunks: u64 = 0;
                                        for chunk in chunks {
                                            let snapshot_chunk = StreamMessage {
                                                message_type: StreamMessageType::SnapshotChunk
                                                    as i32,
                                                payload: Some(StreamPayload::SnapshotChunk(chunk)),
                                                sequence: sequence.fetch_add(1, Ordering::Relaxed),
                                                peer_id: self_name.clone(),
                                            };

                                            if tx.send(snapshot_chunk).await.is_err() {
                                                log::warn!(
                                                    "Failed to send snapshot chunk {} to {}",
                                                    sent_chunks,
                                                    peer_name
                                                );
                                                break;
                                            }

                                            sent_chunks += 1;
                                        }

                                        log::info!(
                                            "Sent {} snapshot chunks for store {:?} to {}",
                                            sent_chunks,
                                            store_type,
                                            peer_name
                                        );
                                    }
                                }
                                StreamMessageType::Ack => {
                                    log::trace!(
                                        "Received ACK from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                }
                                StreamMessageType::Nack => {
                                    log::warn!(
                                        "Received NACK from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                }
                                StreamMessageType::SnapshotComplete => {
                                    log::debug!(
                                        "Received message type {:?} from {}",
                                        msg.message_type,
                                        peer_name
                                    );
                                }
                                StreamMessageType::StreamBatch => {
                                    if let Some(mesh_kv) = &mesh_kv {
                                        if let Some(StreamPayload::StreamBatch(batch)) =
                                            msg.payload
                                        {
                                            dispatch_stream_batch(
                                                mesh_kv,
                                                &msg.peer_id,
                                                batch.entries,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            log::error!("Error receiving from sync_stream with {}: {}", peer_name, e);
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            log::warn!(
                                "sync_stream to {peer_name} idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
                            );
                            break;
                        }
                    }
                }

                incremental_sender_handle.abort();
                let _ = incremental_sender_handle.await;
                log::debug!(
                    peer = %peer_name,
                    "sync_stream_handler exited — handler dropped"
                );
            }
            .instrument(span),
        )
    }

    /// Start a sync_stream connection to a peer
    async fn start_sync_stream_connection(&self, peer: NodeState) -> Result<()> {
        let peer_name = peer.name.clone();
        let peer_addr = peer.address.clone();

        // Check if connection already exists
        {
            let connections = self.sync_connections.lock().await;
            if connections.contains_key(&peer_name) {
                log::debug!("Sync stream connection to {} already exists", peer_name);
                return Ok(());
            }
        }

        // Check if we should initiate connection (avoid duplicates)
        if !self.should_initiate_connection(&peer_name) {
            log::debug!(
                "Skipping sync_stream to {} (peer should initiate)",
                peer_name
            );
            return Ok(());
        }

        log::info!(
            "Starting sync_stream connection to peer {} at address {}",
            peer_name,
            peer_addr
        );

        // Connect to peer's gRPC service via Endpoint so TLS can be configured.
        let connect_url = if self.mtls_manager.is_some() {
            format!("https://{peer_addr}")
        } else {
            format!("http://{peer_addr}")
        };
        log::info!("Connecting to URL: {}", connect_url);

        let mut endpoint = Endpoint::from_shared(connect_url.clone())
            .map_err(|e| anyhow::anyhow!("Invalid peer endpoint {connect_url}: {e}"))?;

        if let Some(mtls_manager) = self.mtls_manager.clone() {
            let tls_domain = endpoint
                .uri()
                .host()
                .map(str::to_owned)
                .unwrap_or_else(|| peer_name.clone());
            endpoint = configure_mtls_endpoint_for_peer(
                endpoint,
                mtls_manager,
                &peer_name,
                &peer_addr,
                tls_domain,
            )
            .await?;
        }

        let channel = endpoint.connect().await.map_err(|e| {
            log::warn!(
                "Failed to connect to peer {} for sync_stream: {}",
                peer_name,
                e
            );
            anyhow::anyhow!("Connection failed: {e}")
        })?;
        let mut client = GossipClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);

        // Create bidirectional stream
        let (tx, rx) = mpsc::channel::<StreamMessage>(128);
        let outgoing_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let response = client.sync_stream(outgoing_stream).await.map_err(|e| {
            log::error!("Failed to establish sync_stream with {}: {}", peer_name, e);
            anyhow::anyhow!("sync_stream RPC failed: {e}")
        })?;

        let incoming_stream = response.into_inner();

        // Spawn task to handle the bidirectional stream
        let self_name = self.self_name.clone();
        let peer_name_clone = peer_name.clone();

        let handle =
            self.spawn_sync_stream_handler(incoming_stream, tx, self_name, peer_name_clone);

        // Store the task handle
        {
            let mut connections = self.sync_connections.lock().await;
            connections.insert(peer_name.clone(), handle);
        }

        log::info!("Sync stream connection to {} established", peer_name);
        Ok(())
    }
}

// TODO: Support weighted random selection. e.g. nodes in INIT state should be more likely to be selected.
fn get_random_values_refs<K, V>(map: &BTreeMap<K, V>, k: usize) -> Vec<&V> {
    let values: Vec<&V> = map.values().collect();

    if k >= values.len() {
        let mut all_values = values;
        all_values.shuffle(&mut rand::rng());
        return all_values;
    }

    let mut rng = rand::rng();

    values.choose_multiple(&mut rng, k).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str, status: NodeStatus, version: u64) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: format!("10.0.0.1:944{}", version % 10),
            status: status as i32,
            version,
            metadata: HashMap::new(),
        }
    }

    fn snapshot(nodes: Vec<NodeState>) -> BTreeMap<String, NodeState> {
        nodes.into_iter().map(|n| (n.name.clone(), n)).collect()
    }

    #[test]
    fn pick_down_probe_target_returns_none_when_empty() {
        let snap = snapshot(vec![]);
        assert!(MeshController::pick_down_probe_target(&snap, "self").is_none());
    }

    #[test]
    fn pick_down_probe_target_returns_none_when_no_down_peers() {
        let snap = snapshot(vec![
            node("self", NodeStatus::Alive, 1),
            node("a", NodeStatus::Alive, 1),
            node("b", NodeStatus::Suspected, 2),
            node("c", NodeStatus::Leaving, 1),
        ]);
        assert!(MeshController::pick_down_probe_target(&snap, "self").is_none());
    }

    #[test]
    fn pick_down_probe_target_returns_the_one_down_peer() {
        let snap = snapshot(vec![
            node("self", NodeStatus::Alive, 1),
            node("a", NodeStatus::Alive, 1),
            node("b", NodeStatus::Down, 3),
        ]);
        let picked = MeshController::pick_down_probe_target(&snap, "self").unwrap();
        assert_eq!(picked.name, "b");
        assert_eq!(picked.status, NodeStatus::Down as i32);
    }

    #[test]
    fn pick_down_probe_target_excludes_self_even_if_marked_down() {
        // Defensive: if the local node ever sees itself as Down (it shouldn't
        // past self-incarnation refresh, but during the same round a stale
        // entry could remain), don't probe ourselves.
        let snap = snapshot(vec![node("self", NodeStatus::Down, 3)]);
        assert!(MeshController::pick_down_probe_target(&snap, "self").is_none());
    }

    #[test]
    fn pick_down_probe_target_picks_only_from_down_when_multiple_present() {
        let snap = snapshot(vec![
            node("self", NodeStatus::Alive, 1),
            node("a", NodeStatus::Down, 3),
            node("b", NodeStatus::Down, 5),
            node("c", NodeStatus::Alive, 1),
            node("d", NodeStatus::Suspected, 2),
        ]);
        // Run several iterations so randomness doesn't flake — every pick
        // must be one of the two Down peers, never the Alive/Suspected ones.
        for _ in 0..32 {
            let picked = MeshController::pick_down_probe_target(&snap, "self").unwrap();
            assert!(
                picked.name == "a" || picked.name == "b",
                "picked unexpected peer {} (status {})",
                picked.name,
                picked.status,
            );
        }
    }
}
