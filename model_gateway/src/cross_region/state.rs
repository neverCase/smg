use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::{ClientLatencySignal, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal};

/// Version and freshness metadata for a materialized signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalVersion {
    pub version: u64,
    pub updated_at_ms: i64,
}

/// In-memory materialized view for remote cross-region signals.
#[derive(Debug, Clone, Default)]
pub struct CrossRegionState {
    readiness: HashMap<String, (SmgReadinessSignal, SignalVersion)>,
    worker_health: HashMap<(String, String), (WorkerHealthSignal, SignalVersion)>,
    worker_load: HashMap<(String, String), (WorkerLoadSignal, SignalVersion)>,
    client_latency: HashMap<(String, String), (ClientLatencySignal, SignalVersion)>,
}

impl CrossRegionState {
    /// Create an empty materialized signal state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return true when the materialized view has no remote signals.
    pub fn is_empty(&self) -> bool {
        self.readiness.is_empty()
            && self.worker_health.is_empty()
            && self.worker_load.is_empty()
            && self.client_latency.is_empty()
    }

    /// Return the readiness signal for a region when present.
    pub fn readiness(&self, region_id: &str) -> Option<&SmgReadinessSignal> {
        self.readiness.get(region_id).map(|(signal, _)| signal)
    }

    /// Return the readiness signal with its freshness version when present.
    pub fn readiness_with_version(
        &self,
        region_id: &str,
    ) -> Option<(&SmgReadinessSignal, SignalVersion)> {
        self.readiness
            .get(region_id)
            .map(|(signal, version)| (signal, *version))
    }

    /// Return the worker health signal for a region/worker when present.
    pub fn worker_health(&self, region_id: &str, worker_id: &str) -> Option<&WorkerHealthSignal> {
        self.worker_health
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return the worker health signal with its freshness version when present.
    pub fn worker_health_with_version(
        &self,
        region_id: &str,
        worker_id: &str,
    ) -> Option<(&WorkerHealthSignal, SignalVersion)> {
        self.worker_health
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, version)| (signal, *version))
    }

    /// Return the worker load signal for a region/worker when present.
    pub fn worker_load(&self, region_id: &str, worker_id: &str) -> Option<&WorkerLoadSignal> {
        self.worker_load
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return the worker load signal with its freshness version when present.
    pub fn worker_load_with_version(
        &self,
        region_id: &str,
        worker_id: &str,
    ) -> Option<(&WorkerLoadSignal, SignalVersion)> {
        self.worker_load
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, version)| (signal, *version))
    }

    /// Return the client latency signal for a client/target region pair when present.
    pub fn client_latency(
        &self,
        client_region: &str,
        target_region: &str,
    ) -> Option<&ClientLatencySignal> {
        self.client_latency
            .get(&(client_region.to_string(), target_region.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return the client latency signal with its freshness version when present.
    pub fn client_latency_with_version(
        &self,
        client_region: &str,
        target_region: &str,
    ) -> Option<(&ClientLatencySignal, SignalVersion)> {
        self.client_latency
            .get(&(client_region.to_string(), target_region.to_string()))
            .map(|(signal, version)| (signal, *version))
    }

    /// Return all regions represented by materialized remote signals in stable order.
    pub fn regions(&self) -> Vec<&str> {
        let mut regions = BTreeSet::new();
        regions.extend(self.readiness.keys().map(String::as_str));
        regions.extend(self.worker_health.keys().map(|(region, _)| region.as_str()));
        regions.extend(self.worker_load.keys().map(|(region, _)| region.as_str()));
        regions.extend(
            self.client_latency
                .keys()
                .map(|(_, target_region)| target_region.as_str()),
        );
        regions.into_iter().collect()
    }

    /// Return all worker ids represented by health or load signals for one region.
    pub fn worker_ids(&self, region_id: &str) -> Vec<&str> {
        let mut worker_ids = BTreeSet::new();
        worker_ids.extend(
            self.worker_health
                .keys()
                .filter_map(|(region, worker)| (region == region_id).then_some(worker.as_str())),
        );
        worker_ids.extend(
            self.worker_load
                .keys()
                .filter_map(|(region, worker)| (region == region_id).then_some(worker.as_str())),
        );
        worker_ids.into_iter().collect()
    }

    /// Insert or replace a readiness signal in the materialized view.
    pub fn upsert_readiness(&mut self, signal: SmgReadinessSignal, version: SignalVersion) {
        self.readiness
            .insert(signal.region_id.clone(), (signal, version));
    }

    /// Insert or replace a worker health signal in the materialized view.
    pub fn upsert_worker_health(&mut self, signal: WorkerHealthSignal, version: SignalVersion) {
        self.worker_health.insert(
            (signal.region_id.clone(), signal.worker_id.clone()),
            (signal, version),
        );
    }

    /// Insert or replace a worker load signal in the materialized view.
    pub fn upsert_worker_load(&mut self, signal: WorkerLoadSignal, version: SignalVersion) {
        self.worker_load.insert(
            (signal.region_id.clone(), signal.worker_id.clone()),
            (signal, version),
        );
    }

    /// Insert or replace a client latency signal in the materialized view.
    pub fn upsert_client_latency(&mut self, signal: ClientLatencySignal, version: SignalVersion) {
        self.client_latency.insert(
            (signal.client_region.clone(), signal.target_region.clone()),
            (signal, version),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_empty() {
        let state = CrossRegionState::new();

        assert!(state.is_empty());
        assert!(state.readiness("us-chicago-1").is_none());
    }
}
