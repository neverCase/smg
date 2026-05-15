//! Region-readiness adapter — emits `smg-readiness/{region}/{server_name}`.
//!
//! Publishes this replica's current `ready` boolean on reconcile.

use std::sync::Arc;

use crate::cross_region::{
    CrossRegionResult, CrossRegionSyncService, SignalKey, SignalKind, SmgReadinessSignal,
};

/// Default freshness window: 30 s with a 5 s reconcile.
pub const DEFAULT_READINESS_STALE_AFTER_MS: u32 = 30_000;

/// Region-readiness producer.
#[derive(Debug, Clone)]
pub struct RegionReadinessAdapter {
    sync: Arc<CrossRegionSyncService>,
    stale_after_ms: u32,
}

impl RegionReadinessAdapter {
    /// Build an adapter rooted at the given sync service.
    pub fn new(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            sync,
            stale_after_ms: DEFAULT_READINESS_STALE_AFTER_MS,
        }
    }

    /// Override the freshness window stamped on each published envelope.
    pub fn with_stale_after_ms(mut self, stale_after_ms: u32) -> Self {
        self.stale_after_ms = stale_after_ms;
        self
    }

    /// Publish the current readiness state.
    pub fn publish_ready(&self, ready: bool) -> CrossRegionResult<()> {
        let region_id = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::SmgReadiness {
            region_id: region_id.clone(),
            server_name: server_name.clone(),
        };
        let body = SmgReadinessSignal {
            region_id,
            server_name,
            ready,
        };
        self.sync
            .publish_signal(key, SignalKind::SmgReadiness(body), self.stale_after_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_region::adapters::test_support::{live_envelopes, service, single_live};

    #[test]
    fn publish_ready_emits_per_replica_envelope() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone());

        adapter.publish_ready(true).expect("publish ok");
        let env = single_live(&svc);
        assert!(matches!(env.signal, Some(SignalKind::SmgReadiness(_))));
        match &env.key {
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            } => {
                assert_eq!(region_id, "us-ashburn-1");
                assert_eq!(server_name, "smg-router-a");
            }
            _ => panic!("unexpected key kind: {:?}", env.key),
        }
        assert_eq!(env.actor, "smg-router-a");
        assert_eq!(env.stale_after_ms, DEFAULT_READINESS_STALE_AFTER_MS);
    }

    #[test]
    fn publish_ready_toggles_value() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone());

        adapter.publish_ready(true).unwrap();
        let first_version = single_live(&svc).version;
        adapter.publish_ready(false).unwrap();
        let env = single_live(&svc);
        match env.signal {
            Some(SignalKind::SmgReadiness(s)) => assert!(!s.ready),
            other => panic!("unexpected signal: {other:?}"),
        }
        assert!(env.version > first_version);
        // Mesh CRDTs collapse same-key writes, so the namespace only ever
        // holds one envelope per `SignalKey`.
        assert_eq!(live_envelopes(&svc).len(), 1);
    }

    #[test]
    fn with_stale_after_ms_overrides_default() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone()).with_stale_after_ms(5_000);
        adapter.publish_ready(true).unwrap();
        assert_eq!(single_live(&svc).stale_after_ms, 5_000);
    }
}
