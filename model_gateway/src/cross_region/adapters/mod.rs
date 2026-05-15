//! Local-producer adapters for the four cross-region signals.
//!
//! Adapters read local gateway state, publish through `CrossRegionSyncService`,
//! and let the runtime subscriber apply inbound mesh deliveries.

mod client_latency;
mod orchestrator;
mod region_readiness;
mod worker_health;
mod worker_load;

#[cfg(test)]
pub(super) mod test_support;

pub use client_latency::ClientLatencyAdapter;
pub use orchestrator::{CrossRegionProducers, ProducerCadences, ProducerHandles};
pub use region_readiness::RegionReadinessAdapter;
pub use worker_health::WorkerHealthAdapter;
pub use worker_load::WorkerLoadAdapter;
