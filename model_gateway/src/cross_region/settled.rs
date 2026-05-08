use super::{CrossRegionError, CrossRegionResult, RegionPeerRegistry, SettledRequestContext};

/// Authenticated inbound Region Agent identity derived from the mTLS peer certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedPeerIdentity {
    region_id: String,
    mtls_identity: String,
}

impl AuthenticatedPeerIdentity {
    /// Build an authenticated peer identity after rejecting blank fields.
    pub fn new(
        region_id: impl Into<String>,
        mtls_identity: impl Into<String>,
    ) -> CrossRegionResult<Self> {
        let identity = Self {
            region_id: region_id.into(),
            mtls_identity: mtls_identity.into(),
        };
        if identity.region_id.trim().is_empty() {
            return Err(CrossRegionError::UnauthorizedPeer {
                reason: "authenticated peer region must not be empty".to_string(),
            });
        }
        if identity.mtls_identity.trim().is_empty() {
            return Err(CrossRegionError::UnauthorizedPeer {
                reason: "authenticated peer mTLS identity must not be empty".to_string(),
            });
        }
        Ok(identity)
    }

    /// Return the authenticated peer region extracted from mTLS identity.
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    /// Return the authenticated peer URI SAN extracted from mTLS identity.
    pub fn mtls_identity(&self) -> &str {
        &self.mtls_identity
    }
}

/// Validate that a settled request can only execute in the local target region.
pub fn validate_settled_local_execution(
    context: &SettledRequestContext,
    local_region_id: &str,
    peer_identity: &AuthenticatedPeerIdentity,
    peer_registry: &RegionPeerRegistry,
    request_model_id: &str,
) -> CrossRegionResult<()> {
    if local_region_id.trim().is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: "cross_region.region_id is required for settled execution".to_string(),
        });
    }

    if context.route.target_region != local_region_id {
        return Err(CrossRegionError::InvalidHeader {
            reason: format!("x-target-region must match local region_id '{local_region_id}'"),
        });
    }

    if context.route.committed_model != request_model_id {
        return Err(CrossRegionError::InvalidHeader {
            reason: "x-committed-model must match request model".to_string(),
        });
    }

    if peer_identity.region_id() != context.common.entry_region {
        return Err(CrossRegionError::UnauthorizedPeer {
            reason: "authenticated peer region must match x-entry-region".to_string(),
        });
    }

    let peer = peer_registry.request_target(&context.common.entry_region)?;
    if peer_identity.mtls_identity() != peer.expected_mtls_identity() {
        return Err(CrossRegionError::UnauthorizedPeer {
            reason: "authenticated peer mTLS identity does not match configured peer".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_region::{
        CrossRegionCommonHeaders, RegionPeer, RequestMode, SettledRouteMetadata,
    };

    /// Build a valid settled request context for validation tests.
    fn settled_context() -> SettledRequestContext {
        SettledRequestContext {
            common: CrossRegionCommonHeaders {
                contract_version: 1,
                source_service: "smg".to_string(),
                opc_request_id: "opc-request-1".to_string(),
                entry_region: "us-chicago-1".to_string(),
                request_mode: RequestMode::Settled,
            },
            route: SettledRouteMetadata {
                target_region: "us-ashburn-1".to_string(),
                committed_model: "cohere.command-r-plus".to_string(),
                route_id: "route-1".to_string(),
                attempt: 1,
            },
        }
    }

    /// Build a peer registry containing the source Region Agent.
    fn peer_registry() -> RegionPeerRegistry {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should build");
        RegionPeerRegistry::new(vec![peer]).expect("registry should build")
    }

    /// Build an authenticated peer identity for the configured source region.
    fn peer_identity(region_id: &str) -> AuthenticatedPeerIdentity {
        AuthenticatedPeerIdentity::new(
            region_id,
            format!(
                "spiffe://oraclecorp.com/oci/oc1/prod/region/{region_id}/service/smg-region-agent"
            ),
        )
        .expect("peer identity should build")
    }

    #[test]
    fn valid_settled_metadata_allows_local_execution() {
        let context = settled_context();

        validate_settled_local_execution(
            &context,
            "us-ashburn-1",
            &peer_identity("us-chicago-1"),
            &peer_registry(),
            "cohere.command-r-plus",
        )
        .expect("settled execution should validate");
    }

    #[test]
    fn target_region_mismatch_is_rejected() {
        let mut context = settled_context();
        context.route.target_region = "us-phoenix-1".to_string();

        let error = validate_settled_local_execution(
            &context,
            "us-ashburn-1",
            &peer_identity("us-chicago-1"),
            &peer_registry(),
            "cohere.command-r-plus",
        )
        .expect_err("target mismatch should fail");

        assert!(error.to_string().contains("x-target-region"));
    }

    #[test]
    fn authenticated_peer_region_mismatch_is_rejected() {
        let context = settled_context();

        let error = validate_settled_local_execution(
            &context,
            "us-ashburn-1",
            &peer_identity("us-phoenix-1"),
            &peer_registry(),
            "cohere.command-r-plus",
        )
        .expect_err("peer region mismatch should fail");

        assert!(error.to_string().contains("authenticated peer region"));
    }

    #[test]
    fn authenticated_peer_identity_mismatch_is_rejected() {
        let context = settled_context();
        let wrong_identity = AuthenticatedPeerIdentity::new(
            "us-chicago-1",
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/other",
        )
        .expect("identity should build");

        let error = validate_settled_local_execution(
            &context,
            "us-ashburn-1",
            &wrong_identity,
            &peer_registry(),
            "cohere.command-r-plus",
        )
        .expect_err("mTLS identity mismatch should fail");

        assert!(error.to_string().contains("mTLS identity"));
    }

    #[test]
    fn committed_model_mismatch_is_rejected() {
        let context = settled_context();

        let error = validate_settled_local_execution(
            &context,
            "us-ashburn-1",
            &peer_identity("us-chicago-1"),
            &peer_registry(),
            "other-model",
        )
        .expect_err("model mismatch should fail");

        assert!(error.to_string().contains("x-committed-model"));
    }
}
