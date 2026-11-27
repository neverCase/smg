//! Optional worker metadata for prefill and time-to-first-token prediction.
//!
//! These types describe model artifacts, deployment configuration and calibration.
//! They do not collect data or change scheduling. Missing values stay unset; field
//! metadata distinguishes missing, automatic and explicitly unspecified values.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Model and deployment facts used to build a worker's prefill predictor.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct WorkerPrefillSpec {
    /// Version of this metadata schema, independent of the engine version.
    pub schema_version: u32,
    /// Model-specific facts, keyed by the model ID served by this worker.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelPrefillSpec>,
    pub engine: Option<PrefillEngineSpec>,
    pub hardware: Option<PrefillHardwareSpec>,
    pub predictor: Option<PrefillPredictorSpec>,
    /// Provenance keyed by paths within this block, for example
    /// `/models/org~1model/execution/model_dtype` (JSON Pointer escaping).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub field_metadata: BTreeMap<String, PrefillFieldMetadata>,
}

impl Default for WorkerPrefillSpec {
    fn default() -> Self {
        Self {
            schema_version: 1,
            models: BTreeMap::new(),
            engine: None,
            hardware: None,
            predictor: None,
            field_metadata: BTreeMap::new(),
        }
    }
}

/// File declarations and execution facts for one model, kept separately.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ModelPrefillSpec {
    pub artifact: Option<PrefillArtifactSpec>,
    pub architecture: Option<PrefillArchitectureSpec>,
    pub checkpoint: Option<PrefillCheckpointSpec>,
    pub execution: Option<PrefillExecutionSpec>,
}

/// Identity of the model files used for feature extraction.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillArtifactSpec {
    /// Path as seen by the process that reads the model files.
    pub local_path: Option<String>,
    pub repo_id: Option<String>,
    /// Revision or commit identifying the checkpoint.
    pub revision: Option<String>,
    /// Configuration content hash; include the hash algorithm in the value.
    pub config_hash: Option<String>,
}

/// Language-model geometry, including hybrid-attention and MoE structure.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillArchitectureSpec {
    pub num_hidden_layers: Option<u32>,
    pub hidden_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    /// Explicit head dimension, which need not be hidden_size / head count.
    pub head_dim: Option<u32>,
    pub vocab_size: Option<u32>,
    /// Per-layer attention kinds in model order. An empty list means unknown.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layer_types: Vec<String>,
    /// Attention window length in tokens, when the architecture declares one.
    pub sliding_window: Option<u64>,
    pub full_attention_interval: Option<u32>,
    pub moe_config: Option<PrefillMoeSpec>,
    pub linear_attention_config: Option<PrefillLinearAttentionSpec>,
}

/// Normalized MoE geometry plus any architecture-specific declarations.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillMoeSpec {
    pub num_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub num_shared_experts: Option<u32>,
    pub shared_expert_intermediate_size: Option<u32>,
    pub raw_config: Option<serde_json::Value>,
}

/// Linear-attention dimensions, separate from full-attention geometry.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillLinearAttentionSpec {
    pub linear_conv_kernel_dim: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub mamba_ssm_dtype: Option<String>,
}

/// Properties declared by checkpoint files, not necessarily runtime settings.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillCheckpointSpec {
    pub declared_dtype: Option<String>,
    pub quantization: Option<PrefillQuantizationSpec>,
    /// On-disk tensor storage bytes, not GPU-resident memory or parameter count.
    pub weight_storage_bytes: Option<u64>,
}

/// Quantization declarations; leave scalar values unset for mixed formats.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillQuantizationSpec {
    pub method: Option<String>,
    pub bits: Option<u32>,
    /// Signed because some formats use -1 for a full-row group.
    pub group_size: Option<i64>,
    pub mode: Option<String>,
    /// Preserve module-specific or mixed-precision configuration here.
    pub raw_config: Option<serde_json::Value>,
}

/// Model settings reported by the engine rather than inferred from files.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillExecutionSpec {
    pub model_dtype: Option<String>,
    pub quantization_method: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub max_model_len: Option<u64>,
}

/// Engine configuration affecting prefill batching, kernels and parallelism.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillEngineSpec {
    pub version: Option<String>,
    pub tp_size: Option<u32>,
    pub pp_size: Option<u32>,
    pub dp_size: Option<u32>,
    pub expert_parallel_config: Option<PrefillExpertParallelSpec>,
    pub attention_backend: Option<String>,
    pub prefix_cache_enabled: Option<bool>,
    pub chunked_prefill_enabled: Option<bool>,
    /// Per-step token budget; retain signed backend sentinel values.
    pub max_num_batched_tokens: Option<i64>,
    /// Backend chunk size, kept separate from a per-step token budget.
    pub chunked_prefill_size: Option<i64>,
    pub max_running_requests: Option<u32>,
    /// SGLang prefill-token limit, separate from chunk size and vLLM's step budget.
    pub max_prefill_tokens: Option<i64>,
}

/// Expert parallelism need not have the same width as tensor parallelism.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillExpertParallelSpec {
    pub enabled: Option<bool>,
    pub size: Option<u32>,
    pub raw_config: Option<serde_json::Value>,
}

/// Devices assigned to this worker, not devices on the gateway host.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillHardwareSpec {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<PrefillDeviceSpec>,
}

/// Stable device identity and placement; memory sizes are always bytes.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillDeviceSpec {
    pub gpu_model: Option<String>,
    /// Allocated GPU compute percentage (100 means a whole GPU), not CUDA core count.
    pub gpu_cores: Option<u32>,
    /// GPU or MIG-device UUID, as appropriate for this allocation.
    pub uuid: Option<String>,
    pub memory_bytes: Option<u64>,
    /// Global distributed rank within the engine deployment.
    pub rank: Option<u32>,
    pub node: Option<String>,
    /// Interconnect description, such as NVLink or PCIe topology.
    pub interconnect: Option<String>,
}

/// Calibration identity and parameters; no prediction is performed here.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillPredictorSpec {
    pub profile_key: Option<String>,
    pub calibration_version: Option<String>,
    /// Named coefficients; units and feature definitions belong to the calibration.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub coefficients: BTreeMap<String, f64>,
    pub lookup_table_ref: Option<String>,
    pub sample_count: Option<u64>,
}

/// Provenance and availability of one field, including an absent value.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PrefillFieldMetadata {
    pub status: PrefillFieldStatus,
    /// Source identifier, such as an endpoint or model file and field path.
    pub source: Option<String>,
    /// Collection time in RFC 3339 format.
    pub collected_at: Option<String>,
    /// Optional explanation for missing, overridden or failed observations.
    pub message: Option<String>,
}

/// Distinguishes an unavailable observation from an explicit backend setting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PrefillFieldStatus {
    /// No observation or availability information yet.
    #[default]
    Unknown,
    /// The source did not contain the field.
    Missing,
    /// The backend reported an automatic setting rather than a resolved value.
    Auto,
    /// The source explicitly left the setting unspecified, for example JSON null.
    Unspecified,
    /// The backend explicitly disabled the feature or selected no quantization.
    Disabled,
    /// A model-file or launch-configuration declaration.
    Declared,
    /// A setting resolved and reported by the engine.
    Resolved,
    /// An explicitly supplied override.
    Overridden,
    /// Reading the source failed.
    Error,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{PrefillFieldStatus, WorkerPrefillSpec};
    use crate::worker::WorkerSpec;

    #[test]
    fn legacy_worker_specs_keep_prefill_unset() {
        let legacy = json!({"url": "http://worker:8000"});
        let absent: WorkerSpec = serde_json::from_value(legacy.clone()).unwrap();
        let null: WorkerSpec = serde_json::from_value(json!({
            "url": "http://worker:8000", "prefill": null,
        }))
        .unwrap();
        let constructed = WorkerSpec::new("http://worker:8000");

        for spec in [absent, null, constructed] {
            assert!(spec.prefill.is_none());
            assert!(serde_json::to_value(spec).unwrap().get("prefill").is_none());
        }
    }

    #[test]
    fn partial_metadata_preserves_unknowns_and_explicit_zero_or_false() {
        let spec: WorkerSpec = serde_json::from_value(json!({
            "url": "http://worker:8000",
            "prefill": {
                "models": {"org/model": {"execution": {"model_dtype": null}}},
                "engine": {"prefix_cache_enabled": false, "chunked_prefill_size": -1},
                "hardware": {"devices": [{"rank": 0}]},
                "predictor": {"sample_count": 0},
                "field_metadata": {
                    "/models/org~1model/execution/model_dtype": {
                        "status": "missing", "source": "engine"
                    },
                    "/models/org~1model/execution/quantization_method": {
                        "status": "unspecified"
                    }
                }
            }
        }))
        .unwrap();
        let prefill = spec.prefill.as_ref().unwrap();
        assert_eq!(prefill.schema_version, 1);
        let execution = prefill.models["org/model"].execution.as_ref().unwrap();
        assert_eq!(execution.model_dtype, None);
        assert_eq!(execution.quantization_method, None);
        assert_eq!(prefill.engine.as_ref().unwrap().tp_size, None);
        assert_eq!(
            prefill.field_metadata["/models/org~1model/execution/model_dtype"].status,
            PrefillFieldStatus::Missing
        );
        let output = serde_json::to_value(&spec).unwrap();
        let block = &output["prefill"];
        assert!(block["models"]["org/model"]["execution"]
            .get("model_dtype")
            .is_none());
        assert_eq!(block["engine"]["prefix_cache_enabled"], false);
        assert_eq!(block["engine"]["chunked_prefill_size"], -1);
        assert_eq!(block["hardware"]["devices"][0]["rank"], 0);
        assert_eq!(block["predictor"]["sample_count"], 0);
        assert_eq!(
            block["field_metadata"]["/models/org~1model/execution/quantization_method"]["status"],
            "unspecified"
        );

        let empty: WorkerPrefillSpec = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, WorkerPrefillSpec::default());
    }

    #[test]
    fn worker_metadata_round_trips_multiple_models_and_calibration() {
        let prefill = json!({
            "schema_version": 1,
            "models": {
                "org/hybrid": {
                    "artifact": {
                        "local_path": "/models/org/hybrid", "repo_id": "org/hybrid",
                        "revision": "checkpoint-revision", "config_hash": "sha256:example"
                    },
                    "architecture": {
                        "num_hidden_layers": 64, "hidden_size": 5120,
                        "intermediate_size": 17408, "num_attention_heads": 24,
                        "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                        "layer_types": ["linear_attention", "full_attention"],
                        "sliding_window": 4096, "full_attention_interval": 4,
                        "linear_attention_config": {
                            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                            "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32"
                        }
                    },
                    "checkpoint": {
                        "declared_dtype": "bfloat16", "weight_storage_bytes": 54000000000_u64,
                        "quantization": {
                            "bits": 4, "group_size": 64, "mode": "affine",
                            "raw_config": {"bits": 4, "group_size": 64, "mode": "affine"}
                        }
                    },
                    "execution": {
                        "model_dtype": "float16", "quantization_method": "awq",
                        "kv_cache_dtype": "fp8", "max_model_len": 32768
                    }
                },
                "org/moe": {
                    "architecture": {"moe_config": {
                        "num_experts": 64, "num_experts_per_tok": 8, "intermediate_size": 2048,
                        "num_shared_experts": 1, "shared_expert_intermediate_size": 4096,
                        "raw_config": {"n_routed_experts": 64}
                    }},
                    "checkpoint": {"quantization": {
                        "method": "compressed-tensors", "group_size": -1,
                        "raw_config": {"config_groups": {"group_0": {
                            "weights": {"num_bits": 4}, "input_activations": {"num_bits": 8}
                        }}}
                    }}
                }
            },
            "engine": {
                "version": "engine-version", "tp_size": 2, "pp_size": 1, "dp_size": 1,
                "expert_parallel_config": {
                    "enabled": true, "size": 2, "raw_config": {"placement": "round_robin"}
                },
                "attention_backend": "flashinfer", "prefix_cache_enabled": true,
                "chunked_prefill_enabled": true, "max_num_batched_tokens": 8192,
                "chunked_prefill_size": 4096, "max_running_requests": 128
            },
            "hardware": {"devices": [{
                "gpu_model": "test-gpu", "gpu_cores": 100, "uuid": "GPU-example", "memory_bytes": 80000000000_u64,
                "rank": 0, "node": "node-1", "interconnect": "NVLink"
            }]},
            "predictor": {
                "profile_key": "deployment-profile", "calibration_version": "v1",
                "coefficients": {"intercept_ms": 0.0, "linear_ms_per_token": 0.01},
                "lookup_table_ref": "profiles/prefill-v1.json", "sample_count": 42
            },
            "field_metadata": {
                "/models/org~1hybrid/checkpoint/declared_dtype": {
                    "status": "declared", "source": "config.json#/text_config/dtype",
                    "collected_at": "2026-11-26T08:00:00Z"
                },
                "/models/org~1hybrid/execution/model_dtype": {
                    "status": "resolved", "source": "engine", "message": "Engine reported value"
                }
            }
        });
        let spec: WorkerSpec = serde_json::from_value(json!({
            "url": "grpc://worker:50051", "prefill": prefill.clone(),
        }))
        .unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap()["prefill"], prefill);
        let block = spec.prefill.as_ref().unwrap();
        assert_eq!(block.models.len(), 2);
        assert_eq!(
            block.models["org/moe"]
                .checkpoint
                .as_ref()
                .unwrap()
                .quantization
                .as_ref()
                .unwrap()
                .bits,
            None
        );
    }

    #[test]
    fn worker_json_schema_includes_prefill_metadata() {
        let schema = serde_json::to_value(schemars::schema_for!(WorkerSpec)).unwrap();
        assert!(schema["properties"].get("prefill").is_some());
        assert!(schema["$defs"].get("WorkerPrefillSpec").is_some());
        assert!(schema["$defs"].get("PrefillFieldStatus").is_some());
        assert!(!schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("prefill")));
    }
}
