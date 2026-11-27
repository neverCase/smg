//! Best-effort model-file facts for prefill prediction. No weight data is loaded.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

use openai_protocol::prefill::{
    ModelPrefillSpec, PrefillArchitectureSpec, PrefillArtifactSpec, PrefillCheckpointSpec,
    PrefillFieldMetadata, PrefillFieldStatus, PrefillLinearAttentionSpec, PrefillMoeSpec,
    PrefillQuantizationSpec, WorkerPrefillSpec,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::time::Instant;

// The entire lookup shares a deadline inside create_local_worker's five seconds.
const FILE_LOOKUP_TIMEOUT: Duration = Duration::from_secs(3);
const QUANTIZATION_FILES: &[&str] = &[
    "quantize_config.json",
    "quant_config.json",
    "hf_quant_config.json",
];

pub(super) struct ModelFileMetadata {
    pub model: ModelPrefillSpec,
    /// JSON Pointers relative to this model's entry in WorkerPrefillSpec.models.
    fields: BTreeMap<String, PrefillFieldMetadata>,
}

fn observation(
    status: PrefillFieldStatus,
    source: impl Into<String>,
    message: Option<String>,
) -> PrefillFieldMetadata {
    PrefillFieldMetadata {
        status,
        source: Some(source.into()),
        collected_at: Some(chrono::Utc::now().to_rfc3339()),
        message,
    }
}

pub(super) fn precision_string(value: Option<&str>) -> Option<String> {
    let value = value?.trim().to_ascii_lowercase();
    let value = value.strip_prefix("torch.").unwrap_or(&value);
    match value {
        "" | "auto" | "unknown" | "none" | "null" => None,
        _ => Some(value.to_owned()),
    }
}

/// Do not send inaccessible engine-side paths to the Hub as repository IDs.
pub(super) fn is_hub_repo_id(model: &str) -> bool {
    !model.starts_with(['/', '.', '~'])
        && model.matches('/').count() <= 1
        && model.split('/').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

/// Reads individual fields independently: one invalid or missing field must
/// not discard the other model facts, and missing numeric fields stay unset.
struct ConfigReader<'a> {
    value: &'a Value,
    source: String,
    fallback: Option<(&'a Value, String)>,
    fields: &'a mut BTreeMap<String, PrefillFieldMetadata>,
}

impl ConfigReader<'_> {
    fn read<T: DeserializeOwned>(&mut self, target: &str, aliases: &[&str]) -> Option<T> {
        let found = aliases
            .iter()
            .find_map(|key| self.value.get(*key).map(|v| (*key, v, &self.source)))
            .or_else(|| {
                self.fallback.as_ref().and_then(|(value, source)| {
                    aliases
                        .iter()
                        .find_map(|key| value.get(*key).map(|v| (*key, v, source)))
                })
            });
        let (value, status, message, source) = match found {
            Some((key, Value::Null, source)) => (
                None,
                PrefillFieldStatus::Unspecified,
                None,
                format!("{source}/{key}"),
            ),
            Some((key, value, source)) => match serde_json::from_value(value.clone()) {
                Ok(value) => (
                    Some(value),
                    PrefillFieldStatus::Declared,
                    None,
                    format!("{source}/{key}"),
                ),
                Err(error) => (
                    None,
                    PrefillFieldStatus::Error,
                    Some(error.to_string()),
                    format!("{source}/{key}"),
                ),
            },
            None => (None, PrefillFieldStatus::Missing, None, self.source.clone()),
        };
        self.fields
            .insert(target.to_owned(), observation(status, source, message));
        value
    }
}

fn parse_config(config: &Value, path: &Path, raw: &[u8]) -> ModelFileMetadata {
    let mut fields = BTreeMap::new();
    let text = config
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    let section = if std::ptr::eq(text, config) {
        ""
    } else {
        "/text_config"
    };
    let mut reader = ConfigReader {
        value: text,
        source: format!("{}#{section}", path.display()),
        fallback: (!section.is_empty()).then(|| (config, format!("{}#", path.display()))),
        fields: &mut fields,
    };
    let mut effective = config.clone();
    if let (Some(root), Some(text)) = (effective.as_object_mut(), text.as_object()) {
        root.extend(text.clone());
    }
    let mut architecture = PrefillArchitectureSpec {
        num_hidden_layers: reader.read(
            "/architecture/num_hidden_layers",
            &["num_hidden_layers", "n_layer", "n_layers"],
        ),
        hidden_size: reader.read(
            "/architecture/hidden_size",
            &["hidden_size", "n_embd", "d_model"],
        ),
        intermediate_size: reader.read(
            "/architecture/intermediate_size",
            &["intermediate_size", "n_inner", "ffn_dim"],
        ),
        num_attention_heads: reader.read(
            "/architecture/num_attention_heads",
            &["num_attention_heads", "n_head", "n_heads"],
        ),
        num_key_value_heads: reader.read(
            "/architecture/num_key_value_heads",
            &["num_key_value_heads", "n_kv_heads"],
        ),
        head_dim: reader.read(
            "/architecture/head_dim",
            &["head_dim", "attention_head_dim"],
        ),
        vocab_size: reader.read("/architecture/vocab_size", &["vocab_size"]),
        layer_types: reader
            .read("/architecture/layer_types", &["layer_types"])
            .unwrap_or_default(),
        sliding_window: reader.read("/architecture/sliding_window", &["sliding_window"]),
        full_attention_interval: reader.read(
            "/architecture/full_attention_interval",
            &["full_attention_interval"],
        ),
        ..Default::default()
    };
    let expert_keys = [
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "num_selected_experts",
        "moe_top_k",
        "num_shared_experts",
        "n_shared_experts",
    ];
    if expert_keys.iter().any(|key| {
        effective
            .get(key)
            .and_then(Value::as_u64)
            .is_some_and(|n| n > 0)
    }) {
        let raw_config = effective.as_object().map(|obj| {
            Value::Object(
                obj.iter()
                    .filter(|(key, _)| {
                        key.contains("expert")
                            || key.starts_with("moe_")
                            || key.as_str() == "norm_topk_prob"
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )
        });
        architecture.moe_config = Some(PrefillMoeSpec {
            num_experts: reader.read(
                "/architecture/moe_config/num_experts",
                &["num_experts", "num_local_experts", "n_routed_experts"],
            ),
            num_experts_per_tok: reader.read(
                "/architecture/moe_config/num_experts_per_tok",
                &["num_experts_per_tok", "num_selected_experts", "moe_top_k"],
            ),
            intermediate_size: reader.read(
                "/architecture/moe_config/intermediate_size",
                &[
                    "moe_intermediate_size",
                    "expert_intermediate_size",
                    "intermediate_size",
                ],
            ),
            num_shared_experts: reader.read(
                "/architecture/moe_config/num_shared_experts",
                &["num_shared_experts", "n_shared_experts"],
            ),
            shared_expert_intermediate_size: reader.read(
                "/architecture/moe_config/shared_expert_intermediate_size",
                &[
                    "shared_expert_intermediate_size",
                    "shared_experts_intermediate_size",
                ],
            ),
            raw_config,
        });
    } else {
        reader.fields.insert(
            "/architecture/moe_config".into(),
            observation(
                PrefillFieldStatus::Missing,
                &reader.source,
                Some("No positive expert counts declared".into()),
            ),
        );
    }
    let linear_keys = [
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
    ];
    if linear_keys
        .iter()
        .any(|key| effective.get(key).is_some_and(|v| !v.is_null()))
    {
        architecture.linear_attention_config = Some(PrefillLinearAttentionSpec {
            linear_conv_kernel_dim: reader.read(
                "/architecture/linear_attention_config/linear_conv_kernel_dim",
                &["linear_conv_kernel_dim"],
            ),
            linear_key_head_dim: reader.read(
                "/architecture/linear_attention_config/linear_key_head_dim",
                &["linear_key_head_dim"],
            ),
            linear_num_key_heads: reader.read(
                "/architecture/linear_attention_config/linear_num_key_heads",
                &["linear_num_key_heads"],
            ),
            linear_num_value_heads: reader.read(
                "/architecture/linear_attention_config/linear_num_value_heads",
                &["linear_num_value_heads"],
            ),
            linear_value_head_dim: reader.read(
                "/architecture/linear_attention_config/linear_value_head_dim",
                &["linear_value_head_dim"],
            ),
            mamba_ssm_dtype: reader.read(
                "/architecture/linear_attention_config/mamba_ssm_dtype",
                &["mamba_ssm_dtype"],
            ),
        });
    } else {
        reader.fields.insert(
            "/architecture/linear_attention_config".into(),
            observation(PrefillFieldStatus::Missing, &reader.source, None),
        );
    }

    let dtype_pointers = [
        "/text_config/dtype",
        "/text_config/torch_dtype",
        "/dtype",
        "/torch_dtype",
    ];
    let dtype_pointer = dtype_pointers
        .into_iter()
        .find(|pointer| config.pointer(pointer).is_some_and(|v| !v.is_null()))
        .or_else(|| {
            dtype_pointers
                .into_iter()
                .find(|pointer| config.pointer(pointer).is_some())
        });
    let declared_dtype =
        dtype_pointer.and_then(|p| precision_string(config.pointer(p).and_then(Value::as_str)));
    fields.insert(
        "/checkpoint/declared_dtype".into(),
        observation(
            match dtype_pointer.and_then(|p| config.pointer(p)) {
                None => PrefillFieldStatus::Missing,
                Some(Value::Null) => PrefillFieldStatus::Unspecified,
                Some(Value::String(value)) if value.trim().eq_ignore_ascii_case("auto") => {
                    PrefillFieldStatus::Auto
                }
                Some(Value::String(_)) if declared_dtype.is_some() => PrefillFieldStatus::Declared,
                Some(Value::String(_)) => PrefillFieldStatus::Unspecified,
                Some(_) => PrefillFieldStatus::Error,
            },
            format!("{}#{}", path.display(), dtype_pointer.unwrap_or("")),
            None,
        ),
    );
    let quant_configs: Vec<_> = [
        "/quantization_config",
        "/quantization",
        "/text_config/quantization_config",
        "/text_config/quantization",
        "/compression_config",
        "/text_config/compression_config",
    ]
    .into_iter()
    .filter_map(|pointer| {
        config
            .pointer(pointer)
            .filter(|value| value.as_object().is_some_and(|obj| !obj.is_empty()))
            .map(|value| (pointer, value))
    })
    .collect();
    let quantization = quant_configs.first().map(|(pointer, value)| {
        let mut quant = parse_quantization(value, &format!("{}#{pointer}", path.display()), &mut fields);
        if quant_configs.iter().any(|(_, other)| other != value) {
            let alternatives: Vec<_> = quant_configs.iter().map(|(pointer, value)| {
                parse_quantization(value, &format!("{}#{pointer}", path.display()), &mut BTreeMap::new())
            }).collect();
            let mut conflicting = Vec::new();
            if alternatives.iter().any(|q| q.method != quant.method) { quant.method = None; conflicting.push("method"); }
            if alternatives.iter().any(|q| q.bits != quant.bits) { quant.bits = None; conflicting.push("bits"); }
            if alternatives.iter().any(|q| q.group_size != quant.group_size) { quant.group_size = None; conflicting.push("group_size"); }
            if alternatives.iter().any(|q| q.mode != quant.mode) { quant.mode = None; conflicting.push("mode"); }
            quant.raw_config = Some(Value::Object(quant_configs.iter().map(|(pointer, value)| {
                ((*pointer).to_owned(), (*value).clone())
            }).collect()));
            for field in conflicting {
                fields.insert(format!("/checkpoint/quantization/{field}"), observation(
                    PrefillFieldStatus::Unknown, path.display().to_string(),
                    Some("Quantization declarations differ across configuration scopes; inspect raw_config".into()),
                ));
            }
        }
        quant
    });
    if quantization.is_none() {
        fields.insert(
            "/checkpoint/quantization".into(),
            observation(
                PrefillFieldStatus::Missing,
                path.display().to_string(),
                None,
            ),
        );
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    let (repo_id, revision) = cache_identity(dir);
    let artifact = PrefillArtifactSpec {
        local_path: Some(dir.to_string_lossy().into_owned()),
        repo_id: repo_id.or_else(|| {
            config
                .get("_name_or_path")
                .and_then(Value::as_str)
                .filter(|name| name.contains('/') && is_hub_repo_id(name))
                .map(str::to_owned)
        }),
        revision: revision.or_else(|| {
            config
                .get("_commit_hash")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
        }),
        config_hash: Some(format!("blake3:{}", blake3::hash(raw).to_hex())),
    };
    fields.insert(
        "/artifact/config_hash".into(),
        observation(
            PrefillFieldStatus::Declared,
            path.display().to_string(),
            Some("BLAKE3 of the saved config.json bytes".into()),
        ),
    );
    ModelFileMetadata {
        model: ModelPrefillSpec {
            artifact: Some(artifact),
            architecture: Some(architecture),
            checkpoint: Some(PrefillCheckpointSpec {
                declared_dtype,
                quantization,
                ..Default::default()
            }),
            execution: None,
        },
        fields,
    }
}

fn parse_quantization(
    value: &Value,
    source: &str,
    fields: &mut BTreeMap<String, PrefillFieldMetadata>,
) -> PrefillQuantizationSpec {
    let mut reader = ConfigReader {
        value,
        source: source.to_owned(),
        fallback: None,
        fields,
    };
    let method: Option<String> = reader.read(
        "/checkpoint/quantization/method",
        &["quant_method", "quantization_method"],
    );
    let normalized_method = precision_string(method.as_deref());
    if method.is_some() && normalized_method.is_none() {
        if let Some(metadata) = reader.fields.get_mut("/checkpoint/quantization/method") {
            metadata.status = if method
                .as_ref()
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("auto"))
            {
                PrefillFieldStatus::Auto
            } else {
                PrefillFieldStatus::Unspecified
            };
        }
    }
    reader.fields.insert(
        "/checkpoint/quantization/raw_config".into(),
        observation(PrefillFieldStatus::Declared, source, None),
    );
    PrefillQuantizationSpec {
        method: normalized_method,
        bits: reader.read(
            "/checkpoint/quantization/bits",
            &["bits", "num_bits", "w_bit"],
        ),
        group_size: reader.read(
            "/checkpoint/quantization/group_size",
            &["group_size", "q_group_size"],
        ),
        mode: reader.read("/checkpoint/quantization/mode", &["mode"]),
        raw_config: Some(value.clone()),
    }
}

fn cache_identity(dir: &Path) -> (Option<String>, Option<String>) {
    for ancestor in dir.ancestors() {
        if ancestor
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "snapshots")
        {
            let revision = ancestor
                .file_name()
                .map(|s| s.to_string_lossy().into_owned());
            let repo_id = ancestor
                .parent()
                .and_then(Path::parent)
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("models--"))
                .map(|s| s.replace("--", "/"));
            return (repo_id, revision);
        }
    }
    (None, None)
}

async fn resolve_file(
    repo: &str,
    filename: &str,
    revision: Option<&str>,
    deadline: Instant,
) -> Result<PathBuf, String> {
    let fetch = async {
        match revision {
            Some(revision) => {
                llm_tokenizer::hub::fetch_file_at_revision(repo, filename, revision).await
            }
            None => llm_tokenizer::hub::fetch_file(repo, filename).await,
        }
    };
    tokio::time::timeout_at(deadline, fetch)
        .await
        .map_err(|_| format!("Model metadata lookup timed out: {repo}/{filename}"))?
        .map_err(|error| format!("Could not fetch {repo}/{filename}: {error}"))
}

async fn optional_json(
    dir: &Path,
    repo: Option<&str>,
    revision: Option<&str>,
    filename: &str,
    deadline: Instant,
) -> Result<Option<Value>, String> {
    let local = dir.join(filename);
    let path = match tokio::fs::metadata(&local).await {
        Ok(_) => local,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => match repo {
            Some(repo) => resolve_file(repo, filename, revision, deadline).await?,
            None => return Ok(None),
        },
        Err(error) => return Err(format!("Could not inspect {}: {error}", local.display())),
    };
    let raw = tokio::fs::read(&path)
        .await
        .map_err(|e| format!("Reading {} failed: {e}", path.display()))?;
    serde_json::from_slice(&raw)
        .map(Some)
        .map_err(|e| format!("Parsing {} failed: {e}", path.display()))
}

/// Accept integral JSON floats used by some exporters, without accepting
/// negative, fractional, overflowing or imprecise byte counts.
fn storage_bytes(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        let n = value.as_f64()?;
        (n.is_finite() && n >= 0.0 && n <= 9_007_199_254_740_991.0 && n.fract() == 0.0)
            .then_some(n as u64)
    })
}

/// Unsharded checkpoints need not have an index. Inspect only local tensor
/// headers in that case; file length includes header/padding and is not a
/// substitute for tensor-storage bytes.
async fn local_safetensors_bytes(dir: &Path) -> Result<Option<u64>, String> {
    let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| e.to_string())?;
    let mut total = None;
    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "safetensors")
            || entry.file_name().to_string_lossy().starts_with("adapter_")
        {
            continue;
        }
        let mut file = tokio::fs::File::open(&path)
            .await
            .map_err(|e| e.to_string())?;
        let length = file.metadata().await.map_err(|e| e.to_string())?.len();
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .await
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > 32 * 1024 * 1024 || header_len > length.saturating_sub(8) {
            return Err(format!(
                "Invalid safetensors header length: {}",
                path.display()
            ));
        }
        let mut bytes = vec![0; header_len as usize];
        file.read_exact(&mut bytes)
            .await
            .map_err(|e| e.to_string())?;
        let header: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let header = header
            .as_object()
            .ok_or("Safetensors header is not an object")?;
        let mut ranges = Vec::new();
        for (key, tensor) in header {
            if key == "__metadata__" {
                continue;
            }
            let offsets = tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or("Missing tensor data_offsets")?;
            let [start, end] = offsets.as_slice() else {
                return Err("Invalid tensor data_offsets".into());
            };
            let start = start.as_u64().ok_or("Invalid tensor offset")?;
            let end = end.as_u64().ok_or("Invalid tensor offset")?;
            if end < start || end > length - 8 - header_len {
                return Err("Tensor data_offsets exceed file length".into());
            }
            ranges.push((start, end));
        }
        ranges.sort_unstable();
        let mut previous_end = 0;
        for (start, end) in ranges {
            if start < previous_end {
                return Err("Overlapping safetensors tensors".into());
            }
            previous_end = end;
            total = Some(
                total
                    .unwrap_or(0_u64)
                    .checked_add(end - start)
                    .ok_or("Tensor-storage size overflow")?,
            );
        }
    }
    Ok(total)
}

pub(super) async fn load_model_file_metadata(
    source: &str,
    revision: Option<&str>,
) -> Result<ModelFileMetadata, String> {
    let deadline = Instant::now() + FILE_LOOKUP_TIMEOUT;
    let local = Path::new(source);
    let repo = if local.is_dir() {
        None
    } else if !local.exists() && is_hub_repo_id(source) {
        Some(source)
    } else {
        return Err(format!("Model directory is not accessible: {source}"));
    };
    let path = match repo {
        Some(repo) => resolve_file(repo, "config.json", revision, deadline).await?,
        None => local.join("config.json"),
    };
    let raw = tokio::fs::read(&path)
        .await
        .map_err(|e| format!("Reading {} failed: {e}", path.display()))?;
    let config: Value = serde_json::from_slice(&raw)
        .map_err(|e| format!("Parsing {} failed: {e}", path.display()))?;
    if !config.is_object() {
        return Err(format!("{} must contain a JSON object", path.display()));
    }
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let dir = absolute
        .parent()
        .ok_or("config.json has no parent directory")?;
    let mut result = parse_config(&config, &absolute, &raw);
    let artifact = result
        .model
        .artifact
        .as_mut()
        .ok_or("Missing parsed artifact")?;
    if let Some(repo) = repo {
        artifact.repo_id = Some(repo.to_owned());
    }
    let resolved_revision = artifact
        .revision
        .clone()
        .or_else(|| revision.map(str::to_owned));
    artifact.revision.clone_from(&resolved_revision);
    for (key, value) in [
        ("local_path", artifact.local_path.as_ref()),
        ("repo_id", artifact.repo_id.as_ref()),
        ("revision", artifact.revision.as_ref()),
    ] {
        result.fields.insert(
            format!("/artifact/{key}"),
            observation(
                if value.is_some() {
                    PrefillFieldStatus::Declared
                } else {
                    PrefillFieldStatus::Missing
                },
                source,
                None,
            ),
        );
    }
    let checkpoint = result
        .model
        .checkpoint
        .as_mut()
        .ok_or("Missing parsed checkpoint")?;

    // Inspect only small JSON sidecars; never download .safetensors or .bin weights.
    match optional_json(
        dir,
        repo,
        resolved_revision.as_deref(),
        "model.safetensors.index.json",
        deadline,
    )
    .await
    {
        Ok(Some(index)) => {
            checkpoint.weight_storage_bytes = index
                .pointer("/metadata/total_size")
                .and_then(storage_bytes);
            let status = if checkpoint.weight_storage_bytes.is_some() {
                PrefillFieldStatus::Declared
            } else if index.pointer("/metadata/total_size").is_some() {
                PrefillFieldStatus::Error
            } else {
                PrefillFieldStatus::Missing
            };
            result.fields.insert(
                "/checkpoint/weight_storage_bytes".into(),
                observation(
                    status,
                    format!(
                        "{}/model.safetensors.index.json#/metadata/total_size",
                        dir.display()
                    ),
                    None,
                ),
            );
        }
        Ok(None) => {
            result.fields.insert(
                "/checkpoint/weight_storage_bytes".into(),
                observation(PrefillFieldStatus::Missing, dir.display().to_string(), None),
            );
        }
        Err(error) => {
            result.fields.insert(
                "/checkpoint/weight_storage_bytes".into(),
                observation(
                    PrefillFieldStatus::Error,
                    dir.display().to_string(),
                    Some(error),
                ),
            );
        }
    }
    if checkpoint.weight_storage_bytes.is_none() {
        match local_safetensors_bytes(dir).await {
            Ok(Some(bytes)) => {
                checkpoint.weight_storage_bytes = Some(bytes);
                result.fields.insert(
                    "/checkpoint/weight_storage_bytes".into(),
                    observation(
                        PrefillFieldStatus::Declared,
                        dir.display().to_string(),
                        Some("Summed tensor data_offsets from local safetensors headers".into()),
                    ),
                );
            }
            Ok(None) => {}
            Err(error) => {
                result.fields.insert(
                    "/checkpoint/weight_storage_bytes".into(),
                    observation(
                        PrefillFieldStatus::Error,
                        dir.display().to_string(),
                        Some(error),
                    ),
                );
            }
        }
    }
    if checkpoint.quantization.is_none() {
        for filename in QUANTIZATION_FILES {
            match optional_json(dir, repo, resolved_revision.as_deref(), filename, deadline).await {
                Ok(Some(config)) if config.as_object().is_some_and(|obj| !obj.is_empty()) => {
                    let q = [
                        "/quantization_config",
                        "/quantization",
                        "/compression_config",
                    ]
                    .into_iter()
                    .find_map(|p| config.pointer(p).filter(|v| v.is_object()))
                    .unwrap_or(&config);
                    checkpoint.quantization = Some(parse_quantization(
                        q,
                        &format!("{}/{filename}#", dir.display()),
                        &mut result.fields,
                    ));
                    result.fields.remove("/checkpoint/quantization");
                    break;
                }
                Ok(_) => {}
                Err(error) => {
                    result.fields.insert(
                        "/checkpoint/quantization".into(),
                        observation(
                            PrefillFieldStatus::Error,
                            format!("{}/{filename}", dir.display()),
                            Some(error),
                        ),
                    );
                    if Instant::now() >= deadline {
                        break;
                    }
                }
            }
        }
    }
    Ok(result)
}

/// Merge per field, retaining explicit registration values and their provenance.
pub(super) fn merge_model_file_metadata(
    prefill: &mut WorkerPrefillSpec,
    id: &str,
    discovered: ModelFileMetadata,
) {
    fn fill<T>(target: &mut Option<T>, value: Option<T>) {
        if target.is_none() {
            *target = value;
        }
    }
    let current = prefill.models.entry(id.to_owned()).or_default();
    let existing = serde_json::to_value(&*current).unwrap_or_default();
    if let Some(value) = discovered.model.artifact {
        let target = current.artifact.get_or_insert_with(Default::default);
        fill(&mut target.local_path, value.local_path);
        fill(&mut target.repo_id, value.repo_id);
        fill(&mut target.revision, value.revision);
        fill(&mut target.config_hash, value.config_hash);
    }
    if let Some(value) = discovered.model.architecture {
        let target = current.architecture.get_or_insert_with(Default::default);
        fill(&mut target.num_hidden_layers, value.num_hidden_layers);
        fill(&mut target.hidden_size, value.hidden_size);
        fill(&mut target.intermediate_size, value.intermediate_size);
        fill(&mut target.num_attention_heads, value.num_attention_heads);
        fill(&mut target.num_key_value_heads, value.num_key_value_heads);
        fill(&mut target.head_dim, value.head_dim);
        fill(&mut target.vocab_size, value.vocab_size);
        fill(&mut target.sliding_window, value.sliding_window);
        fill(
            &mut target.full_attention_interval,
            value.full_attention_interval,
        );
        if target.layer_types.is_empty() {
            target.layer_types = value.layer_types;
        }
        if let Some(value) = value.moe_config {
            let target = target.moe_config.get_or_insert_with(Default::default);
            fill(&mut target.num_experts, value.num_experts);
            fill(&mut target.num_experts_per_tok, value.num_experts_per_tok);
            fill(&mut target.intermediate_size, value.intermediate_size);
            fill(&mut target.num_shared_experts, value.num_shared_experts);
            fill(
                &mut target.shared_expert_intermediate_size,
                value.shared_expert_intermediate_size,
            );
            fill(&mut target.raw_config, value.raw_config);
        }
        if let Some(value) = value.linear_attention_config {
            let target = target
                .linear_attention_config
                .get_or_insert_with(Default::default);
            fill(
                &mut target.linear_conv_kernel_dim,
                value.linear_conv_kernel_dim,
            );
            fill(&mut target.linear_key_head_dim, value.linear_key_head_dim);
            fill(&mut target.linear_num_key_heads, value.linear_num_key_heads);
            fill(
                &mut target.linear_num_value_heads,
                value.linear_num_value_heads,
            );
            fill(
                &mut target.linear_value_head_dim,
                value.linear_value_head_dim,
            );
            fill(&mut target.mamba_ssm_dtype, value.mamba_ssm_dtype);
        }
    }
    if let Some(value) = discovered.model.checkpoint {
        let target = current.checkpoint.get_or_insert_with(Default::default);
        fill(&mut target.declared_dtype, value.declared_dtype);
        fill(&mut target.weight_storage_bytes, value.weight_storage_bytes);
        if let Some(value) = value.quantization {
            let target = target.quantization.get_or_insert_with(Default::default);
            fill(&mut target.method, value.method);
            fill(&mut target.bits, value.bits);
            fill(&mut target.group_size, value.group_size);
            fill(&mut target.mode, value.mode);
            fill(&mut target.raw_config, value.raw_config);
        }
    }
    let prefix = format!("/models/{}", id.replace('~', "~0").replace('/', "~1"));
    for (path, metadata) in discovered.fields {
        if existing.pointer(&path).is_none_or(Value::is_null) {
            prefill
                .field_metadata
                .entry(format!("{prefix}{path}"))
                .or_insert(metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn loads_hybrid_model_geometry_identity_and_integral_float_size() {
        let dir = tempfile::tempdir().unwrap();
        let layers: Vec<_> = (0..64)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let config = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "dtype": "bfloat16", "num_hidden_layers": 64, "hidden_size": 5120,
                "intermediate_size": 17408, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "full_attention_interval": 4, "layer_types": layers,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32"
            },
            "vision_config": {"hidden_size": 1152}
        });
        let raw = serde_json::to_vec(&config).unwrap();
        std::fs::write(dir.path().join("config.json"), &raw).unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":55562855904.0}}"#,
        )
        .unwrap();
        let result = load_model_file_metadata(dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let artifact = result.model.artifact.as_ref().unwrap();
        assert_eq!(artifact.local_path.as_deref(), dir.path().to_str());
        assert_eq!(
            artifact.config_hash,
            Some(format!("blake3:{}", blake3::hash(&raw)))
        );
        assert_eq!(artifact.repo_id, None);
        assert_eq!(artifact.revision, None);
        let geometry = result.model.architecture.as_ref().unwrap();
        assert_eq!(geometry.num_hidden_layers, Some(64));
        assert_eq!(geometry.hidden_size, Some(5120));
        assert_eq!(geometry.head_dim, Some(256));
        assert_eq!(geometry.num_attention_heads, Some(24));
        assert_eq!(geometry.layer_types.len(), 64);
        assert_eq!(
            geometry
                .layer_types
                .iter()
                .filter(|k| *k == "full_attention")
                .count(),
            16
        );
        assert_eq!(geometry.moe_config, None);
        assert_eq!(geometry.sliding_window, None);
        assert_eq!(
            geometry
                .linear_attention_config
                .as_ref()
                .unwrap()
                .linear_num_value_heads,
            Some(48)
        );
        let checkpoint = result.model.checkpoint.as_ref().unwrap();
        assert_eq!(checkpoint.declared_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(checkpoint.quantization, None);
        assert_eq!(checkpoint.weight_storage_bytes, Some(55_562_855_904));
        assert_eq!(result.model.execution, None);
        assert_eq!(
            result.fields["/architecture/sliding_window"].status,
            PrefillFieldStatus::Missing
        );
        assert!(result.fields["/architecture/hidden_size"]
            .source
            .as_ref()
            .unwrap()
            .ends_with("#/text_config/hidden_size"));
        assert!(result.fields["/architecture/hidden_size"]
            .collected_at
            .is_some());
    }

    #[tokio::test]
    async fn loads_moe_aliases_and_a_separate_quantization_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&json!({
                "torch_dtype": "float16", "n_layer": 32, "hidden_size": 4096,
                "n_routed_experts": 64, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
                "n_shared_experts": 1, "shared_expert_intermediate_size": 4096
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("quant_config.json"),
            r#"{"quant_method":"awq","w_bit":4,"q_group_size":128}"#,
        )
        .unwrap();
        let result = load_model_file_metadata(dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let architecture = result.model.architecture.unwrap();
        assert_eq!(architecture.num_hidden_layers, Some(32));
        assert_eq!(architecture.head_dim, None);
        let moe = architecture.moe_config.unwrap();
        assert_eq!(moe.num_experts, Some(64));
        assert_eq!(moe.num_experts_per_tok, Some(8));
        assert_eq!(moe.intermediate_size, Some(2048));
        assert_eq!(moe.num_shared_experts, Some(1));
        let quant = result.model.checkpoint.unwrap().quantization.unwrap();
        assert_eq!(quant.method.as_deref(), Some("awq"));
        assert_eq!(quant.bits, Some(4));
        assert_eq!(quant.group_size, Some(128));
        assert!(result.fields["/checkpoint/quantization/bits"]
            .source
            .as_ref()
            .unwrap()
            .contains("quant_config.json"));
    }

    #[test]
    fn retains_mixed_quantization_without_fabricating_a_uniform_bit_width() {
        let config = json!({"quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"one": {"weights": {"num_bits": 4}}, "two": {"weights": {"num_bits": 8}}}
        }});
        let result = parse_config(&config, Path::new("/models/mixed/config.json"), b"config");
        let quant = result.model.checkpoint.unwrap().quantization.unwrap();
        assert_eq!(quant.method.as_deref(), Some("compressed-tensors"));
        assert_eq!(quant.bits, None);
        assert_eq!(quant.raw_config.unwrap(), config["quantization_config"]);
    }

    #[test]
    fn text_config_overrides_root_geometry_and_inherits_missing_fields() {
        let config = json!({
            "hidden_size": 1152, "vocab_size": 248320, "num_attention_heads": 24,
            "text_config": {"hidden_size": 5120, "head_dim": 256}
        });
        let result = parse_config(&config, Path::new("/models/hybrid/config.json"), b"config");
        let architecture = result.model.architecture.unwrap();
        assert_eq!(architecture.hidden_size, Some(5120));
        assert_eq!(architecture.vocab_size, Some(248320));
        assert_eq!(architecture.num_attention_heads, Some(24));
        assert_eq!(architecture.head_dim, Some(256));
        assert!(result.fields["/architecture/vocab_size"]
            .source
            .as_ref()
            .unwrap()
            .ends_with("#/vocab_size"));
        assert!(result.fields["/architecture/hidden_size"]
            .source
            .as_ref()
            .unwrap()
            .ends_with("#/text_config/hidden_size"));
    }

    #[test]
    fn preserves_differing_quantization_scopes_without_claiming_a_uniform_format() {
        let config = json!({
            "quantization_config": {"quant_method":"awq", "bits":8},
            "text_config": {"quantization_config":{"quant_method":"awq", "bits":4}}
        });
        let result = parse_config(&config, Path::new("/models/mixed/config.json"), b"config");
        let quant = result.model.checkpoint.unwrap().quantization.unwrap();
        assert_eq!(quant.method.as_deref(), Some("awq"));
        assert_eq!(quant.bits, None);
        assert_eq!(
            quant.raw_config.unwrap()["/text_config/quantization_config"]["bits"],
            4
        );
        assert_eq!(
            result.fields["/checkpoint/quantization/bits"].status,
            PrefillFieldStatus::Unknown
        );
    }

    #[test]
    fn file_precision_preserves_automatic_null_and_invalid_states() {
        for (config, status) in [
            (json!({"dtype":"auto"}), PrefillFieldStatus::Auto),
            (json!({"dtype":null}), PrefillFieldStatus::Unspecified),
            (json!({"dtype":32}), PrefillFieldStatus::Error),
            (json!({}), PrefillFieldStatus::Missing),
        ] {
            let result = parse_config(&config, Path::new("/models/model/config.json"), b"config");
            assert_eq!(result.model.checkpoint.unwrap().declared_dtype, None);
            assert_eq!(result.fields["/checkpoint/declared_dtype"].status, status);
        }
    }

    #[tokio::test]
    async fn malformed_optional_files_do_not_discard_geometry_or_precision() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"dtype":"bfloat16","hidden_size":5120,"num_hidden_layers":"bad"}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"), b"not JSON").unwrap();
        let result = load_model_file_metadata(dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        assert_eq!(
            result.model.architecture.as_ref().unwrap().hidden_size,
            Some(5120)
        );
        assert_eq!(
            result
                .model
                .architecture
                .as_ref()
                .unwrap()
                .num_hidden_layers,
            None
        );
        assert_eq!(
            result
                .model
                .checkpoint
                .as_ref()
                .unwrap()
                .declared_dtype
                .as_deref(),
            Some("bfloat16")
        );
        assert_eq!(
            result.fields["/architecture/num_hidden_layers"].status,
            PrefillFieldStatus::Error
        );
        assert_eq!(
            result.fields["/checkpoint/weight_storage_bytes"].status,
            PrefillFieldStatus::Error
        );
    }

    #[tokio::test]
    async fn unsharded_storage_uses_tensor_offsets_without_reading_tensor_data() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let header = serde_json::to_vec(&json!({
            "weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        }))
        .unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&[0; 4]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();
        let result = load_model_file_metadata(dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        assert_eq!(
            result.model.checkpoint.unwrap().weight_storage_bytes,
            Some(4)
        );
    }

    #[test]
    fn merges_partial_overrides_without_replacing_runtime_or_hardware() {
        let config = json!({"hidden_size":5120, "num_hidden_layers":64, "dtype":"bfloat16"});
        let discovered = parse_config(&config, Path::new("/models/model/config.json"), b"config");
        let mut profile: WorkerPrefillSpec = serde_json::from_value(json!({
            "models": {"org/model~v1": {
                "architecture": {"hidden_size": 8192},
                "execution": {"model_dtype": "float16", "quantization_method": "fp8"},
                "artifact": {"repo_id": "explicit/model"}
            }},
            "hardware": {"devices": [{"uuid": "GPU-explicit"}]},
            "engine": {"prefix_cache_enabled": false}
        }))
        .unwrap();
        let hardware = profile.hardware.clone();
        let engine = profile.engine.clone();
        merge_model_file_metadata(&mut profile, "org/model~v1", discovered);
        let model = &profile.models["org/model~v1"];
        assert_eq!(model.architecture.as_ref().unwrap().hidden_size, Some(8192));
        assert_eq!(
            model.architecture.as_ref().unwrap().num_hidden_layers,
            Some(64)
        );
        assert_eq!(
            model.execution.as_ref().unwrap().model_dtype.as_deref(),
            Some("float16")
        );
        assert_eq!(
            model
                .execution
                .as_ref()
                .unwrap()
                .quantization_method
                .as_deref(),
            Some("fp8")
        );
        assert_eq!(
            model.artifact.as_ref().unwrap().repo_id.as_deref(),
            Some("explicit/model")
        );
        assert_eq!(profile.hardware, hardware);
        assert_eq!(profile.engine, engine);
        assert!(profile
            .field_metadata
            .contains_key("/models/org~1model~0v1/architecture/num_hidden_layers"));
        assert!(!profile
            .field_metadata
            .contains_key("/models/org~1model~0v1/architecture/hidden_size"));
    }

    #[test]
    fn resolves_cache_identity_without_guessing_for_plain_directories() {
        assert_eq!(
            cache_identity(Path::new("/cache/models--org--model/snapshots/abcdef")),
            (Some("org/model".into()), Some("abcdef".into()))
        );
        assert_eq!(cache_identity(Path::new("/models/org/model")), (None, None));
        assert!(!is_hub_repo_id("/models/not-mounted"));
        assert!(!is_hub_repo_id("C:\\models\\not-mounted"));
        assert!(is_hub_repo_id("org/model"));
    }

    #[test]
    fn validates_storage_counts_and_does_not_infer_unknown_geometry() {
        assert_eq!(
            storage_bytes(&json!(55_562_855_904.0)),
            Some(55_562_855_904)
        );
        for invalid in [json!(-1), json!(4.5), json!(18_446_744_073_709_551_616.0)] {
            assert_eq!(storage_bytes(&invalid), None);
        }
        let result = parse_config(&json!({}), Path::new("/models/empty/config.json"), b"{}");
        assert_eq!(result.model.architecture.as_ref().unwrap().head_dim, None);
        assert_eq!(
            result
                .model
                .architecture
                .as_ref()
                .unwrap()
                .num_key_value_heads,
            None
        );
        assert_eq!(result.model.checkpoint.as_ref().unwrap().quantization, None);
    }
}
