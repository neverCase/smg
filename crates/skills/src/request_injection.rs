//! Tier-1 skill listing construction and outbound-request injection.
//!
//! Request-time skill *resolution* (see [`crate::resolution`]) produces a
//! [`ResolvedSkillManifest`], but that manifest has no effect until the gateway
//! discloses it to the model. This module owns the read-side "Tier-1" disclosure
//! step for the Responses surface: it formats a compact listing of the attached
//! skills (name + short description) under a configurable token budget and
//! prepends it to the request's system `instructions` before the first model
//! turn (issue #1193).
//!
//! The async lookups needed to enrich storage-backed refs into displayable
//! `(name, description)` pairs live in the gateway (it owns the
//! [`crate::SkillService`]); this module is pure formatting + injection so the
//! budget math and wire shaping stay unit-testable without IO.

use openai_protocol::responses::ResponsesRequest;

use crate::{
    api::SkillService,
    config::SkillsBudgetLimit,
    resolution::{ResolvedSkillManifest, ResolvedSkillRef},
};

/// Heading the Tier-1 listing is injected under. Mirrors the
/// `skills_instructions` vocabulary the reserved read tools reference in their
/// descriptions (see [`crate::tool_schemas`]).
const TIER1_LISTING_HEADING: &str = "# skills_instructions";

/// Preamble appended after the heading to orient the model.
const TIER1_LISTING_PREAMBLE: &str =
    "The following skills are attached to this request. Each entry lists a skill \
     id, name, and description. When the user's task matches a skill, use the \
     skill's instructions; if read tools are available, call them to load the \
     full SKILL.md before acting.";

/// A single skill rendered into the Tier-1 listing.
///
/// The gateway builds these from a [`ResolvedSkillManifest`], resolving
/// storage-backed refs to their pinned name/description via the
/// [`crate::SkillService`]. Pass-through provider refs that SMG cannot describe
/// are represented with whatever identifying text is available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillListingEntry {
    /// Stable identifier the model uses to name the skill in read-tool calls.
    pub skill_id: String,
    /// Human-readable skill name.
    pub name: String,
    /// Short description; empty when none is known.
    pub description: String,
}

impl SkillListingEntry {
    /// Render this entry into a single compact listing line.
    fn render(&self) -> String {
        let description = self.description.trim();
        if description.is_empty() {
            format!("- {} ({})", self.name.trim(), self.skill_id.trim())
        } else {
            format!(
                "- {} ({}): {}",
                self.name.trim(),
                self.skill_id.trim(),
                description
            )
        }
    }
}

/// Identify the storage-backed manifest entries whose name/description the
/// gateway still has to resolve via the [`crate::SkillService`].
///
/// Storage-backed refs only carry an id + pinned version at resolution time, so
/// the gateway must enrich them; this helper exposes the ids that need a lookup
/// so callers do not have to re-match the enum.
#[must_use]
pub fn manifest_storage_skill_ids(manifest: &ResolvedSkillManifest) -> Vec<String> {
    manifest
        .refs()
        .iter()
        .filter_map(|skill_ref| match skill_ref {
            ResolvedSkillRef::SmgStorage { skill_id, .. } => Some(skill_id.clone()),
            _ => None,
        })
        .collect()
}

/// Build the compact Tier-1 listing text from already-resolved listing entries,
/// honoring `budget`.
///
/// Returns `None` when there is nothing to disclose or the budget is too small
/// to fit even a single entry. The budget is a *token* budget; we approximate
/// tokens as `ceil(chars / 4)`, the conventional rough ratio, so this stays
/// dependency-free and deterministic. Entries are appended in order until the
/// next one would exceed the budget, at which point the listing is truncated and
/// a trailing note records how many skills were omitted.
#[must_use]
pub fn build_tier1_listing(
    entries: &[SkillListingEntry],
    budget: SkillsBudgetLimit,
) -> Option<String> {
    if entries.is_empty() {
        return None;
    }

    let max_tokens = match budget {
        SkillsBudgetLimit::Unlimited => None,
        SkillsBudgetLimit::Tokens(0) => return None,
        SkillsBudgetLimit::Tokens(tokens) => Some(tokens as usize),
    };

    let header = format!("{TIER1_LISTING_HEADING}\n{TIER1_LISTING_PREAMBLE}");
    let mut body = String::new();
    let mut included = 0usize;

    for entry in entries {
        let candidate_line = entry.render();
        let projected = format!("{header}\n{body}\n{candidate_line}");
        if let Some(max_tokens) = max_tokens {
            if approximate_token_count(&projected) > max_tokens {
                break;
            }
        }
        body.push('\n');
        body.push_str(&candidate_line);
        included += 1;
    }

    if included == 0 {
        // Budget could not fit even the first entry alongside the header.
        return None;
    }

    let omitted = entries.len() - included;
    let mut listing = format!("{header}{body}");
    if omitted > 0 {
        listing.push_str(&format!(
            "\n- (+{omitted} more skill(s) omitted to fit the instruction budget)"
        ));
    }
    Some(listing)
}

/// Resolve a [`ResolvedSkillManifest`] into displayable Tier-1 listing entries.
///
/// Storage-backed (`SmgStorage`) refs are enriched with the pinned version's
/// name/description via `service`; a lookup failure degrades gracefully to the
/// bare skill id rather than failing the whole request. Client-local refs use
/// their declared name/description. Opaque provider pass-through objects are
/// skipped — SMG does not own their disclosure and cannot describe them.
pub async fn collect_manifest_listing_entries(
    service: Option<&SkillService>,
    tenant_id: &str,
    manifest: &ResolvedSkillManifest,
) -> Vec<SkillListingEntry> {
    let mut entries = Vec::with_capacity(manifest.refs().len());
    for skill_ref in manifest.refs() {
        match skill_ref {
            ResolvedSkillRef::SmgStorage {
                skill_id, pinned, ..
            } => {
                let (name, description) = match service {
                    Some(service) => service
                        .get_skill_version(tenant_id, skill_id, &pinned.version)
                        .await
                        .map(|record| {
                            let description = record
                                .short_description
                                .filter(|value| !value.trim().is_empty())
                                .unwrap_or(record.description);
                            (record.name, description)
                        })
                        .unwrap_or_else(|_| (skill_id.clone(), String::new())),
                    None => (skill_id.clone(), String::new()),
                };
                entries.push(SkillListingEntry {
                    skill_id: skill_id.clone(),
                    name,
                    description,
                });
            }
            ResolvedSkillRef::ClientLocalPath {
                name, description, ..
            } => entries.push(SkillListingEntry {
                skill_id: name.clone(),
                name: name.clone(),
                description: description.clone(),
            }),
            // Provider-owned refs are disclosed by the provider itself; SMG has
            // no local name/description to surface, so they are not listed.
            ResolvedSkillRef::AnthropicProvider { .. }
            | ResolvedSkillRef::OpenAIProvider { .. }
            | ResolvedSkillRef::OpenAIOpaquePassThrough { .. } => {}
        }
    }
    entries
}

/// Prepend an already-built Tier-1 listing to a Responses request's system
/// `instructions`.
///
/// Existing instructions are preserved and pushed below the listing so the
/// skill disclosure is the first thing the model reads. A no-op when `listing`
/// is empty.
pub fn inject_responses_tier1_listing(request: &mut ResponsesRequest, listing: &str) {
    if listing.is_empty() {
        return;
    }

    request.instructions = Some(match request.instructions.take() {
        Some(existing) if !existing.trim().is_empty() => format!("{listing}\n\n{existing}"),
        _ => listing.to_string(),
    });
}

/// Approximate the token count of `text` using the conventional ~4 chars/token
/// heuristic. Rounds up so a non-empty string always costs at least one token.
fn approximate_token_count(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponsesRequest};

    use super::*;
    use crate::{PinnedSkillVersion, ResolvedSkillManifest, ResolvedSkillRef};

    fn entry(skill_id: &str, name: &str, description: &str) -> SkillListingEntry {
        SkillListingEntry {
            skill_id: skill_id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
        }
    }

    #[test]
    fn build_listing_includes_each_skill_name_and_description() {
        let entries = vec![
            entry("skill_a", "acme:map", "Map the repo"),
            entry("skill_b", "acme:search", "Search the repo"),
        ];

        let listing = build_tier1_listing(&entries, SkillsBudgetLimit::Unlimited)
            .expect("listing should be produced");

        assert!(listing.contains(TIER1_LISTING_HEADING));
        assert!(listing.contains("acme:map"));
        assert!(listing.contains("skill_a"));
        assert!(listing.contains("Map the repo"));
        assert!(listing.contains("acme:search"));
        assert!(listing.contains("Search the repo"));
    }

    #[test]
    fn build_listing_returns_none_for_empty_entries() {
        assert!(build_tier1_listing(&[], SkillsBudgetLimit::Unlimited).is_none());
    }

    #[test]
    fn build_listing_returns_none_for_zero_budget() {
        let entries = vec![entry("skill_a", "acme:map", "Map the repo")];
        assert!(build_tier1_listing(&entries, SkillsBudgetLimit::Tokens(0)).is_none());
    }

    #[test]
    fn build_listing_truncates_to_budget_and_records_omitted() {
        let entries = vec![
            entry(
                "skill_a",
                "acme:map",
                "Map the repo with ripgrep and friends",
            ),
            entry("skill_b", "acme:search", "Search the repo thoroughly"),
            entry("skill_c", "acme:lint", "Lint everything in the repo"),
        ];

        // A tight budget that fits the header + first entry (~84 tokens) but
        // not the second (~97 tokens), exercising mid-list truncation.
        let listing = build_tier1_listing(&entries, SkillsBudgetLimit::Tokens(90))
            .expect("at least one entry should fit");

        assert!(listing.contains("acme:map"));
        assert!(
            listing.contains("more skill(s) omitted"),
            "expected truncation note, got: {listing}"
        );
        // Skills beyond the budget must not appear once truncated.
        assert!(!listing.contains("acme:search"));
        assert!(!listing.contains("acme:lint"));
    }

    #[test]
    fn build_listing_returns_none_when_budget_cannot_fit_first_entry() {
        let entries = vec![entry(
            "skill_a",
            "acme:map",
            "Map the repo with ripgrep and friends",
        )];
        // Header alone is ~68 tokens; a budget below header+entry0 cannot fit
        // even the first line, so nothing is disclosed.
        assert!(build_tier1_listing(&entries, SkillsBudgetLimit::Tokens(70)).is_none());
    }

    #[test]
    fn build_listing_handles_missing_description() {
        let entries = vec![entry("skill_a", "acme:map", "")];
        let listing = build_tier1_listing(&entries, SkillsBudgetLimit::Unlimited)
            .expect("listing should be produced");
        assert!(listing.contains("acme:map (skill_a)"));
    }

    #[test]
    fn inject_prepends_listing_above_existing_instructions() {
        let mut request = ResponsesRequest {
            model: "gpt-5.1".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            instructions: Some("Existing system prompt.".to_string()),
            ..Default::default()
        };

        inject_responses_tier1_listing(&mut request, "LISTING-TEXT");

        let instructions = request.instructions.expect("instructions set");
        assert!(instructions.starts_with("LISTING-TEXT"));
        assert!(instructions.contains("Existing system prompt."));
        // Listing must precede the prior prompt.
        let listing_idx = instructions.find("LISTING-TEXT").unwrap();
        let existing_idx = instructions.find("Existing system prompt.").unwrap();
        assert!(listing_idx < existing_idx);
    }

    #[test]
    fn inject_sets_instructions_when_absent() {
        let mut request = ResponsesRequest {
            model: "gpt-5.1".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            instructions: None,
            ..Default::default()
        };

        inject_responses_tier1_listing(&mut request, "LISTING-TEXT");
        assert_eq!(request.instructions.as_deref(), Some("LISTING-TEXT"));
    }

    #[test]
    fn inject_is_noop_for_empty_listing() {
        let mut request = ResponsesRequest {
            model: "gpt-5.1".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            instructions: Some("Keep me.".to_string()),
            ..Default::default()
        };
        inject_responses_tier1_listing(&mut request, "");
        assert_eq!(request.instructions.as_deref(), Some("Keep me."));
    }

    #[tokio::test]
    async fn collect_entries_enriches_storage_refs_via_service() -> Result<(), anyhow::Error> {
        use std::sync::Arc;

        use smg_blob_storage::FilesystemBlobStore;
        use tempfile::TempDir;

        use crate::{CreateSkillRequest, SkillService, SkillUpload, UploadedSkillFile};

        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);
        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let manifest = ResolvedSkillManifest::new(vec![
            ResolvedSkillRef::SmgStorage {
                skill_id: created.skill.skill_id.clone(),
                requested_version: None,
                pinned: PinnedSkillVersion {
                    version: created.version.version.clone(),
                    version_number: created.version.version_number,
                },
            },
            ResolvedSkillRef::ClientLocalPath {
                name: "repo".to_string(),
                description: "local checkout".to_string(),
                path: "/workspace".to_string(),
            },
            // Provider pass-through must not be listed.
            ResolvedSkillRef::OpenAIProvider {
                skill_id: "openai-spreadsheets".to_string(),
                raw_version: None,
            },
        ]);

        let entries = collect_manifest_listing_entries(Some(&service), "tenant-a", &manifest).await;

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].skill_id, created.skill.skill_id);
        assert_eq!(entries[0].name, "acme:map");
        assert_eq!(entries[0].description, "Map the repo");
        assert_eq!(entries[1].name, "repo");
        assert_eq!(entries[1].description, "local checkout");
        Ok(())
    }

    #[tokio::test]
    async fn collect_entries_degrades_to_skill_id_on_lookup_failure() {
        // No service available: storage refs degrade to their bare id with an
        // empty description rather than panicking or dropping the entry.
        let manifest = ResolvedSkillManifest::new(vec![ResolvedSkillRef::SmgStorage {
            skill_id: "skill_missing".to_string(),
            requested_version: None,
            pinned: PinnedSkillVersion {
                version: "v1".to_string(),
                version_number: 1,
            },
        }]);

        let entries = collect_manifest_listing_entries(None, "tenant-a", &manifest).await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].skill_id, "skill_missing");
        assert_eq!(entries[0].name, "skill_missing");
        assert!(entries[0].description.is_empty());
    }

    #[test]
    fn manifest_storage_ids_only_returns_smg_storage_refs() {
        let manifest = ResolvedSkillManifest::new(vec![
            ResolvedSkillRef::SmgStorage {
                skill_id: "skill_storage".to_string(),
                requested_version: None,
                pinned: PinnedSkillVersion {
                    version: "v1".to_string(),
                    version_number: 1,
                },
            },
            ResolvedSkillRef::ClientLocalPath {
                name: "repo".to_string(),
                description: "local".to_string(),
                path: "/workspace".to_string(),
            },
            ResolvedSkillRef::OpenAIProvider {
                skill_id: "openai-skill".to_string(),
                raw_version: None,
            },
        ]);

        assert_eq!(manifest_storage_skill_ids(&manifest), vec!["skill_storage"]);
    }
}
