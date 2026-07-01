use serde::{Deserialize, Serialize};
use validator::Validate;
use super::validated::Normalizable;

use super::common::{default_true, GenerationRequest, StringOrArray};

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rerank_request"))]
pub struct RerankRequest {
    /// The query text to rank documents against
    #[validate(custom(function = "validate_query"))]
    pub query: String,

    /// List of documents to be ranked
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// Model to use for reranking
    pub model: String,

    /// Maximum number of documents to return (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_true")]
    pub return_documents: bool,

    // SGLang specific extensions
    /// Request ID for tracking
    pub rid: Option<StringOrArray>,

    /// User identifier
    pub user: Option<String>,
}

impl GenerationRequest for RerankRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl super::validated::Normalizable for RerankRequest {
    // Use default no-op normalization
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates that the query is not empty
fn validate_query(query: &str) -> Result<(), validator::ValidationError> {
    if query.trim().is_empty() {
        return Err(validator::ValidationError::new("query cannot be empty"));
    }
    Ok(())
}

/// Validates that the documents list is not empty
fn validate_documents(documents: &[String]) -> Result<(), validator::ValidationError> {
    if documents.is_empty() {
        return Err(validator::ValidationError::new(
            "documents list cannot be empty",
        ));
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
#[expect(
    clippy::unnecessary_wraps,
    reason = "validator crate requires Result return type"
)]
fn validate_rerank_request(req: &RerankRequest) -> Result<(), validator::ValidationError> {
    // Validate top_k if specified
    if let Some(k) = req.top_k {
        if k > req.documents.len() {
            tracing::warn!(
                "top_k ({}) is greater than number of documents ({})",
                k,
                req.documents.len()
            );
        }
    }
    Ok(())
}

impl RerankRequest {
    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

/// Individual rerank result (Jina AI v1 compatible)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResult {
    /// Original index of the document in the request
    pub index: usize,

    /// Relevance score for the document
    #[serde(alias = "score")]
    pub relevance_score: f32,

    /// The document (if return_documents was true)
    #[serde(default, deserialize_with = "deserialize_rerank_document")]
    pub document: Option<RerankDocument>,
}

/// Document wrapper for rerank results (Jina AI v1 format: `{"text": "..."}`)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct RerankDocument {
    pub text: String,
}

/// Usage information specific to rerank responses.
/// Backends typically only report `total_tokens`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankUsageInfo {
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
}

/// Rerank response (Jina AI v1 compatible)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResponse {
    /// Model used for reranking
    pub model: String,

    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Usage information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<RerankUsageInfo>,

    /// Response ID (optional, for request tracking)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<StringOrArray>,
}

impl RerankResponse {
    /// Create a new RerankResponse with the given results and model
    pub fn new(
        results: Vec<RerankResult>,
        model: String,
        request_id: Option<StringOrArray>,
    ) -> Self {
        RerankResponse {
            results,
            model,
            usage: None,
            id: request_id,
        }
    }

    /// Apply top_k limit to results
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
}

/// Custom deserializer for document field that accepts both:
/// - Plain string: `"text content"` (legacy format)
/// - Object: `{"text": "text content"}` (Jina v1 standard)
/// - null / missing → None
fn deserialize_rerank_document<'de, D>(
    deserializer: D,
) -> Result<Option<RerankDocument>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde_json::Value;

    let value: Option<Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(RerankDocument { text: s })),
        Some(Value::Object(map)) => {
            if let Some(Value::String(text)) = map.get("text") {
                Ok(Some(RerankDocument {
                    text: text.clone(),
                }))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// V1 API compatibility format for rerank requests
/// Matches Python's V1RerankReqInput
#[derive(Debug, Clone, Serialize, Deserialize, Validate, schemars::JsonSchema)]
pub struct V1RerankReqInput {
    #[validate(length(min = 1, message = "query cannot be empty"))]  // ✅ 不用 required
    pub query: String,

    #[validate(length(min = 1, message = "documents cannot be empty"))]  // ✅ 验证 Vec 长度
    pub documents: Vec<String>,

    #[validate(length(min = 1, message = "model cannot be empty"))]  // ✅ 不用 required
    pub model: String,

    #[validate(range(min = 1, message = "top_k must be at least 1"))]  // ✅ 验证数值范围
    pub top_k: Option<usize>,
}

impl Normalizable for V1RerankReqInput {
    fn normalize(&mut self) {
        // 如果有需要标准化的字段，在这里处理
        // 例如：trim 字符串、去除多余空格等
        self.query = self.query.trim().to_string();
        self.model = self.model.trim().to_string();
        // documents 可以逐个 trim
        for doc in &mut self.documents {
            *doc = doc.trim().to_string();
        }
    }
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: v1.model,
            top_k: v1.top_k,
            return_documents: true,
            rid: None,
            user: None,
        }
    }
}
