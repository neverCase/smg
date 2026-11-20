use std::sync::Arc;
use tracing::warn;
use llm_multimodal::MediaPartOrder;
use llm_tokenizer::TokenizerRegistry;
use openai_protocol::chat::ChatCompletionRequest;
use crate::policies::{KvcacheConductorPolicy, LoadBalancingPolicy};
use crate::routers::grpc::{multimodal, utils};

pub(crate) async fn prepare_conductor_tokens(
    policy: &dyn LoadBalancingPolicy,
    tokenizer_registry: &TokenizerRegistry,
    model_id: &str,
    request: Option<&ChatCompletionRequest>
) -> Option<Arc<[u32]>> {
    if !policy.as_any().is::<KvcacheConductorPolicy>() {
        return None;
    }

    let request = request?;

    match tokenizer_conductor_chat(tokenizer_registry, model_id, request).await {
        Ok(tokens) => Some(tokens.into()),
        Err(error) => {
            warn!(
                model = model_id,
                %error,
                "Conductor tokenization unavailable"
            );
            None
        }
    }
}

pub(crate) async fn tokenizer_conductor_chat(
    tokenizer_registry: &TokenizerRegistry,
    model_id: &str,
    request: &ChatCompletionRequest,
) -> Result<Vec<u32>, String> {

    if !multimodal::media_plan_chat(&request.messages).is_empty() {
        return Err(
            "Conductor tokenization currently supports text-only chat"
                .to_string(),
        );
    }

    let tokenizer = tokenizer_registry.get(model_id).ok_or_else(|| {
        format!("Tokenizer not found for model: {model_id}")
    })?;

    utils::validate_chat_content_parts(&request.messages)?;

    let render_request = utils::request_as_rendered(request);

    let (processed, prompt_encoding) =
        utils::process_chat_messages_with_placeholders(
            &render_request.as_ref(),
            tokenizer.as_ref(),
            None,
            MediaPartOrder::MediaFirst
        )?;

    let encoding = utils::encode_prompt_blocking(
        tokenizer,
        &processed.text,
        prompt_encoding,
    )
        .await
        .map_err(|e| format!("Tokenizer encoding failed: {e}"))?;

    Ok(encoding.token_ids().to_vec())
}