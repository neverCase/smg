use std::time::Instant;

use serde_json::Value;
use tracing::warn;

use crate::{
    observability::metrics::Metrics,
    routers::{
        common::sse::{SseDecoder, SseFrame},
        http::chat_metrics::{ChatTokenUsage, RoutedChatMetricsContext},
    },
};

pub(crate) struct ChatStreamTtftObserver {
    context: RoutedChatMetricsContext,
    decoder: SseDecoder,

    ttft_recorded: bool,
    tpot_recorded: bool,
    tpm_recorded: bool,
    finished: bool,
    decode_failed: bool,

    /// 收到首个有效输出 token 的时间。
    first_token_at: Option<Instant>,

    /// 最近一次观察到的 completion token 数及其到达时间。
    ///
    /// 使用“最近一次”是为了兼容 continuous_usage_stats：
    /// 中间 chunk 可能也带 usage，最终一次才是完整 token 数。
    completion_usage: Option<(u32, Instant)>,

    /// 最近一次累计 usage，用于在流结束时只记录一次 TPM counter。
    token_usage: Option<ChatTokenUsage>,
}

impl ChatStreamTtftObserver {
    pub(crate) fn new(context: RoutedChatMetricsContext) -> Self {
        warn!(
            model = %context.model,
            worker = %context.worker,
            worker_uid = %context.worker_uid,
            "TPOT_DIAG observer_created"
        );

        Self {
            context,
            decoder: SseDecoder::new(),
            ttft_recorded: false,
            tpot_recorded: false,
            tpm_recorded: false,
            finished: false,
            decode_failed: false,
            first_token_at: None,
            completion_usage: None,
            token_usage: None,
        }
    }

    fn record_ttft(&mut self, first_token_at: Instant) {
        if self.ttft_recorded || self.finished {
            return;
        }

        let ttft = first_token_at.saturating_duration_since(self.context.started_at);

        Metrics::record_http_chat_ttft(
            &self.context.model,
            &self.context.worker,
            &self.context.worker_uid,
            ttft,
        );

        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            ttft_seconds = ttft.as_secs_f64(),
            "TPOT_DIAG ttft_recorded"
        );

        self.first_token_at = Some(first_token_at);
        self.ttft_recorded = true;
    }

    pub(crate) fn observe_chunk(&mut self, chunk: &[u8]) {
        // 不能再因为 TTFT 已记录就返回；还需要继续寻找最终 usage。
        if self.decode_failed || self.finished {
            return;
        }

        if let Err(error) = self.decoder.push(chunk) {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                chunk_len = chunk.len(),
                error = %error,
                "TPOT_DIAG decoder_push_failed"
            );

            self.decode_failed = true;
            return;
        }

        while let Some(frame) = self.decoder.next_frame() {
            match frame {
                Ok(frame) => {
                    self.observe_frame(&frame);

                    // [DONE] 会在 observe_frame() 中完成指标记录。
                    if self.finished {
                        break;
                    }
                }

                Err(error) => {
                    warn!(
                        model = %self.context.model,
                        worker = %self.context.worker,
                        worker_uid = %self.context.worker_uid,
                        error = %error,
                        "TPOT_DIAG next_frame_failed"
                    );

                    self.decode_failed = true;
                    break;
                }
            }
        }

        self.decoder.compact();
    }

    fn observe_frame(&mut self, frame: &SseFrame<'_>) {
        if frame.is_done() {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                ttft_recorded = self.ttft_recorded,
                completion_tokens = ?self
                    .completion_usage
                    .as_ref()
                    .map(|(tokens, _)| *tokens),
                "TPOT_DIAG done_observed"
            );

            let missing_reason = if self.ttft_recorded {
                "usage_missing"
            } else {
                "done_without_token"
            };

            self.finish(missing_reason);
            return;
        }

        let value = match frame.decode_data::<Value>() {
            Ok(value) => value,
            Err(error) => {
                // Do not log frame.data: it can contain user-generated content.
                warn!(
                    model = %self.context.model,
                    worker = %self.context.worker,
                    worker_uid = %self.context.worker_uid,
                    event_type = ?frame.event_type.as_deref(),
                    data_len = frame.data.len(),
                    error = %error,
                    "TPOT_DIAG json_decode_failed"
                );
                return;
            }
        };

        let observed_at = Instant::now();

        if !self.ttft_recorded && value_has_output_token(&value) {
            self.record_ttft(observed_at);
        }

        if let Some(usage) = ChatTokenUsage::from_value(&value) {
            let choices_len = value.get("choices").and_then(Value::as_array).map(Vec::len);
            let merged_usage = self
                .token_usage
                .map_or(usage, |previous| previous.merged_with(usage));

            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                input_tokens = ?merged_usage.input_tokens,
                output_tokens = ?merged_usage.output_tokens,
                choices_len = ?choices_len,
                "TPOT_DIAG usage_observed"
            );

            // 覆盖旧值，保留最后一次 usage。这样兼容
            // continuous_usage_stats=true 的累计 usage。
            self.token_usage = Some(merged_usage);

            if let Some(completion_tokens) = usage
                .output_tokens
                .and_then(|tokens| u32::try_from(tokens).ok())
            {
                self.completion_usage = Some((completion_tokens, observed_at));
            }
        }
    }

    pub(crate) fn finish_eof(&mut self) {
        if self.finished {
            return;
        }

        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            ttft_recorded = self.ttft_recorded,
            completion_tokens = ?self
                .completion_usage
                .as_ref()
                .map(|(tokens, _)| *tokens),
            decode_failed = self.decode_failed,
            "TPOT_DIAG finish_eof"
        );

        // observe_chunk() 已经循环调用 next_frame() 直到 None。
        // flush 最后一个没有空行结尾的 SSE frame。
        if !self.decode_failed {
            let flushed = self.decoder.flush();

            match flushed {
                Some(Ok(frame)) => {
                    self.observe_frame(&frame);
                }

                Some(Err(error)) => {
                    warn!(
                        model = %self.context.model,
                        worker = %self.context.worker,
                        worker_uid = %self.context.worker_uid,
                        error = %error,
                        "TPOT_DIAG flush_failed"
                    );
                    self.decode_failed = true;
                }

                None => {}
            }
        }

        // flush 中如果遇到 [DONE]，已经完成记录。
        if self.finished {
            return;
        }

        let missing_reason = if self.decode_failed {
            "sse_decode_error"
        } else if !self.ttft_recorded {
            "eof_without_token"
        } else {
            "usage_missing"
        };

        self.finish(missing_reason);
    }

    pub(crate) fn finish_stream_error(&mut self) {
        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            "TPOT_DIAG finish_stream_error"
        );
        self.finish("upstream_stream_error");
    }

    pub(crate) fn finish_disconnected(&mut self) {
        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            "TPOT_DIAG finish_disconnected"
        );
        self.finish("client_disconnected");
    }

    fn finish(&mut self, missing_reason: &'static str) {
        if self.finished {
            return;
        }

        self.finished = true;

        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            ttft_recorded = self.ttft_recorded,
            completion_tokens = ?self
                .completion_usage
                .as_ref()
                .map(|(tokens, _)| *tokens),
            fallback_reason = missing_reason,
            "TPOT_DIAG finish_entered"
        );

        if let Some(usage) = self.token_usage {
            Metrics::record_http_chat_tokens(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                true,
                usage.input_tokens,
                usage.output_tokens,
            );

            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                input_tokens = ?usage.input_tokens,
                output_tokens = ?usage.output_tokens,
                "TPM_DIAG tokens_recorded"
            );

            self.tpm_recorded = true;
        }

        if !self.ttft_recorded {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                reason = missing_reason,
                "TPOT_DIAG tpot_missing_no_ttft"
            );

            Metrics::record_http_chat_ttft_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                missing_reason,
            );

            Metrics::record_http_chat_tpot_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                missing_reason,
            );

            return;
        }

        let Some(first_token_at) = self.first_token_at else {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                "TPOT_DIAG tpot_missing_first_token_timestamp"
            );

            Metrics::record_http_chat_tpot_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                "first_token_timestamp_missing",
            );
            return;
        };

        let Some((completion_tokens, usage_observed_at)) = self.completion_usage else {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                reason = missing_reason,
                "TPOT_DIAG tpot_missing_usage"
            );

            Metrics::record_http_chat_tpot_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                missing_reason,
            );
            return;
        };

        // 一个 token 没有相邻 token interval，所以 TPOT 无定义。
        if completion_tokens <= 1 {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                completion_tokens,
                "TPOT_DIAG tpot_missing_insufficient_tokens"
            );

            Metrics::record_http_chat_tpot_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                "insufficient_output_tokens",
            );
            return;
        }

        let generation_after_first = usage_observed_at.saturating_duration_since(first_token_at);

        let token_intervals = completion_tokens - 1;
        let tpot = generation_after_first / token_intervals;

        Metrics::record_http_chat_tpot(
            &self.context.model,
            &self.context.worker,
            &self.context.worker_uid,
            tpot,
        );

        warn!(
            model = %self.context.model,
            worker = %self.context.worker,
            worker_uid = %self.context.worker_uid,
            completion_tokens,
            token_intervals,
            generation_after_first_seconds = generation_after_first.as_secs_f64(),
            tpot_seconds = tpot.as_secs_f64(),
            "TPOT_DIAG tpot_recorded"
        );

        self.tpot_recorded = true;
    }
}

impl Drop for ChatStreamTtftObserver {
    fn drop(&mut self) {
        if !self.finished {
            warn!(
                model = %self.context.model,
                worker = %self.context.worker,
                worker_uid = %self.context.worker_uid,
                ttft_recorded = self.ttft_recorded,
                tpot_recorded = self.tpot_recorded,
                tpm_recorded = self.tpm_recorded,
                decode_failed = self.decode_failed,
                completion_tokens = ?self
                    .completion_usage
                    .as_ref()
                    .map(|(tokens, _)| *tokens),
                buffered_bytes = self.decoder.buffered_len(),
                "TPOT_DIAG observer_dropped_without_finish"
            );
        }
    }
}

fn value_has_output_token(value: &Value) -> bool {
    value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| choices.iter().any(choice_has_output))
}

fn choice_has_output(choice: &Value) -> bool {
    let Some(delta) = choice.get("delta") else {
        return false;
    };

    has_non_empty_string(delta.get("content"))
        || has_non_empty_string(delta.get("reasoning_content"))
        || has_non_empty_string(delta.get("refusal"))
        || has_non_empty_array(delta.get("tool_calls"))
        || has_non_empty_object(delta.get("function_call"))
}

fn has_non_empty_string(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
}

fn has_non_empty_array(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}

fn has_non_empty_object(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_object)
        .is_some_and(|object| !object.is_empty())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn observer() -> ChatStreamTtftObserver {
        ChatStreamTtftObserver::new(RoutedChatMetricsContext {
            started_at: Instant::now(),
            model: "test-model".to_string(),
            worker: "worker-0".to_string(),
            worker_uid: "uid-0".to_string(),
        })
    }

    #[test]
    fn eof_flushes_unterminated_token_frame() {
        let mut observer = observer();

        observer.observe_chunk(br#"data: {"choices":[{"delta":{"content":"a"}}]}"#);

        // 没有结尾空行，普通 next_frame() 暂时读不到。
        assert!(!observer.ttft_recorded);

        observer.finish_eof();

        assert!(observer.ttft_recorded);
        assert!(observer.finished);
        assert!(!observer.decode_failed);

        // 没有 usage，因此不能记录 TPOT。
        assert!(!observer.tpot_recorded);
        assert!(!observer.tpm_recorded);
    }

    #[test]
    fn records_tpot_from_final_usage() {
        let mut observer = observer();

        observer.observe_chunk(
            br#"data: {"choices":[{"index":0,"delta":{"content":"a"}}]}

data: {"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}

data: [DONE]

"#,
        );

        assert!(observer.ttft_recorded);
        assert!(observer.tpot_recorded);
        assert!(observer.tpm_recorded);
        assert!(observer.finished);
        assert_eq!(
            observer
                .completion_usage
                .as_ref()
                .map(|(tokens, _)| *tokens),
            Some(2)
        );
        assert_eq!(
            observer.token_usage,
            Some(ChatTokenUsage {
                input_tokens: Some(10),
                output_tokens: Some(2),
            })
        );
    }

    #[test]
    fn keeps_latest_continuous_usage_value() {
        let mut observer = observer();

        observer.observe_chunk(
            br#"data: {"choices":[{"delta":{"content":"a"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}

data: {"choices":[{"delta":{"content":"b"}}],"usage":{"completion_tokens":2}}

data: {"choices":[],"usage":{"completion_tokens":3}}

data: [DONE]

"#,
        );

        assert!(observer.tpot_recorded);
        assert!(observer.tpm_recorded);
        assert_eq!(
            observer
                .completion_usage
                .as_ref()
                .map(|(tokens, _)| *tokens),
            Some(3)
        );
        assert_eq!(
            observer.token_usage,
            Some(ChatTokenUsage {
                input_tokens: Some(10),
                output_tokens: Some(3),
            })
        );
    }

    #[test]
    fn does_not_record_tpot_without_usage() {
        let mut observer = observer();

        observer.observe_chunk(
            br#"data: {"choices":[{"delta":{"content":"a"}}]}

data: [DONE]

"#,
        );

        assert!(observer.ttft_recorded);
        assert!(!observer.tpot_recorded);
        assert!(!observer.tpm_recorded);
        assert!(observer.finished);
    }

    #[test]
    fn one_output_token_has_no_tpot() {
        let mut observer = observer();

        observer.observe_chunk(
            br#"data: {"choices":[{"delta":{"content":"a"}}]}

data: {"choices":[],"usage":{"completion_tokens":1}}

data: [DONE]

"#,
        );

        assert!(observer.ttft_recorded);
        assert!(!observer.tpot_recorded);
        assert!(observer.tpm_recorded);
        assert!(observer.finished);
    }

    #[test]
    fn records_tpm_without_an_observable_output_chunk() {
        let mut observer = observer();

        observer.observe_chunk(
            br#"data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":0}}

data: [DONE]

"#,
        );

        assert!(!observer.ttft_recorded);
        assert!(!observer.tpot_recorded);
        assert!(observer.tpm_recorded);
        assert!(observer.finished);
        assert_eq!(
            observer.token_usage,
            Some(ChatTokenUsage {
                input_tokens: Some(8),
                output_tokens: Some(0),
            })
        );
    }
}
