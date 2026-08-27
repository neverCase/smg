use serde_json::Value;

use crate::{
    observability::metrics::Metrics,
    routers::{
        common::sse::{SseDecoder, SseFrame},
        http::chat_metrics::RoutedChatMetricsContext,
    },
};

pub(crate) struct ChatStreamTtftObserver {
    context: RoutedChatMetricsContext,
    decoder: SseDecoder,
    recorded: bool,
    finished: bool,
    decode_failed: bool,
}

impl ChatStreamTtftObserver {
    pub(crate) fn new(context: RoutedChatMetricsContext) -> Self {
        Self {
            context,
            decoder: SseDecoder::new(),
            recorded: false,
            finished: false,
            decode_failed: false,
        }
    }

    fn record_ttft(&mut self) {
        if self.recorded || self.finished {
            return;
        }

        Metrics::record_http_chat_ttft(
            &self.context.model,
            &self.context.worker,
            &self.context.worker_uid,
            self.context.started_at.elapsed(),
        );

        self.recorded = true;
    }
    pub(crate) fn observe_chunk(&mut self, chunk: &[u8]) {
        if self.recorded || self.decode_failed || self.finished {
            return;
        }

        if self.decoder.push(chunk).is_err() {
            self.decode_failed = true;
            return;
        }

        while let Some(frame) = self.decoder.next_frame() {
            match frame {
                Ok(frame) if frame_has_output_token(&frame) => {
                    self.record_ttft();
                    break;
                }

                Ok(_) => {}

                Err(_) => {
                    self.decode_failed = true;
                    break;
                }
            }
        }

        self.decoder.compact();
    }

    pub(crate) fn finish_eof(&mut self) {
        if self.finished {
            return;
        }

        // observe_chunk() 已经循环调用 next_frame() 直到 None，
        // 所以这里可以安全 flush 最后一个没有空行结束的 frame。
        if !self.recorded && !self.decode_failed {
            match self.decoder.flush() {
                Some(Ok(frame)) if frame_has_output_token(&frame) => {
                    self.record_ttft();
                }

                Some(Ok(_)) | None => {}

                Some(Err(_)) => {
                    self.decode_failed = true;
                }
            }
        }

        let missing_reason = if self.decode_failed {
            "sse_decode_error"
        } else {
            "eof_without_token"
        };

        self.finish(missing_reason);
    }

    pub(crate) fn finish_stream_error(&mut self) {
        self.finish("upstream_stream_error");
    }

    pub(crate) fn finish_disconnected(&mut self) {
        self.finish("client_disconnected");
    }

    fn finish(&mut self, missing_reason: &'static str) {
        if self.finished {
            return;
        }

        self.finished = true;

        if !self.recorded {
            Metrics::record_http_chat_ttft_missing(
                &self.context.model,
                &self.context.worker,
                &self.context.worker_uid,
                missing_reason,
            );
        }
    }
}

fn frame_has_output_token(frame: &SseFrame<'_>) -> bool {
    if frame.is_done() {
        return false;
    }

    let Ok(value) = frame.decode_data::<Value>() else {
        return false;
    };

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
        assert!(!observer.recorded);

        observer.finish_eof();

        assert!(observer.recorded);
        assert!(observer.finished);
        assert!(!observer.decode_failed);
    }
}
