use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use serde_json::Value;

/// Request-scoped handoff for the authoritative SSE time-to-first-token value.
///
/// The router-side SSE observer and the server-side audit body stream run in
/// different tasks, so the value needs shared ownership and a one-shot write.
#[derive(Clone, Debug, Default)]
pub(crate) struct ChatStreamTtftRecorder {
    value: Arc<OnceLock<Duration>>,
}

impl ChatStreamTtftRecorder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record(&self, ttft: Duration) {
        let _ = self.value.set(ttft);
    }

    pub(crate) fn get(&self) -> Option<Duration> {
        self.value.get().copied()
    }

    pub(crate) fn get_ms(&self) -> Option<u64> {
        self.get()
            .map(|ttft| u64::try_from(ttft.as_millis()).unwrap_or(u64::MAX))
    }
}

pub(crate) struct ChatMetricsContext {
    pub started_at: Instant,
    pub model: String,
    pub ttft_recorder: Option<ChatStreamTtftRecorder>,
}

#[derive(Clone)]
pub(crate) struct RoutedChatMetricsContext {
    pub started_at: Instant,
    pub model: String,
    pub worker: String,
    pub worker_uid: String,
    pub ttft_recorder: Option<ChatStreamTtftRecorder>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ChatTokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

impl ChatTokenUsage {
    pub(crate) fn from_value(value: &Value) -> Option<Self> {
        let usage = value.get("usage")?;

        let input_tokens = usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64);
        let output_tokens = usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64);

        (input_tokens.is_some() || output_tokens.is_some()).then_some(Self {
            input_tokens,
            output_tokens,
        })
    }

    pub(crate) fn from_json_slice(bytes: &[u8]) -> Option<Self> {
        let value = serde_json::from_slice(bytes).ok()?;
        Self::from_value(&value)
    }

    /// Continuous usage chunks are cumulative. Keep the newest value for each
    /// field while retaining a field omitted by a later backend-specific chunk.
    pub(crate) fn merged_with(self, newer: Self) -> Self {
        Self {
            input_tokens: newer.input_tokens.or(self.input_tokens),
            output_tokens: newer.output_tokens.or(self.output_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_chat_completion_usage() {
        let usage = ChatTokenUsage::from_value(&json!({
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18
            }
        }));

        assert_eq!(
            usage,
            Some(ChatTokenUsage {
                input_tokens: Some(11),
                output_tokens: Some(7),
            })
        );
    }

    #[test]
    fn parses_input_output_token_fallbacks() {
        let usage =
            ChatTokenUsage::from_json_slice(br#"{"usage":{"input_tokens":13,"output_tokens":5}}"#);

        assert_eq!(
            usage,
            Some(ChatTokenUsage {
                input_tokens: Some(13),
                output_tokens: Some(5),
            })
        );
    }

    #[test]
    fn merges_partial_continuous_usage() {
        let previous = ChatTokenUsage {
            input_tokens: Some(20),
            output_tokens: Some(2),
        };
        let newer = ChatTokenUsage {
            input_tokens: None,
            output_tokens: Some(3),
        };

        assert_eq!(
            previous.merged_with(newer),
            ChatTokenUsage {
                input_tokens: Some(20),
                output_tokens: Some(3),
            }
        );
    }
}
