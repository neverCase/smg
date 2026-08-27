use std::time::Instant;

pub(crate) struct ChatMetricsContext {
    pub started_at: Instant,
    pub model: String,
}

#[derive(Clone)]
pub(crate) struct RoutedChatMetricsContext {
    pub started_at: Instant,
    pub model: String,
    pub worker: String,
    pub worker_uid: String,
}
