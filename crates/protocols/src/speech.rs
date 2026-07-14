use serde::{Deserialize, Serialize};
use crate::common::GenerationRequest;

/// Transcription request - compatible with OpenAI's /v1/audio/sppech API.
///
/// The audio file itself is carried out-of-band because the endpoint uses
/// multipart/form-data, not JSON.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct SpeechRequest {
    /// input: string
    // The text to generate audio for. The maximum length is 4096 characters.
    // maxLength4096
    pub input: String,

    /// model: string or SpeechModel
    // One of the available TTS models: tts-1, tts-1-hd, gpt-4o-mini-tts, or gpt-4o-mini-tts-2025-12-15.
    pub model: String,

    /// voice: string or "alloy" or "ash" or "ballad" or 7 more or object
    // The voice to use when generating the audio. Supported built-in voices are alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse, marin, and cedar. You may also provide a custom voice object with an id, for example { "id": "voice_1234" }. Previews of the voices are available in the Text to speech guide.
    pub voice: Option<String>,

    // instructions: optional string
    // Control the voice of your generated audio with additional instructions. Does not work with tts-1 or tts-1-hd.
    // maxLength 4096
    pub instructions: Option<String>,

    /// response_format: optional "mp3" or "opus" or "aac" or 3 more
    // The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm.
    pub response_format: Option<String>,

    /// speed: optional number
    // The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.
    // minimum 0.25, maximum 4
    pub speed: Option<f64>,

    /// stream_format: optional "sse" or "audio"
    // The format to stream the audio in. Supported formats are sse and audio. sse is not supported for tts-1 or tts-1-hd.
    pub stream_format: Option<String>,

    pub prompt_text: Option<String>,
}

impl GenerationRequest for SpeechRequest {
    fn is_stream(&self) -> bool {
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Audio bytes are not visible here; use the optional prompt as a
        // rough cache-aware routing hint when present.
        self.input.clone()
    }
}
