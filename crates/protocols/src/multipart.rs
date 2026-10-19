//! Axum extractors for multipart/form-data inference endpoints.
//!
//! JSON endpoints get [`crate::validated::ValidatedJson`]; the
//! `/v1/audio/transcriptions` endpoint uses multipart/form-data and gets
//! [`AudioTranscriptionMultipart`], which parses the form into a typed
//! `(TranscriptionRequest, AudioFile)` pair before the handler runs.

#[cfg(feature = "axum")]
use axum::{
    extract::{multipart::MultipartError, FromRequest, Multipart, Request},
    http::StatusCode,
    response::Response,
};

#[cfg(feature = "axum")]
use crate::{
    images::{ImageEditRequest, ImageFile, ImageVariationRequest, Mask},
    speech::SpeechRequest,
    transcription::{AudioFile, TranscriptionRequest},
};

/// Extractor for `/v1/audio/transcriptions` requests.
///
/// Parses `multipart/form-data` into a [`TranscriptionRequest`] (text fields)
/// plus an [`AudioFile`] (the `file` part). Returns `400 Bad Request` on
/// malformed parts, missing/empty `file`, missing/blank `model`, or
/// out-of-range `temperature`.
#[cfg(feature = "axum")]
pub struct AudioTranscriptionMultipart {
    pub request: TranscriptionRequest,
    pub audio: AudioFile,
}

#[cfg(feature = "axum")]
pub struct ImageEditMultipart {
    pub request: ImageEditRequest,
    pub images: Vec<ImageFile>,
}

#[cfg(feature = "axum")]
pub struct ImageVariationMultipart {
    pub request: ImageVariationRequest,
    pub image: ImageFile,
}

#[cfg(feature = "axum")]
pub struct AudioSpeechMultipart {
    pub request: SpeechRequest,
    pub prompt_speech: Option<AudioFile>,
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for AudioTranscriptionMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(|e| bad_request("invalid_multipart", None, e.body_text()))?;

        let mut file_bytes: Option<bytes::Bytes> = None;
        let mut file_name: Option<String> = None;
        let mut file_content_type: Option<String> = None;
        let mut request = TranscriptionRequest::default();
        let mut timestamp_granularities: Vec<String> = Vec::new();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(
                        "invalid_multipart",
                        None,
                        format!("Failed to read multipart field: {e}"),
                    ));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "file" => {
                    file_name = field.file_name().map(str::to_string);
                    file_content_type = field.content_type().map(str::to_string);
                    match field.bytes().await {
                        Ok(b) => file_bytes = Some(b),
                        Err(e) => {
                            return Err(bad_request(
                                "invalid_multipart",
                                Some("file"),
                                format!("Failed to read audio file bytes: {e}"),
                            ));
                        }
                    }
                }
                "model" => match field.text().await {
                    Ok(t) => request.model = t,
                    Err(e) => return Err(bad_text_field("model", e)),
                },
                "language" => match field.text().await {
                    Ok(t) => request.language = Some(t),
                    Err(e) => return Err(bad_text_field("language", e)),
                },
                "prompt" => match field.text().await {
                    Ok(t) => request.prompt = Some(t),
                    Err(e) => return Err(bad_text_field("prompt", e)),
                },
                "response_format" => match field.text().await {
                    Ok(t) => request.response_format = Some(t),
                    Err(e) => return Err(bad_text_field("response_format", e)),
                },
                "temperature" => match field.text().await {
                    Ok(t) => match t.trim().parse::<f32>() {
                        Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                            request.temperature = Some(v);
                        }
                        Ok(v) => {
                            return Err(bad_request(
                                "invalid_value",
                                Some("temperature"),
                                format!("Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"),
                            ));
                        }
                        Err(e) => {
                            return Err(bad_request(
                                "invalid_value",
                                Some("temperature"),
                                format!("Invalid 'temperature' value: {e}"),
                            ));
                        }
                    },
                    Err(e) => return Err(bad_text_field("temperature", e)),
                },
                "timestamp_granularities" | "timestamp_granularities[]" => {
                    match field.text().await {
                        Ok(t) => timestamp_granularities.push(t),
                        Err(e) => return Err(bad_text_field("timestamp_granularities", e)),
                    }
                }
                "stream" => {
                    match field.text().await {
                        Ok(t) => match t.as_str() {
                            "true" | "True" | "TRUE" | "1" => request.stream = Some(true),
                            "false" | "False" | "FALSE" | "0" => request.stream = Some(false),
                            other => {
                                return Err(bad_request(
                                "invalid_value",
                                Some("stream"),
                                format!("Invalid 'stream' value: '{other}' (expected true/false/1/0)"),
                            ));
                            }
                        },
                        Err(e) => return Err(bad_text_field("stream", e)),
                    }
                }
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        if request.model.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("model"),
                "Missing required 'model' field".to_string(),
            ));
        }
        request.model = request.model.trim().to_string();

        let bytes = match file_bytes {
            Some(b) if !b.is_empty() => b,
            Some(_) => {
                return Err(bad_request(
                    "invalid_value",
                    Some("file"),
                    "Uploaded 'file' part is empty".to_string(),
                ));
            }
            None => {
                return Err(bad_request(
                    "missing_required_parameter",
                    Some("file"),
                    "Missing required 'file' part".to_string(),
                ));
            }
        };

        if !timestamp_granularities.is_empty() {
            request.timestamp_granularities = Some(timestamp_granularities);
        }

        let audio = AudioFile {
            bytes,
            file_name: file_name.unwrap_or_else(|| "audio".to_string()),
            content_type: file_content_type,
        };

        Ok(AudioTranscriptionMultipart { request, audio })
    }
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for ImageEditMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(|e| bad_request("invalid_multipart", None, e.body_text()))?;

        let mut images = Vec::new();
        let mut request = ImageEditRequest::default();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(
                        "invalid_multipart",
                        None,
                        format!("Failed to read multipart field: {e}"),
                    ));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                // 支持 "image" 和 "image[]" 两种字段名
                "image" | "image[]" => {
                    let file_name = field.file_name().map(str::to_string);
                    let content_type = field.content_type().map(str::to_string);
                    let bytes = field.bytes().await.map_err(|e| {
                        bad_request(
                            "invalid_multipart",
                            Some("image"),
                            format!("Failed to read image file bytes: {e}"),
                        )
                    })?;

                    if bytes.is_empty() {
                        return Err(bad_request(
                            "invalid_value",
                            Some("image"),
                            "Uploaded 'image' part is empty".to_string(),
                        ));
                    }

                    images.push(ImageFile {
                        bytes,
                        file_name: file_name.unwrap_or_else(|| "image".to_string()),
                        content_type,
                    });
                }
                "prompt" => {
                    request.prompt = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("prompt", e))?;
                }
                "model" => {
                    request.model = field.text().await.map_err(|e| bad_text_field("model", e))?;
                }
                "background" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("background", e))?;
                    if !v.is_empty() {
                        request.background = Some(v);
                    }
                }
                "input_fidelity" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("input_fidelity", e))?;
                    if !v.is_empty() {
                        request.input_fidelity = Some(v);
                    }
                }
                "moderation" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("moderation", e))?;
                    if !v.is_empty() {
                        request.moderation = Some(v);
                    }
                }
                "n" => {
                    let v = field.text().await.map_err(|e| bad_text_field("n", e))?;
                    if !v.is_empty() {
                        let n = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(
                                "invalid_value",
                                Some("n"),
                                format!("Invalid 'n' value: '{v}' (must be 1-10)"),
                            )
                        })?;
                        if !(1..=10).contains(&n) {
                            return Err(bad_request(
                                "invalid_value",
                                Some("n"),
                                format!("Invalid 'n' value: {n} (must be between 1 and 10)"),
                            ));
                        }
                        request.n = Some(n);
                    }
                }
                "output_compression" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("output_compression", e))?;
                    if !v.is_empty() {
                        let c = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(
                                "invalid_value",
                                Some("output_compression"),
                                format!(
                                    "Invalid 'output_compression' value: '{v}' (must be 0-100)"
                                ),
                            )
                        })?;
                        if c > 100 {
                            return Err(bad_request("invalid_value", Some("output_compression"), format!(
                                "Invalid 'output_compression' value: {c} (must be between 0 and 100)"
                            )));
                        }
                        request.output_compression = Some(c);
                    }
                }
                "output_format" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("output_format", e))?;
                    if !v.is_empty() {
                        request.output_format = Some(v);
                    }
                }
                "partial_images" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("partial_images", e))?;
                    if !v.is_empty() {
                        let p = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(
                                "invalid_value",
                                Some("partial_images"),
                                format!("Invalid 'partial_images' value: '{v}' (must be 0-3)"),
                            )
                        })?;
                        if p > 3 {
                            return Err(bad_request(
                                "invalid_value",
                                Some("partial_images"),
                                format!(
                                    "Invalid 'partial_images' value: {p} (must be between 0 and 3)"
                                ),
                            ));
                        }
                        request.partial_images = Some(p);
                    }
                }
                "quality" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("quality", e))?;
                    if !v.is_empty() {
                        request.quality = Some(v);
                    }
                }
                "size" => {
                    let v = field.text().await.map_err(|e| bad_text_field("size", e))?;
                    if !v.is_empty() {
                        request.size = Some(v);
                    }
                }
                "stream" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("stream", e))?;
                    if !v.is_empty() {
                        request.stream = match v.as_str() {
                            "true" | "True" | "TRUE" | "1" => Some(true),
                            "false" | "False" | "FALSE" | "0" => Some(false),
                            other => {
                                return Err(bad_request(
                                    "invalid_value",
                                    Some("stream"),
                                    format!(
                                    "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                                ),
                                ));
                            }
                        };
                    }
                }
                "user" => {
                    let v = field.text().await.map_err(|e| bad_text_field("user", e))?;
                    if !v.is_empty() {
                        request.user = Some(v);
                    }
                }
                // mask 特殊处理（需要解析 JSON）
                "mask" => {
                    let text = field.text().await.map_err(|e| bad_text_field("mask", e))?;
                    if !text.is_empty() {
                        match serde_json::from_str::<Mask>(&text) {
                            Ok(mask) => request.mask = Some(mask),
                            Err(e) => {
                                return Err(bad_request("invalid_value", Some("mask"), format!(
                                    "Invalid 'mask' JSON: {e}. Expected {{\"file_id\": \"...\"}} or {{\"image_url\": \"...\"}}"
                                )));
                            }
                        }
                    }
                }
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        // ============ 校验 ============
        if request.model.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("model"),
                "Missing required 'model' field".to_string(),
            ));
        }
        request.model = request.model.trim().to_string();

        if request.prompt.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("prompt"),
                "Missing required 'prompt' field".to_string(),
            ));
        }
        request.prompt = request.prompt.trim().to_string();

        if images.is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("image"),
                "Missing required 'image' part(s)".to_string(),
            ));
        }
        if images.len() > 16 {
            return Err(bad_request(
                "invalid_value",
                Some("image"),
                format!("Too many images: {} (maximum 16)", images.len()),
            ));
        }

        Ok(ImageEditMultipart { request, images })
    }
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for ImageVariationMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(|e| bad_request("invalid_multipart", None, e.body_text()))?;

        let mut images = Vec::new();
        let mut request = ImageVariationRequest::default();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(
                        "invalid_multipart",
                        None,
                        format!("Failed to read multipart field: {e}"),
                    ));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "image" => {
                    let file_name = field.file_name().map(str::to_string);
                    let content_type = field.content_type().map(str::to_string);
                    let bytes = field.bytes().await.map_err(|e| {
                        bad_request(
                            "invalid_multipart",
                            Some("image"),
                            format!("Failed to read image file bytes: {e}"),
                        )
                    })?;

                    if bytes.is_empty() {
                        return Err(bad_request(
                            "invalid_value",
                            Some("image"),
                            "Uploaded 'image' part is empty".to_string(),
                        ));
                    }

                    images.push(ImageFile {
                        bytes,
                        file_name: file_name.unwrap_or_else(|| "image".to_string()),
                        content_type,
                    });
                }
                "model" => {
                    request.model = field.text().await.map_err(|e| bad_text_field("model", e))?;
                }
                "n" => {
                    let v = field.text().await.map_err(|e| bad_text_field("n", e))?;
                    if !v.is_empty() {
                        let n = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(
                                "invalid_value",
                                Some("n"),
                                format!("Invalid 'n' value: '{v}' (must be 1-10)"),
                            )
                        })?;
                        if !(1..=10).contains(&n) {
                            return Err(bad_request(
                                "invalid_value",
                                Some("n"),
                                format!("Invalid 'n' value: {n} (must be between 1 and 10)"),
                            ));
                        }
                        request.n = Some(n);
                    }
                }
                "response_format" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("response_format", e))?;
                    if !v.is_empty() {
                        request.response_format = Some(v);
                    }
                }
                "size" => {
                    let v = field.text().await.map_err(|e| bad_text_field("size", e))?;
                    if !v.is_empty() {
                        request.size = Some(v);
                    }
                }
                "stream" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("stream", e))?;
                    if !v.is_empty() {
                        request.stream = match v.as_str() {
                            "true" | "True" | "TRUE" | "1" => Some(true),
                            "false" | "False" | "FALSE" | "0" => Some(false),
                            other => {
                                return Err(bad_request(
                                    "invalid_value",
                                    Some("stream"),
                                    format!(
                                    "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                                ),
                                ));
                            }
                        };
                    }
                }
                "user" => {
                    let v = field.text().await.map_err(|e| bad_text_field("user", e))?;
                    if !v.is_empty() {
                        request.user = Some(v);
                    }
                }
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        // ============ 校验 ============
        if request.model.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("model"),
                "Missing required 'model' field".to_string(),
            ));
        }
        request.model = request.model.trim().to_string();

        if images.is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("image"),
                "Missing required 'image' part(s)".to_string(),
            ));
        }
        let image = images.into_iter().next().unwrap();

        Ok(ImageVariationMultipart { request, image })
    }
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for AudioSpeechMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(|e| bad_request("invalid_multipart", None, e.body_text()))?;

        let mut audios = Vec::new();
        let mut request = SpeechRequest::default();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(
                        "invalid_multipart",
                        None,
                        format!("Failed to read multipart field: {e}"),
                    ));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "prompt_speech" => {
                    let file_name = field.file_name().map(str::to_string);
                    let content_type = field.content_type().map(str::to_string);
                    let bytes = field.bytes().await.map_err(|e| {
                        bad_request(
                            "invalid_multipart",
                            Some("prompt_speech"),
                            format!("Failed to read prompt_speech file bytes: {e}"),
                        )
                    })?;

                    if bytes.is_empty() {
                        return Err(bad_request(
                            "invalid_value",
                            Some("prompt_speech"),
                            "Uploaded 'prompt_speech' part is empty".to_string(),
                        ));
                    }

                    audios.push(AudioFile {
                        bytes,
                        file_name: file_name.unwrap_or_else(|| "prompt_speech".to_string()),
                        content_type,
                    });
                }
                "input" => {
                    request.input = field.text().await.map_err(|e| bad_text_field("input", e))?;
                }
                "model" => {
                    request.model = field.text().await.map_err(|e| bad_text_field("model", e))?;
                }
                "voice" => {
                    let v = field.text().await.map_err(|e| bad_text_field("voice", e))?;
                    if !v.is_empty() {
                        request.voice = Some(v);
                    }
                }
                "instructions" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("instructions", e))?;
                    if !v.is_empty() {
                        request.instructions = Some(v);
                    }
                }
                "response_format" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("response_format", e))?;
                    if !v.is_empty() {
                        request.response_format = Some(v);
                    }
                }
                "speed" => match field.text().await {
                    Ok(t) => match t.trim().parse::<f64>() {
                        Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                            request.speed = Some(v);
                        }
                        Ok(v) => {
                            return Err(bad_request("invalid_value", Some("speed"), format!(
                                "Invalid 'speed' value: {v} (must be a finite number in [0.25 to 4.0. 1.0])"
                            )));
                        }
                        Err(e) => {
                            return Err(bad_request(
                                "invalid_value",
                                Some("speed"),
                                format!("Invalid 'speed' value: {e}"),
                            ));
                        }
                    },
                    Err(e) => return Err(bad_text_field("speed", e)),
                },
                "stream_format" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("stream_format", e))?;
                    if !v.is_empty() {
                        match v.as_str() {
                            "sse" | "audio" => {
                                request.stream_format = Some(v);
                            }
                            other => {
                                return Err(bad_request("invalid_value", Some("stream_format"), format!(
                                    "Invalid 'stream_format' value: '{other}' (expected 'sse' or 'audio')"
                                )));
                            }
                        }
                    }
                }
                "prompt_text" => {
                    let v = field
                        .text()
                        .await
                        .map_err(|e| bad_text_field("prompt_text", e))?;
                    if !v.is_empty() {
                        request.prompt_text = Some(v);
                    }
                }
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        // ============ 校验 ============
        if request.model.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("model"),
                "Missing required 'model' field".to_string(),
            ));
        }
        request.model = request.model.trim().to_string();

        let prompt_speech = audios.into_iter().next();

        // 校验：如果没有 prompt_speech，则 voice 必填
        if prompt_speech.is_none() && request.voice.is_none() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("voice"),
                "Either 'prompt_speech' audio or 'voice' field must be provided".to_string(),
            ));
        }

        Ok(AudioSpeechMultipart {
            request,
            prompt_speech,
        })
    }
}

/// The refusal of a form the parser cannot take: the gateway's JSON error
/// object, written as a JSON body's rejection is, with `param` the field the
/// refusal is about when there is one.
#[cfg(feature = "axum")]
fn bad_request(code: &str, param: Option<&str>, message: String) -> Response {
    crate::validated::rejection(
        StatusCode::BAD_REQUEST,
        code,
        param.map(str::to_owned),
        message,
    )
}

#[cfg(feature = "axum")]
fn bad_text_field(field: &str, e: MultipartError) -> Response {
    bad_request(
        "invalid_multipart",
        Some(field),
        format!("Failed to read '{field}' field: {e}"),
    )
}

#[cfg(test)]
#[cfg(feature = "axum")]
mod tests {
    use axum::{
        body::Body,
        extract::{FromRequest, Request},
    };

    use super::*;

    async fn endpoint_refusal<T: FromRequest<(), Rejection = Response>>(
        form: &str,
    ) -> serde_json::Value {
        let request = Request::builder()
            .method("POST")
            .header("content-type", "multipart/form-data; boundary=cut")
            .body(Body::from(form.replace('\n', "\r\n")))
            .expect("a request");
        let response = match T::from_request(request, &()).await {
            Ok(_) => panic!("the invalid form was accepted"),
            Err(response) => response,
        };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain error body");
        serde_json::from_slice(&body).expect("JSON error envelope")
    }

    #[tokio::test]
    async fn image_and_speech_forms_report_missing_models_in_the_json_error_object() {
        let form = "--cut--\n";
        for body in [
            endpoint_refusal::<ImageEditMultipart>(form).await,
            endpoint_refusal::<ImageVariationMultipart>(form).await,
            endpoint_refusal::<AudioSpeechMultipart>(form).await,
        ] {
            assert_eq!(
                body["error"]["code"], "missing_required_parameter",
                "{body}"
            );
            assert_eq!(body["error"]["param"], "model", "{body}");
        }
    }

    #[tokio::test]
    async fn image_and_speech_forms_name_the_invalid_field_in_the_json_error_object() {
        let image_form = "--cut\nContent-Disposition: form-data; name=\"n\"\n\ninvalid\n--cut--\n";
        let speech_form =
            "--cut\nContent-Disposition: form-data; name=\"speed\"\n\ninvalid\n--cut--\n";
        for (body, field) in [
            (
                endpoint_refusal::<ImageEditMultipart>(image_form).await,
                "n",
            ),
            (
                endpoint_refusal::<ImageVariationMultipart>(image_form).await,
                "n",
            ),
            (
                endpoint_refusal::<AudioSpeechMultipart>(speech_form).await,
                "speed",
            ),
        ] {
            assert_eq!(body["error"]["code"], "invalid_value", "{body}");
            assert_eq!(body["error"]["param"], field, "{body}");
        }
    }

    /// The parser's refusal of `form` (CRLF line ends, boundary `cut`): its
    /// status and JSON body.
    async fn refusal(form: &str) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method("POST")
            .header("content-type", "multipart/form-data; boundary=cut")
            .body(Body::from(form.replace('\n', "\r\n")))
            .expect("a request");
        refusal_of(request).await
    }

    /// The parser's refusal of `request`: its status and JSON body.
    async fn refusal_of(request: Request) -> (StatusCode, serde_json::Value) {
        let response = match AudioTranscriptionMultipart::from_request(request, &()).await {
            Ok(_) => panic!("the form was taken"),
            Err(refusal) => refusal,
        };
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        (
            status,
            serde_json::from_slice(&body).expect("json error body"),
        )
    }

    /// A request that is no multipart form at all (a JSON body here) is
    /// refused by the extractor before any part is read, with the same JSON
    /// error object and code as a part the parser cannot read.
    #[tokio::test]
    async fn a_request_that_is_not_a_form_is_refused_with_the_json_error_object() {
        let request = Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "whisper"}"#))
            .expect("a request");
        let (status, body) = refusal_of(request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(body["error"]["code"], "invalid_multipart", "{body}");
        assert!(body["error"]["param"].is_null(), "{body}");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("boundary"), "{body}");
    }

    #[tokio::test]
    async fn a_form_without_its_file_is_refused_with_the_json_error_object() {
        let (status, body) =
            refusal("--cut\nContent-Disposition: form-data; name=\"model\"\n\nwhisper\n--cut--\n")
                .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(
            body["error"]["code"], "missing_required_parameter",
            "{body}"
        );
        assert_eq!(body["error"]["param"], "file", "{body}");
        assert_eq!(
            body["error"]["message"], "Missing required 'file' part",
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_form_without_a_model_is_refused_with_the_json_error_object() {
        let (status, body) = refusal(
            "--cut\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\nContent-Type: audio/wav\n\nRIFF\n--cut--\n",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(
            body["error"]["code"], "missing_required_parameter",
            "{body}"
        );
        assert_eq!(body["error"]["param"], "model", "{body}");
        assert_eq!(
            body["error"]["message"], "Missing required 'model' field",
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_temperature_that_is_not_a_number_is_refused_with_the_json_error_object() {
        let (status, body) = refusal(
            "--cut\nContent-Disposition: form-data; name=\"temperature\"\n\nwarm\n--cut--\n",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "invalid_value", "{body}");
        assert_eq!(body["error"]["param"], "temperature", "{body}");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.starts_with("Invalid 'temperature' value:"),
            "{body}"
        );
    }
}
