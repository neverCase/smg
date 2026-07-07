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
    response::{IntoResponse, Response},
};

#[cfg(feature = "axum")]
use crate::{
    transcription::{AudioFile, TranscriptionRequest},
    images::{ImageFile, ImageEditRequest, Mask},
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
impl<S: Send + Sync> FromRequest<S> for AudioTranscriptionMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;

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
                    return Err(bad_request(format!("Failed to read multipart field: {e}")));
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
                            return Err(bad_request(format!(
                                "Failed to read audio file bytes: {e}"
                            )));
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
                            return Err(bad_request(format!(
                                "Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"
                            )));
                        }
                        Err(e) => {
                            return Err(bad_request(format!("Invalid 'temperature' value: {e}")));
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
                "stream" => match field.text().await {
                    Ok(t) => match t.as_str() {
                        "true" | "True" | "TRUE" | "1" => request.stream = Some(true),
                        "false" | "False" | "FALSE" | "0" => request.stream = Some(false),
                        other => {
                            return Err(bad_request(format!(
                                "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                            )));
                        }
                    },
                    Err(e) => return Err(bad_text_field("stream", e)),
                },
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        if request.model.trim().is_empty() {
            return Err(bad_request("Missing required 'model' field".to_string()));
        }
        request.model = request.model.trim().to_string();

        let bytes = match file_bytes {
            Some(b) if !b.is_empty() => b,
            Some(_) => {
                return Err(bad_request("Uploaded 'file' part is empty".to_string()));
            }
            None => {
                return Err(bad_request("Missing required 'file' part".to_string()));
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
            .map_err(IntoResponse::into_response)?;

        let mut images = Vec::new();
        let mut request = ImageEditRequest::default();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(format!("Failed to read multipart field: {e}")));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                // 支持 "image" 和 "image[]" 两种字段名
                "image" | "image[]" => {
                    let file_name = field.file_name().map(str::to_string);
                    let content_type = field.content_type().map(str::to_string);
                    let bytes = field.bytes().await.map_err(|e| {
                        bad_request(format!("Failed to read image file bytes: {e}"))
                    })?;

                    if bytes.is_empty() {
                        return Err(bad_request("Uploaded 'image' part is empty".to_string()));
                    }

                    images.push(ImageFile {
                        bytes,
                        file_name: file_name.unwrap_or_else(|| "image".to_string()),
                        content_type,
                    });
                },
                "prompt" => {
                    request.prompt = field.text().await.map_err(|e| bad_text_field("prompt", e))?;
                }
                "model" => {
                    request.model = field.text().await.map_err(|e| bad_text_field("model", e))?;
                }
                "background" => {
                    let v = field.text().await.map_err(|e| bad_text_field("background", e))?;
                    if !v.is_empty() {
                        request.background = Some(v);
                    }
                }
                "input_fidelity" => {
                    let v = field.text().await.map_err(|e| bad_text_field("input_fidelity", e))?;
                    if !v.is_empty() {
                        request.input_fidelity = Some(v);
                    }
                }
                "moderation" => {
                    let v = field.text().await.map_err(|e| bad_text_field("moderation", e))?;
                    if !v.is_empty() {
                        request.moderation = Some(v);
                    }
                }
                "n" => {
                    let v = field.text().await.map_err(|e| bad_text_field("n", e))?;
                    if !v.is_empty() {
                        let n = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(format!("Invalid 'n' value: '{v}' (must be 1-10)"))
                        })?;
                        if !(1..=10).contains(&n) {
                            return Err(bad_request(format!(
                                "Invalid 'n' value: {n} (must be between 1 and 10)"
                            )));
                        }
                        request.n = Some(n);
                    }
                }
                "output_compression" => {
                    let v = field.text().await.map_err(|e| bad_text_field("output_compression", e))?;
                    if !v.is_empty() {
                        let c = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(format!("Invalid 'output_compression' value: '{v}' (must be 0-100)"))
                        })?;
                        if c > 100 {
                            return Err(bad_request(format!(
                                "Invalid 'output_compression' value: {c} (must be between 0 and 100)"
                            )));
                        }
                        request.output_compression = Some(c);
                    }
                }
                "output_format" => {
                    let v = field.text().await.map_err(|e| bad_text_field("output_format", e))?;
                    if !v.is_empty() {
                        request.output_format = Some(v);
                    }
                }
                "partial_images" => {
                    let v = field.text().await.map_err(|e| bad_text_field("partial_images", e))?;
                    if !v.is_empty() {
                        let p = v.trim().parse::<u8>().map_err(|_| {
                            bad_request(format!("Invalid 'partial_images' value: '{v}' (must be 0-3)"))
                        })?;
                        if p > 3 {
                            return Err(bad_request(format!(
                                "Invalid 'partial_images' value: {p} (must be between 0 and 3)"
                            )));
                        }
                        request.partial_images = Some(p);
                    }
                }
                "quality" => {
                    let v = field.text().await.map_err(|e| bad_text_field("quality", e))?;
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
                    let v = field.text().await.map_err(|e| bad_text_field("stream", e))?;
                    if !v.is_empty() {
                        request.stream = match v.as_str() {
                            "true" | "True" | "TRUE" | "1" => Some(true),
                            "false" | "False" | "FALSE" | "0" => Some(false),
                            other => {
                                return Err(bad_request(format!(
                                    "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                                )));
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
                                return Err(bad_request(format!(
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
            return Err(bad_request("Missing required 'model' field".to_string()));
        }
        request.model = request.model.trim().to_string();

        if request.prompt.trim().is_empty() {
            return Err(bad_request("Missing required 'prompt' field".to_string()));
        }
        request.prompt = request.prompt.trim().to_string();

        if images.is_empty() {
            return Err(bad_request("Missing required 'image' part(s)".to_string()));
        }
        if images.len() > 16 {
            return Err(bad_request(format!(
                "Too many images: {} (maximum 16)",
                images.len()
            )));
        }

        Ok(ImageEditMultipart { request, images })
    }
}

#[cfg(feature = "axum")]
fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}

#[cfg(feature = "axum")]
fn bad_text_field(field: &str, e: MultipartError) -> Response {
    bad_request(format!("Failed to read '{field}' field: {e}"))
}

