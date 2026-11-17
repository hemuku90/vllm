// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! SageMaker's single inference entry point.
//!
//! Original Python definition:
//! <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/sagemaker/api_router.py>

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::Value;

use crate::error::{ApiError, invalid_request};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{self, ChatCompletionRequest, CompletionRequest};
use crate::state::AppState;

/// For SageMaker, routes requests based on the request type.
///
/// The body goes to the first endpoint whose request type it validates as:
/// chat completions, then completions. If none fits, the error is that of
/// the first type whose required fields the body has, i.e. the type the
/// client most likely meant, reported as that endpoint would report it.
// TODO: also try the pooling request types once their endpoints exist.
pub async fn invocations(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    // `messages` is the only required field of a chat completion request;
    // a completion request has none.
    let has_chat_fields = body.get("messages").is_some();

    let chat_error = match ValidatedJson::<ChatCompletionRequest>::from_value(body.clone()) {
        Ok(request) => return openai::chat_completions(state, headers, request).await,
        Err(error) => error,
    };
    let completion_error = match ValidatedJson::<CompletionRequest>::from_value(body.clone()) {
        Ok(request) => return openai::completions(state, headers, request).await,
        Err(error) => error,
    };

    Err(match body {
        Value::Object(_) if has_chat_fields => chat_error,
        Value::Object(_) => completion_error,
        _ => invalid_request!(
            "Cannot find suitable handler for request. Expected one of: \
             ['ChatCompletionRequest', 'CompletionRequest']"
        ),
    })
}
