// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Validated JSON extractor for automatic request validation.
//! Variation of https://github.com/lightseekorg/smg/blob/main/crates/protocols/src/validated.rs

use axum::Json;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;
use validator::Validate;

use super::types::Normalizable;
use crate::error::{ApiError, invalid_request, json_parse_error};

/// A JSON extractor that automatically validates and normalizes the request
/// body.
///
/// This extractor deserializes the request body and automatically calls
/// `.validate()` on types that implement the `Validate` trait. If validation
/// fails, it returns [`ApiError::InvalidRequest`] with details about the
/// validation errors.
pub struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(data) = Json::<T>::from_request(req, state).await?;
        Self::validated(data)
    }
}

impl<T: DeserializeOwned + Validate + Normalizable> ValidatedJson<T> {
    /// Deserialize and validate an already-parsed JSON body, failing the same
    /// way the extractor does.
    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self, ApiError> {
        let data = serde_path_to_error::deserialize(value).map_err(|error| {
            json_parse_error!("Failed to deserialize the JSON body into the target type: {error}")
        })?;
        Self::validated(data)
    }

    fn validated(mut data: T) -> Result<Self, ApiError> {
        data.normalize();

        data.validate()
            .map_err(|validation_errors| invalid_request!("{}", validation_errors))?;

        Ok(ValidatedJson(data))
    }
}

impl<T> std::ops::Deref for ValidatedJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ValidatedJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
