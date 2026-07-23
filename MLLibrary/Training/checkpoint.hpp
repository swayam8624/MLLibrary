#pragma once

#include <string>

#include "model.hpp"

/// Persists only trainable model parameters in a versioned binary checkpoint.
/// The write uses a temporary sibling file then rename, so a failed write does
/// not replace a prior checkpoint. Load validates count and matrix shapes
/// before copying any parameter data.
bool model_save_checkpoint(const ModelContext* model, const char* path, std::string* error = nullptr);
bool model_load_checkpoint(ModelContext* model, const char* path, std::string* error = nullptr);
