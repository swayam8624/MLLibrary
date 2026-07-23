//
//  training.hpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#pragma once

#include <cstdint>
#include "matrix.hpp"
#include "model.hpp"

using u32 = std::uint32_t;
using f32 = float;

//======================
// Optimizers
//======================

enum ModelOptimizerKind {
    MODEL_OPTIMIZER_SGD = 0,
    MODEL_OPTIMIZER_MOMENTUM_SGD,
    MODEL_OPTIMIZER_NESTEROV,
    MODEL_OPTIMIZER_RMSPROP,
    MODEL_OPTIMIZER_ADAM,
    MODEL_OPTIMIZER_ADAMW,
};

//======================
// Training descriptor
//======================

struct ModelTrainingDesc {
    Matrix* train_images;
    Matrix* train_labels;
    Matrix* test_images;
    Matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;

    ModelOptimizerKind optimizer = MODEL_OPTIMIZER_SGD;
    f32 momentum = 0.9f;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    f32 epsilon = 1e-8f;
    f32 weight_decay = 0.0f;
    // A non-positive value disables global-norm clipping.
    f32 max_gradient_norm = 0.0f;

    // Optional CSV path for visual training dashboards.
    // Columns: epoch,accuracy,cost,learning_rate
    const char* metrics_csv_path = nullptr;
};

//======================
// API
//======================

void model_train(
    ModelContext* model,
    const ModelTrainingDesc* desc
);
