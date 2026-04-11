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
};

//======================
// API
//======================

void model_train(
    ModelContext* model,
    const ModelTrainingDesc* desc
);
