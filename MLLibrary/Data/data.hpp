//
//  data.hpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

// TEACHING PUBLIC API: legacy MNIST raw Float32 loader and diagnostic
// helpers, implemented by Data/data.cpp. Distinct from typed Dataset/CSV.
// See docs/DATA_TO_AUTOML_LAB.md and the MNIST training walkthrough.
#pragma once

#include "matrix.hpp"

class MemArena;

//======================
// Loading
//======================

// Load exactly rows*cols raw Float32 values into arena-owned Matrix storage.
// Returns nullptr on invalid input/IO failure; no IDX parsing or byte swap.
Matrix* mat_load(
    MemArena* arena,
    u32 rows,
    u32 cols,
    const char* filename
);

//======================
// Utilities
//======================

// Print one 28x28 sample to a color-capable terminal; assumes >=784 floats.
void draw_mnist_digit(const f32* data);

// One-hot encoding
// Writes out[row,class]=1 for valid unsigned class indices; silently skips
// classes >= num_classes after conversion. Validate labels before calling.
void one_hot_encode(
    Matrix* out,
    const Matrix* labels,
    u32 num_classes
);
