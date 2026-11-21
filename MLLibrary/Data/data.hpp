// SOURCE STUDY NOTE:
// Input/data API for the baseline demonstration. Follow data.cpp for format handling and conversion
// to matrix buffers, then main.cpp and Training/training.cpp for consumers.
// Cross-file reading guide: docs/SOURCE_READING_ORDER.md.

//
//  data.hpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#pragma once

#include "matrix.hpp"

class MemArena;

//======================
// Loading
//======================

Matrix* mat_load(
    MemArena* arena,
    u32 rows,
    u32 cols,
    const char* filename
);

//======================
// Utilities
//======================

void draw_mnist_digit(const f32* data);

// One-hot encoding
void one_hot_encode(
    Matrix* out,
    const Matrix* labels,
    u32 num_classes
);
