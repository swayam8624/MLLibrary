//
//  data.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

// TEACHING BRIDGE: main.cpp's fixed-size raw MNIST loader. Unlike the typed
// CSV Dataset in dataset.cpp, this path expects unheadered binary Float32
// matrices of exactly caller-declared shape. Memory belongs to MemArena and
// all callers must preserve its lifetime. See docs/DATA_TO_AUTOML_LAB.md.
#include "data.hpp"
#include "arena.h"

#include <cstdio>
#include <cstring>

//======================
// Load matrix from file
//======================

// Open a raw binary file, check its exact byte length against rows*cols
// Float32 values, allocate an arena-backed Matrix via Math/matrix.cpp and
// read all bytes. Failure returns nullptr; a failed read does not roll back
// the arena's allocations. No CSV, IDX, endianness or dtype conversion here.
Matrix *mat_load(
    MemArena *arena,
    u32 rows,
    u32 cols,
    const char *filename)
{
    if (!arena || !filename || rows == 0 || cols == 0)
        return nullptr;

    FILE *file = std::fopen(filename, "rb");
    if (!file)
        return nullptr;

    if (std::fseek(file, 0, SEEK_END) != 0)
    {
        std::fclose(file);
        return nullptr;
    }
    const long end = std::ftell(file);
    if (end < 0 || std::fseek(file, 0, SEEK_SET) != 0)
    {
        std::fclose(file);
        return nullptr;
    }

    const u64 expected_size = sizeof(f32) * static_cast<u64>(rows) * cols;
    if (static_cast<u64>(end) != expected_size)
    {
        std::fclose(file);
        return nullptr;
    }

    Matrix *mat = mat_create(arena, rows, cols);
    if (!mat)
    {
        std::fclose(file);
        return nullptr;
    }

    const u64 read_size = std::fread(mat->data, 1, expected_size, file);
    const bool close_succeeded = std::fclose(file) == 0;
    if (read_size != expected_size || !close_succeeded)
        return nullptr;

    return mat;
}

//======================
// Visualization
//======================

// Terminal-only debugging view of a 28x28 Float32 image. Assumes the pointer
// addresses at least 784 elements in the expected pixel order; this function
// has no length parameter and does not validate its caller's buffer extent.
void draw_mnist_digit(const f32 *data)
{
    for (u32 y = 0; y < 28; y++)
    {
        for (u32 x = 0; x < 28; x++)
        {
            f32 val = data[x + y * 28];
            u32 col = 232 + (u32)(val * 23);

            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

//======================
// One-hot encoding
//======================

// Convert each scalar class index from labels into a row-wise one-hot Matrix.
// Output must have the same sample count and num_classes columns; main.cpp
// allocates it in MemArena before model training. The loop reads the first
// label element in each row and silently skips integer class IDs >= num_classes.
// It does not report invalid/fractional labels, so callers must validate them
// before conversion; this void API supplies no error result.
void one_hot_encode(
    Matrix *out,
    const Matrix *labels,
    u32 num_classes)
{
    if (!out || !labels || !out->data || !labels->data || out->cols != num_classes)
        return;
    if (out->rows != labels->rows || labels->cols == 0)
        return;

    mat_clear(out);

    for (u32 i = 0; i < labels->rows; i++)
    {
        u32 cls = (u32)labels->data[i];
        if (cls >= num_classes)
            continue;

        out->data[i * num_classes + cls] = 1.0f;
    }
}
