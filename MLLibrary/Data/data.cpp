//
//  data.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#include "data.hpp"
#include "arena.h"

#include <cstdio>
#include <cstring>

//======================
// Load matrix from file
//======================

Matrix *mat_load(
    MemArena *arena,
    u32 rows,
    u32 cols,
    const char *filename)
{
    Matrix *mat = mat_create(arena, rows, cols);
    if (!mat)
        return nullptr;

    FILE *f = std::fopen(filename, "rb");
    if (!f)
        return nullptr;

    std::fseek(f, 0, SEEK_END);
    u64 size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    u64 max_size = sizeof(f32) * (u64)rows * cols;
    if (size > max_size)
        size = max_size;

    u64 read_size = std::fread(mat->data, 1, size, f);
    std::fclose(f);

    if (read_size != size)
        return nullptr;

    return mat;
}

//======================
// Visualization
//======================

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
