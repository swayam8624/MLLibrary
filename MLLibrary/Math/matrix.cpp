//
//  matrix.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

#include "matrix.hpp"
#include "arena.h"
#include "prng.h"

#include <cstring>
#include <cmath>
#include <algorithm>
//======================
// Creation
//======================

Matrix *mat_create(MemArena *arena, u32 rows, u32 cols)
{
    Matrix *mat = push_struct<Matrix>(arena);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = push_array<f32>(arena, (u64)rows * cols);

    return mat;
}

b32 mat_copy(Matrix *dst, const Matrix *src)
{
    if (dst->rows != src->rows || dst->cols != src->cols)
    {
        return false;
    }

    std::memcpy(dst->data, src->data,
                sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}

//======================
// Basic ops
//======================

void mat_clear(Matrix *mat)
{
    std::memset(mat->data, 0,
                sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(Matrix *mat, f32 x)
{
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] = x;
    }
}

void mat_fill_rand(Matrix *mat, f32 lower, f32 upper)
{
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}

void mat_scale(Matrix *mat, f32 scale)
{
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(const Matrix *mat)
{
    u64 size = (u64)mat->rows * mat->cols;

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++)
    {
        sum += mat->data[i];
    }
    return sum;
}

u64 mat_argmax(const Matrix *mat)
{
    u64 size = (u64)mat->rows * mat->cols;

    u64 max_i = 0;
    for (u64 i = 0; i < size; i++)
    {
        if (mat->data[i] > mat->data[max_i])
        {
            max_i = i;
        }
    }
    return max_i;
}

//======================
// Math ops
//======================

b32 mat_add(Matrix *out, const Matrix *a, const Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
        return false;
    if (out->rows != a->rows || out->cols != a->cols)
        return false;

    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 mat_sub(Matrix *out, const Matrix *a, const Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
        return false;
    if (out->rows != a->rows || out->cols != a->cols)
        return false;

    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

//======================
// Matmul kernels
//======================

b32 mat_mul(
    Matrix *out, const Matrix *a, const Matrix *b,
    b32 zero_out, b32 transpose_a, b32 transpose_b)
{
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows)
        return false;
    if (out->rows != a_rows || out->cols != b_cols)
        return false;

    if (zero_out)
        mat_clear(out);

    // Compute with transpose awareness
    for (u32 i = 0; i < a_rows; i++)
    {
        for (u32 j = 0; j < b_cols; j++)
        {
            f32 sum = 0.0f;
            for (u32 k = 0; k < a_cols; k++)
            {
                // If transposed, swap the row/col indexing mapping
                u32 index_a = transpose_a ? (i + k * a->cols) : (k + i * a->cols);
                u32 index_b = transpose_b ? (k + j * b->cols) : (j + k * b->cols);

                sum += a->data[index_a] * b->data[index_b];
            }
            out->data[j + i * out->cols] += sum;
        }
    }

    return true;
}

//======================
// Activations
//======================

b32 mat_relu(Matrix *out, const Matrix *in)
{
    if (out->rows != in->rows || out->cols != in->cols)
        return false;

    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = std::max(0.0f, in->data[i]);
    }

    return true;
}

b32 mat_softmax(Matrix *out, const Matrix *in)
{
    u64 size = (u64)out->rows * out->cols;

    // find max
    f32 max_val = in->data[0];
    for (u64 i = 1; i < size; i++)
    {
        if (in->data[i] > max_val)
            max_val = in->data[i];
    }

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = std::exp(in->data[i] - max_val); // ✅ FIX
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);
    return true;
}

b32 mat_cross_entropy(Matrix *out, const Matrix *p, const Matrix *q)
{
    if (p->rows != q->rows || p->cols != q->cols)
        return false;
    if (out->rows != p->rows || out->cols != p->cols)
        return false;

    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        // Add 1e-7f to q->data[i] to prevent log(0) which causes -inf and NaN costs
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -std::log(q->data[i] + 1e-7f);
    }

    return true;
}

b32 mat_relu_add_grad(Matrix *out, const Matrix *in, const Matrix *grad)
{
    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        if (in->data[i] > 0.0f)
        {
            out->data[i] += grad->data[i];
        }
    }

    return true;
}

b32 mat_softmax_add_grad(Matrix *out, const Matrix *softmax_out, const Matrix *grad)
{
    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] += grad->data[i]; // simplified (works with cross-entropy)
    }

    return true;
}

b32 mat_cross_entropy_add_grad(
    Matrix *p_grad, Matrix *q_grad,
    const Matrix *p, const Matrix *q, const Matrix *grad)
{
    u64 size = (u64)p->rows * p->cols;

    for (u64 i = 0; i < size; i++)
    {
        // Because softmax backward is a pass-through, we calculate the
        // mathematically combined gradient (q - p) here.
        // q is the prediction, p is the target.
        q_grad->data[i] += (q->data[i] - p->data[i]) * grad->data[i];
    }

    return true;
}