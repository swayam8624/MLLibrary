//
//  matrix.hpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//
#pragma once

#include <cstdint>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using b32 = std::int32_t;

class MemArena;

//======================
// Matrix
//======================

struct Matrix {
    u32 rows;
    u32 cols;
    f32* data;
};

//======================
// Creation
//======================

Matrix* mat_create(MemArena* arena, u32 rows, u32 cols);
b32     mat_copy(Matrix* dst, const Matrix* src);

//======================
// Basic ops
//======================

void mat_clear(Matrix* mat);
void mat_fill(Matrix* mat, f32 x);
void mat_fill_rand(Matrix* mat, f32 lower, f32 upper);

void mat_scale(Matrix* mat, f32 scale);

f32 mat_sum(const Matrix* mat);
u64 mat_argmax(const Matrix* mat);

//======================
// Math ops
//======================

b32 mat_add(Matrix* out, const Matrix* a, const Matrix* b);
b32 mat_sub(Matrix* out, const Matrix* a, const Matrix* b);

b32 mat_mul(
    Matrix* out, const Matrix* a, const Matrix* b,
    b32 zero_out, b32 transpose_a, b32 transpose_b
);

//======================
// Activations and losses
//======================

b32 mat_relu(Matrix* out, const Matrix* in);

// Legacy scalar-runtime semantics: every element of `in` belongs to one
// categorical distribution, regardless of the matrix shape. Batched/axis-aware
// softmax belongs to the Tensor runtime rather than this compact matrix layer.
b32 mat_softmax(Matrix* out, const Matrix* in);

// Elementwise -p * log(q + epsilon). The output has the same shape as p and q.
b32 mat_cross_entropy(Matrix* out, const Matrix* p, const Matrix* q);

// Numerically stable scalar cross entropy computed directly from logits.
// `out` must be 1x1. All target/logit elements form one distribution.
b32 mat_softmax_cross_entropy(
    Matrix* out,
    const Matrix* targets,
    const Matrix* logits
);

//======================
// Gradients
//======================

b32 mat_relu_add_grad(Matrix* out, const Matrix* in, const Matrix* grad);

// Adds J_softmax^T * grad to `out` for the all-elements distribution used by
// mat_softmax.
b32 mat_softmax_add_grad(Matrix* out, const Matrix* softmax_out, const Matrix* grad);

// Adds the true derivatives of elementwise cross entropy. Composing this with
// mat_softmax_add_grad yields the usual softmax-cross-entropy derivative.
b32 mat_cross_entropy_add_grad(
    Matrix* p_grad, Matrix* q_grad,
    const Matrix* p, const Matrix* q, const Matrix* grad
);

// Adds gradients for the numerically stable scalar fused loss. `grad` must be
// 1x1. For normalized targets, logits_grad receives softmax(logits) - targets.
b32 mat_softmax_cross_entropy_add_grad(
    Matrix* target_grad,
    Matrix* logits_grad,
    const Matrix* targets,
    const Matrix* logits,
    const Matrix* grad
);
