//
//  matrix.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

// TEACHING NUMERICAL KERNELS — see docs/MNIST_TRAINING_FROM_FIRST_PRINCIPLES.md.
// Model/model.cpp calls these during forward and reverse passes; training.cpp
// uses reductions and parameter updates. Matrices are arena-backed Float32
// row-major buffers, not owning std::vector objects. Most public kernels
// return b32 false on invalid shapes/pointers rather than throwing.
// Not all APIs can safely accept partially overlapping source/dest ranges.
#include "matrix.hpp"
#include "arena.h"
#include "prng.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
constexpr f32 cross_entropy_epsilon = 1e-7f;

bool same_shape(const Matrix* a, const Matrix* b)
{
    return a && b && a->rows == b->rows && a->cols == b->cols;
}

u64 element_count(const Matrix* matrix)
{
    return matrix ? static_cast<u64>(matrix->rows) * matrix->cols : 0;
}
}

//======================
// Creation
//======================

// Allocate a Matrix descriptor AND its Float32 data from the caller's arena.
// The lifetime follows MemArena, not an independent Matrix destructor;
// shape multiplication is widened to u64 for push_array's allocation guard.
Matrix *mat_create(MemArena *arena, u32 rows, u32 cols)
{
    Matrix *mat = push_struct<Matrix>(arena);
    if (!mat)
        return nullptr;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = push_array<f32>(arena, static_cast<u64>(rows) * cols);
    if (!mat->data)
        return nullptr;

    return mat;
}

// Equal-shape raw byte copy. std::memcpy does not support partially
// overlapping buffers; use nonoverlapping storage (or a memmove-style API).
b32 mat_copy(Matrix *dst, const Matrix *src)
{
    if (!dst || !src || !dst->data || !src->data)
        return false;
    if (!same_shape(dst, src))
        return false;

    std::memcpy(dst->data, src->data,
                sizeof(f32) * element_count(dst));

    return true;
}

//======================
// Basic ops
//======================

// Zero every Float32 bit pattern. training.cpp clears all parameter and
// intermediate gradients before a new batch; reverse mode then accumulates.
void mat_clear(Matrix *mat)
{
    if (!mat || !mat->data)
        return;
    std::memset(mat->data, 0,
                sizeof(f32) * element_count(mat));
}

// Assign one scalar to every logical element; model.cpp seeds the root loss
// gradient using this operation before reverse traversal.
void mat_fill(Matrix *mat, f32 x)
{
    if (!mat || !mat->data)
        return;
    const u64 size = element_count(mat);
    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] = x;
    }
}

// Weight initialization draws from the GLOBAL PRNG wrapper in Core/prng.cpp.
// Reproducibility therefore depends on when prng_seed is called, separate
// from the training descriptor's independently seeded batch-shuffle stream.
void mat_fill_rand(Matrix *mat, f32 lower, f32 upper)
{
    if (!mat || !mat->data)
        return;
    const u64 size = element_count(mat);

    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}

// In-place scaling used by probability normalization and reductions;
// modifies the arena-owned value buffer, not a separate result allocation.
void mat_scale(Matrix *mat, f32 scale)
{
    if (!mat || !mat->data)
        return;
    const u64 size = element_count(mat);

    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] *= scale;
    }
}

// Scalar reduction over the entire flat matrix. main/model training uses it
// to sum the elementwise cross-entropy matrix into sample loss.
f32 mat_sum(const Matrix *mat)
{
    if (!mat || !mat->data)
        return 0.0f;
    const u64 size = element_count(mat);

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++)
    {
        sum += mat->data[i];
    }
    return sum;
}

// Return flat maximum index for evaluation. Empty/null matrices return 0,
// so callers must validate nonempty data before interpreting index 0 as class.
u64 mat_argmax(const Matrix *mat)
{
    if (!mat || !mat->data)
        return 0;
    const u64 size = element_count(mat);
    if (size == 0)
        return 0;

    u64 max_i = 0;
    for (u64 i = 1; i < size; i++)
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

// Shape-preserving elementwise add; model.cpp uses it both in the residual
// forward path and to accumulate gradient contributions into parent buffers.
b32 mat_add(Matrix *out, const Matrix *a, const Matrix *b)
{
    if (!out || !a || !b || !out->data || !a->data || !b->data)
        return false;
    if (!same_shape(a, b) || !same_shape(out, a))
        return false;

    const u64 size = element_count(out);

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

// Shape-preserving elementwise subtract. In reverse-mode differentiation,
// out may alias an operand's gradient buffer for accumulation.
b32 mat_sub(Matrix *out, const Matrix *a, const Matrix *b)
{
    if (!out || !a || !b || !out->data || !a->data || !b->data)
        return false;
    if (!same_shape(a, b) || !same_shape(out, a))
        return false;

    const u64 size = element_count(out);

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

//======================
// Matmul kernels
//======================

// Core GEMM: transpose_a/transpose_b select the logical operand layouts
// without materializing transposes. zero_out=true overwrites C; false adds
// the dot-product contribution into existing C. model.cpp uses the latter
// for shared-graph gradient accumulation (dA=dC*B^T, dB=A^T*dC).
// Invalid shape/pointer returns false; the model interpreter currently does
// not propagate this bool as a typed exception.
b32 mat_mul(
    Matrix *out, const Matrix *a, const Matrix *b,
    b32 zero_out, b32 transpose_a, b32 transpose_b)
{
    if (!out || !a || !b || !out->data || !a->data || !b->data)
        return false;

    const u32 a_rows = transpose_a ? a->cols : a->rows;
    const u32 a_cols = transpose_a ? a->rows : a->cols;
    const u32 b_rows = transpose_b ? b->cols : b->rows;
    const u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows)
        return false;
    if (out->rows != a_rows || out->cols != b_cols)
        return false;

    for (u32 i = 0; i < a_rows; i++)
    {
        for (u32 j = 0; j < b_cols; j++)
        {
            f32 sum = 0.0f;
            for (u32 k = 0; k < a_cols; k++)
            {
                const u32 index_a = transpose_a ? (i + k * a->cols) : (k + i * a->cols);
                const u32 index_b = transpose_b ? (k + j * b->cols) : (j + k * b->cols);

                sum += a->data[index_a] * b->data[index_b];
            }

            f32 *dst = out->data + j + i * out->cols;
            if (zero_out)
                *dst = sum;
            else
                *dst += sum;
        }
    }

    return true;
}

//======================
// Activations and losses
//======================

// Pointwise max(0,x) on identical shapes. The derivative helper below uses
// the input's positive region; the kink at zero is nondifferentiable.
b32 mat_relu(Matrix *out, const Matrix *in)
{
    if (!out || !in || !out->data || !in->data)
        return false;
    if (!same_shape(out, in))
        return false;

    const u64 size = element_count(out);

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = std::max(0.0f, in->data[i]);
    }

    return true;
}

// Stable softmax over the ENTIRE flattened matrix (not one row per sample).
// Subtracting the maximum avoids large positive exponent overflow for finite
// inputs. model.cpp's MV_OP_SOFTMAX dispatch uses this implementation.
b32 mat_softmax(Matrix *out, const Matrix *in)
{
    if (!out || !in || !out->data || !in->data)
        return false;
    if (!same_shape(out, in))
        return false;

    const u64 size = element_count(out);
    if (size == 0)
        return false;

    f32 max_val = in->data[0];
    for (u64 i = 1; i < size; i++)
    {
        max_val = std::max(max_val, in->data[i]);
    }
    if (!std::isfinite(max_val))
        return false;

    double sum = 0.0;
    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = std::exp(in->data[i] - max_val);
        sum += out->data[i];
    }

    if (!(sum > 0.0) || !std::isfinite(sum))
        return false;

    mat_scale(out, static_cast<f32>(1.0 / sum));
    return true;
}

// Elementwise target/probability cross-entropy; output keeps input shape.
// The training loop explicitly calls mat_sum to obtain one sample cost.
// This is distinct from the scalar logits-level fused loss below.
b32 mat_cross_entropy(Matrix *out, const Matrix *p, const Matrix *q)
{
    if (!out || !p || !q || !out->data || !p->data || !q->data)
        return false;
    if (!same_shape(p, q) || !same_shape(out, p))
        return false;

    const u64 size = element_count(out);

    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = p->data[i] == 0.0f
            ? 0.0f
            : -p->data[i] * std::log(q->data[i] + cross_entropy_epsilon);
    }

    return true;
}

// Numerically focused fused alternative: accepts targets and raw logits,
// produces one [1,1] loss value. main.cpp currently constructs softmax
// followed by elementwise cross entropy instead of this fused operation.
b32 mat_softmax_cross_entropy(
    Matrix* out,
    const Matrix* targets,
    const Matrix* logits)
{
    if (!out || !targets || !logits || !out->data || !targets->data || !logits->data)
        return false;
    if (!same_shape(targets, logits) || out->rows != 1 || out->cols != 1)
        return false;

    const u64 size = element_count(logits);
    if (size == 0)
        return false;

    f32 max_logit = logits->data[0];
    for (u64 i = 1; i < size; ++i)
    {
        max_logit = std::max(max_logit, logits->data[i]);
    }
    if (!std::isfinite(max_logit))
        return false;

    double exponential_sum = 0.0;
    double target_sum = 0.0;
    double weighted_logits = 0.0;
    for (u64 i = 0; i < size; ++i)
    {
        exponential_sum += std::exp(static_cast<double>(logits->data[i] - max_logit));
        target_sum += targets->data[i];
        weighted_logits += static_cast<double>(targets->data[i]) * logits->data[i];
    }

    if (!(exponential_sum > 0.0) || !std::isfinite(exponential_sum))
        return false;

    const double log_sum_exp = static_cast<double>(max_logit) + std::log(exponential_sum);
    const double loss = target_sum * log_sum_exp - weighted_logits;
    if (!std::isfinite(loss))
        return false;

    out->data[0] = static_cast<f32>(loss);
    return true;
}

//======================
// Gradients
//======================

// Reverse-mode vector-Jacobian product for ReLU. ADD to out instead of
// assigning: a graph node can have multiple downstream gradient consumers.
b32 mat_relu_add_grad(Matrix *out, const Matrix *in, const Matrix *grad)
{
    if (!out || !in || !grad || !out->data || !in->data || !grad->data)
        return false;
    if (!same_shape(out, in) || !same_shape(out, grad))
        return false;

    const u64 size = element_count(out);

    for (u64 i = 0; i < size; i++)
    {
        if (in->data[i] > 0.0f)
        {
            out->data[i] += grad->data[i];
        }
    }

    return true;
}

// Full-softmax vector-Jacobian product. Uses dot(probability, incoming_grad)
// to account for every output's dependence on all logits, then ADDS to out.
b32 mat_softmax_add_grad(Matrix *out, const Matrix *softmax_out, const Matrix *grad)
{
    if (!out || !softmax_out || !grad || !out->data || !softmax_out->data || !grad->data)
        return false;
    if (!same_shape(out, softmax_out) || !same_shape(out, grad))
        return false;

    const u64 size = element_count(out);
    double projection = 0.0;
    for (u64 i = 0; i < size; ++i)
    {
        projection += static_cast<double>(softmax_out->data[i]) * grad->data[i];
    }

    for (u64 i = 0; i < size; ++i)
    {
        out->data[i] += softmax_out->data[i]
            * (grad->data[i] - static_cast<f32>(projection));
    }

    return true;
}

// Cross-entropy reverse path: p_grad/q_grad are optional independently;
// the target may intentionally be nondifferentiable while predicted
// probabilities receive gradients. Shapes are checked before writes.
b32 mat_cross_entropy_add_grad(
    Matrix *p_grad, Matrix *q_grad,
    const Matrix *p, const Matrix *q, const Matrix *grad)
{
    if (!p || !q || !grad || !p->data || !q->data || !grad->data)
        return false;
    if (!same_shape(p, q) || !same_shape(grad, p))
        return false;

    if (p_grad && (!p_grad->data || !same_shape(p_grad, p)))
        return false;
    if (q_grad && (!q_grad->data || !same_shape(q_grad, q)))
        return false;
    if (!p_grad && !q_grad)
        return true;

    const u64 size = element_count(p);

    for (u64 i = 0; i < size; i++)
    {
        const f32 probability = q->data[i] + cross_entropy_epsilon;
        if (q_grad)
            q_grad->data[i] += (-p->data[i] / probability) * grad->data[i];

        if (p_grad)
            p_grad->data[i] += -std::log(probability) * grad->data[i];
    }

    return true;
}

// Fused logits-level loss backward; root gradient is scalar, while the
// computed logits gradient has target/logit shape and accumulates into
// existing storage. Compare with a finite-difference reference.
b32 mat_softmax_cross_entropy_add_grad(
    Matrix* target_grad,
    Matrix* logits_grad,
    const Matrix* targets,
    const Matrix* logits,
    const Matrix* grad)
{
    if (!targets || !logits || !grad || !targets->data || !logits->data || !grad->data)
        return false;
    if (!same_shape(targets, logits) || grad->rows != 1 || grad->cols != 1)
        return false;
    if (target_grad && (!target_grad->data || !same_shape(target_grad, targets)))
        return false;
    if (logits_grad && (!logits_grad->data || !same_shape(logits_grad, logits)))
        return false;
    if (!target_grad && !logits_grad)
        return true;

    const u64 size = element_count(logits);
    if (size == 0)
        return false;

    f32 max_logit = logits->data[0];
    for (u64 i = 1; i < size; ++i)
    {
        max_logit = std::max(max_logit, logits->data[i]);
    }
    if (!std::isfinite(max_logit))
        return false;

    double exponential_sum = 0.0;
    double target_sum = 0.0;
    for (u64 i = 0; i < size; ++i)
    {
        exponential_sum += std::exp(static_cast<double>(logits->data[i] - max_logit));
        target_sum += targets->data[i];
    }
    if (!(exponential_sum > 0.0) || !std::isfinite(exponential_sum))
        return false;

    const double log_sum_exp = static_cast<double>(max_logit) + std::log(exponential_sum);
    const f32 upstream = grad->data[0];
    for (u64 i = 0; i < size; ++i)
    {
        if (logits_grad)
        {
            const double probability = std::exp(
                static_cast<double>(logits->data[i] - max_logit)) / exponential_sum;
            logits_grad->data[i] += static_cast<f32>(
                (probability * target_sum - targets->data[i]) * upstream);
        }

        if (target_grad)
        {
            target_grad->data[i] += static_cast<f32>(
                (log_sum_exp - logits->data[i]) * upstream);
        }
    }

    return true;
}
