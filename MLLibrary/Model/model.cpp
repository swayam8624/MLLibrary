//
//  model.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#include "model.hpp"
#include "arena.h"

#include <cstring>

//======================
// Creation
//======================

// API / ownership: construct the graph registry inside MemArena. The caller
// owns the arena; all ModelVar pointers and compiled programs borrow its
// lifetime. main.cpp creates this before create_mnist_model().
ModelContext *model_create(MemArena *arena)
{
    return push_struct<ModelContext>(arena);
}

//======================
// Variable creation
//======================

// API / memory: allocate one graph node and its value matrix, optionally a
// gradient matrix when MV_FLAG_REQUIRES_GRAD is present. The role flags also
// register the input/output/target/cost in ModelContext; a failure returns
// nullptr and previously bumped arena allocations are not individually freed.
ModelVar *mv_create(
    MemArena *arena, ModelContext *model,
    u32 rows, u32 cols, u32 flags)
{
    if (!arena || !model || rows == 0 || cols == 0)
        return nullptr;

    ModelVar *out = push_struct<ModelVar>(arena);
    if (!out)
        return nullptr;

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;

    out->val = mat_create(arena, rows, cols);
    if (!out->val)
        return nullptr;

    if (flags & MV_FLAG_REQUIRES_GRAD)
    {
        out->grad = mat_create(arena, rows, cols);
        if (!out->grad)
            return nullptr;
        mat_clear(out->grad);
    }
    else
    {
        out->grad = nullptr;
    }

    if (flags & MV_FLAG_INPUT)
        model->input = out;
    if (flags & MV_FLAG_OUTPUT)
        model->output = out;
    if (flags & MV_FLAG_DESIRED_OUTPUT)
        model->desired_output = out;
    if (flags & MV_FLAG_COST)
        model->cost = out;

    return out;
}

//======================
// Internal helpers
//======================

// Internal graph builder: unary nodes retain one input pointer and inherit
// its differentiability, but do not compute any values yet. model_prog_compute
// executes the operation later, after graph compilation.
static ModelVar *mv_unary(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags, ModelVarOp op)
{
    if (!input || !input->val)
        return nullptr;

    if (input->flags & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar *out = mv_create(
        arena, model,
        input->val->rows,
        input->val->cols,
        flags);
    if (!out)
        return nullptr;

    out->op = op;
    out->inputs[0] = input;

    return out;
}

// Internal graph builder: retain both operand nodes; automatic differentiation
// is required when either operand needs gradients. Callers validate operation-
// specific shapes before allocating this result node.
static ModelVar *mv_binary(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b,
    u32 rows, u32 cols,
    u32 flags, ModelVarOp op)
{
    if (!a || !b || !a->val || !b->val)
        return nullptr;

    if ((a->flags | b->flags) & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar *out = mv_create(arena, model, rows, cols, flags);
    if (!out)
        return nullptr;

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

//======================
// Operations
//======================

// ReLU graph node: same matrix shape as input. The eventual backward pass
// calls mat_relu_add_grad, accumulating only where input was positive.
ModelVar *mv_relu(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags)
{
    return mv_unary(arena, model, input, flags, MV_OP_RELU);
}

// Softmax graph node: the compact runtime normalizes the full Matrix, not
// independent batches/rows. For a logits-level fused loss see the separate
// mv_softmax_cross_entropy graph node.
ModelVar *mv_softmax(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags)
{
    return mv_unary(arena, model, input, flags, MV_OP_SOFTMAX);
}

// Elementwise addition: both input matrices must have identical dimensions;
// returns nullptr on mismatch, no implicit broadcasting. Backprop must add
// the incoming gradient to both parents (important for residual branches).
ModelVar *mv_add(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (!a || !b || !a->val || !b->val
        || a->val->rows != b->val->rows
        || a->val->cols != b->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD);
}

// Elementwise subtraction: same-shape input contract. In reverse mode the
// left parent receives +gradient and the right parent receives -gradient.
ModelVar *mv_sub(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (!a || !b || !a->val || !b->val
        || a->val->rows != b->val->rows
        || a->val->cols != b->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB);
}

// Matrix multiplication node: [M,K] times [K,N] produces [M,N]. Forward
// delegates to matrix.cpp::mat_mul; backward computes dA=dC*B^T and
// dB=A^T*dC, accumulating into existing gradient matrices.
ModelVar *mv_matmul(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (!a || !b || !a->val || !b->val
        || a->val->cols != b->val->rows)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL);
}

// Probability-level cross-entropy node: inputs have identical shape and the
// result retains that shape; main.cpp sums its elements for the sample cost.
// The target is typically not differentiable, but probabilities are.
ModelVar *mv_cross_entropy(
    MemArena *arena, ModelContext *model,
    ModelVar *p, ModelVar *q, u32 flags)
{
    if (!p || !q || !p->val || !q->val
        || p->val->rows != q->val->rows
        || p->val->cols != q->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY);
}

// Alternative fused logits-level loss: targets and logits have equal shapes;
// this builder returns a scalar [1,1] loss node instead of an elementwise
// loss matrix. The current MNIST main uses the separate softmax + CE path.
ModelVar *mv_softmax_cross_entropy(
    MemArena *arena, ModelContext *model,
    ModelVar *targets, ModelVar *logits, u32 flags)
{
    if (!targets || !logits || !targets->val || !logits->val
        || targets->val->rows != logits->val->rows
        || targets->val->cols != logits->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, targets, logits,
        1, 1,
        flags, MV_OP_SOFTMAX_CROSS_ENTROPY);
}

//======================
// Program creation (topological sort)
//======================

// Compile reachable graph nodes into an arena-owned dependency-first program
// using an explicit stack and visited/index bookkeeping. Forward execution
// assumes parents precede consumers; reverse-mode walks this order backwards.
// This routine returns an empty program on invalid input or allocation/stack
// failure, so clients must validate successful compilation before training.
ModelProgram model_prog_create(
    MemArena *arena, ModelContext *model, ModelVar *out_var)
{
    ModelProgram empty{};
    if (!arena || !model || !out_var || model->num_vars == 0)
        return empty;

    b32 *visited = push_array<b32>(arena, model->num_vars);
    b32 *added = push_array<b32>(arena, model->num_vars);
    if (!visited || !added)
        return empty;

    std::memset(visited, 0, sizeof(b32) * model->num_vars);
    std::memset(added, 0, sizeof(b32) * model->num_vars);

    const u32 max_stack = model->num_vars * 4;
    ModelVar **stack = push_array<ModelVar *>(arena, max_stack);
    ModelVar **out = push_array<ModelVar *>(arena, model->num_vars);
    if (!stack || !out)
        return empty;

    u32 stack_size = 0;
    u32 out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0)
    {
        ModelVar *cur = stack[--stack_size];
        if (!cur || cur->index >= model->num_vars)
            continue;

        if (visited[cur->index])
        {
            if (!added[cur->index])
            {
                out[out_size++] = cur;
                added[cur->index] = true;
            }
            continue;
        }

        visited[cur->index] = true;
        if (stack_size >= max_stack)
            return empty;
        stack[stack_size++] = cur;

        const u32 num_inputs = mv_num_inputs(cur->op);
        for (u32 i = 0; i < num_inputs; i++)
        {
            ModelVar *input = cur->inputs[i];
            if (!input || input->index >= model->num_vars || visited[input->index])
                continue;
            if (stack_size >= max_stack)
                return empty;
            stack[stack_size++] = input;
        }
    }

    ModelProgram prog{};
    prog.size = out_size;
    prog.vars = push_array<ModelVar *>(arena, out_size);
    if (!prog.vars)
        return empty;

    std::memcpy(prog.vars, out, sizeof(ModelVar *) * out_size);
    return prog;
}

//======================
// Forward
//======================

// Forward interpreter: each ModelVar::op dispatches to scalar matrix.cpp
// kernels, writing the node's preallocated value buffer. Inputs and params
// are already populated by main.cpp/training.cpp. This is NOT a KairoSIMD
// dispatcher, and this void API does not propagate matrix helper bool errors.
void model_prog_compute(ModelProgram *prog)
{
    if (!prog || !prog->vars)
        return;

    for (u32 i = 0; i < prog->size; i++)
    {
        ModelVar *cur = prog->vars[i];
        if (!cur)
            continue;

        ModelVar *a = cur->inputs[0];
        ModelVar *b = cur->inputs[1];

        switch (cur->op)
        {
        case MV_OP_NULL:
        case MV_OP_CREATE:
            break;

        case MV_OP_RELU:
            mat_relu(cur->val, a->val);
            break;

        case MV_OP_SOFTMAX:
            mat_softmax(cur->val, a->val);
            break;

        case MV_OP_ADD:
            mat_add(cur->val, a->val, b->val);
            break;

        case MV_OP_SUB:
            mat_sub(cur->val, a->val, b->val);
            break;

        case MV_OP_MATMUL:
            mat_mul(cur->val, a->val, b->val, true, false, false);
            break;

        case MV_OP_CROSS_ENTROPY:
            mat_cross_entropy(cur->val, a->val, b->val);
            break;

        case MV_OP_SOFTMAX_CROSS_ENTROPY:
            mat_softmax_cross_entropy(cur->val, a->val, b->val);
            break;
        }
    }
}

//======================
// Backward
//======================

// Reverse-mode autodiff: clear nonparameter intermediates, seed the final
// scalar/elementwise loss gradient with 1, then visit nodes in reverse order.
// Parameter gradients are deliberately NOT cleared here: training.cpp clears
// all gradients once per batch and calls this per sample to accumulate them.
// mat_mul(..., zero_out=false) adds matrix contributions; add/sub and nonlinear
// gradient helpers likewise accumulate through shared/residual graph edges.
void model_prog_compute_grads(ModelProgram *prog)
{
    if (!prog || !prog->vars || prog->size == 0)
        return;

    for (u32 i = 0; i < prog->size; i++)
    {
        ModelVar *cur = prog->vars[i];
        if (!cur || !(cur->flags & MV_FLAG_REQUIRES_GRAD))
            continue;
        if (cur->flags & MV_FLAG_PARAMETER)
            continue;
        mat_clear(cur->grad);
    }

    ModelVar *root = prog->vars[prog->size - 1];
    if (!root || !root->grad)
        return;
    mat_fill(root->grad, 1.0f);

    for (i32 i = static_cast<i32>(prog->size) - 1; i >= 0; i--)
    {
        ModelVar *cur = prog->vars[i];
        if (!cur || !(cur->flags & MV_FLAG_REQUIRES_GRAD))
            continue;

        ModelVar *a = cur->inputs[0];
        ModelVar *b = cur->inputs[1];

        switch (cur->op)
        {
        case MV_OP_RELU:
            if (a->flags & MV_FLAG_REQUIRES_GRAD)
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            break;

        case MV_OP_SOFTMAX:
            if (a->flags & MV_FLAG_REQUIRES_GRAD)
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            break;

        case MV_OP_ADD:
            if (a->flags & MV_FLAG_REQUIRES_GRAD)
                mat_add(a->grad, a->grad, cur->grad);

            if (b->flags & MV_FLAG_REQUIRES_GRAD)
                mat_add(b->grad, b->grad, cur->grad);
            break;

        case MV_OP_SUB:
            if (a->flags & MV_FLAG_REQUIRES_GRAD)
                mat_add(a->grad, a->grad, cur->grad);

            if (b->flags & MV_FLAG_REQUIRES_GRAD)
                mat_sub(b->grad, b->grad, cur->grad);
            break;

        case MV_OP_MATMUL:
            if (a->flags & MV_FLAG_REQUIRES_GRAD)
                mat_mul(a->grad, cur->grad, b->val, false, false, true);

            if (b->flags & MV_FLAG_REQUIRES_GRAD)
                mat_mul(b->grad, a->val, cur->grad, false, true, false);
            break;

        case MV_OP_CROSS_ENTROPY:
            mat_cross_entropy_add_grad(
                (a->flags & MV_FLAG_REQUIRES_GRAD) ? a->grad : nullptr,
                (b->flags & MV_FLAG_REQUIRES_GRAD) ? b->grad : nullptr,
                a->val, b->val, cur->grad);
            break;

        case MV_OP_SOFTMAX_CROSS_ENTROPY:
            mat_softmax_cross_entropy_add_grad(
                (a->flags & MV_FLAG_REQUIRES_GRAD) ? a->grad : nullptr,
                (b->flags & MV_FLAG_REQUIRES_GRAD) ? b->grad : nullptr,
                a->val, b->val, cur->grad);
            break;

        default:
            break;
        }
    }
}

//======================
// Compile + forward
//======================

// Build two independent reachable execution lists: one ending at the model
// output for inference, another ending at cost for training and gradients.
// training.cpp requires a nonempty cost_prog; callers should check it.
void model_compile(MemArena *arena, ModelContext *model)
{
    if (!arena || !model)
        return;

    if (model->output)
    {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    if (model->cost)
    {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

// Inference facade: execute the compiled output program using current input
// and parameter buffers. It does not update gradients or train weights.
void model_feedforward(ModelContext *model)
{
    if (!model)
        return;
    model_prog_compute(&model->forward_prog);
}
