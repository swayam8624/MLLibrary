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

ModelContext *model_create(MemArena *arena)
{
    return push_struct<ModelContext>(arena);
}

//======================
// Variable creation
//======================

ModelVar *mv_create(
    MemArena *arena, ModelContext *model,
    u32 rows, u32 cols, u32 flags)
{
    ModelVar *out = push_struct<ModelVar>(arena);

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;

    out->val = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD)
    {
        out->grad = mat_create(arena, rows, cols);
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

static ModelVar *mv_unary(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags, ModelVarOp op)
{
    if (input->flags & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar *out = mv_create(
        arena, model,
        input->val->rows,
        input->val->cols,
        flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

static ModelVar *mv_binary(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b,
    u32 rows, u32 cols,
    u32 flags, ModelVarOp op)
{
    if ((a->flags | b->flags) & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar *out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

//======================
// Operations
//======================

ModelVar *mv_relu(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags)
{
    return mv_unary(arena, model, input, flags, MV_OP_RELU);
}

ModelVar *mv_softmax(
    MemArena *arena, ModelContext *model,
    ModelVar *input, u32 flags)
{
    return mv_unary(arena, model, input, flags, MV_OP_SOFTMAX);
}

ModelVar *mv_add(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (a->val->rows != b->val->rows ||
        a->val->cols != b->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD);
}

ModelVar *mv_sub(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (a->val->rows != b->val->rows ||
        a->val->cols != b->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB);
}

ModelVar *mv_matmul(
    MemArena *arena, ModelContext *model,
    ModelVar *a, ModelVar *b, u32 flags)
{
    if (a->val->cols != b->val->rows)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL);
}

ModelVar *mv_cross_entropy(
    MemArena *arena, ModelContext *model,
    ModelVar *p, ModelVar *q, u32 flags)
{
    if (p->val->rows != q->val->rows ||
        p->val->cols != q->val->cols)
    {
        return nullptr;
    }

    return mv_binary(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY);
}

//======================
// Program creation (topological sort)
//======================

ModelProgram model_prog_create(
    MemArena *arena, ModelContext *model, ModelVar *out_var)
{
    b32 *visited = push_array<b32>(arena, model->num_vars);
    b32 *added = push_array<b32>(arena, model->num_vars); // Track what's in 'out'

    std::memset(visited, 0, sizeof(b32) * model->num_vars);
    std::memset(added, 0, sizeof(b32) * model->num_vars);

    // Increase stack capacity to prevent overflow from DAG edges
    u32 max_stack = model->num_vars * 4;
    ModelVar **stack = push_array<ModelVar *>(arena, max_stack);

    // Out size will now safely never exceed num_vars
    ModelVar **out = push_array<ModelVar *>(arena, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0)
    {
        ModelVar *cur = stack[--stack_size];

        if (cur->index >= model->num_vars)
            continue;

        if (visited[cur->index])
        {
            // Crucial Fix: Only add to 'out' once
            if (!added[cur->index])
            {
                out[out_size++] = cur;
                added[cur->index] = true;
            }
            continue;
        }

        visited[cur->index] = true;

        stack[stack_size++] = cur;

        u32 num_inputs = mv_num_inputs(cur->op);

        for (u32 i = 0; i < num_inputs; i++)
        {
            ModelVar *input = cur->inputs[i];

            if (input->index >= model->num_vars ||
                visited[input->index])
            {
                continue;
            }

            stack[stack_size++] = input;
        }
    }

    ModelProgram prog{};
    prog.size = out_size;
    prog.vars = push_array<ModelVar *>(arena, out_size);

    std::memcpy(prog.vars, out, sizeof(ModelVar *) * out_size);

    return prog;
}
//======================
// Forward
//======================

void model_prog_compute(ModelProgram *prog)
{
    for (u32 i = 0; i < prog->size; i++)
    {
        ModelVar *cur = prog->vars[i];

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
            mat_mul(cur->val, a->val, b->val, 1, 0, 0);
            break;

        case MV_OP_CROSS_ENTROPY:
            mat_cross_entropy(cur->val, a->val, b->val);
            break;
        }
    }
}

//======================
// Backward
//======================

void model_prog_compute_grads(ModelProgram *prog)
{
    for (u32 i = 0; i < prog->size; i++)
    {
        ModelVar *cur = prog->vars[i];

        if (!(cur->flags & MV_FLAG_REQUIRES_GRAD))
            continue;
        if (cur->flags & MV_FLAG_PARAMETER)
            continue;

        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    for (i32 i = (i32)prog->size - 1; i >= 0; i--)
    {
        ModelVar *cur = prog->vars[i];

        if (!(cur->flags & MV_FLAG_REQUIRES_GRAD))
            continue;

        ModelVar *a = cur->inputs[0];
        ModelVar *b = cur->inputs[1];

        switch (cur->op)
        {
        case MV_OP_RELU:
            mat_relu_add_grad(a->grad, a->val, cur->grad);
            break;

        case MV_OP_SOFTMAX:
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
                mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);

            if (b->flags & MV_FLAG_REQUIRES_GRAD)
                mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
            break;

        case MV_OP_CROSS_ENTROPY:
            mat_cross_entropy_add_grad(
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

void model_compile(MemArena *arena, ModelContext *model)
{
    if (model->output)
    {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    if (model->cost)
    {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

void model_feedforward(ModelContext *model)
{
    model_prog_compute(&model->forward_prog);
}
