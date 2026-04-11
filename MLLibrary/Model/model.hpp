//
//  model.hpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//
#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch"

#include <cstdint>
#include "matrix.hpp"

class MemArena;

using u32 = std::uint32_t;
using b32 = std::int32_t;

//======================
// Flags
//======================

enum ModelVarFlags {
    MV_FLAG_NONE = 0,
    MV_FLAG_REQUIRES_GRAD  = (1 << 0),
    MV_FLAG_PARAMETER      = (1 << 1),
    MV_FLAG_INPUT          = (1 << 2),
    MV_FLAG_OUTPUT         = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST           = (1 << 5),
};

//======================
// Operations
//======================

enum ModelVarOp {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,

    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
};

constexpr u32 MODEL_VAR_MAX_INPUTS = 2;

inline u32 mv_num_inputs(ModelVarOp op) {
    return (op < _MV_OP_UNARY_START) ? 0 :
           (op < _MV_OP_BINARY_START ? 1 : 2);
}

//======================
// Core structures
//======================

struct ModelVar {
    u32 index;
    u32 flags;

    Matrix* val;
    Matrix* grad;

    ModelVarOp op;
    ModelVar* inputs[MODEL_VAR_MAX_INPUTS];
};

struct ModelProgram {
    ModelVar** vars;
    u32 size;
};

struct ModelContext {
    u32 num_vars = 0;

    ModelVar* input = nullptr;
    ModelVar* output = nullptr;
    ModelVar* desired_output = nullptr;
    ModelVar* cost = nullptr;

    ModelProgram forward_prog{};
    ModelProgram cost_prog{};
};

//======================
// API
//======================

ModelContext* model_create(MemArena* arena);
void model_compile(MemArena* arena, ModelContext* model);
void model_feedforward(ModelContext* model);

//======================
// Variable creation
//======================

ModelVar* mv_create(
    MemArena* arena, ModelContext* model,
    u32 rows, u32 cols, u32 flags
);

ModelVar* mv_relu(
    MemArena* arena, ModelContext* model,
    ModelVar* input, u32 flags
);

ModelVar* mv_softmax(
    MemArena* arena, ModelContext* model,
    ModelVar* input, u32 flags
);

ModelVar* mv_add(
    MemArena* arena, ModelContext* model,
    ModelVar* a, ModelVar* b, u32 flags
);

ModelVar* mv_sub(
    MemArena* arena, ModelContext* model,
    ModelVar* a, ModelVar* b, u32 flags
);

ModelVar* mv_matmul(
    MemArena* arena, ModelContext* model,
    ModelVar* a, ModelVar* b, u32 flags
);

ModelVar* mv_cross_entropy(
    MemArena* arena, ModelContext* model,
    ModelVar* p, ModelVar* q, u32 flags
);

//======================
// Execution
//======================

ModelProgram model_prog_create(
    MemArena* arena, ModelContext* model, ModelVar* out_var
);

void model_prog_compute(ModelProgram* prog);
void model_prog_compute_grads(ModelProgram* prog);
