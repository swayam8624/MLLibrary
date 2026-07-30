#include <cmath>
#include <cstdio>
#include <vector>

#include "arena.h"
#include "base.h"
#include "matrix.hpp"
#include "model.hpp"

namespace {

constexpr f32 finite_difference_step = 1e-3f;
constexpr f32 gradient_tolerance = 4e-3f;

bool nearly_equal(f32 a, f32 b, f32 tolerance = gradient_tolerance)
{
    const f32 scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= tolerance * scale;
}

void clear_program_gradients(ModelProgram& program)
{
    for (u32 index = 0; index < program.size; ++index)
    {
        ModelVar* variable = program.vars[index];
        if (variable && variable->grad) mat_clear(variable->grad);
    }
}

bool gradient_matches_finite_difference(
    ModelProgram& program,
    ModelVar* objective,
    ModelVar* variable,
    const char* label)
{
    if (!objective || !objective->val || !variable || !variable->val || !variable->grad)
        return false;

    clear_program_gradients(program);
    model_prog_compute(&program);
    model_prog_compute_grads(&program);

    const u64 count = static_cast<u64>(variable->val->rows) * variable->val->cols;
    std::vector<f32> analytic(count);
    for (u64 index = 0; index < count; ++index)
        analytic[index] = variable->grad->data[index];

    for (u64 index = 0; index < count; ++index)
    {
        const f32 original = variable->val->data[index];

        variable->val->data[index] = original + finite_difference_step;
        model_prog_compute(&program);
        const f32 plus = mat_sum(objective->val);

        variable->val->data[index] = original - finite_difference_step;
        model_prog_compute(&program);
        const f32 minus = mat_sum(objective->val);

        variable->val->data[index] = original;
        const f32 numerical = (plus - minus) / (2.0f * finite_difference_step);
        if (!nearly_equal(analytic[index], numerical))
        {
            std::fprintf(
                stderr,
                "%s gradient mismatch at %llu: analytic=%g numerical=%g\n",
                label,
                static_cast<unsigned long long>(index),
                static_cast<double>(analytic[index]),
                static_cast<double>(numerical));
            return false;
        }
    }

    model_prog_compute(&program);
    return true;
}

bool test_composed_elementwise_and_matmul_gradients()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* x = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* bias = mv_create(arena, model, 3, 1, MV_FLAG_NONE);
    ModelVar* weights = mv_create(arena, model, 1, 3, MV_FLAG_NONE);
    if (!model || !x || !bias || !weights)
    {
        MemArena::destroy(arena);
        return false;
    }

    x->val->data[0] = -0.7f;
    x->val->data[1] = 0.4f;
    x->val->data[2] = 1.2f;
    bias->val->data[0] = 0.2f;
    bias->val->data[1] = -0.1f;
    bias->val->data[2] = 0.3f;
    weights->val->data[0] = 0.5f;
    weights->val->data[1] = -1.2f;
    weights->val->data[2] = 0.7f;

    ModelVar* shifted = mv_add(arena, model, x, bias, MV_FLAG_NONE);
    ModelVar* activated = mv_relu(arena, model, shifted, MV_FLAG_NONE);
    ModelVar* adjusted = mv_sub(arena, model, activated, bias, MV_FLAG_NONE);
    ModelVar* objective = mv_matmul(arena, model, weights, adjusted, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    const bool passed = program.size > 0
        && gradient_matches_finite_difference(program, objective, x, "add/sub/relu/matmul");
    MemArena::destroy(arena);
    return passed;
}

bool test_softmax_jacobian_vector_product()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* logits = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* weights = mv_create(arena, model, 1, 3, MV_FLAG_NONE);
    if (!model || !logits || !weights)
    {
        MemArena::destroy(arena);
        return false;
    }

    logits->val->data[0] = 0.3f;
    logits->val->data[1] = -0.8f;
    logits->val->data[2] = 1.1f;
    weights->val->data[0] = 0.7f;
    weights->val->data[1] = -1.1f;
    weights->val->data[2] = 0.4f;

    ModelVar* probabilities = mv_softmax(arena, model, logits, MV_FLAG_NONE);
    ModelVar* objective = mv_matmul(arena, model, weights, probabilities, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    model_prog_compute(&program);
    const bool normalized = probabilities
        && nearly_equal(mat_sum(probabilities->val), 1.0f, 1e-5f);
    const bool passed = normalized && program.size > 0
        && gradient_matches_finite_difference(program, objective, logits, "softmax JVP");
    MemArena::destroy(arena);
    return passed;
}

bool test_probability_cross_entropy_gradient()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* targets = mv_create(arena, model, 3, 1, MV_FLAG_NONE);
    ModelVar* probabilities = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (!model || !targets || !probabilities)
    {
        MemArena::destroy(arena);
        return false;
    }

    targets->val->data[0] = 0.0f;
    targets->val->data[1] = 1.0f;
    targets->val->data[2] = 0.0f;
    probabilities->val->data[0] = 0.2f;
    probabilities->val->data[1] = 0.5f;
    probabilities->val->data[2] = 0.3f;

    ModelVar* objective = mv_cross_entropy(
        arena, model, targets, probabilities, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    const bool passed = program.size > 0
        && gradient_matches_finite_difference(
            program, objective, probabilities, "probability cross entropy");
    MemArena::destroy(arena);
    return passed;
}

bool test_composed_softmax_cross_entropy_gradient()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* targets = mv_create(arena, model, 3, 1, MV_FLAG_NONE);
    ModelVar* logits = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (!model || !targets || !logits)
    {
        MemArena::destroy(arena);
        return false;
    }

    targets->val->data[0] = 0.0f;
    targets->val->data[1] = 1.0f;
    targets->val->data[2] = 0.0f;
    logits->val->data[0] = 0.4f;
    logits->val->data[1] = -0.2f;
    logits->val->data[2] = 1.0f;

    ModelVar* probabilities = mv_softmax(arena, model, logits, MV_FLAG_NONE);
    ModelVar* objective = mv_cross_entropy(
        arena, model, targets, probabilities, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    const bool passed = program.size > 0
        && gradient_matches_finite_difference(
            program, objective, logits, "composed softmax cross entropy");
    MemArena::destroy(arena);
    return passed;
}

bool test_fused_softmax_cross_entropy_gradient()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* targets = mv_create(arena, model, 3, 1, MV_FLAG_NONE);
    ModelVar* logits = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (!model || !targets || !logits)
    {
        MemArena::destroy(arena);
        return false;
    }

    targets->val->data[0] = 0.0f;
    targets->val->data[1] = 1.0f;
    targets->val->data[2] = 0.0f;
    logits->val->data[0] = 0.4f;
    logits->val->data[1] = -0.2f;
    logits->val->data[2] = 1.0f;

    ModelVar* objective = mv_softmax_cross_entropy(
        arena, model, targets, logits, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    const bool passed = objective && objective->val->rows == 1 && objective->val->cols == 1
        && program.size > 0
        && gradient_matches_finite_difference(
            program, objective, logits, "fused softmax cross entropy");
    MemArena::destroy(arena);
    return passed;
}

bool test_shared_residual_fanout_accumulates_gradients()
{
    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext* model = model_create(arena);
    ModelVar* x = mv_create(
        arena, model, 3, 1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* weights = mv_create(arena, model, 1, 3, MV_FLAG_NONE);
    if (!model || !x || !weights)
    {
        MemArena::destroy(arena);
        return false;
    }

    x->val->data[0] = 0.2f;
    x->val->data[1] = 0.8f;
    x->val->data[2] = 1.4f;
    weights->val->data[0] = 0.3f;
    weights->val->data[1] = -0.6f;
    weights->val->data[2] = 1.1f;

    ModelVar* branch = mv_relu(arena, model, x, MV_FLAG_NONE);
    ModelVar* residual = mv_add(arena, model, branch, branch, MV_FLAG_NONE);
    ModelVar* objective = mv_matmul(arena, model, weights, residual, MV_FLAG_NONE);
    ModelProgram program = model_prog_create(arena, model, objective);

    const bool passed = program.size > 0
        && gradient_matches_finite_difference(program, objective, x, "shared residual fanout");
    MemArena::destroy(arena);
    return passed;
}

} // namespace

int main()
{
    if (!test_composed_elementwise_and_matmul_gradients())
    {
        std::fputs("composed elementwise/matmul gradient test failed\n", stderr);
        return 1;
    }
    if (!test_softmax_jacobian_vector_product())
    {
        std::fputs("softmax Jacobian-vector product test failed\n", stderr);
        return 1;
    }
    if (!test_probability_cross_entropy_gradient())
    {
        std::fputs("probability cross entropy gradient test failed\n", stderr);
        return 1;
    }
    if (!test_composed_softmax_cross_entropy_gradient())
    {
        std::fputs("composed softmax cross entropy test failed\n", stderr);
        return 1;
    }
    if (!test_fused_softmax_cross_entropy_gradient())
    {
        std::fputs("fused softmax cross entropy test failed\n", stderr);
        return 1;
    }
    if (!test_shared_residual_fanout_accumulates_gradients())
    {
        std::fputs("shared residual fanout gradient test failed\n", stderr);
        return 1;
    }
    return 0;
}
