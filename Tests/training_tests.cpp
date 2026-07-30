#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "arena.h"
#include "base.h"
#include "matrix.hpp"
#include "model.hpp"
#include "Training/training.hpp"

namespace {

struct TinyTrainingFixture final {
    MemArena* arena = nullptr;
    ModelContext* model = nullptr;
    ModelVar* weights = nullptr;
    Matrix* images = nullptr;
    Matrix* labels = nullptr;
};

TinyTrainingFixture create_fixture()
{
    TinyTrainingFixture fixture;
    fixture.arena = MemArena::create(MiB(16), KiB(64));
    if (!fixture.arena) return fixture;

    fixture.model = model_create(fixture.arena);
    ModelVar* input = mv_create(fixture.arena, fixture.model, 1, 1, MV_FLAG_INPUT);
    fixture.weights = mv_create(
        fixture.arena,
        fixture.model,
        2,
        1,
        MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* logits = mv_matmul(fixture.arena, fixture.model, fixture.weights, input, 0);
    ModelVar* prediction = logits
        ? mv_softmax(fixture.arena, fixture.model, logits, MV_FLAG_OUTPUT)
        : nullptr;
    ModelVar* target = mv_create(
        fixture.arena, fixture.model, 2, 1, MV_FLAG_DESIRED_OUTPUT);
    ModelVar* cost = prediction
        ? mv_cross_entropy(fixture.arena, fixture.model, target, prediction, MV_FLAG_COST)
        : nullptr;

    fixture.images = mat_create(fixture.arena, 3, 1);
    fixture.labels = mat_create(fixture.arena, 3, 2);
    if (!fixture.model || !input || !fixture.weights || !logits || !prediction
        || !target || !cost || !fixture.images || !fixture.labels)
    {
        MemArena::destroy(fixture.arena);
        return {};
    }

    fixture.weights->val->data[0] = 0.1f;
    fixture.weights->val->data[1] = -0.1f;
    fixture.images->data[0] = 1.0f;
    fixture.images->data[1] = 2.0f;
    fixture.images->data[2] = -1.0f;
    fixture.labels->data[0] = 1.0f;
    fixture.labels->data[1] = 0.0f;
    fixture.labels->data[2] = 0.0f;
    fixture.labels->data[3] = 1.0f;
    fixture.labels->data[4] = 1.0f;
    fixture.labels->data[5] = 0.0f;
    model_compile(fixture.arena, fixture.model);
    return fixture;
}

ModelTrainingDesc make_descriptor(const TinyTrainingFixture& fixture)
{
    ModelTrainingDesc desc{};
    desc.train_images = fixture.images;
    desc.train_labels = fixture.labels;
    desc.test_images = fixture.images;
    desc.test_labels = fixture.labels;
    desc.epochs = 2;
    desc.batch_size = 2;
    desc.learning_rate = 0.05f;
    desc.optimizer = MODEL_OPTIMIZER_SGD;
    desc.seed = 123456u;
    return desc;
}

bool run_deterministic_training(
    std::array<float, 2>& weights,
    ModelTrainingResult& result)
{
    TinyTrainingFixture fixture = create_fixture();
    if (!fixture.arena) return false;
    ModelTrainingDesc desc = make_descriptor(fixture);
    result = model_train(fixture.model, &desc);
    weights = { fixture.weights->val->data[0], fixture.weights->val->data[1] };
    const bool passed = result.success
        && result.completed_epochs == 2
        && result.completed_steps == 4
        && std::isfinite(result.final_loss)
        && std::isfinite(result.final_accuracy);
    MemArena::destroy(fixture.arena);
    return passed;
}

bool test_partial_batches_and_deterministic_seed()
{
    std::array<float, 2> first_weights{};
    std::array<float, 2> second_weights{};
    ModelTrainingResult first_result;
    ModelTrainingResult second_result;
    if (!run_deterministic_training(first_weights, first_result)
        || !run_deterministic_training(second_weights, second_result))
        return false;

    return first_weights == second_weights
        && first_result.completed_steps == second_result.completed_steps
        && first_result.final_loss == second_result.final_loss
        && first_result.final_accuracy == second_result.final_accuracy;
}

bool test_invalid_descriptor_is_rejected_before_mutation()
{
    TinyTrainingFixture fixture = create_fixture();
    if (!fixture.arena) return false;
    const std::array<float, 2> before = {
        fixture.weights->val->data[0], fixture.weights->val->data[1]
    };
    ModelTrainingDesc desc = make_descriptor(fixture);
    desc.batch_size = 0;
    const ModelTrainingResult result = model_train(fixture.model, &desc);
    const std::array<float, 2> after = {
        fixture.weights->val->data[0], fixture.weights->val->data[1]
    };
    const bool passed = !result.success
        && result.completed_steps == 0
        && result.completed_epochs == 0
        && !result.error.empty()
        && before == after;
    MemArena::destroy(fixture.arena);
    return passed;
}

bool test_non_finite_batch_is_rejected_before_update()
{
    TinyTrainingFixture fixture = create_fixture();
    if (!fixture.arena) return false;
    fixture.images->data[0] = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 2> before = {
        fixture.weights->val->data[0], fixture.weights->val->data[1]
    };
    ModelTrainingDesc desc = make_descriptor(fixture);
    desc.epochs = 1;
    desc.batch_size = 3;
    desc.reject_non_finite = true;
    const ModelTrainingResult result = model_train(fixture.model, &desc);
    const std::array<float, 2> after = {
        fixture.weights->val->data[0], fixture.weights->val->data[1]
    };
    const bool passed = !result.success
        && result.completed_steps == 0
        && !result.error.empty()
        && before == after;
    MemArena::destroy(fixture.arena);
    return passed;
}

} // namespace

int main()
{
    if (!test_partial_batches_and_deterministic_seed())
    {
        std::fputs("partial-batch or deterministic training test failed\n", stderr);
        return 1;
    }
    if (!test_invalid_descriptor_is_rejected_before_mutation())
    {
        std::fputs("training validation test failed\n", stderr);
        return 1;
    }
    if (!test_non_finite_batch_is_rejected_before_update())
    {
        std::fputs("non-finite training rejection test failed\n", stderr);
        return 1;
    }
    return 0;
}
