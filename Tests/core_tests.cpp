#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "arena.h"
#include "base.h"
#include "matrix.hpp"
#include "model.hpp"
#include "Training/training.hpp"
#include "Training/checkpoint.hpp"
#include "classical.hpp"

namespace {

bool nearly_equal(f32 a, f32 b, f32 tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

bool test_cross_entropy_propagates_without_target_gradient()
{
    MemArena *arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;

    ModelContext *model = model_create(arena);
    ModelVar *target = mv_create(arena, model, 2, 1, MV_FLAG_DESIRED_OUTPUT);
    ModelVar *logits = mv_create(arena, model, 2, 1, MV_FLAG_REQUIRES_GRAD);
    ModelVar *prediction = mv_softmax(arena, model, logits, MV_FLAG_OUTPUT);
    ModelVar *cost = mv_cross_entropy(arena, model, target, prediction, MV_FLAG_COST);
    if (!prediction || !cost) {
        MemArena::destroy(arena);
        return false;
    }

    target->val->data[0] = 1.0f;
    target->val->data[1] = 0.0f;
    logits->val->data[0] = 0.0f;
    logits->val->data[1] = 0.0f;
    model_compile(arena, model);
    model_prog_compute(&model->cost_prog);
    model_prog_compute_grads(&model->cost_prog);

    const bool passed = nearly_equal(prediction->val->data[0], 0.5f)
        && nearly_equal(prediction->grad->data[0], -0.5f)
        && nearly_equal(prediction->grad->data[1], 0.5f)
        && nearly_equal(logits->grad->data[0], -0.5f)
        && nearly_equal(logits->grad->data[1], 0.5f);
    MemArena::destroy(arena);
    return passed;
}

bool test_matmul_transpose_accumulates()
{
    MemArena *arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;
    Matrix *a = mat_create(arena, 2, 2);
    Matrix *b = mat_create(arena, 2, 1);
    Matrix *out = mat_create(arena, 2, 1);
    if (!a || !b || !out) {
        MemArena::destroy(arena);
        return false;
    }
    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;
    b->data[0] = 5.0f; b->data[1] = 6.0f;
    const bool passed = mat_mul(out, a, b, true, false, false)
        && nearly_equal(out->data[0], 17.0f)
        && nearly_equal(out->data[1], 39.0f);
    MemArena::destroy(arena);
    return passed;
}

bool test_each_optimizer_updates_a_trainable_logit()
{
    const ModelOptimizerKind optimizers[] = { MODEL_OPTIMIZER_SGD, MODEL_OPTIMIZER_MOMENTUM_SGD,
        MODEL_OPTIMIZER_NESTEROV, MODEL_OPTIMIZER_RMSPROP, MODEL_OPTIMIZER_ADAM, MODEL_OPTIMIZER_ADAMW };
    for (ModelOptimizerKind optimizer : optimizers) {
        MemArena *arena = MemArena::create(MiB(16), KiB(64));
        ModelContext *model = model_create(arena);
        mv_create(arena, model, 1, 1, MV_FLAG_INPUT);
        ModelVar *logits = mv_create(arena, model, 2, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
        ModelVar *prediction = mv_softmax(arena, model, logits, MV_FLAG_OUTPUT);
        ModelVar *target = mv_create(arena, model, 2, 1, MV_FLAG_DESIRED_OUTPUT);
        mv_cross_entropy(arena, model, target, prediction, MV_FLAG_COST);
        Matrix *images = mat_create(arena, 1, 1);
        Matrix *labels = mat_create(arena, 1, 2);
        labels->data[0] = 1.0f;
        labels->data[1] = 0.0f;
        mat_clear(logits->val);
        model_compile(arena, model);
        ModelTrainingDesc desc{};
        desc.train_images = images; desc.train_labels = labels;
        desc.test_images = images; desc.test_labels = labels;
        desc.epochs = 3; desc.batch_size = 1; desc.learning_rate = 0.05f;
        desc.optimizer = optimizer; desc.max_gradient_norm = 1.0f;
        desc.weight_decay = optimizer == MODEL_OPTIMIZER_ADAMW ? 0.01f : 0.0f;
        model_train(model, &desc);
        const bool updated = std::isfinite(logits->val->data[0]) && std::isfinite(logits->val->data[1])
            && logits->val->data[0] > logits->val->data[1];
        MemArena::destroy(arena);
        if (!updated) return false;
    }
    return true;
}

bool test_classical_preprocessing_and_learners()
{
    const DenseTable samples = { { -2.0f, -1.0f }, { -1.0f, -2.0f }, { 1.0f, 2.0f }, { 2.0f, 1.0f } };
    StandardScaler scaler;
    scaler.fit(samples);
    const DenseTable standardized = scaler.transform(samples);
    if (!nearly_equal(standardized[0][0] + standardized[3][0], 0.0f)) return false;

    KMeans clusters(2, 50);
    clusters.fit(samples);
    const std::size_t left = clusters.predict({ -1.5f, -1.5f });
    const std::size_t right = clusters.predict({ 1.5f, 1.5f });
    if (left == right) return false;

    LogisticRegression classifier(0.2f, 300, 0.001f);
    classifier.fit(samples, { 0, 0, 1, 1 });
    if (!(classifier.predict({ -1.5f, -1.5f }) == 0
        && classifier.predict({ 1.5f, 1.5f }) == 1
        && classifier.predict_probability({ 1.5f, 1.5f }) > 0.9f)) return false;

    KNearestNeighbors nearest(3);
    nearest.fit(samples, { 0, 0, 1, 1 });
    if (nearest.predict({ -1.5f, -1.5f }) != 0 || nearest.predict({ 1.5f, 1.5f }) != 1) return false;

    GaussianNaiveBayes bayes;
    bayes.fit(samples, { 0, 0, 1, 1 });
    return bayes.predict({ -1.5f, -1.5f }) == 0 && bayes.predict({ 1.5f, 1.5f }) == 1;
}

bool test_parameter_checkpoint_round_trip()
{
    const std::filesystem::path path = "/tmp/mllibrary-checkpoint-test.bin";
    std::filesystem::remove(path);
    MemArena *arena = MemArena::create(MiB(8), KiB(64));
    ModelContext* model = model_create(arena);
    ModelVar* parameter = mv_create(arena, model, 2, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* target = mv_create(arena, model, 2, 1, MV_FLAG_DESIRED_OUTPUT);
    mv_cross_entropy(arena, model, target, parameter, MV_FLAG_COST);
    model_compile(arena, model);
    parameter->val->data[0] = 1.25f;
    parameter->val->data[1] = -3.5f;
    std::string error;
    const bool saved = model_save_checkpoint(model, path.c_str(), &error);
    parameter->val->data[0] = 0.0f;
    parameter->val->data[1] = 0.0f;
    const bool loaded = model_load_checkpoint(model, path.c_str(), &error);
    const bool passed = saved && loaded && nearly_equal(parameter->val->data[0], 1.25f)
        && nearly_equal(parameter->val->data[1], -3.5f);
    std::filesystem::remove(path);
    MemArena::destroy(arena);
    return passed;
}

} // namespace

int main()
{
    if (!test_cross_entropy_propagates_without_target_gradient()) {
        std::fputs("cross entropy gradient test failed\n", stderr);
        return 1;
    }
    if (!test_matmul_transpose_accumulates()) {
        std::fputs("matrix multiply test failed\n", stderr);
        return 1;
    }
    if (!test_each_optimizer_updates_a_trainable_logit()) {
        std::fputs("optimizer update test failed\n", stderr);
        return 1;
    }
    if (!test_classical_preprocessing_and_learners()) {
        std::fputs("classical ML test failed\n", stderr);
        return 1;
    }
    if (!test_parameter_checkpoint_round_trip()) {
        std::fputs("checkpoint round-trip test failed\n", stderr);
        return 1;
    }
    return 0;
}
