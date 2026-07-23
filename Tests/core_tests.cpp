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
#include "runtime_contracts.hpp"

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

    PCA pca(1);
    pca.fit({ { -2.0f, -2.0f }, { -1.0f, -1.0f }, { 1.0f, 1.0f }, { 2.0f, 2.0f } });
    const DenseTable projected = pca.transform({ { -2.0f, -2.0f }, { 2.0f, 2.0f } });
    if (!(std::fabs(projected[0][0]) > 2.0f && nearly_equal(projected[0][0] + projected[1][0], 0.0f, 1e-4f))) return false;

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
    if (bayes.predict({ -1.5f, -1.5f }) != 0 || bayes.predict({ 1.5f, 1.5f }) != 1) return false;

    LinearRegression regression(0.08f, 1500);
    regression.fit({ { 0.0f }, { 1.0f }, { 2.0f }, { 3.0f } }, { 1.0f, 3.0f, 5.0f, 7.0f });
    if (!nearly_equal(regression.predict({ 4.0f }), 9.0f, 0.02f)) return false;

    const DenseTable xorSamples = {
        { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
        { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }
    };
    const std::vector<int> xorLabels = { 10, 20, 20, 10, 10, 20, 20, 10 };
    DecisionTreeClassifier tree(4);
    tree.fit(xorSamples, xorLabels);
    if (tree.node_count() < 3) return false;
    for (std::size_t index = 0; index < xorSamples.size(); ++index)
        if (tree.predict(xorSamples[index]) != xorLabels[index]) return false;

    RandomForestClassifier firstForest(31, 6, 2, 1, 2, 12345u);
    RandomForestClassifier secondForest(31, 6, 2, 1, 2, 12345u);
    firstForest.fit(xorSamples, xorLabels);
    secondForest.fit(xorSamples, xorLabels);
    if (firstForest.tree_count() != 31 || secondForest.tree_count() != 31) return false;
    for (const auto& sample : xorSamples)
    {
        const int first = firstForest.predict(sample);
        if (first != secondForest.predict(sample) || (first != 10 && first != 20)) return false;
    }
    return true;
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

bool test_runtime_contracts_and_capability_selection()
{
    using namespace mllibrary::contracts;
    TensorDescriptor images{
        .shape = { 32, 28, 28, 1 },
        .strides = {},
        .dataType = DataType::Float32,
        .layout = TensorLayout::NHWC,
        .device = DeviceType::CPUScalar,
        .ownership = StorageOwnership::Owned
    };
    images.validate();
    if (!images.contiguous() || images.element_count() != 32u * 28u * 28u
        || images.byte_size() != images.element_count() * sizeof(float)) return false;

    DatasetFormat dataset{
        .version = DatasetFormat::CurrentVersion,
        .identifier = "mnist-batch",
        .samples = images,
        .labels = TensorDescriptor{
            .shape = { 32 },
            .strides = {},
            .dataType = DataType::Int64,
            .layout = TensorLayout::RowMajor,
            .device = DeviceType::CPUScalar,
            .ownership = StorageOwnership::Owned
        },
        .sampleCount = 32
    };
    dataset.validate();
    CheckpointFormat{}.validate();

    ModelCapabilityRegistry registry;
    registry.register_model({
        .modelID = "mnist.cpu",
        .task = ModelTask::Classification,
        .dataTypes = { DataType::Float32 },
        .devices = { DeviceType::CPUScalar },
        .maximumRank = 4,
        .maximumInputBytes = MiB(8),
        .supportsInference = true,
        .supportsTraining = true
    });
    registry.register_model({
        .modelID = "mnist.metal",
        .task = ModelTask::Classification,
        .dataTypes = { DataType::Float32, DataType::Float16 },
        .devices = { DeviceType::Metal },
        .maximumRank = 4,
        .maximumInputBytes = MiB(32),
        .supportsInference = true,
        .supportsTraining = false
    });
    if (registry.require("mnist.cpu").supportsTraining != true) return false;
    if (registry.select(ModelTask::Classification, DataType::Float16, DeviceType::Metal,
            false, 4, images.byte_size()).modelID != "mnist.metal") return false;

    bool rejected = false;
    try
    {
        TensorDescriptor invalid = images;
        invalid.shape = { 32, 28, 28 };
        invalid.validate();
    }
    catch (const ContractError& error)
    {
        rejected = error.code() == ContractErrorCode::UnsupportedLayout;
    }
    return rejected;
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
    if (!test_runtime_contracts_and_capability_selection()) {
        std::fputs("runtime contract test failed\n", stderr);
        return 1;
    }
    return 0;
}
