#include <cstdio>
#include <cstring>
#include <cmath>

#include "arena.h"
#include "prng.h"
#include "base.h"

#include "matrix.hpp"
#include "model.hpp"
#include "training.hpp"
#include "data.hpp"

//======================
// Model definition
//======================

static void create_mnist_model(MemArena *arena, ModelContext *model)
{
    ModelVar *input = mv_create(arena, model, 784, 1, MV_FLAG_INPUT);

    ModelVar *W0 = mv_create(arena, model, 16, 784,
                             MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    ModelVar *W1 = mv_create(arena, model, 16, 16,
                             MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    ModelVar *W2 = mv_create(arena, model, 10, 16,
                             MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Xavier init
    f32 b0 = std::sqrt(6.0f / (784 + 16));
    f32 b1 = std::sqrt(6.0f / (16 + 16));
    f32 b2 = std::sqrt(6.0f / (16 + 10));

    mat_fill_rand(W0->val, -b0, b0);
    mat_fill_rand(W1->val, -b1, b1);
    mat_fill_rand(W2->val, -b2, b2);

    ModelVar *bias0 = mv_create(arena, model, 16, 1,
                                MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    ModelVar *bias1 = mv_create(arena, model, 16, 1,
                                MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    ModelVar *bias2 = mv_create(arena, model, 10, 1,
                                MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Layer 1
    ModelVar *z0 = mv_add(arena, model,
                          mv_matmul(arena, model, W0, input, 0),
                          bias0, 0);

    ModelVar *a0 = mv_relu(arena, model, z0, 0);

    // Layer 2 (residual style)
    ModelVar *z1 = mv_add(arena, model,
                          mv_matmul(arena, model, W1, a0, 0),
                          bias1, 0);

    ModelVar *a1 = mv_add(arena, model,
                          a0,
                          mv_relu(arena, model, z1, 0),
                          0);

    // Output
    ModelVar *z2 = mv_add(arena, model,
                          mv_matmul(arena, model, W2, a1, 0),
                          bias2, 0);

    ModelVar *output = mv_softmax(arena, model, z2, MV_FLAG_OUTPUT);

    // Target
    ModelVar *y = mv_create(arena, model, 10, 1, MV_FLAG_DESIRED_OUTPUT);

    // Loss
    mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
}

//======================
// Main
//======================

int main()
{
    //======================
    // Memory
    //======================

    MemArena *arena = MemArena::create(GiB(1), MiB(1));

    //======================
    // Load data
    //======================

    Matrix *train_images = mat_load(arena, 60000, 784, "train_images.mat");
    Matrix *test_images = mat_load(arena, 10000, 784, "test_images.mat");

    Matrix *train_labels_raw = mat_load(arena, 60000, 1, "train_labels.mat");
    Matrix *test_labels_raw = mat_load(arena, 10000, 1, "test_labels.mat");

    Matrix *train_labels = mat_create(arena, 60000, 10);
    Matrix *test_labels = mat_create(arena, 10000, 10);

    one_hot_encode(train_labels, train_labels_raw, 10);
    one_hot_encode(test_labels, test_labels_raw, 10);

    if (!train_images || !train_labels_raw ||
        !test_images || !test_labels_raw)
    {
        printf("❌ Failed to load data files\n");
        return 1;
    }

    //======================
    // Debug print
    //======================

    draw_mnist_digit(test_images->data);

    for (u32 i = 0; i < 10; i++)
    {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    //======================
    // Model
    //======================

    ModelContext *model = model_create(arena);
    create_mnist_model(arena, model);

    model_compile(arena, model);

    //======================
    // Pre-training check
    //======================

    std::memcpy(
        model->input->val->data,
        test_images->data,
        sizeof(f32) * 784);

    model_feedforward(model);

    printf("Pre-training output:\n");
    for (u32 i = 0; i < 10; i++)
    {
        printf("%.3f ", model->output->val->data[i]);
    }
    printf("\n\n");

    //======================
    // Training
    //======================

    ModelTrainingDesc desc{};
    desc.train_images = train_images;
    desc.train_labels = train_labels;
    desc.test_images = test_images;
    desc.test_labels = test_labels;
    desc.epochs = 10;
    desc.batch_size = 50;
    desc.learning_rate = 0.01f;

    model_train(model, &desc);

    //======================
    // Post-training check
    //======================

    std::memcpy(
        model->input->val->data,
        test_images->data,
        sizeof(f32) * 784);

    model_feedforward(model);

    printf("\nPost-training output:\n");
    for (u32 i = 0; i < 10; i++)
    {
        printf("%.3f ", model->output->val->data[i]);
    }
    printf("\n");

    //======================
    // Cleanup
    //======================

    MemArena::destroy(arena);

    return 0;
}
