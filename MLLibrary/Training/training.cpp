//
//  training.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#include "training.hpp"
#include "arena.h"
#include "prng.h"

#include <cstring>
#include <cstdio>
#include <cmath>

//======================
// Training
//======================

void model_train(
    ModelContext *model,
    const ModelTrainingDesc *desc)
{
    Matrix *train_images = desc->train_images;
    Matrix *train_labels = desc->train_labels;
    Matrix *test_images = desc->test_images;
    Matrix *test_labels = desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / desc->batch_size;

    //======================
    // Scratch memory
    //======================

    MemArena::Temp scratch = MemArena::scratch_get(nullptr, 0);
    u32 *order = push_array<u32>(scratch.arena(), num_examples, false);
    if (!order)
    {
        printf("Failed to allocate training order buffer\n");
        return;
    }

    Matrix **first_moment_by_var = nullptr;
    Matrix **second_moment_by_var = nullptr;
    const bool needs_first_moment = desc->optimizer == MODEL_OPTIMIZER_MOMENTUM_SGD
        || desc->optimizer == MODEL_OPTIMIZER_NESTEROV
        || desc->optimizer == MODEL_OPTIMIZER_ADAM
        || desc->optimizer == MODEL_OPTIMIZER_ADAMW;
    const bool needs_second_moment = desc->optimizer == MODEL_OPTIMIZER_RMSPROP
        || desc->optimizer == MODEL_OPTIMIZER_ADAM
        || desc->optimizer == MODEL_OPTIMIZER_ADAMW;
    if (needs_first_moment)
    {
        first_moment_by_var = push_array<Matrix *>(scratch.arena(), model->num_vars, true);
        if (!first_moment_by_var)
        {
            printf("Failed to allocate optimizer first-moment table\n");
            return;
        }
    }
    if (needs_second_moment)
    {
        second_moment_by_var = push_array<Matrix *>(scratch.arena(), model->num_vars, true);
        if (!second_moment_by_var)
        {
            printf("Failed to allocate optimizer second-moment table\n");
            return;
        }
    }

    for (u32 i = 0; i < model->cost_prog.size; i++)
    {
        ModelVar *cur = model->cost_prog.vars[i];
        if (!(cur->flags & MV_FLAG_PARAMETER))
            continue;
        if (needs_first_moment)
        {
            first_moment_by_var[cur->index] = mat_create(scratch.arena(), cur->val->rows, cur->val->cols);
            if (!first_moment_by_var[cur->index])
            {
                printf("Failed to allocate optimizer first-moment matrix\n");
                return;
            }
            mat_clear(first_moment_by_var[cur->index]);
        }
        if (needs_second_moment)
        {
            second_moment_by_var[cur->index] = mat_create(scratch.arena(), cur->val->rows, cur->val->cols);
            if (!second_moment_by_var[cur->index])
            {
                printf("Failed to allocate optimizer second-moment matrix\n");
                return;
            }
            mat_clear(second_moment_by_var[cur->index]);
        }
    }

    FILE *metrics_file = nullptr;
    if (desc->metrics_csv_path)
    {
        metrics_file = std::fopen(desc->metrics_csv_path, "w");
        if (metrics_file)
        {
            std::fprintf(metrics_file, "epoch,accuracy,cost,learning_rate\n");
            std::fflush(metrics_file);
        }
    }

    for (u32 i = 0; i < num_examples; i++)
    {
        order[i] = i;
    }

    //======================
    // Training loop
    //======================

    for (u32 epoch = 0; epoch < desc->epochs; epoch++)
    {

        // Shuffle
        for (u32 i = 0; i < num_examples; i++)
        {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = order[a];
            order[a] = order[b];
            order[b] = tmp;
        }

        // Batches
        for (u32 batch = 0; batch < num_batches; batch++)
        {

            // Clear parameter grads
            for (u32 i = 0; i < model->cost_prog.size; i++)
            {
                ModelVar *cur = model->cost_prog.vars[i];

                if (cur->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;

            for (u32 i = 0; i < desc->batch_size; i++)
            {
                u32 idx = order[batch * desc->batch_size + i];

                // Copy input
                std::memcpy(
                    model->input->val->data,
                    train_images->data + idx * input_size,
                    sizeof(f32) * input_size);

                // Copy label
                std::memcpy(
                    model->desired_output->val->data,
                    train_labels->data + idx * output_size,
                    sizeof(f32) * output_size);

                // Forward + backward
                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }

            avg_cost /= (f32)desc->batch_size;

            f32 gradient_scale = 1.0f / static_cast<f32>(desc->batch_size);
            if (desc->max_gradient_norm > 0.0f)
            {
                f32 squared_norm = 0.0f;
                for (u32 i = 0; i < model->cost_prog.size; i++)
                {
                    ModelVar *cur = model->cost_prog.vars[i];
                    if (!(cur->flags & MV_FLAG_PARAMETER)) continue;
                    const u64 size = static_cast<u64>(cur->grad->rows) * cur->grad->cols;
                    for (u64 j = 0; j < size; j++)
                    {
                        const f32 g = cur->grad->data[j] * gradient_scale;
                        squared_norm += g * g;
                    }
                }
                const f32 norm = std::sqrt(squared_norm);
                if (norm > desc->max_gradient_norm)
                    gradient_scale *= desc->max_gradient_norm / norm;
            }

            const u64 optimizer_step = static_cast<u64>(epoch) * num_batches + batch + 1;
            const f32 adam_bias1 = 1.0f - std::pow(desc->beta1, static_cast<f32>(optimizer_step));
            const f32 adam_bias2 = 1.0f - std::pow(desc->beta2, static_cast<f32>(optimizer_step));

            for (u32 i = 0; i < model->cost_prog.size; i++)
            {
                ModelVar *cur = model->cost_prog.vars[i];
                if (!(cur->flags & MV_FLAG_PARAMETER))
                    continue;
                const u64 size = static_cast<u64>(cur->val->rows) * cur->val->cols;
                Matrix *first = needs_first_moment ? first_moment_by_var[cur->index] : nullptr;
                Matrix *second = needs_second_moment ? second_moment_by_var[cur->index] : nullptr;
                for (u64 j = 0; j < size; j++)
                {
                    f32 gradient = cur->grad->data[j] * gradient_scale;
                    if (desc->optimizer == MODEL_OPTIMIZER_ADAM && desc->weight_decay != 0.0f)
                        gradient += desc->weight_decay * cur->val->data[j];

                    switch (desc->optimizer)
                    {
                    case MODEL_OPTIMIZER_SGD:
                        cur->val->data[j] -= desc->learning_rate * gradient;
                        break;
                    case MODEL_OPTIMIZER_MOMENTUM_SGD:
                        first->data[j] = desc->momentum * first->data[j] + gradient;
                        cur->val->data[j] -= desc->learning_rate * first->data[j];
                        break;
                    case MODEL_OPTIMIZER_NESTEROV:
                        first->data[j] = desc->momentum * first->data[j] + gradient;
                        cur->val->data[j] -= desc->learning_rate * (desc->momentum * first->data[j] + gradient);
                        break;
                    case MODEL_OPTIMIZER_RMSPROP:
                        second->data[j] = desc->beta2 * second->data[j]
                            + (1.0f - desc->beta2) * gradient * gradient;
                        cur->val->data[j] -= desc->learning_rate * gradient
                            / (std::sqrt(second->data[j]) + desc->epsilon);
                        break;
                    case MODEL_OPTIMIZER_ADAM:
                    case MODEL_OPTIMIZER_ADAMW:
                        first->data[j] = desc->beta1 * first->data[j] + (1.0f - desc->beta1) * gradient;
                        second->data[j] = desc->beta2 * second->data[j]
                            + (1.0f - desc->beta2) * gradient * gradient;
                        if (desc->optimizer == MODEL_OPTIMIZER_ADAMW && desc->weight_decay != 0.0f)
                            cur->val->data[j] *= 1.0f - desc->learning_rate * desc->weight_decay;
                        cur->val->data[j] -= desc->learning_rate * (first->data[j] / adam_bias1)
                            / (std::sqrt(second->data[j] / adam_bias2) + desc->epsilon);
                        break;
                    }
                }
            }

            printf(
                "Epoch %2u/%2u, Batch %4u/%4u, Cost: %.4f\r",
                epoch + 1, desc->epochs,
                batch + 1, num_batches,
                avg_cost);
            fflush(stdout);
        }

        printf("\n");

        //======================
        // Evaluation
        //======================

        u32 correct = 0;
        f32 avg_cost = 0.0f;

        for (u32 i = 0; i < num_tests; i++)
        {
            std::memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size);

            std::memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size);

            model_prog_compute(&model->cost_prog);

            avg_cost += mat_sum(model->cost->val);

            if (mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val))
            {
                correct++;
            }
        }

        avg_cost /= (f32)num_tests;

        printf(
            "Test: %u/%u (%.2f%%), Cost: %.4f\n",
            correct, num_tests,
            (f32)correct / num_tests * 100.0f,
            avg_cost);

        if (metrics_file)
        {
            std::fprintf(
                metrics_file,
                "%u,%.6f,%.6f,%.8f\n",
                epoch + 1,
                (f32)correct / num_tests,
                avg_cost,
                desc->learning_rate);
            std::fflush(metrics_file);
        }
    }

    if (metrics_file)
    {
        std::fclose(metrics_file);
    }
}
