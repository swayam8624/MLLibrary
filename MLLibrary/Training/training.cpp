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

    // MemArena::Temp scratch = nullptr; // placeholder (we'll fix below)

    // If you kept scratch system:
    // MemArena::Temp scratch = arena_scratch_get(NULL, 0);

    // For now assume you pass arena if needed
    u32 *order = new u32[num_examples];

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

            //======================
            // SGD update
            //======================

            for (u32 i = 0; i < model->cost_prog.size; i++)
            {
                ModelVar *cur = model->cost_prog.vars[i];

                if (!(cur->flags & MV_FLAG_PARAMETER))
                    continue;

                mat_scale(
                    cur->grad,
                    desc->learning_rate / desc->batch_size);

                mat_sub(cur->val, cur->val, cur->grad);
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
    }

    delete[] order;
}
