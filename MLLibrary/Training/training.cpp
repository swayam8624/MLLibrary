//
//  training.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 12/04/26.
//

#include "training.hpp"
#include "arena.h"
#include "prng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

bool matrix_ready(const Matrix* matrix)
{
    return matrix && matrix->data && matrix->rows > 0 && matrix->cols > 0;
}

std::string validate_training_request(
    const ModelContext* model,
    const ModelTrainingDesc* desc)
{
    if (!model) return "Training requires a model.";
    if (!desc) return "Training requires a descriptor.";
    if (!model->input || !model->output || !model->desired_output || !model->cost)
        return "Training requires compiled input, output, target, and cost variables.";
    if (!matrix_ready(model->input->val) || !matrix_ready(model->output->val)
        || !matrix_ready(model->desired_output->val) || !matrix_ready(model->cost->val))
        return "Training model variables must own non-empty value matrices.";
    if (!model->cost_prog.vars || model->cost_prog.size == 0)
        return "Training requires a compiled cost program.";

    if (!matrix_ready(desc->train_images) || !matrix_ready(desc->train_labels)
        || !matrix_ready(desc->test_images) || !matrix_ready(desc->test_labels))
        return "Training and test matrices must be non-empty and own storage.";
    if (desc->train_images->rows != desc->train_labels->rows)
        return "Training image and label row counts must match.";
    if (desc->test_images->rows != desc->test_labels->rows)
        return "Test image and label row counts must match.";
    if (desc->train_images->cols != desc->test_images->cols)
        return "Training and test input feature counts must match.";
    if (desc->train_labels->cols != desc->test_labels->cols)
        return "Training and test label widths must match.";

    const u64 input_values = static_cast<u64>(model->input->val->rows) * model->input->val->cols;
    const u64 output_values = static_cast<u64>(model->desired_output->val->rows)
        * model->desired_output->val->cols;
    if (desc->train_images->cols != input_values)
        return "Dataset feature count does not match the model input size.";
    if (desc->train_labels->cols != output_values)
        return "Dataset label width does not match the model target size.";
    if (static_cast<u64>(model->output->val->rows) * model->output->val->cols != output_values)
        return "Model output and target sizes must match.";

    if (desc->epochs == 0) return "Training requires at least one epoch.";
    if (desc->batch_size == 0) return "Training batch size must be positive.";
    if (!(desc->learning_rate > 0.0f) || !std::isfinite(desc->learning_rate))
        return "Training learning rate must be positive and finite.";

    const int optimizer = static_cast<int>(desc->optimizer);
    if (optimizer < static_cast<int>(MODEL_OPTIMIZER_SGD)
        || optimizer > static_cast<int>(MODEL_OPTIMIZER_ADAMW))
        return "Training optimizer is not supported.";
    if (!(desc->momentum >= 0.0f && desc->momentum < 1.0f) || !std::isfinite(desc->momentum))
        return "Training momentum must be finite and in [0, 1).";
    if (!(desc->beta1 >= 0.0f && desc->beta1 < 1.0f) || !std::isfinite(desc->beta1))
        return "Training beta1 must be finite and in [0, 1).";
    if (!(desc->beta2 >= 0.0f && desc->beta2 < 1.0f) || !std::isfinite(desc->beta2))
        return "Training beta2 must be finite and in [0, 1).";
    if (!(desc->epsilon > 0.0f) || !std::isfinite(desc->epsilon))
        return "Training epsilon must be positive and finite.";
    if (desc->weight_decay < 0.0f || !std::isfinite(desc->weight_decay))
        return "Training weight decay must be non-negative and finite.";
    if (desc->max_gradient_norm < 0.0f || !std::isfinite(desc->max_gradient_norm))
        return "Training maximum gradient norm must be non-negative and finite.";
    if (desc->metrics_csv_path && !*desc->metrics_csv_path)
        return "Training metrics path must not be empty.";

    bool found_parameter = false;
    for (u32 index = 0; index < model->cost_prog.size; ++index)
    {
        const ModelVar* variable = model->cost_prog.vars[index];
        if (!variable || !matrix_ready(variable->val))
            return "Training cost program contains an invalid variable.";
        if ((variable->flags & MV_FLAG_REQUIRES_GRAD) && !matrix_ready(variable->grad))
            return "A differentiable training variable is missing gradient storage.";
        if (variable->flags & MV_FLAG_PARAMETER) found_parameter = true;
    }
    if (!found_parameter) return "Training requires at least one trainable parameter.";

    return {};
}

u32 bounded_random(PRNG& random, u32 bound)
{
    const u32 threshold = static_cast<u32>(-bound) % bound;
    for (;;)
    {
        const u32 value = random.rand();
        if (value >= threshold) return value % bound;
    }
}

} // namespace

ModelTrainingResult model_train(
    ModelContext* model,
    const ModelTrainingDesc* desc)
{
    ModelTrainingResult result;
    const std::string validation_error = validate_training_request(model, desc);
    if (!validation_error.empty())
    {
        result.error = validation_error;
        return result;
    }

    Matrix* train_images = desc->train_images;
    Matrix* train_labels = desc->train_labels;
    Matrix* test_images = desc->test_images;
    Matrix* test_labels = desc->test_labels;

    const u32 num_examples = train_images->rows;
    const u32 input_size = train_images->cols;
    const u32 output_size = train_labels->cols;
    const u32 num_tests = test_images->rows;
    const u32 num_batches = num_examples / desc->batch_size
        + static_cast<u32>(num_examples % desc->batch_size != 0);

    FILE* metrics_file = nullptr;
    auto fail = [&](std::string message)
    {
        if (metrics_file)
        {
            std::fclose(metrics_file);
            metrics_file = nullptr;
        }
        result.success = false;
        result.error = std::move(message);
        return result;
    };

    MemArena::Temp scratch = MemArena::scratch_get(nullptr, 0);
    if (!scratch.arena()) return fail("Failed to acquire training scratch memory.");

    u32* order = push_array<u32>(scratch.arena(), num_examples, false);
    if (!order) return fail("Failed to allocate the training order buffer.");

    Matrix** first_moment_by_var = nullptr;
    Matrix** second_moment_by_var = nullptr;
    const bool needs_first_moment = desc->optimizer == MODEL_OPTIMIZER_MOMENTUM_SGD
        || desc->optimizer == MODEL_OPTIMIZER_NESTEROV
        || desc->optimizer == MODEL_OPTIMIZER_ADAM
        || desc->optimizer == MODEL_OPTIMIZER_ADAMW;
    const bool needs_second_moment = desc->optimizer == MODEL_OPTIMIZER_RMSPROP
        || desc->optimizer == MODEL_OPTIMIZER_ADAM
        || desc->optimizer == MODEL_OPTIMIZER_ADAMW;

    if (needs_first_moment)
    {
        first_moment_by_var = push_array<Matrix*>(scratch.arena(), model->num_vars, true);
        if (!first_moment_by_var) return fail("Failed to allocate the optimizer first-moment table.");
    }
    if (needs_second_moment)
    {
        second_moment_by_var = push_array<Matrix*>(scratch.arena(), model->num_vars, true);
        if (!second_moment_by_var) return fail("Failed to allocate the optimizer second-moment table.");
    }

    for (u32 index = 0; index < model->cost_prog.size; ++index)
    {
        ModelVar* variable = model->cost_prog.vars[index];
        if (!(variable->flags & MV_FLAG_PARAMETER)) continue;
        if (needs_first_moment)
        {
            first_moment_by_var[variable->index] = mat_create(
                scratch.arena(), variable->val->rows, variable->val->cols);
            if (!first_moment_by_var[variable->index])
                return fail("Failed to allocate an optimizer first-moment matrix.");
        }
        if (needs_second_moment)
        {
            second_moment_by_var[variable->index] = mat_create(
                scratch.arena(), variable->val->rows, variable->val->cols);
            if (!second_moment_by_var[variable->index])
                return fail("Failed to allocate an optimizer second-moment matrix.");
        }
    }

    if (desc->metrics_csv_path)
    {
        metrics_file = std::fopen(desc->metrics_csv_path, "w");
        if (!metrics_file) return fail("Cannot open the training metrics CSV file.");
        if (std::fprintf(metrics_file, "epoch,accuracy,cost,learning_rate\n") < 0
            || std::fflush(metrics_file) != 0)
            return fail("Cannot initialize the training metrics CSV file.");
    }

    for (u32 index = 0; index < num_examples; ++index) order[index] = index;
    PRNG random(desc->seed, desc->seed ^ 0xda3e39cb94b95bdbULL);

    for (u32 epoch = 0; epoch < desc->epochs; ++epoch)
    {
        for (u32 remaining = num_examples; remaining > 1; --remaining)
        {
            const u32 selected = bounded_random(random, remaining);
            std::swap(order[remaining - 1], order[selected]);
        }

        for (u32 batch = 0; batch < num_batches; ++batch)
        {
            const u32 batch_start = batch * desc->batch_size;
            const u32 current_batch_size = std::min(
                desc->batch_size, num_examples - batch_start);

            for (u32 index = 0; index < model->cost_prog.size; ++index)
            {
                ModelVar* variable = model->cost_prog.vars[index];
                if (variable->flags & MV_FLAG_REQUIRES_GRAD) mat_clear(variable->grad);
            }

            double batch_cost = 0.0;
            for (u32 sample = 0; sample < current_batch_size; ++sample)
            {
                const u32 dataset_index = order[batch_start + sample];
                std::memcpy(
                    model->input->val->data,
                    train_images->data + static_cast<u64>(dataset_index) * input_size,
                    sizeof(f32) * input_size);
                std::memcpy(
                    model->desired_output->val->data,
                    train_labels->data + static_cast<u64>(dataset_index) * output_size,
                    sizeof(f32) * output_size);

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                const f32 sample_cost = mat_sum(model->cost->val);
                if (desc->reject_non_finite && !std::isfinite(sample_cost))
                    return fail("Training produced a non-finite batch loss before applying an update.");
                batch_cost += sample_cost;
            }

            const f32 average_cost = static_cast<f32>(
                batch_cost / static_cast<double>(current_batch_size));
            f32 gradient_scale = 1.0f / static_cast<f32>(current_batch_size);
            double squared_norm = 0.0;

            for (u32 index = 0; index < model->cost_prog.size; ++index)
            {
                ModelVar* variable = model->cost_prog.vars[index];
                if (!(variable->flags & MV_FLAG_PARAMETER)) continue;
                const u64 size = static_cast<u64>(variable->grad->rows) * variable->grad->cols;
                for (u64 element = 0; element < size; ++element)
                {
                    const f32 gradient = variable->grad->data[element] * gradient_scale;
                    if (desc->reject_non_finite && !std::isfinite(gradient))
                        return fail("Training produced a non-finite parameter gradient before applying an update.");
                    squared_norm += static_cast<double>(gradient) * gradient;
                }
            }

            if (desc->reject_non_finite && !std::isfinite(squared_norm))
                return fail("Training produced a non-finite global gradient norm.");
            if (desc->max_gradient_norm > 0.0f)
            {
                const double norm = std::sqrt(squared_norm);
                if (norm > desc->max_gradient_norm)
                    gradient_scale *= static_cast<f32>(desc->max_gradient_norm / norm);
            }

            const std::uint64_t optimizer_step = result.completed_steps + 1;
            const f32 adam_bias1 = 1.0f
                - std::pow(desc->beta1, static_cast<f32>(optimizer_step));
            const f32 adam_bias2 = 1.0f
                - std::pow(desc->beta2, static_cast<f32>(optimizer_step));

            auto compute_update = [&](
                ModelVar* variable,
                Matrix* first,
                Matrix* second,
                u64 element,
                f32& next_value,
                f32& next_first,
                f32& next_second)
            {
                f32 gradient = variable->grad->data[element] * gradient_scale;
                next_value = variable->val->data[element];
                next_first = first ? first->data[element] : 0.0f;
                next_second = second ? second->data[element] : 0.0f;

                if (desc->optimizer == MODEL_OPTIMIZER_ADAM && desc->weight_decay != 0.0f)
                    gradient += desc->weight_decay * next_value;

                switch (desc->optimizer)
                {
                case MODEL_OPTIMIZER_SGD:
                    next_value -= desc->learning_rate * gradient;
                    break;
                case MODEL_OPTIMIZER_MOMENTUM_SGD:
                    next_first = desc->momentum * next_first + gradient;
                    next_value -= desc->learning_rate * next_first;
                    break;
                case MODEL_OPTIMIZER_NESTEROV:
                    next_first = desc->momentum * next_first + gradient;
                    next_value -= desc->learning_rate
                        * (desc->momentum * next_first + gradient);
                    break;
                case MODEL_OPTIMIZER_RMSPROP:
                    next_second = desc->beta2 * next_second
                        + (1.0f - desc->beta2) * gradient * gradient;
                    next_value -= desc->learning_rate * gradient
                        / (std::sqrt(next_second) + desc->epsilon);
                    break;
                case MODEL_OPTIMIZER_ADAM:
                case MODEL_OPTIMIZER_ADAMW:
                    next_first = desc->beta1 * next_first
                        + (1.0f - desc->beta1) * gradient;
                    next_second = desc->beta2 * next_second
                        + (1.0f - desc->beta2) * gradient * gradient;
                    if (desc->optimizer == MODEL_OPTIMIZER_ADAMW
                        && desc->weight_decay != 0.0f)
                        next_value *= 1.0f - desc->learning_rate * desc->weight_decay;
                    next_value -= desc->learning_rate * (next_first / adam_bias1)
                        / (std::sqrt(next_second / adam_bias2) + desc->epsilon);
                    break;
                }

                return std::isfinite(next_value)
                    && (!first || std::isfinite(next_first))
                    && (!second || std::isfinite(next_second));
            };

            if (desc->reject_non_finite)
            {
                for (u32 index = 0; index < model->cost_prog.size; ++index)
                {
                    ModelVar* variable = model->cost_prog.vars[index];
                    if (!(variable->flags & MV_FLAG_PARAMETER)) continue;
                    Matrix* first = needs_first_moment ? first_moment_by_var[variable->index] : nullptr;
                    Matrix* second = needs_second_moment ? second_moment_by_var[variable->index] : nullptr;
                    const u64 size = static_cast<u64>(variable->val->rows) * variable->val->cols;
                    for (u64 element = 0; element < size; ++element)
                    {
                        f32 next_value = 0.0f;
                        f32 next_first = 0.0f;
                        f32 next_second = 0.0f;
                        if (!compute_update(variable, first, second, element,
                                next_value, next_first, next_second))
                            return fail("Training produced a non-finite optimizer update; the batch was not applied.");
                    }
                }
            }

            for (u32 index = 0; index < model->cost_prog.size; ++index)
            {
                ModelVar* variable = model->cost_prog.vars[index];
                if (!(variable->flags & MV_FLAG_PARAMETER)) continue;
                Matrix* first = needs_first_moment ? first_moment_by_var[variable->index] : nullptr;
                Matrix* second = needs_second_moment ? second_moment_by_var[variable->index] : nullptr;
                const u64 size = static_cast<u64>(variable->val->rows) * variable->val->cols;
                for (u64 element = 0; element < size; ++element)
                {
                    f32 next_value = 0.0f;
                    f32 next_first = 0.0f;
                    f32 next_second = 0.0f;
                    compute_update(variable, first, second, element,
                        next_value, next_first, next_second);
                    variable->val->data[element] = next_value;
                    if (first) first->data[element] = next_first;
                    if (second) second->data[element] = next_second;
                }
            }
            ++result.completed_steps;

            std::printf(
                "Epoch %2u/%2u, Batch %4u/%4u, Cost: %.4f\r",
                epoch + 1, desc->epochs,
                batch + 1, num_batches,
                average_cost);
            std::fflush(stdout);
        }

        std::printf("\n");

        u32 correct = 0;
        double evaluation_cost = 0.0;
        for (u32 sample = 0; sample < num_tests; ++sample)
        {
            std::memcpy(
                model->input->val->data,
                test_images->data + static_cast<u64>(sample) * input_size,
                sizeof(f32) * input_size);
            std::memcpy(
                model->desired_output->val->data,
                test_labels->data + static_cast<u64>(sample) * output_size,
                sizeof(f32) * output_size);

            model_prog_compute(&model->cost_prog);
            const f32 sample_cost = mat_sum(model->cost->val);
            if (desc->reject_non_finite && !std::isfinite(sample_cost))
                return fail("Evaluation produced a non-finite loss.");
            evaluation_cost += sample_cost;

            if (mat_argmax(model->output->val)
                == mat_argmax(model->desired_output->val))
                ++correct;
        }

        result.final_loss = static_cast<f32>(
            evaluation_cost / static_cast<double>(num_tests));
        result.final_accuracy = static_cast<f32>(correct)
            / static_cast<f32>(num_tests);
        result.completed_epochs = epoch + 1;

        std::printf(
            "Test: %u/%u (%.2f%%), Cost: %.4f\n",
            correct, num_tests,
            result.final_accuracy * 100.0f,
            result.final_loss);

        if (metrics_file)
        {
            if (std::fprintf(
                    metrics_file,
                    "%u,%.6f,%.6f,%.8f\n",
                    epoch + 1,
                    result.final_accuracy,
                    result.final_loss,
                    desc->learning_rate) < 0
                || std::fflush(metrics_file) != 0)
                return fail("Writing the training metrics CSV file failed.");
        }
    }

    if (metrics_file)
    {
        if (std::fclose(metrics_file) != 0)
        {
            metrics_file = nullptr;
            return fail("Closing the training metrics CSV file failed.");
        }
        metrics_file = nullptr;
    }

    result.success = true;
    return result;
}
