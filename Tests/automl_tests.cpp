#include "automl.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mllibrary::automl;
using namespace mllibrary::data;
using namespace mllibrary::experiment;
using namespace mllibrary::search;

namespace {

Dataset binary_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"city", ColumnType::Categorical, false, false},
        {"note", ColumnType::Text, false, false},
        {"label", ColumnType::Boolean, false, true},
    }};
    std::vector<DataRow> rows;
    for (int index = 0; index < 30; ++index)
    {
        const bool label = index >= 15;
        rows.push_back({
            label ? 1.0 + index * 0.03 : -1.0 - index * 0.03,
            std::string(index % 2 == 0 ? "Kolkata" : "Delhi"),
            std::string("free form note ") + std::to_string(index),
            label,
        });
    }
    return Dataset(std::move(schema), std::move(rows), "memory://automl-binary");
}

Dataset multiclass_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"label", ColumnType::Categorical, false, true},
    }};
    std::vector<DataRow> rows;
    const char* labels[] = {"alpha", "beta", "gamma"};
    for (int index = 0; index < 30; ++index)
    {
        rows.push_back({
            static_cast<double>(index % 10) + static_cast<double>(index / 10) * 10.0,
            std::string(labels[index % 3]),
        });
    }
    return Dataset(std::move(schema), std::move(rows), "memory://automl-multiclass");
}

Dataset regression_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"target", ColumnType::Numeric, false, true},
    }};
    std::vector<DataRow> rows;
    for (int index = 0; index < 30; ++index)
    {
        const double x = static_cast<double>(index) / 10.0;
        rows.push_back({x, 2.0 * x + 1.0});
    }
    return Dataset(std::move(schema), std::move(rows), "memory://automl-regression");
}

Dataset numeric_classification_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"target", ColumnType::Numeric, false, true},
    }};
    std::vector<DataRow> rows;
    for (int index = 0; index < 40; ++index)
    {
        rows.push_back({
            static_cast<double>(index),
            index < 20 ? 0.0 : 1.0,
        });
    }
    return Dataset(std::move(schema), std::move(rows), "memory://automl-numeric-class");
}

bool has_candidate(const AutoMLPlan& plan, const std::string& name)
{
    return std::any_of(
        plan.candidates.begin(), plan.candidates.end(),
        [&](const AutoMLCandidate& candidate) { return candidate.name == name; });
}

bool has_warning_fragment(const AutoMLPlan& plan, const std::string& fragment)
{
    return std::any_of(
        plan.warnings.begin(), plan.warnings.end(),
        [&](const std::string& warning)
        {
            return warning.find(fragment) != std::string::npos;
        });
}

bool test_binary_plan()
{
    AutoMLConfig config;
    config.maximum_total_trials = 20;
    const AutoMLPlan plan = AutoMLPlanner(config).plan(binary_dataset());
    return plan.success
        && plan.task == TaskKind::Classification
        && plan.estimated_feature_count == 3
        && has_candidate(plan, "logistic_regression")
        && has_candidate(plan, "random_forest")
        && has_candidate(plan, "k_nearest_neighbors")
        && has_warning_fragment(plan, "Text column 'note'");
}

bool test_multiclass_skips_binary_logistic()
{
    AutoMLConfig config;
    config.maximum_total_trials = 20;
    const AutoMLPlan plan = AutoMLPlanner(config).plan(multiclass_dataset());
    return plan.success
        && plan.task == TaskKind::Classification
        && !has_candidate(plan, "logistic_regression")
        && has_candidate(plan, "gaussian_naive_bayes")
        && has_warning_fragment(plan, "binary-only");
}

bool test_regression_plan_is_honest()
{
    AutoMLConfig config;
    config.maximum_total_trials = 10;
    const AutoMLPlan plan = AutoMLPlanner(config).plan(regression_dataset());
    return plan.success
        && plan.task == TaskKind::Regression
        && plan.candidates.size() == 1
        && plan.candidates.front().name == "linear_regression"
        && has_warning_fragment(plan, "tree and forest regressors");
}

bool test_numeric_class_inference()
{
    AutoMLConfig config;
    config.maximum_total_trials = 10;
    config.numeric_classification_limit = 5;
    config.numeric_classification_ratio = 0.05;
    const AutoMLPlan plan = AutoMLPlanner(config).plan(
        numeric_classification_dataset());
    return plan.success && plan.task == TaskKind::Classification;
}

bool test_end_to_end_selection_and_artifacts()
{
    AutoMLConfig config;
    config.search_strategy = SearchStrategy::Random;
    config.folds = 3;
    config.seed = 42;
    config.maximum_candidates = 2;
    config.maximum_total_trials = 4;
    config.include_knn = false;
    config.include_random_forest = false;
    const auto output = std::filesystem::temp_directory_path()
        / "mllibrary-automl-test";
    std::filesystem::remove_all(output);
    config.output_directory = output.string();

    const AutoMLResult result = AutoMLRunner(config).run(binary_dataset());
    const bool ok = result.success
        && result.best_candidate() != nullptr
        && result.best_candidate()->search.best_trial() != nullptr
        && result.candidates.size() == 2
        && std::filesystem::exists(output / "automl.json")
        && std::filesystem::file_size(output / "automl.json") > 0
        && result.to_json().find("best_candidate_index") != std::string::npos;
    std::filesystem::remove_all(output);
    return ok;
}

bool test_missing_target_is_rejected()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"y", ColumnType::Numeric, false, false},
    }};
    Dataset dataset(std::move(schema), {{1.0, 2.0}, {2.0, 4.0}});
    const AutoMLPlan plan = AutoMLPlanner().plan(dataset);
    return !plan.success && plan.error.find("target") != std::string::npos;
}

} // namespace

int main()
{
    if (!test_binary_plan())
    {
        std::fputs("binary AutoML plan test failed\n", stderr);
        return 1;
    }
    if (!test_multiclass_skips_binary_logistic())
    {
        std::fputs("multiclass AutoML plan test failed\n", stderr);
        return 1;
    }
    if (!test_regression_plan_is_honest())
    {
        std::fputs("regression AutoML plan test failed\n", stderr);
        return 1;
    }
    if (!test_numeric_class_inference())
    {
        std::fputs("numeric-class inference test failed\n", stderr);
        return 1;
    }
    if (!test_end_to_end_selection_and_artifacts())
    {
        std::fputs("end-to-end AutoML test failed\n", stderr);
        return 1;
    }
    if (!test_missing_target_is_rejected())
    {
        std::fputs("missing-target AutoML test failed\n", stderr);
        return 1;
    }
    return 0;
}
