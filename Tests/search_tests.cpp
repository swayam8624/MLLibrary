#include "search.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mllibrary::data;
using namespace mllibrary::evaluation;
using namespace mllibrary::experiment;
using namespace mllibrary::search;

namespace {

Dataset classification_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"label", ColumnType::Boolean, false, true},
    }};
    std::vector<DataRow> rows;
    for (int index = 0; index < 24; ++index)
    {
        const bool label = index >= 12;
        const double x = label
            ? 1.0 + static_cast<double>(index - 12) * 0.1
            : -2.1 + static_cast<double>(index) * 0.1;
        rows.push_back({x, label});
    }
    return Dataset(std::move(schema), std::move(rows), "memory://search-classification");
}

CrossValidationConfig base_classification_evaluation()
{
    CrossValidationConfig evaluation;
    evaluation.task = TaskKind::Classification;
    evaluation.estimator.kind = EstimatorKind::LogisticRegression;
    evaluation.strategy = FoldStrategy::Stratified;
    evaluation.primary_metric = PrimaryMetric::MacroF1;
    evaluation.folds = 3;
    evaluation.seed = 42;
    return evaluation;
}

bool test_grid_search_and_artifact()
{
    SearchConfig config;
    config.strategy = SearchStrategy::Grid;
    config.evaluation = base_classification_evaluation();
    config.space.parameters = {
        {Hyperparameter::LearningRate, {0.05, 0.2}},
        {Hyperparameter::Iterations, {100.0, 300.0}},
    };
    config.budget.maximum_trials = 10;
    const auto output = std::filesystem::temp_directory_path()
        / "mllibrary-search-grid";
    std::filesystem::remove_all(output);
    config.output_directory = output.string();

    const SearchResult result = HyperparameterSearch(config).run(classification_dataset());
    const bool ok = result.success
        && result.trials.size() == 4
        && result.best_trial() != nullptr
        && result.best_trial()->fold_budget == 3
        && std::filesystem::exists(output / "search.json")
        && std::filesystem::file_size(output / "search.json") > 0;
    std::filesystem::remove_all(output);
    return ok;
}

bool same_parameters(
    const std::vector<AppliedHyperparameter>& left,
    const std::vector<AppliedHyperparameter>& right)
{
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].parameter != right[index].parameter
            || left[index].value != right[index].value)
        {
            return false;
        }
    }
    return true;
}

bool test_random_search_is_deterministic()
{
    SearchConfig config;
    config.strategy = SearchStrategy::Random;
    config.evaluation = base_classification_evaluation();
    config.evaluation.estimator.kind = EstimatorKind::DecisionTreeClassifier;
    config.space.parameters = {
        {Hyperparameter::MaxDepth, {1.0, 2.0, 3.0}},
        {Hyperparameter::MinSamplesLeaf, {1.0, 2.0}},
    };
    config.budget.maximum_trials = 4;
    config.seed = 777;

    const SearchResult first = HyperparameterSearch(config).run(classification_dataset());
    const SearchResult second = HyperparameterSearch(config).run(classification_dataset());
    if (!first.success || !second.success || first.trials.size() != second.trials.size())
        return false;
    for (std::size_t index = 0; index < first.trials.size(); ++index)
    {
        if (!same_parameters(first.trials[index].parameters, second.trials[index].parameters)
            || first.trials[index].state != second.trials[index].state
            || first.trials[index].score != second.trials[index].score)
        {
            return false;
        }
    }
    return true;
}

bool test_failed_trials_do_not_abort_study()
{
    SearchConfig config;
    config.strategy = SearchStrategy::Grid;
    config.evaluation = base_classification_evaluation();
    config.evaluation.estimator.kind = EstimatorKind::DecisionTreeClassifier;
    config.space.parameters = {
        {Hyperparameter::MaxFeatures, {0.0, 99.0}},
    };
    config.budget.maximum_trials = 2;

    const SearchResult result = HyperparameterSearch(config).run(classification_dataset());
    std::size_t completed = 0;
    std::size_t failed = 0;
    for (const SearchTrial& trial : result.trials)
    {
        if (trial.state == TrialState::Completed) ++completed;
        if (trial.state == TrialState::Failed) ++failed;
    }
    return result.success && completed == 1 && failed == 1;
}

bool test_successive_halving_promotes_candidates()
{
    SearchConfig config;
    config.strategy = SearchStrategy::SuccessiveHalving;
    config.evaluation = base_classification_evaluation();
    config.evaluation.estimator.kind = EstimatorKind::DecisionTreeClassifier;
    config.evaluation.folds = 4;
    config.space.parameters = {
        {Hyperparameter::MaxDepth, {1.0, 2.0, 3.0, 4.0}},
    };
    config.budget.maximum_trials = 4;
    config.budget.minimum_folds = 2;
    config.budget.reduction_factor = 2;
    config.seed = 19;

    const SearchResult result = HyperparameterSearch(config).run(classification_dataset());
    bool saw_pruned = false;
    bool saw_full_budget = false;
    for (const SearchTrial& trial : result.trials)
    {
        saw_pruned = saw_pruned || trial.state == TrialState::Pruned;
        saw_full_budget = saw_full_budget
            || (trial.state == TrialState::Completed && trial.fold_budget == 4);
    }
    return result.success
        && result.trials.size() == 6
        && saw_pruned
        && saw_full_budget
        && result.best_trial()
        && result.best_trial()->fold_budget == 4;
}

bool test_validation_and_metric_direction()
{
    if (metric_is_maximized(PrimaryMetric::RootMeanSquaredError)) return false;
    if (!metric_is_maximized(PrimaryMetric::R2)) return false;

    bool rejected = false;
    try
    {
        SearchSpace space{{
            {Hyperparameter::Iterations, {0.0}},
        }};
        space.validate();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    return rejected;
}

} // namespace

int main()
{
    if (!test_grid_search_and_artifact())
    {
        std::fputs("grid search test failed\n", stderr);
        return 1;
    }
    if (!test_random_search_is_deterministic())
    {
        std::fputs("random search determinism test failed\n", stderr);
        return 1;
    }
    if (!test_failed_trials_do_not_abort_study())
    {
        std::fputs("failed-trial isolation test failed\n", stderr);
        return 1;
    }
    if (!test_successive_halving_promotes_candidates())
    {
        std::fputs("successive-halving test failed\n", stderr);
        return 1;
    }
    if (!test_validation_and_metric_direction())
    {
        std::fputs("search validation test failed\n", stderr);
        return 1;
    }
    return 0;
}
