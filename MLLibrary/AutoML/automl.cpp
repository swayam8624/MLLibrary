#include "automl.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mllibrary::automl {
namespace {

using mllibrary::data::ColumnProfile;
using mllibrary::data::ColumnType;
using mllibrary::data::DatasetProfileOptions;
using mllibrary::data::DatasetSchema;
using mllibrary::data::dataset_profile_to_json;
using mllibrary::data::profile_dataset;
using mllibrary::evaluation::CrossValidationConfig;
using mllibrary::experiment::EstimatorKind;
using mllibrary::preprocessing::NumericScaling;
using mllibrary::search::Hyperparameter;
using mllibrary::search::HyperparameterSearch;
using mllibrary::search::SearchConfig;
using mllibrary::search::metric_is_maximized;

std::string escape_json(const std::string& input)
{
    std::ostringstream output;
    for (unsigned char character : input)
    {
        switch (character)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20)
            {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned>(character) << std::dec;
            }
            else
            {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

const ColumnProfile& target_profile(
    const Dataset& dataset,
    const DatasetProfile& profile)
{
    const auto target = dataset.schema().target_column();
    if (!target)
        throw std::invalid_argument("AutoML requires one declared target column.");
    if (*target >= profile.columns.size())
        throw std::logic_error("Dataset profile does not contain the target column.");
    return profile.columns[*target];
}

TaskKind infer_task(
    const Dataset& dataset,
    const DatasetProfile& profile,
    const AutoMLConfig& config)
{
    const auto target_index = dataset.schema().target_column();
    if (!target_index)
        throw std::invalid_argument("AutoML requires one declared target column.");
    const ColumnType target_type = dataset.schema().columns[*target_index].type;
    const ColumnProfile& target = profile.columns[*target_index];

    if (config.task == TaskSelection::Classification)
        return TaskKind::Classification;
    if (config.task == TaskSelection::Regression)
        return TaskKind::Regression;

    if (target_type == ColumnType::Boolean
        || target_type == ColumnType::Categorical
        || target_type == ColumnType::Text)
    {
        return TaskKind::Classification;
    }

    const double ratio = target.non_missing_count == 0
        ? 1.0
        : static_cast<double>(target.unique_count)
            / static_cast<double>(target.non_missing_count);
    if (target.unique_count >= 2
        && target.unique_count <= config.numeric_classification_limit
        && ratio <= config.numeric_classification_ratio)
    {
        return TaskKind::Classification;
    }
    return TaskKind::Regression;
}

FoldStrategy select_fold_strategy(TaskKind task, const AutoMLConfig& config)
{
    if (!config.group_column.empty() && !config.time_column.empty())
        throw std::invalid_argument("AutoML cannot use group and time-series validation simultaneously.");
    if (!config.time_column.empty()) return FoldStrategy::TimeSeries;
    if (!config.group_column.empty()) return FoldStrategy::Group;
    return task == TaskKind::Classification
        ? FoldStrategy::Stratified
        : FoldStrategy::KFold;
}

std::size_t estimate_feature_count(
    const Dataset& dataset,
    const DatasetProfile& profile,
    std::vector<std::string>& warnings)
{
    const auto target = dataset.schema().target_column();
    std::size_t features = 0;
    for (std::size_t index = 0; index < profile.columns.size(); ++index)
    {
        if (target && index == *target) continue;
        const ColumnProfile& column = profile.columns[index];
        switch (column.type)
        {
        case ColumnType::Numeric:
        case ColumnType::Boolean:
            ++features;
            break;
        case ColumnType::Categorical:
            features += column.unique_count;
            if (column.missing_count != 0) ++features;
            if (column.unique_count > 128)
            {
                warnings.push_back(
                    "Categorical column '" + column.name + "' has "
                    + std::to_string(column.unique_count)
                    + " unique values and may create a wide one-hot feature block.");
            }
            break;
        case ColumnType::Text:
            warnings.push_back(
                "Text column '" + column.name
                + "' is excluded because text tokenization is not implemented yet.");
            break;
        }
    }
    return features;
}

AutoMLCandidate make_logistic(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed)
{
    AutoMLCandidate candidate;
    candidate.name = "logistic_regression";
    candidate.rationale = "Fast, interpretable binary baseline for scaled tabular features.";
    candidate.task = TaskKind::Classification;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::LogisticRegression;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::Standard;
    candidate.search_space.parameters = {
        {Hyperparameter::LearningRate, {0.01, 0.05, 0.1}},
        {Hyperparameter::Iterations, {250.0, 500.0, 1000.0}},
        {Hyperparameter::L2, {0.0, 0.0001, 0.01}},
    };
    return candidate;
}

AutoMLCandidate make_naive_bayes(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed)
{
    AutoMLCandidate candidate;
    candidate.name = "gaussian_naive_bayes";
    candidate.rationale = "Low-cost probabilistic baseline that often works well on small tabular datasets.";
    candidate.task = TaskKind::Classification;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::GaussianNaiveBayes;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::Standard;
    candidate.search_space.parameters = {
        {Hyperparameter::VarianceSmoothing, {1e-9, 1e-6, 1e-3}},
    };
    return candidate;
}

AutoMLCandidate make_tree(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed)
{
    AutoMLCandidate candidate;
    candidate.name = "decision_tree";
    candidate.rationale = "Non-linear baseline that handles interactions without requiring feature scaling.";
    candidate.task = TaskKind::Classification;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::DecisionTreeClassifier;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::None;
    candidate.search_space.parameters = {
        {Hyperparameter::MaxDepth, {4.0, 8.0, 16.0}},
        {Hyperparameter::MinSamplesSplit, {2.0, 4.0, 8.0}},
        {Hyperparameter::MinSamplesLeaf, {1.0, 2.0, 4.0}},
    };
    return candidate;
}

AutoMLCandidate make_forest(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed)
{
    AutoMLCandidate candidate;
    candidate.name = "random_forest";
    candidate.rationale = "Robust non-linear ensemble for mixed tabular features and feature interactions.";
    candidate.task = TaskKind::Classification;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::RandomForestClassifier;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::None;
    candidate.search_space.parameters = {
        {Hyperparameter::Trees, {50.0, 100.0, 200.0}},
        {Hyperparameter::MaxDepth, {8.0, 16.0}},
        {Hyperparameter::MinSamplesLeaf, {1.0, 2.0, 4.0}},
    };
    return candidate;
}

AutoMLCandidate make_knn(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed,
    std::size_t row_count)
{
    AutoMLCandidate candidate;
    candidate.name = "k_nearest_neighbors";
    candidate.rationale = "Local distance baseline for moderate-size datasets after standard scaling.";
    candidate.task = TaskKind::Classification;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::KNearestNeighbors;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::Standard;
    std::vector<double> neighbors{1.0, 3.0, 5.0, 9.0, 15.0};
    neighbors.erase(
        std::remove_if(neighbors.begin(), neighbors.end(),
            [row_count](double value)
            {
                return value > static_cast<double>(std::max<std::size_t>(1, row_count / 2));
            }),
        neighbors.end());
    if (neighbors.empty()) neighbors.push_back(1.0);
    candidate.search_space.parameters = {
        {Hyperparameter::Neighbors, std::move(neighbors)},
    };
    return candidate;
}

AutoMLCandidate make_linear_regression(
    PrimaryMetric metric,
    FoldStrategy strategy,
    std::uint32_t seed)
{
    AutoMLCandidate candidate;
    candidate.name = "linear_regression";
    candidate.rationale = "Current native regression baseline for scaled numeric and encoded tabular features.";
    candidate.task = TaskKind::Regression;
    candidate.primary_metric = metric;
    candidate.fold_strategy = strategy;
    candidate.estimator.kind = EstimatorKind::LinearRegression;
    candidate.estimator.seed = seed;
    candidate.pipeline.numeric_scaling = NumericScaling::Standard;
    candidate.search_space.parameters = {
        {Hyperparameter::LearningRate, {0.005, 0.01, 0.05}},
        {Hyperparameter::Iterations, {500.0, 1000.0, 2000.0}},
        {Hyperparameter::L2, {0.0, 0.0001, 0.01}},
    };
    return candidate;
}

void validate_config(const AutoMLConfig& config)
{
    if (config.folds < 2)
        throw std::invalid_argument("AutoML requires at least two folds.");
    if (config.maximum_total_trials == 0)
        throw std::invalid_argument("maximum_total_trials must be at least one.");
    if (!std::isfinite(config.maximum_seconds) || config.maximum_seconds < 0.0)
        throw std::invalid_argument("maximum_seconds must be finite and non-negative.");
    if (config.numeric_classification_limit < 2)
        throw std::invalid_argument("numeric_classification_limit must be at least two.");
    if (!std::isfinite(config.numeric_classification_ratio)
        || config.numeric_classification_ratio < 0.0
        || config.numeric_classification_ratio > 1.0)
    {
        throw std::invalid_argument("numeric_classification_ratio must be in [0, 1].");
    }
}

bool better_candidate(
    const AutoMLCandidateResult& left,
    const AutoMLCandidateResult& right,
    bool maximize)
{
    const auto* left_trial = left.search.best_trial();
    const auto* right_trial = right.search.best_trial();
    if (!left_trial) return false;
    if (!right_trial) return true;
    if (left_trial->score == right_trial->score)
        return left.candidate.name < right.candidate.name;
    return maximize
        ? left_trial->score > right_trial->score
        : left_trial->score < right_trial->score;
}

} // namespace

std::string AutoMLPlan::to_json() const
{
    std::ostringstream output;
    output << "{\n"
           << "  \"success\": " << (success ? "true" : "false") << ",\n"
           << "  \"error\": \"" << escape_json(error) << "\",\n"
           << "  \"task\": \"" << mllibrary::experiment::task_kind_name(task) << "\",\n"
           << "  \"primary_metric\": \""
           << mllibrary::evaluation::primary_metric_name(primary_metric) << "\",\n"
           << "  \"fold_strategy\": \""
           << mllibrary::evaluation::fold_strategy_name(fold_strategy) << "\",\n"
           << "  \"estimated_feature_count\": " << estimated_feature_count << ",\n"
           << "  \"warnings\": [";
    for (std::size_t index = 0; index < warnings.size(); ++index)
    {
        if (index != 0) output << ", ";
        output << "\"" << escape_json(warnings[index]) << "\"";
    }
    output << "],\n  \"candidates\": [\n";
    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        const AutoMLCandidate& candidate = candidates[index];
        output << "    {\"name\": \"" << escape_json(candidate.name)
               << "\", \"rationale\": \"" << escape_json(candidate.rationale)
               << "\", \"estimator\": \""
               << mllibrary::experiment::estimator_kind_name(candidate.estimator.kind)
               << "\", \"search_cardinality\": "
               << candidate.search_space.cardinality() << '}';
        if (index + 1 != candidates.size()) output << ',';
        output << '\n';
    }
    output << "  ],\n  \"profile\": " << dataset_profile_to_json(profile) << "}\n";
    return output.str();
}

const AutoMLCandidateResult* AutoMLResult::best_candidate() const noexcept
{
    if (!best_candidate_index || *best_candidate_index >= candidates.size())
        return nullptr;
    return &candidates[*best_candidate_index];
}

std::string AutoMLResult::to_json() const
{
    std::ostringstream output;
    output << "{\n"
           << "  \"success\": " << (success ? "true" : "false") << ",\n"
           << "  \"error\": \"" << escape_json(error) << "\",\n"
           << "  \"best_candidate_index\": ";
    if (best_candidate_index) output << *best_candidate_index; else output << "null";
    output << ",\n  \"plan\": " << plan.to_json() << ",\n"
           << "  \"candidate_results\": [\n";
    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        output << "    {\"candidate\": \""
               << escape_json(candidates[index].candidate.name)
               << "\", \"search\": " << candidates[index].search.to_json() << '}';
        if (index + 1 != candidates.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
    return output.str();
}

void AutoMLResult::write_json(const std::string& path) const
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path())
        std::filesystem::create_directories(destination.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot open AutoML report '" + path + "'.");
    output << to_json();
    if (!output)
        throw std::runtime_error("Failed while writing AutoML report '" + path + "'.");
}

AutoMLPlanner::AutoMLPlanner(AutoMLConfig config)
    : config_(std::move(config))
{
}

AutoMLPlan AutoMLPlanner::plan(const Dataset& dataset) const
{
    AutoMLPlan plan;
    try
    {
        validate_config(config_);
        plan.profile = profile_dataset(dataset, DatasetProfileOptions{});
        plan.warnings = plan.profile.warnings;
        plan.task = infer_task(dataset, plan.profile, config_);
        plan.fold_strategy = select_fold_strategy(plan.task, config_);
        plan.primary_metric = plan.task == TaskKind::Classification
            ? PrimaryMetric::MacroF1
            : PrimaryMetric::RootMeanSquaredError;

        if (!config_.group_column.empty()
            && dataset.schema().find_column(config_.group_column) == DatasetSchema::npos)
        {
            throw std::invalid_argument(
                "Group column '" + config_.group_column + "' does not exist.");
        }
        if (!config_.time_column.empty()
            && dataset.schema().find_column(config_.time_column) == DatasetSchema::npos)
        {
            throw std::invalid_argument(
                "Time column '" + config_.time_column + "' does not exist.");
        }

        const ColumnProfile& target = target_profile(dataset, plan.profile);
        if (target.missing_count != 0)
            throw std::invalid_argument("AutoML does not accept missing target values.");
        if (plan.task == TaskKind::Classification && target.unique_count < 2)
            throw std::invalid_argument("Classification requires at least two target classes.");
        if (plan.task == TaskKind::Regression && target.type != ColumnType::Numeric)
            throw std::invalid_argument("Regression requires a numeric target column.");

        plan.estimated_feature_count = estimate_feature_count(
            dataset,
            plan.profile,
            plan.warnings);
        if (plan.estimated_feature_count == 0)
            throw std::invalid_argument("No supported feature columns remain after preprocessing.");

        const std::uint32_t seed = static_cast<std::uint32_t>(config_.seed);
        if (plan.task == TaskKind::Classification)
        {
            if (target.unique_count == 2)
                plan.candidates.push_back(make_logistic(plan.primary_metric, plan.fold_strategy, seed));
            else
                plan.warnings.push_back("Logistic regression is skipped because the current adapter is binary-only.");

            plan.candidates.push_back(make_naive_bayes(plan.primary_metric, plan.fold_strategy, seed));
            plan.candidates.push_back(make_tree(plan.primary_metric, plan.fold_strategy, seed));
            if (config_.include_random_forest)
                plan.candidates.push_back(make_forest(plan.primary_metric, plan.fold_strategy, seed));
            if (config_.include_knn
                && dataset.row_count() <= 50000
                && plan.estimated_feature_count <= 1024)
            {
                plan.candidates.push_back(make_knn(
                    plan.primary_metric,
                    plan.fold_strategy,
                    seed,
                    dataset.row_count()));
            }
            else if (config_.include_knn)
            {
                plan.warnings.push_back(
                    "K-nearest neighbors is skipped because the dataset is too large or too wide for the current exact implementation.");
            }
        }
        else
        {
            plan.candidates.push_back(make_linear_regression(
                plan.primary_metric,
                plan.fold_strategy,
                seed));
            plan.warnings.push_back(
                "Only linear regression is currently available for regression; tree and forest regressors are not implemented yet.");
        }

        if (config_.maximum_candidates != 0
            && plan.candidates.size() > config_.maximum_candidates)
        {
            plan.candidates.resize(config_.maximum_candidates);
        }
        if (plan.candidates.size() > config_.maximum_total_trials)
        {
            plan.warnings.push_back(
                "Candidate count was reduced so every candidate receives at least one trial.");
            plan.candidates.resize(config_.maximum_total_trials);
        }
        if (plan.candidates.empty())
            throw std::invalid_argument("AutoML generated no eligible model candidates.");

        plan.success = true;
    }
    catch (const std::exception& error)
    {
        plan.success = false;
        plan.error = error.what();
    }
    return plan;
}

AutoMLRunner::AutoMLRunner(AutoMLConfig config)
    : config_(std::move(config))
{
}

AutoMLResult AutoMLRunner::run(const Dataset& dataset) const
{
    AutoMLResult result;
    result.plan = AutoMLPlanner(config_).plan(dataset);
    if (!result.plan.success)
    {
        result.error = result.plan.error;
        return result;
    }

    try
    {
        const std::size_t candidate_count = result.plan.candidates.size();
        const std::size_t base_trials = config_.maximum_total_trials / candidate_count;
        const std::size_t remainder = config_.maximum_total_trials % candidate_count;
        const double per_candidate_seconds = config_.maximum_seconds > 0.0
            ? config_.maximum_seconds / static_cast<double>(candidate_count)
            : 0.0;

        for (std::size_t index = 0; index < candidate_count; ++index)
        {
            const AutoMLCandidate& candidate = result.plan.candidates[index];
            SearchConfig search_config;
            search_config.strategy = config_.search_strategy;
            search_config.seed = config_.seed + index;
            search_config.space = candidate.search_space;
            search_config.evaluation.task = candidate.task;
            search_config.evaluation.estimator = candidate.estimator;
            search_config.evaluation.pipeline = candidate.pipeline;
            search_config.evaluation.strategy = candidate.fold_strategy;
            search_config.evaluation.primary_metric = candidate.primary_metric;
            search_config.evaluation.folds = config_.folds;
            search_config.evaluation.seed = config_.seed + index;
            search_config.evaluation.group_column = config_.group_column;
            search_config.evaluation.time_column = config_.time_column;
            search_config.evaluation.exclude_split_column = true;
            search_config.budget.maximum_trials = base_trials + (index < remainder ? 1 : 0);
            search_config.budget.maximum_seconds = per_candidate_seconds;
            search_config.budget.minimum_folds = std::min<std::size_t>(2, config_.folds);
            search_config.budget.reduction_factor = 3;
            if (!config_.output_directory.empty())
            {
                search_config.output_directory = (
                    std::filesystem::path(config_.output_directory) / candidate.name).string();
            }

            AutoMLCandidateResult candidate_result;
            candidate_result.candidate = candidate;
            candidate_result.search = HyperparameterSearch(search_config).run(dataset);
            result.candidates.push_back(std::move(candidate_result));
        }

        const bool maximize = metric_is_maximized(result.plan.primary_metric);
        for (std::size_t index = 0; index < result.candidates.size(); ++index)
        {
            if (!result.candidates[index].search.success)
                continue;
            if (!result.best_candidate_index
                || better_candidate(
                    result.candidates[index],
                    result.candidates[*result.best_candidate_index],
                    maximize))
            {
                result.best_candidate_index = index;
            }
        }

        if (!result.best_candidate_index)
        {
            result.error = "No AutoML candidate completed successfully.";
            return result;
        }
        result.success = true;
        if (!config_.output_directory.empty())
        {
            std::filesystem::create_directories(config_.output_directory);
            result.write_json(
                (std::filesystem::path(config_.output_directory) / "automl.json").string());
        }
    }
    catch (const std::exception& error)
    {
        result.success = false;
        result.error = error.what();
    }
    return result;
}

const char* task_selection_name(TaskSelection task) noexcept
{
    switch (task)
    {
    case TaskSelection::Auto: return "auto";
    case TaskSelection::Classification: return "classification";
    case TaskSelection::Regression: return "regression";
    }
    return "unknown";
}

} // namespace mllibrary::automl
