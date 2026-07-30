#include "cross_validation.hpp"
#include "metrics.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using namespace mllibrary::evaluation;
using namespace mllibrary::experiment;
using namespace mllibrary::preprocessing;
using namespace mllibrary::data;

namespace {

bool nearly_equal(double left, double right, double tolerance = 1e-9)
{
    return std::fabs(left - right) <= tolerance;
}

bool test_classification_metrics()
{
    const ClassificationMetrics metrics = calculate_classification_metrics(
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 1.0, 1.0},
        {0.1, 0.8, 0.9, 0.7});

    return metrics.sample_count == 4
        && nearly_equal(metrics.accuracy, 0.75)
        && metrics.confusion.labels == std::vector<int>({0, 1})
        && metrics.confusion.at(0, 0) == 1
        && metrics.confusion.at(0, 1) == 1
        && metrics.confusion.at(1, 0) == 0
        && metrics.confusion.at(1, 1) == 2
        && nearly_equal(metrics.macro_precision, 5.0 / 6.0)
        && nearly_equal(metrics.macro_recall, 0.75)
        && nearly_equal(metrics.macro_f1, 11.0 / 15.0)
        && nearly_equal(metrics.weighted_f1, 11.0 / 15.0)
        && metrics.log_loss.has_value()
        && metrics.roc_auc.has_value()
        && nearly_equal(*metrics.roc_auc, 0.75);
}

bool test_regression_metrics()
{
    const RegressionMetrics metrics = calculate_regression_metrics(
        {1.0, 2.0, 3.0},
        {1.0, 3.0, 2.0});
    return metrics.sample_count == 3
        && nearly_equal(metrics.mean_absolute_error, 2.0 / 3.0)
        && nearly_equal(metrics.mean_squared_error, 2.0 / 3.0)
        && nearly_equal(metrics.root_mean_squared_error, std::sqrt(2.0 / 3.0))
        && nearly_equal(metrics.r2, 0.0)
        && metrics.mean_absolute_percentage_error.has_value()
        && metrics.mape_sample_count == 3
        && nearly_equal(*metrics.mean_absolute_percentage_error, 5.0 / 18.0);
}

bool test_fold_generators()
{
    const auto kfold = make_k_folds(10, 5, 42, false);
    std::set<std::size_t> covered;
    for (const FoldIndices& fold : kfold)
    {
        if (fold.validation.size() != 2 || fold.training.size() != 8)
            return false;
        covered.insert(fold.validation.begin(), fold.validation.end());
    }
    if (covered.size() != 10) return false;

    const std::vector<std::string> labels = {
        "a", "a", "a", "a", "a", "a",
        "b", "b", "b", "b", "b", "b"
    };
    const auto stratified = make_stratified_folds(labels, 3, 11, false);
    for (const FoldIndices& fold : stratified)
    {
        std::size_t a_count = 0;
        std::size_t b_count = 0;
        for (std::size_t index : fold.validation)
            labels[index] == "a" ? ++a_count : ++b_count;
        if (a_count != 2 || b_count != 2) return false;
    }

    const std::vector<std::string> groups = {
        "a", "a", "b", "b", "c", "c", "d", "d"
    };
    const auto grouped = make_group_folds(groups, 2);
    std::vector<int> group_fold(4, -1);
    for (std::size_t fold_index = 0; fold_index < grouped.size(); ++fold_index)
    {
        for (std::size_t row : grouped[fold_index].validation)
        {
            const std::size_t group_index = row / 2;
            if (group_fold[group_index] != -1
                && group_fold[group_index] != static_cast<int>(fold_index))
                return false;
            group_fold[group_index] = static_cast<int>(fold_index);
        }
    }

    std::vector<std::size_t> order(12);
    for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
    const auto time_folds = make_time_series_folds(order, 3);
    return time_folds[0].training == std::vector<std::size_t>({0, 1, 2})
        && time_folds[0].validation == std::vector<std::size_t>({3, 4, 5})
        && time_folds[1].training.size() == 6
        && time_folds[1].validation == std::vector<std::size_t>({6, 7, 8})
        && time_folds[2].training.size() == 9
        && time_folds[2].validation == std::vector<std::size_t>({9, 10, 11});
}

Dataset make_binary_dataset()
{
    DatasetSchema schema{{
        {"x", ColumnType::Numeric, false, false},
        {"segment", ColumnType::Categorical, false, false},
        {"target", ColumnType::Boolean, false, true}
    }};
    std::vector<DataRow> rows;
    for (int value = -6; value <= -1; ++value)
        rows.push_back({static_cast<double>(value), std::string("negative"), false});
    for (int value = 1; value <= 6; ++value)
        rows.push_back({static_cast<double>(value), std::string("positive"), true});
    return Dataset(std::move(schema), std::move(rows), "memory://binary");
}

bool test_stratified_cross_validation()
{
    CrossValidationConfig config;
    config.task = TaskKind::Classification;
    config.estimator.kind = EstimatorKind::LogisticRegression;
    config.estimator.learning_rate = 0.2;
    config.estimator.iterations = 800;
    config.estimator.l2 = 0.001;
    config.strategy = FoldStrategy::Stratified;
    config.primary_metric = PrimaryMetric::MacroF1;
    config.folds = 3;
    config.seed = 42;

    const CrossValidationResult result = CrossValidator(config).run(make_binary_dataset());
    if (!result.success || result.folds.size() != 3) return false;
    if (result.aggregate.mean < 0.99 || result.aggregate.minimum < 0.99) return false;
    for (const FoldEvaluation& fold : result.folds)
    {
        if (!fold.classification || fold.training_rows != 8 || fold.validation_rows != 4)
            return false;
        if (!fold.classification->roc_auc || *fold.classification->roc_auc < 0.99)
            return false;
    }

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "mllibrary-cross-validation.json";
    result.write_json(path.string());
    const bool exists = std::filesystem::exists(path)
        && std::filesystem::file_size(path) > 0;
    std::filesystem::remove(path);
    return exists;
}

bool test_regression_and_time_series_cross_validation()
{
    DatasetSchema schema{{
        {"time", ColumnType::Numeric, false, false},
        {"x", ColumnType::Numeric, false, false},
        {"target", ColumnType::Numeric, false, true}
    }};
    std::vector<DataRow> rows;
    for (int index = 0; index < 16; ++index)
    {
        const double x = static_cast<double>(index);
        rows.push_back({x, x, 2.0 * x + 1.0});
    }
    Dataset dataset(std::move(schema), std::move(rows), "memory://series");

    CrossValidationConfig config;
    config.task = TaskKind::Regression;
    config.estimator.kind = EstimatorKind::LinearRegression;
    config.estimator.learning_rate = 0.04;
    config.estimator.iterations = 2500;
    config.strategy = FoldStrategy::TimeSeries;
    config.primary_metric = PrimaryMetric::R2;
    config.folds = 3;
    config.time_column = "time";
    config.exclude_split_column = true;

    const CrossValidationResult result = CrossValidator(config).run(dataset);
    return result.success
        && result.folds.size() == 3
        && result.folds[0].training_rows == 4
        && result.folds[0].validation_rows == 4
        && result.folds[2].training_rows == 12
        && result.folds[2].validation_rows == 4
        && result.aggregate.mean > 0.99;
}

bool test_group_cross_validation_keeps_groups_whole()
{
    DatasetSchema schema{{
        {"group", ColumnType::Categorical, false, false},
        {"x", ColumnType::Numeric, false, false},
        {"target", ColumnType::Boolean, false, true}
    }};
    std::vector<DataRow> rows;
    for (int group = 0; group < 6; ++group)
    {
        const bool target = group >= 3;
        const double x = target ? 2.0 + group : -2.0 - group;
        rows.push_back({std::string("g") + std::to_string(group), x, target});
        rows.push_back({std::string("g") + std::to_string(group), x + 0.1, target});
    }
    Dataset dataset(std::move(schema), std::move(rows), "memory://groups");

    CrossValidationConfig config;
    config.task = TaskKind::Classification;
    config.estimator.kind = EstimatorKind::DecisionTreeClassifier;
    config.estimator.max_depth = 3;
    config.strategy = FoldStrategy::Group;
    config.primary_metric = PrimaryMetric::Accuracy;
    config.folds = 3;
    config.group_column = "group";
    config.exclude_split_column = true;

    const CrossValidationResult result = CrossValidator(config).run(dataset);
    return result.success
        && result.folds.size() == 3
        && result.aggregate.mean > 0.99;
}

} // namespace

int main()
{
    if (!test_classification_metrics())
    {
        std::fputs("classification metrics test failed\n", stderr);
        return 1;
    }
    if (!test_regression_metrics())
    {
        std::fputs("regression metrics test failed\n", stderr);
        return 1;
    }
    if (!test_fold_generators())
    {
        std::fputs("fold generator test failed\n", stderr);
        return 1;
    }
    if (!test_stratified_cross_validation())
    {
        std::fputs("stratified cross-validation test failed\n", stderr);
        return 1;
    }
    if (!test_regression_and_time_series_cross_validation())
    {
        std::fputs("time-series cross-validation test failed\n", stderr);
        return 1;
    }
    if (!test_group_cross_validation_keeps_groups_whole())
    {
        std::fputs("group cross-validation test failed\n", stderr);
        return 1;
    }
    return 0;
}
