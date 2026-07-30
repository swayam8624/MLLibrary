#include "experiment.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mllibrary::experiment;
using namespace mllibrary::preprocessing;
using mllibrary::data::ColumnSchema;
using mllibrary::data::ColumnType;

namespace {

PreparedDataset make_classification_dataset(
    std::vector<double> values,
    std::vector<DataValue> targets,
    std::size_t columns = 1)
{
    PreparedDataset dataset;
    dataset.row_count = targets.size();
    dataset.column_count = columns;
    dataset.values = std::move(values);
    for (std::size_t column = 0; column < columns; ++column)
        dataset.feature_names.push_back("x" + std::to_string(column));
    dataset.targets = std::move(targets);
    dataset.target_schema = ColumnSchema{
        "label", ColumnType::Categorical, false, true
    };
    return dataset;
}

bool test_logistic_experiment_and_artifacts()
{
    const PreparedDataset training = make_classification_dataset(
        {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0},
        {std::string("no"), std::string("no"), std::string("no"),
         std::string("yes"), std::string("yes"), std::string("yes")});
    const PreparedDataset validation = make_classification_dataset(
        {-2.5, -0.5, 0.5, 2.5},
        {std::string("no"), std::string("no"),
         std::string("yes"), std::string("yes")});

    EstimatorConfig config;
    config.kind = EstimatorKind::LogisticRegression;
    config.learning_rate = 0.2;
    config.iterations = 800;
    config.l2 = 0.001;

    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "mllibrary-experiment-runner";
    std::filesystem::remove_all(output);

    const ExperimentResult result = ExperimentRunner(
        TaskKind::Classification, config).run(
            training,
            &validation,
            &validation,
            "dataset-123",
            "pipeline-456",
            output.string());

    const bool passed = result.success
        && result.estimator_name == "logistic_regression"
        && result.class_labels == std::vector<std::string>({"no", "yes"})
        && result.validation.has_value()
        && result.validation->accuracy == 1.0
        && result.validation->scores.size() == validation.row_count
        && result.model_state_units == 2
        && std::filesystem::exists(output / "experiment.json")
        && std::filesystem::exists(output / "validation_predictions.csv")
        && std::filesystem::exists(output / "test_predictions.csv");

    std::ifstream manifest(output / "experiment.json", std::ios::binary);
    const std::string json(
        (std::istreambuf_iterator<char>(manifest)),
        std::istreambuf_iterator<char>());
    const bool manifest_ok = json.find("\"dataset_fingerprint\": \"dataset-123\"")
            != std::string::npos
        && json.find("\"accuracy\": 1") != std::string::npos;

    std::filesystem::remove_all(output);
    return passed && manifest_ok;
}

bool test_linear_regression_experiment()
{
    PreparedDataset training;
    training.row_count = 5;
    training.column_count = 1;
    training.values = {0.0, 1.0, 2.0, 3.0, 4.0};
    training.feature_names = {"x"};
    training.targets = {1.0, 3.0, 5.0, 7.0, 9.0};
    training.target_schema = ColumnSchema{
        "y", ColumnType::Numeric, false, true
    };

    PreparedDataset test = training;
    test.row_count = 2;
    test.values = {5.0, 6.0};
    test.targets = {11.0, 13.0};

    EstimatorConfig config;
    config.kind = EstimatorKind::LinearRegression;
    config.learning_rate = 0.04;
    config.iterations = 2500;

    const ExperimentResult result = ExperimentRunner(
        TaskKind::Regression, config).run(training, nullptr, &test);
    return result.success
        && result.test.has_value()
        && result.test->mean_squared_error < 1e-3
        && result.class_labels.empty()
        && result.model_state_units == 2;
}

bool test_multiclass_tree_experiment()
{
    const PreparedDataset dataset = make_classification_dataset(
        {
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
            2.0, 0.0,
            2.0, 1.0,
        },
        {std::string("a"), std::string("a"), std::string("b"),
         std::string("b"), std::string("c"), std::string("c")},
        2);

    EstimatorConfig config;
    config.kind = EstimatorKind::DecisionTreeClassifier;
    config.max_depth = 4;
    const ExperimentResult result = ExperimentRunner(
        TaskKind::Classification, config).run(dataset, &dataset, nullptr);
    return result.success
        && result.validation.has_value()
        && result.validation->accuracy == 1.0
        && result.class_labels.size() == 3
        && result.model_state_units >= 3;
}

bool test_invalid_task_and_unseen_target_fail_cleanly()
{
    PreparedDataset regression;
    regression.row_count = 2;
    regression.column_count = 1;
    regression.values = {0.0, 1.0};
    regression.feature_names = {"x"};
    regression.targets = {0.0, 1.0};

    EstimatorConfig wrong;
    wrong.kind = EstimatorKind::LogisticRegression;
    const ExperimentResult mismatch = ExperimentRunner(
        TaskKind::Regression, wrong).run(regression);
    if (mismatch.success || mismatch.error.find("incompatible") == std::string::npos)
        return false;

    const PreparedDataset training = make_classification_dataset(
        {-1.0, 1.0},
        {std::string("left"), std::string("right")});
    const PreparedDataset validation = make_classification_dataset(
        {0.0},
        {std::string("unknown")});
    EstimatorConfig tree;
    tree.kind = EstimatorKind::DecisionTreeClassifier;
    const ExperimentResult unseen = ExperimentRunner(
        TaskKind::Classification, tree).run(training, &validation, nullptr);
    return !unseen.success
        && unseen.error.find("absent from training") != std::string::npos;
}

} // namespace

int main()
{
    if (!test_logistic_experiment_and_artifacts())
    {
        std::fputs("logistic experiment test failed\n", stderr);
        return 1;
    }
    if (!test_linear_regression_experiment())
    {
        std::fputs("linear experiment test failed\n", stderr);
        return 1;
    }
    if (!test_multiclass_tree_experiment())
    {
        std::fputs("multiclass experiment test failed\n", stderr);
        return 1;
    }
    if (!test_invalid_task_and_unseen_target_fail_cleanly())
    {
        std::fputs("experiment validation test failed\n", stderr);
        return 1;
    }
    return 0;
}
