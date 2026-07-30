#pragma once

#include "pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mllibrary::experiment {

using mllibrary::preprocessing::PreparedDataset;

enum class TaskKind : std::uint8_t {
    Classification,
    Regression,
};

enum class EstimatorKind : std::uint8_t {
    LogisticRegression,
    LinearRegression,
    KNearestNeighbors,
    GaussianNaiveBayes,
    DecisionTreeClassifier,
    RandomForestClassifier,
};

struct EstimatorConfig final {
    EstimatorKind kind = EstimatorKind::LogisticRegression;
    double learning_rate = 0.1;
    std::size_t iterations = 500;
    double l2 = 0.0;
    std::size_t neighbors = 5;
    double variance_smoothing = 1e-6;
    std::size_t trees = 100;
    std::size_t max_depth = 16;
    std::size_t min_samples_split = 2;
    std::size_t min_samples_leaf = 1;
    std::size_t max_features = 0;
    std::uint32_t seed = 5489u;
};

struct DatasetEvaluation final {
    std::size_t row_count = 0;
    std::vector<double> actual;
    std::vector<double> predicted;
    std::vector<double> scores;
    double accuracy = 0.0;
    double mean_squared_error = 0.0;
    double inference_seconds = 0.0;
};

struct ExperimentResult final {
    bool success = false;
    std::string error;
    TaskKind task = TaskKind::Classification;
    EstimatorConfig estimator{};
    std::string estimator_name;
    std::string dataset_fingerprint;
    std::string pipeline_fingerprint;
    std::vector<std::string> class_labels;
    double fit_seconds = 0.0;
    std::size_t model_state_units = 0;
    std::optional<DatasetEvaluation> validation;
    std::optional<DatasetEvaluation> test;

    [[nodiscard]] std::string to_json() const;
    void write_artifacts(const std::string& output_directory) const;
};

class Estimator {
public:
    virtual ~Estimator() = default;

    [[nodiscard]] virtual const char* name() const noexcept = 0;
    [[nodiscard]] virtual TaskKind task() const noexcept = 0;
    virtual void fit(
        const PreparedDataset& training,
        const PreparedDataset* validation = nullptr) = 0;
    [[nodiscard]] virtual std::vector<double> predict(
        const PreparedDataset& dataset) const = 0;
    [[nodiscard]] virtual std::vector<double> predict_scores(
        const PreparedDataset& dataset) const;
    [[nodiscard]] virtual std::vector<double> encode_targets(
        const PreparedDataset& dataset) const = 0;
    [[nodiscard]] virtual const std::vector<std::string>& class_labels() const noexcept = 0;
    [[nodiscard]] virtual std::size_t model_state_units() const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<Estimator> make_estimator(
    const EstimatorConfig& config);

class ExperimentRunner final {
public:
    ExperimentRunner(TaskKind task, EstimatorConfig estimator);

    [[nodiscard]] ExperimentResult run(
        const PreparedDataset& training,
        const PreparedDataset* validation = nullptr,
        const PreparedDataset* test = nullptr,
        std::string dataset_fingerprint = {},
        std::string pipeline_fingerprint = {},
        const std::string& output_directory = {}) const;

private:
    TaskKind task_;
    EstimatorConfig estimator_;
};

[[nodiscard]] const char* task_kind_name(TaskKind task) noexcept;
[[nodiscard]] const char* estimator_kind_name(EstimatorKind estimator) noexcept;

} // namespace mllibrary::experiment
