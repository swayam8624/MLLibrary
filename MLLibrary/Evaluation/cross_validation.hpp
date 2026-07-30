#pragma once

#include "dataset.hpp"
#include "experiment.hpp"
#include "metrics.hpp"
#include "pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mllibrary::evaluation {

using mllibrary::data::Dataset;
using mllibrary::experiment::EstimatorConfig;
using mllibrary::experiment::ExperimentResult;
using mllibrary::experiment::TaskKind;
using mllibrary::preprocessing::PipelineOptions;

enum class FoldStrategy : std::uint8_t {
    KFold,
    Stratified,
    Group,
    TimeSeries,
};

enum class PrimaryMetric : std::uint8_t {
    Accuracy,
    MacroF1,
    WeightedF1,
    LogLoss,
    RocAuc,
    MeanAbsoluteError,
    RootMeanSquaredError,
    R2,
};

struct FoldIndices final {
    std::vector<std::size_t> training;
    std::vector<std::size_t> validation;
};

[[nodiscard]] std::vector<FoldIndices> make_k_folds(
    std::size_t row_count,
    std::size_t fold_count,
    std::uint64_t seed = 5489u,
    bool shuffle = true);

[[nodiscard]] std::vector<FoldIndices> make_stratified_folds(
    const std::vector<std::string>& labels,
    std::size_t fold_count,
    std::uint64_t seed = 5489u,
    bool shuffle = true);

[[nodiscard]] std::vector<FoldIndices> make_group_folds(
    const std::vector<std::string>& groups,
    std::size_t fold_count);

[[nodiscard]] std::vector<FoldIndices> make_time_series_folds(
    const std::vector<std::size_t>& ordered_rows,
    std::size_t fold_count);

struct MetricAggregate final {
    double mean = 0.0;
    double standard_deviation = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct CrossValidationConfig final {
    TaskKind task = TaskKind::Classification;
    EstimatorConfig estimator{};
    PipelineOptions pipeline{};
    FoldStrategy strategy = FoldStrategy::Stratified;
    PrimaryMetric primary_metric = PrimaryMetric::MacroF1;
    std::size_t folds = 5;
    std::uint64_t seed = 5489u;
    bool shuffle = true;
    std::string group_column;
    std::string time_column;
    bool exclude_split_column = true;
};

struct FoldEvaluation final {
    std::size_t fold_index = 0;
    std::size_t training_rows = 0;
    std::size_t validation_rows = 0;
    ExperimentResult experiment;
    std::optional<ClassificationMetrics> classification;
    std::optional<RegressionMetrics> regression;
    double primary_metric_value = 0.0;
};

struct CrossValidationResult final {
    bool success = false;
    std::string error;
    FoldStrategy strategy = FoldStrategy::KFold;
    PrimaryMetric primary_metric = PrimaryMetric::Accuracy;
    std::vector<FoldEvaluation> folds;
    MetricAggregate aggregate;

    [[nodiscard]] std::string to_json() const;
    void write_json(const std::string& path) const;
};

class CrossValidator final {
public:
    explicit CrossValidator(CrossValidationConfig config);

    [[nodiscard]] CrossValidationResult run(const Dataset& dataset) const;

private:
    CrossValidationConfig config_;
};

[[nodiscard]] const char* fold_strategy_name(FoldStrategy strategy) noexcept;
[[nodiscard]] const char* primary_metric_name(PrimaryMetric metric) noexcept;

} // namespace mllibrary::evaluation
