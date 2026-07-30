#pragma once

#include "dataset.hpp"
#include "profiling.hpp"
#include "search.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mllibrary::automl {

using mllibrary::data::Dataset;
using mllibrary::data::DatasetProfile;
using mllibrary::evaluation::FoldStrategy;
using mllibrary::evaluation::PrimaryMetric;
using mllibrary::experiment::EstimatorConfig;
using mllibrary::experiment::TaskKind;
using mllibrary::preprocessing::PipelineOptions;
using mllibrary::search::SearchResult;
using mllibrary::search::SearchSpace;
using mllibrary::search::SearchStrategy;

enum class TaskSelection : std::uint8_t {
    Auto,
    Classification,
    Regression,
};

struct AutoMLConfig final {
    TaskSelection task = TaskSelection::Auto;
    SearchStrategy search_strategy = SearchStrategy::SuccessiveHalving;
    std::size_t folds = 5;
    std::uint64_t seed = 5489u;
    std::size_t maximum_candidates = 0;
    std::size_t maximum_total_trials = 30;
    double maximum_seconds = 0.0;
    std::size_t numeric_classification_limit = 20;
    double numeric_classification_ratio = 0.05;
    bool include_knn = true;
    bool include_random_forest = true;
    std::string group_column;
    std::string time_column;
    std::string output_directory;
};

struct AutoMLCandidate final {
    std::string name;
    std::string rationale;
    TaskKind task = TaskKind::Classification;
    EstimatorConfig estimator{};
    PipelineOptions pipeline{};
    SearchSpace search_space{};
    PrimaryMetric primary_metric = PrimaryMetric::Accuracy;
    FoldStrategy fold_strategy = FoldStrategy::KFold;
};

struct AutoMLPlan final {
    bool success = false;
    std::string error;
    TaskKind task = TaskKind::Classification;
    PrimaryMetric primary_metric = PrimaryMetric::Accuracy;
    FoldStrategy fold_strategy = FoldStrategy::KFold;
    std::size_t estimated_feature_count = 0;
    DatasetProfile profile{};
    std::vector<std::string> warnings;
    std::vector<AutoMLCandidate> candidates;

    [[nodiscard]] std::string to_json() const;
};

struct AutoMLCandidateResult final {
    AutoMLCandidate candidate;
    SearchResult search;
};

struct AutoMLResult final {
    bool success = false;
    std::string error;
    AutoMLPlan plan{};
    std::vector<AutoMLCandidateResult> candidates;
    std::optional<std::size_t> best_candidate_index;

    [[nodiscard]] const AutoMLCandidateResult* best_candidate() const noexcept;
    [[nodiscard]] std::string to_json() const;
    void write_json(const std::string& path) const;
};

class AutoMLPlanner final {
public:
    explicit AutoMLPlanner(AutoMLConfig config = {});

    [[nodiscard]] AutoMLPlan plan(const Dataset& dataset) const;

private:
    AutoMLConfig config_;
};

class AutoMLRunner final {
public:
    explicit AutoMLRunner(AutoMLConfig config = {});

    [[nodiscard]] AutoMLResult run(const Dataset& dataset) const;

private:
    AutoMLConfig config_;
};

[[nodiscard]] const char* task_selection_name(TaskSelection task) noexcept;

} // namespace mllibrary::automl
