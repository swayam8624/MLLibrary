// TEACHING PUBLIC API: bounded discrete hyperparameter trials evaluated
// through CrossValidator and ranked by metric direction. Search-selected
// results are not a substitute for an independent final test set.
#pragma once

#include "cross_validation.hpp"
#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mllibrary::search {

using mllibrary::data::Dataset;
using mllibrary::evaluation::CrossValidationConfig;
using mllibrary::evaluation::CrossValidationResult;
using mllibrary::evaluation::PrimaryMetric;
using mllibrary::experiment::EstimatorConfig;

// Grid tries discrete Cartesian combinations, Random seeded-shuffles
// candidates, and SuccessiveHalving promotes candidates with larger fold budgets.
enum class SearchStrategy : std::uint8_t {
    Grid,
    Random,
    SuccessiveHalving,
};

enum class Hyperparameter : std::uint8_t {
    LearningRate,
    Iterations,
    L2,
    Neighbors,
    VarianceSmoothing,
    Trees,
    MaxDepth,
    MinSamplesSplit,
    MinSamplesLeaf,
    MaxFeatures,
    Seed,
};

enum class TrialState : std::uint8_t {
    Completed,
    Failed,
    Pruned,
};

struct HyperparameterSpec final {
    Hyperparameter parameter = Hyperparameter::LearningRate;
    std::vector<double> values;
};

struct SearchSpace final {
    std::vector<HyperparameterSpec> parameters;

    void validate() const;
    [[nodiscard]] std::size_t cardinality() const;
};

// Maximum trials and optional wall-clock budget. The timer is checked
// between evaluations; an in-flight fold is not preemptively interrupted.
struct SearchBudget final {
    std::size_t maximum_trials = 25;
    double maximum_seconds = 0.0;
    std::size_t minimum_folds = 2;
    std::size_t reduction_factor = 3;
};

struct SearchConfig final {
    CrossValidationConfig evaluation{};
    SearchSpace space{};
    SearchStrategy strategy = SearchStrategy::Random;
    SearchBudget budget{};
    std::uint64_t seed = 5489u;
    std::string output_directory;
};

struct AppliedHyperparameter final {
    Hyperparameter parameter = Hyperparameter::LearningRate;
    double value = 0.0;
};

struct SearchTrial final {
    std::size_t trial_id = 0;
    std::size_t round = 0;
    std::size_t fold_budget = 0;
    TrialState state = TrialState::Failed;
    std::string error;
    EstimatorConfig estimator{};
    std::vector<AppliedHyperparameter> parameters;
    double score = 0.0;
    double elapsed_seconds = 0.0;
    std::optional<CrossValidationResult> evaluation;
};

// Retains completed/failed/pruned trial history and a best-trial ID.
// Failure states must remain visible when interpreting ranking artifacts.
struct SearchResult final {
    bool success = false;
    std::string error;
    SearchStrategy strategy = SearchStrategy::Random;
    PrimaryMetric primary_metric = PrimaryMetric::Accuracy;
    bool maximize = true;
    bool stopped_by_time_budget = false;
    std::optional<std::size_t> best_trial_id;
    std::vector<SearchTrial> trials;

    [[nodiscard]] const SearchTrial* best_trial() const noexcept;
    [[nodiscard]] std::string to_json() const;
    void write_json(const std::string& path) const;
};

// run() builds candidate-specific CV configurations. Each CV fold fits its
// own preprocessing/model state using the *training* indices only.
class HyperparameterSearch final {
public:
    explicit HyperparameterSearch(SearchConfig config);

    [[nodiscard]] SearchResult run(const Dataset& dataset) const;

private:
    SearchConfig config_;
};

[[nodiscard]] EstimatorConfig apply_hyperparameters(
    EstimatorConfig base,
    const std::vector<AppliedHyperparameter>& parameters);

[[nodiscard]] bool metric_is_maximized(PrimaryMetric metric) noexcept;
[[nodiscard]] const char* search_strategy_name(SearchStrategy strategy) noexcept;
[[nodiscard]] const char* hyperparameter_name(Hyperparameter parameter) noexcept;
[[nodiscard]] const char* trial_state_name(TrialState state) noexcept;

} // namespace mllibrary::search
