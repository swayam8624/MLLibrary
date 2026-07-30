#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mllibrary::search {
namespace {

using Clock = std::chrono::steady_clock;
using Candidate = std::vector<AppliedHyperparameter>;

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

std::size_t checked_integer(double value, const char* name, std::size_t minimum)
{
    if (!std::isfinite(value) || value < static_cast<double>(minimum)
        || std::floor(value) != value
        || value > static_cast<double>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument(std::string(name) + " must be an integer of at least "
            + std::to_string(minimum) + '.');
    }
    return static_cast<std::size_t>(value);
}

std::uint32_t checked_seed(double value)
{
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value
        || value > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::invalid_argument("seed must be an integer in uint32 range.");
    }
    return static_cast<std::uint32_t>(value);
}

void enumerate_candidates(
    const SearchSpace& space,
    std::size_t dimension,
    Candidate& current,
    std::vector<Candidate>& output)
{
    if (dimension == space.parameters.size())
    {
        output.push_back(current);
        return;
    }

    const HyperparameterSpec& spec = space.parameters[dimension];
    for (double value : spec.values)
    {
        current.push_back({spec.parameter, value});
        enumerate_candidates(space, dimension + 1, current, output);
        current.pop_back();
    }
}

std::vector<Candidate> make_candidates(
    const SearchConfig& config)
{
    const std::size_t cardinality = config.space.cardinality();
    if (cardinality > 1000000)
        throw std::invalid_argument("Search space exceeds the one-million-combination safety limit.");

    std::vector<Candidate> candidates;
    candidates.reserve(cardinality);
    Candidate current;
    enumerate_candidates(config.space, 0, current, candidates);

    if (config.strategy == SearchStrategy::Random
        || config.strategy == SearchStrategy::SuccessiveHalving)
    {
        std::mt19937_64 generator(config.seed);
        std::shuffle(candidates.begin(), candidates.end(), generator);
    }

    if (candidates.size() > config.budget.maximum_trials)
        candidates.resize(config.budget.maximum_trials);
    return candidates;
}

bool score_better(double left, double right, bool maximize)
{
    return maximize ? left > right : left < right;
}

SearchTrial evaluate_candidate(
    const Dataset& dataset,
    const SearchConfig& config,
    const Candidate& candidate,
    std::size_t trial_id,
    std::size_t round,
    std::size_t fold_budget)
{
    SearchTrial trial;
    trial.trial_id = trial_id;
    trial.round = round;
    trial.fold_budget = fold_budget;
    trial.parameters = candidate;

    const auto started = Clock::now();
    try
    {
        trial.estimator = apply_hyperparameters(config.evaluation.estimator, candidate);
        CrossValidationConfig evaluation = config.evaluation;
        evaluation.estimator = trial.estimator;
        evaluation.folds = fold_budget;

        CrossValidationResult result = mllibrary::evaluation::CrossValidator(evaluation).run(dataset);
        if (!result.success)
            throw std::runtime_error(result.error.empty() ? "Cross-validation failed." : result.error);
        if (!std::isfinite(result.aggregate.mean))
            throw std::runtime_error("Cross-validation produced a non-finite primary metric.");

        trial.score = result.aggregate.mean;
        trial.evaluation = std::move(result);
        trial.state = TrialState::Completed;
    }
    catch (const std::exception& error)
    {
        trial.state = TrialState::Failed;
        trial.error = error.what();
    }
    trial.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return trial;
}

void select_best(SearchResult& result)
{
    const SearchTrial* best = nullptr;
    for (const SearchTrial& trial : result.trials)
    {
        if (trial.state != TrialState::Completed)
            continue;
        if (!best
            || trial.fold_budget > best->fold_budget
            || (trial.fold_budget == best->fold_budget
                && score_better(trial.score, best->score, result.maximize))
            || (trial.fold_budget == best->fold_budget
                && trial.score == best->score
                && trial.trial_id < best->trial_id))
        {
            best = &trial;
        }
    }

    if (best)
    {
        result.best_trial_id = best->trial_id;
        result.success = true;
    }
    else
    {
        result.success = false;
        result.error = "No hyperparameter trial completed successfully.";
    }
}

bool time_budget_reached(
    const SearchConfig& config,
    const Clock::time_point& started)
{
    return config.budget.maximum_seconds > 0.0
        && std::chrono::duration<double>(Clock::now() - started).count()
            >= config.budget.maximum_seconds;
}

} // namespace

void SearchSpace::validate() const
{
    std::set<Hyperparameter> seen;
    for (const HyperparameterSpec& spec : parameters)
    {
        if (spec.values.empty())
        {
            throw std::invalid_argument(
                std::string("Search dimension '") + hyperparameter_name(spec.parameter)
                + "' must contain at least one value.");
        }
        if (!seen.insert(spec.parameter).second)
        {
            throw std::invalid_argument(
                std::string("Duplicate search dimension '")
                + hyperparameter_name(spec.parameter) + "'.");
        }
        for (double value : spec.values)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    std::string("Search dimension '") + hyperparameter_name(spec.parameter)
                    + "' contains a non-finite value.");
            }
            (void)apply_hyperparameters({}, {{spec.parameter, value}});
        }
    }
}

std::size_t SearchSpace::cardinality() const
{
    validate();
    std::size_t result = 1;
    for (const HyperparameterSpec& spec : parameters)
    {
        if (result > std::numeric_limits<std::size_t>::max() / spec.values.size())
            throw std::overflow_error("Search-space cardinality overflow.");
        result *= spec.values.size();
    }
    return result;
}

const SearchTrial* SearchResult::best_trial() const noexcept
{
    if (!best_trial_id)
        return nullptr;
    for (const SearchTrial& trial : trials)
    {
        if (trial.trial_id == *best_trial_id)
            return &trial;
    }
    return nullptr;
}

std::string SearchResult::to_json() const
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n"
           << "  \"success\": " << (success ? "true" : "false") << ",\n"
           << "  \"error\": \"" << escape_json(error) << "\",\n"
           << "  \"strategy\": \"" << search_strategy_name(strategy) << "\",\n"
           << "  \"primary_metric\": \""
           << mllibrary::evaluation::primary_metric_name(primary_metric) << "\",\n"
           << "  \"maximize\": " << (maximize ? "true" : "false") << ",\n"
           << "  \"stopped_by_time_budget\": "
           << (stopped_by_time_budget ? "true" : "false") << ",\n"
           << "  \"best_trial_id\": ";
    if (best_trial_id) output << *best_trial_id; else output << "null";
    output << ",\n  \"trials\": [\n";

    for (std::size_t index = 0; index < trials.size(); ++index)
    {
        const SearchTrial& trial = trials[index];
        output << "    {\"trial_id\": " << trial.trial_id
               << ", \"round\": " << trial.round
               << ", \"fold_budget\": " << trial.fold_budget
               << ", \"state\": \"" << trial_state_name(trial.state) << "\""
               << ", \"score\": " << trial.score
               << ", \"elapsed_seconds\": " << trial.elapsed_seconds
               << ", \"estimator\": \""
               << mllibrary::experiment::estimator_kind_name(trial.estimator.kind) << "\""
               << ", \"error\": \"" << escape_json(trial.error) << "\""
               << ", \"parameters\": {";
        for (std::size_t parameter_index = 0;
             parameter_index < trial.parameters.size(); ++parameter_index)
        {
            if (parameter_index != 0) output << ", ";
            output << "\"" << hyperparameter_name(trial.parameters[parameter_index].parameter)
                   << "\": " << trial.parameters[parameter_index].value;
        }
        output << "}";
        if (trial.evaluation)
            output << ", \"evaluation\": " << trial.evaluation->to_json();
        output << '}';
        if (index + 1 != trials.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
    return output.str();
}

void SearchResult::write_json(const std::string& path) const
{
    const std::filesystem::path destination(path);
    if (destination.has_parent_path())
        std::filesystem::create_directories(destination.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot open hyperparameter-search report '" + path + "'.");
    output << to_json();
    if (!output)
        throw std::runtime_error("Failed while writing hyperparameter-search report '" + path + "'.");
}

HyperparameterSearch::HyperparameterSearch(SearchConfig config)
    : config_(std::move(config))
{
}

SearchResult HyperparameterSearch::run(const Dataset& dataset) const
{
    SearchResult result;
    result.strategy = config_.strategy;
    result.primary_metric = config_.evaluation.primary_metric;
    result.maximize = metric_is_maximized(result.primary_metric);

    try
    {
        config_.space.validate();
        if (config_.budget.maximum_trials == 0)
            throw std::invalid_argument("maximum_trials must be at least one.");
        if (!std::isfinite(config_.budget.maximum_seconds)
            || config_.budget.maximum_seconds < 0.0)
        {
            throw std::invalid_argument("maximum_seconds must be finite and non-negative.");
        }
        if (config_.evaluation.folds < 2)
            throw std::invalid_argument("Hyperparameter search requires at least two evaluation folds.");
        if (config_.budget.minimum_folds < 2)
            throw std::invalid_argument("minimum_folds must be at least two.");
        if (config_.budget.reduction_factor < 2)
            throw std::invalid_argument("reduction_factor must be at least two.");

        std::vector<Candidate> candidates = make_candidates(config_);
        if (candidates.empty())
            throw std::invalid_argument("Hyperparameter search generated no candidates.");

        const auto search_started = Clock::now();
        std::size_t next_trial_id = 0;

        if (config_.strategy != SearchStrategy::SuccessiveHalving)
        {
            for (const Candidate& candidate : candidates)
            {
                if (time_budget_reached(config_, search_started))
                {
                    result.stopped_by_time_budget = true;
                    break;
                }
                result.trials.push_back(evaluate_candidate(
                    dataset,
                    config_,
                    candidate,
                    next_trial_id++,
                    0,
                    config_.evaluation.folds));
            }
        }
        else
        {
            std::vector<std::size_t> active(candidates.size());
            for (std::size_t index = 0; index < active.size(); ++index)
                active[index] = index;

            const std::size_t maximum_folds = config_.evaluation.folds;
            std::size_t fold_budget = std::min(
                std::max<std::size_t>(2, config_.budget.minimum_folds),
                maximum_folds);
            std::size_t round = 0;

            while (!active.empty())
            {
                std::vector<std::pair<std::size_t, std::size_t>> completed;
                for (std::size_t candidate_index : active)
                {
                    if (time_budget_reached(config_, search_started))
                    {
                        result.stopped_by_time_budget = true;
                        break;
                    }
                    const std::size_t trial_position = result.trials.size();
                    result.trials.push_back(evaluate_candidate(
                        dataset,
                        config_,
                        candidates[candidate_index],
                        next_trial_id++,
                        round,
                        fold_budget));
                    if (result.trials.back().state == TrialState::Completed)
                        completed.emplace_back(trial_position, candidate_index);
                }

                if (result.stopped_by_time_budget || completed.empty())
                    break;
                if (fold_budget == maximum_folds)
                    break;

                std::stable_sort(
                    completed.begin(), completed.end(),
                    [&](const auto& left, const auto& right)
                    {
                        const SearchTrial& left_trial = result.trials[left.first];
                        const SearchTrial& right_trial = result.trials[right.first];
                        if (left_trial.score == right_trial.score)
                            return left_trial.trial_id < right_trial.trial_id;
                        return score_better(left_trial.score, right_trial.score, result.maximize);
                    });

                const std::size_t keep = std::max<std::size_t>(
                    1,
                    (completed.size() + config_.budget.reduction_factor - 1)
                        / config_.budget.reduction_factor);
                std::vector<std::size_t> promoted;
                promoted.reserve(keep);
                for (std::size_t index = 0; index < completed.size(); ++index)
                {
                    if (index < keep)
                        promoted.push_back(completed[index].second);
                    else
                        result.trials[completed[index].first].state = TrialState::Pruned;
                }
                active = std::move(promoted);

                if (fold_budget > maximum_folds / config_.budget.reduction_factor)
                    fold_budget = maximum_folds;
                else
                    fold_budget = std::min(
                        maximum_folds,
                        fold_budget * config_.budget.reduction_factor);
                ++round;
            }
        }

        select_best(result);
        if (!config_.output_directory.empty())
        {
            std::filesystem::create_directories(config_.output_directory);
            result.write_json(
                (std::filesystem::path(config_.output_directory) / "search.json").string());
        }
    }
    catch (const std::exception& error)
    {
        result.success = false;
        result.error = error.what();
    }
    return result;
}

EstimatorConfig apply_hyperparameters(
    EstimatorConfig base,
    const std::vector<AppliedHyperparameter>& parameters)
{
    for (const AppliedHyperparameter& parameter : parameters)
    {
        const double value = parameter.value;
        switch (parameter.parameter)
        {
        case Hyperparameter::LearningRate:
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument("learning_rate must be finite and positive.");
            base.learning_rate = value;
            break;
        case Hyperparameter::Iterations:
            base.iterations = checked_integer(value, "iterations", 1);
            break;
        case Hyperparameter::L2:
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument("l2 must be finite and non-negative.");
            base.l2 = value;
            break;
        case Hyperparameter::Neighbors:
            base.neighbors = checked_integer(value, "neighbors", 1);
            break;
        case Hyperparameter::VarianceSmoothing:
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument("variance_smoothing must be finite and positive.");
            base.variance_smoothing = value;
            break;
        case Hyperparameter::Trees:
            base.trees = checked_integer(value, "trees", 1);
            break;
        case Hyperparameter::MaxDepth:
            base.max_depth = checked_integer(value, "max_depth", 1);
            break;
        case Hyperparameter::MinSamplesSplit:
            base.min_samples_split = checked_integer(value, "min_samples_split", 2);
            break;
        case Hyperparameter::MinSamplesLeaf:
            base.min_samples_leaf = checked_integer(value, "min_samples_leaf", 1);
            break;
        case Hyperparameter::MaxFeatures:
            base.max_features = checked_integer(value, "max_features", 0);
            break;
        case Hyperparameter::Seed:
            base.seed = checked_seed(value);
            break;
        }
    }
    return base;
}

bool metric_is_maximized(PrimaryMetric metric) noexcept
{
    switch (metric)
    {
    case PrimaryMetric::LogLoss:
    case PrimaryMetric::MeanAbsoluteError:
    case PrimaryMetric::RootMeanSquaredError:
        return false;
    case PrimaryMetric::Accuracy:
    case PrimaryMetric::MacroF1:
    case PrimaryMetric::WeightedF1:
    case PrimaryMetric::RocAuc:
    case PrimaryMetric::R2:
        return true;
    }
    return true;
}

const char* search_strategy_name(SearchStrategy strategy) noexcept
{
    switch (strategy)
    {
    case SearchStrategy::Grid: return "grid";
    case SearchStrategy::Random: return "random";
    case SearchStrategy::SuccessiveHalving: return "successive_halving";
    }
    return "unknown";
}

const char* hyperparameter_name(Hyperparameter parameter) noexcept
{
    switch (parameter)
    {
    case Hyperparameter::LearningRate: return "learning_rate";
    case Hyperparameter::Iterations: return "iterations";
    case Hyperparameter::L2: return "l2";
    case Hyperparameter::Neighbors: return "neighbors";
    case Hyperparameter::VarianceSmoothing: return "variance_smoothing";
    case Hyperparameter::Trees: return "trees";
    case Hyperparameter::MaxDepth: return "max_depth";
    case Hyperparameter::MinSamplesSplit: return "min_samples_split";
    case Hyperparameter::MinSamplesLeaf: return "min_samples_leaf";
    case Hyperparameter::MaxFeatures: return "max_features";
    case Hyperparameter::Seed: return "seed";
    }
    return "unknown";
}

const char* trial_state_name(TrialState state) noexcept
{
    switch (state)
    {
    case TrialState::Completed: return "completed";
    case TrialState::Failed: return "failed";
    case TrialState::Pruned: return "pruned";
    }
    return "unknown";
}

} // namespace mllibrary::search
