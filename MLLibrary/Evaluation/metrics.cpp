#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace mllibrary::evaluation {
namespace {

int class_id(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Classification labels must be finite.");
    const double rounded = std::round(value);
    if (std::fabs(value - rounded) > 1e-9
        || rounded < static_cast<double>(std::numeric_limits<int>::min())
        || rounded > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument(
            "Classification labels and predictions must be integer-valued class IDs.");
    }
    return static_cast<int>(rounded);
}

void require_parallel_vectors(
    const std::vector<double>& actual,
    const std::vector<double>& predicted)
{
    if (actual.empty())
        throw std::invalid_argument("Metric calculation requires at least one sample.");
    if (actual.size() != predicted.size())
        throw std::invalid_argument("Actual and predicted vectors must have equal length.");
}

std::optional<double> binary_roc_auc(
    const std::vector<int>& actual,
    const std::vector<double>& scores,
    int positive_label)
{
    std::vector<std::pair<double, bool>> ranked;
    ranked.reserve(scores.size());
    std::size_t positives = 0;
    for (std::size_t index = 0; index < scores.size(); ++index)
    {
        if (!std::isfinite(scores[index]) || scores[index] < 0.0 || scores[index] > 1.0)
            throw std::invalid_argument("Binary probability scores must be finite values in [0, 1].");
        const bool positive = actual[index] == positive_label;
        positives += positive ? 1u : 0u;
        ranked.emplace_back(scores[index], positive);
    }

    const std::size_t negatives = scores.size() - positives;
    if (positives == 0 || negatives == 0)
        return std::nullopt;

    std::sort(
        ranked.begin(), ranked.end(),
        [](const auto& left, const auto& right)
        {
            if (left.first != right.first) return left.first < right.first;
            return left.second < right.second;
        });

    double positive_rank_sum = 0.0;
    std::size_t begin = 0;
    while (begin < ranked.size())
    {
        std::size_t end = begin + 1;
        while (end < ranked.size() && ranked[end].first == ranked[begin].first)
            ++end;
        const double average_rank =
            (static_cast<double>(begin + 1) + static_cast<double>(end)) * 0.5;
        for (std::size_t index = begin; index < end; ++index)
        {
            if (ranked[index].second)
                positive_rank_sum += average_rank;
        }
        begin = end;
    }

    const double positive_count = static_cast<double>(positives);
    const double negative_count = static_cast<double>(negatives);
    const double auc = (
        positive_rank_sum - positive_count * (positive_count + 1.0) * 0.5)
        / (positive_count * negative_count);
    return auc;
}

} // namespace

std::size_t ConfusionMatrix::at(
    std::size_t actual_index,
    std::size_t predicted_index) const
{
    if (actual_index >= labels.size() || predicted_index >= labels.size())
        throw std::out_of_range("Confusion-matrix index is out of range.");
    return counts[actual_index * labels.size() + predicted_index];
}

ClassificationMetrics calculate_classification_metrics(
    const std::vector<double>& actual,
    const std::vector<double>& predicted,
    const std::vector<double>& positive_class_scores)
{
    require_parallel_vectors(actual, predicted);
    if (!positive_class_scores.empty()
        && positive_class_scores.size() != actual.size())
    {
        throw std::invalid_argument(
            "Probability-score count must match the classification sample count.");
    }

    std::vector<int> actual_ids;
    std::vector<int> predicted_ids;
    actual_ids.reserve(actual.size());
    predicted_ids.reserve(predicted.size());
    std::set<int> label_set;
    for (double value : actual)
    {
        const int label = class_id(value);
        actual_ids.push_back(label);
        label_set.insert(label);
    }
    for (double value : predicted)
    {
        const int label = class_id(value);
        predicted_ids.push_back(label);
        label_set.insert(label);
    }

    ClassificationMetrics result;
    result.sample_count = actual.size();
    result.confusion.labels.assign(label_set.begin(), label_set.end());
    result.confusion.counts.assign(
        result.confusion.labels.size() * result.confusion.labels.size(), 0);

    std::map<int, std::size_t> index_by_label;
    for (std::size_t index = 0; index < result.confusion.labels.size(); ++index)
        index_by_label.emplace(result.confusion.labels[index], index);

    std::size_t correct = 0;
    for (std::size_t sample = 0; sample < actual_ids.size(); ++sample)
    {
        const std::size_t actual_index = index_by_label.at(actual_ids[sample]);
        const std::size_t predicted_index = index_by_label.at(predicted_ids[sample]);
        ++result.confusion.counts[
            actual_index * result.confusion.labels.size() + predicted_index];
        correct += actual_ids[sample] == predicted_ids[sample] ? 1u : 0u;
    }
    result.accuracy = static_cast<double>(correct)
        / static_cast<double>(result.sample_count);

    result.classes.reserve(result.confusion.labels.size());
    for (std::size_t class_index = 0;
         class_index < result.confusion.labels.size();
         ++class_index)
    {
        const std::size_t true_positive = result.confusion.at(class_index, class_index);
        std::size_t support = 0;
        std::size_t predicted_count = 0;
        for (std::size_t other = 0; other < result.confusion.labels.size(); ++other)
        {
            support += result.confusion.at(class_index, other);
            predicted_count += result.confusion.at(other, class_index);
        }

        ClassMetrics metrics;
        metrics.label = result.confusion.labels[class_index];
        metrics.support = support;
        metrics.precision = predicted_count == 0
            ? 0.0
            : static_cast<double>(true_positive) / static_cast<double>(predicted_count);
        metrics.recall = support == 0
            ? 0.0
            : static_cast<double>(true_positive) / static_cast<double>(support);
        metrics.f1 = metrics.precision + metrics.recall == 0.0
            ? 0.0
            : 2.0 * metrics.precision * metrics.recall
                / (metrics.precision + metrics.recall);

        result.macro_precision += metrics.precision;
        result.macro_recall += metrics.recall;
        result.macro_f1 += metrics.f1;
        result.weighted_f1 += metrics.f1 * static_cast<double>(support);
        result.classes.push_back(metrics);
    }

    const double class_count = static_cast<double>(result.classes.size());
    result.macro_precision /= class_count;
    result.macro_recall /= class_count;
    result.macro_f1 /= class_count;
    result.weighted_f1 /= static_cast<double>(result.sample_count);

    if (!positive_class_scores.empty())
    {
        if (result.confusion.labels.size() != 2)
            throw std::invalid_argument(
                "Probability-based log loss and ROC-AUC currently require exactly two classes.");
        const int positive_label = result.confusion.labels.back();
        constexpr double epsilon = 1e-15;
        double loss = 0.0;
        for (std::size_t index = 0; index < actual_ids.size(); ++index)
        {
            const double score = positive_class_scores[index];
            if (!std::isfinite(score) || score < 0.0 || score > 1.0)
            {
                throw std::invalid_argument(
                    "Binary probability scores must be finite values in [0, 1].");
            }
            const double probability = std::clamp(score, epsilon, 1.0 - epsilon);
            const bool positive = actual_ids[index] == positive_label;
            loss -= positive ? std::log(probability) : std::log(1.0 - probability);
        }
        result.log_loss = loss / static_cast<double>(actual_ids.size());
        result.roc_auc = binary_roc_auc(
            actual_ids, positive_class_scores, positive_label);
    }

    return result;
}

RegressionMetrics calculate_regression_metrics(
    const std::vector<double>& actual,
    const std::vector<double>& predicted,
    double zero_tolerance)
{
    require_parallel_vectors(actual, predicted);
    if (!std::isfinite(zero_tolerance) || zero_tolerance < 0.0)
        throw std::invalid_argument("Regression zero tolerance must be finite and non-negative.");

    RegressionMetrics result;
    result.sample_count = actual.size();

    double actual_mean = 0.0;
    for (double value : actual)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Regression targets must be finite.");
        actual_mean += value;
    }
    actual_mean /= static_cast<double>(actual.size());

    double absolute_error = 0.0;
    double squared_error = 0.0;
    double total_variance = 0.0;
    double percentage_error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        if (!std::isfinite(predicted[index]))
            throw std::invalid_argument("Regression predictions must be finite.");
        const double error = predicted[index] - actual[index];
        absolute_error += std::fabs(error);
        squared_error += error * error;
        const double centered = actual[index] - actual_mean;
        total_variance += centered * centered;
        if (std::fabs(actual[index]) > zero_tolerance)
        {
            percentage_error += std::fabs(error / actual[index]);
            ++result.mape_sample_count;
        }
    }

    const double sample_count = static_cast<double>(result.sample_count);
    result.mean_absolute_error = absolute_error / sample_count;
    result.mean_squared_error = squared_error / sample_count;
    result.root_mean_squared_error = std::sqrt(result.mean_squared_error);
    result.r2 = total_variance <= zero_tolerance
        ? (squared_error <= zero_tolerance ? 1.0 : 0.0)
        : 1.0 - squared_error / total_variance;
    if (result.mape_sample_count != 0)
    {
        result.mean_absolute_percentage_error = percentage_error
            / static_cast<double>(result.mape_sample_count);
    }
    return result;
}

} // namespace mllibrary::evaluation
