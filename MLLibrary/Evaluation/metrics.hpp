// TEACHING PUBLIC API: score held-out predictions; CrossValidator and
// Search use these outputs but these helpers do not fit models. Binary AUC
// and log loss require probability scores and two classes.
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace mllibrary::evaluation {

// Flattened row=actual, column=predicted class counts, with sorted labels.
// at() checks bounds against the label table.
struct ConfusionMatrix final {
    std::vector<int> labels;
    std::vector<std::size_t> counts;

    [[nodiscard]] std::size_t at(
        std::size_t actual_index,
        std::size_t predicted_index) const;
};

struct ClassMetrics final {
    int label = 0;
    std::size_t support = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

// Accuracy and macro/weighted class measures; optional log_loss/roc_auc
// are absent without suitable binary probabilities or defined ROC classes.
struct ClassificationMetrics final {
    std::size_t sample_count = 0;
    double accuracy = 0.0;
    double macro_precision = 0.0;
    double macro_recall = 0.0;
    double macro_f1 = 0.0;
    double weighted_f1 = 0.0;
    std::optional<double> log_loss;
    std::optional<double> roc_auc;
    ConfusionMatrix confusion;
    std::vector<ClassMetrics> classes;
};

// Error statistics on finite prediction/target pairs. MAPE may be absent
// when every denominator is near zero (zero_tolerance policy).
struct RegressionMetrics final {
    std::size_t sample_count = 0;
    double mean_absolute_error = 0.0;
    double mean_squared_error = 0.0;
    double root_mean_squared_error = 0.0;
    double r2 = 0.0;
    std::optional<double> mean_absolute_percentage_error;
    std::size_t mape_sample_count = 0;
};

// Expects same-order, equally sized integral-valued labels. Optional scores
// are positive-class probabilities in [0,1], used for log loss and AUC.
[[nodiscard]] ClassificationMetrics calculate_classification_metrics(
    const std::vector<double>& actual,
    const std::vector<double>& predicted,
    const std::vector<double>& positive_class_scores = {});

// Calculates MAE/MSE/RMSE/R2 and near-zero-aware MAPE; not a fitting API.
[[nodiscard]] RegressionMetrics calculate_regression_metrics(
    const std::vector<double>& actual,
    const std::vector<double>& predicted,
    double zero_tolerance = 1e-12);

} // namespace mllibrary::evaluation
