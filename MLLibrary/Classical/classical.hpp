#pragma once

#include <cstddef>
#include <vector>

using DenseTable = std::vector<std::vector<float>>;

/// Standardizes tabular columns using training-set mean and population scale.
/// Constant columns are mapped to zero and retain a scale of one.
class StandardScaler final {
public:
    void fit(const DenseTable& samples);
    [[nodiscard]] DenseTable transform(const DenseTable& samples) const;
    [[nodiscard]] const std::vector<float>& mean() const noexcept { return mean_; }
    [[nodiscard]] const std::vector<float>& scale() const noexcept { return scale_; }

private:
    std::vector<float> mean_;
    std::vector<float> scale_;
};

/// Lloyd k-means with deterministic first-k initialization and an explicit
/// iteration cap. The fitted centers remain in feature space supplied to fit.
class KMeans final {
public:
    KMeans(std::size_t clusters, std::size_t maxIterations = 300, float tolerance = 1e-4f);
    void fit(const DenseTable& samples);
    [[nodiscard]] std::size_t predict(const std::vector<float>& sample) const;
    [[nodiscard]] const DenseTable& centers() const noexcept { return centers_; }

private:
    std::size_t clusters_;
    std::size_t maxIterations_;
    float tolerance_;
    DenseTable centers_;
};

/// Binary logistic regression optimized by deterministic full-batch gradient
/// descent. Labels must be exactly zero or one.
class LogisticRegression final {
public:
    LogisticRegression(float learningRate = 0.1f, std::size_t iterations = 500, float l2 = 0.0f);
    void fit(const DenseTable& samples, const std::vector<int>& labels);
    [[nodiscard]] float predict_probability(const std::vector<float>& sample) const;
    [[nodiscard]] int predict(const std::vector<float>& sample, float threshold = 0.5f) const;
    [[nodiscard]] const std::vector<float>& weights() const noexcept { return weights_; }
    [[nodiscard]] float bias() const noexcept { return bias_; }

private:
    float learningRate_;
    std::size_t iterations_;
    float l2_;
    std::vector<float> weights_;
    float bias_ = 0.0f;
};
