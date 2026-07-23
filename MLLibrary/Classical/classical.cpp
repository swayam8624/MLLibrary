#include "classical.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

std::size_t validate_table(const DenseTable& samples)
{
    if (samples.empty() || samples.front().empty()) throw std::invalid_argument("Samples must be non-empty with at least one feature.");
    const std::size_t features = samples.front().size();
    for (const auto& sample : samples) if (sample.size() != features) throw std::invalid_argument("Samples must be rectangular.");
    return features;
}

float squared_distance(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::invalid_argument("Feature count mismatch.");
    float result = 0.0f;
    for (std::size_t i = 0; i < lhs.size(); ++i) { const float delta = lhs[i] - rhs[i]; result += delta * delta; }
    return result;
}

} // namespace

void StandardScaler::fit(const DenseTable& samples)
{
    const std::size_t features = validate_table(samples);
    mean_.assign(features, 0.0f);
    scale_.assign(features, 0.0f);
    for (const auto& sample : samples) for (std::size_t feature = 0; feature < features; ++feature) mean_[feature] += sample[feature];
    for (float& value : mean_) value /= static_cast<float>(samples.size());
    for (const auto& sample : samples) for (std::size_t feature = 0; feature < features; ++feature) { const float delta = sample[feature] - mean_[feature]; scale_[feature] += delta * delta; }
    for (float& value : scale_) { value = std::sqrt(value / static_cast<float>(samples.size())); if (value <= std::numeric_limits<float>::epsilon()) value = 1.0f; }
}

DenseTable StandardScaler::transform(const DenseTable& samples) const
{
    if (mean_.empty()) throw std::logic_error("StandardScaler must be fitted before transform.");
    const std::size_t features = validate_table(samples);
    if (features != mean_.size()) throw std::invalid_argument("Feature count differs from fitted scaler.");
    DenseTable result = samples;
    for (auto& sample : result) for (std::size_t feature = 0; feature < features; ++feature) sample[feature] = (sample[feature] - mean_[feature]) / scale_[feature];
    return result;
}

KMeans::KMeans(std::size_t clusters, std::size_t maxIterations, float tolerance)
    : clusters_(clusters), maxIterations_(maxIterations), tolerance_(tolerance)
{
    if (clusters == 0 || maxIterations == 0 || tolerance < 0.0f) throw std::invalid_argument("KMeans parameters are invalid.");
}

void KMeans::fit(const DenseTable& samples)
{
    const std::size_t features = validate_table(samples);
    if (clusters_ > samples.size()) throw std::invalid_argument("KMeans cluster count exceeds sample count.");
    centers_.clear();
    centers_.push_back(samples.front());
    while (centers_.size() < clusters_)
    {
        std::size_t farthest = 0;
        float farthestDistance = -1.0f;
        for (std::size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex)
        {
            float nearestDistance = squared_distance(samples[sampleIndex], centers_.front());
            for (std::size_t center = 1; center < centers_.size(); ++center)
            {
                nearestDistance = std::min(nearestDistance, squared_distance(samples[sampleIndex], centers_[center]));
            }
            if (nearestDistance > farthestDistance)
            {
                farthestDistance = nearestDistance;
                farthest = sampleIndex;
            }
        }
        centers_.push_back(samples[farthest]);
    }
    std::vector<std::size_t> assignments(samples.size());
    for (std::size_t iteration = 0; iteration < maxIterations_; ++iteration)
    {
        DenseTable next(clusters_, std::vector<float>(features, 0.0f));
        std::vector<std::size_t> counts(clusters_, 0);
        for (std::size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex)
        {
            assignments[sampleIndex] = predict(samples[sampleIndex]);
            ++counts[assignments[sampleIndex]];
            for (std::size_t feature = 0; feature < features; ++feature) next[assignments[sampleIndex]][feature] += samples[sampleIndex][feature];
        }
        float movement = 0.0f;
        for (std::size_t cluster = 0; cluster < clusters_; ++cluster)
        {
            if (counts[cluster] == 0) { next[cluster] = centers_[cluster]; continue; }
            for (float& value : next[cluster]) value /= static_cast<float>(counts[cluster]);
            movement += squared_distance(centers_[cluster], next[cluster]);
        }
        centers_ = std::move(next);
        if (movement <= tolerance_ * tolerance_) return;
    }
}

std::size_t KMeans::predict(const std::vector<float>& sample) const
{
    if (centers_.empty()) throw std::logic_error("KMeans must be fitted before predict.");
    std::size_t best = 0;
    float bestDistance = squared_distance(sample, centers_[0]);
    for (std::size_t cluster = 1; cluster < centers_.size(); ++cluster) { const float distance = squared_distance(sample, centers_[cluster]); if (distance < bestDistance) { best = cluster; bestDistance = distance; } }
    return best;
}

LogisticRegression::LogisticRegression(float learningRate, std::size_t iterations, float l2)
    : learningRate_(learningRate), iterations_(iterations), l2_(l2)
{
    if (!(learningRate > 0.0f) || iterations == 0 || l2 < 0.0f) throw std::invalid_argument("LogisticRegression parameters are invalid.");
}

void LogisticRegression::fit(const DenseTable& samples, const std::vector<int>& labels)
{
    const std::size_t features = validate_table(samples);
    if (labels.size() != samples.size()) throw std::invalid_argument("LogisticRegression labels must match samples.");
    for (int label : labels) if (label != 0 && label != 1) throw std::invalid_argument("LogisticRegression labels must be zero or one.");
    weights_.assign(features, 0.0f); bias_ = 0.0f;
    for (std::size_t iteration = 0; iteration < iterations_; ++iteration)
    {
        std::vector<float> gradient(features, 0.0f); float biasGradient = 0.0f;
        for (std::size_t row = 0; row < samples.size(); ++row)
        {
            const float error = predict_probability(samples[row]) - static_cast<float>(labels[row]);
            for (std::size_t feature = 0; feature < features; ++feature) gradient[feature] += error * samples[row][feature];
            biasGradient += error;
        }
        const float inverseCount = 1.0f / static_cast<float>(samples.size());
        for (std::size_t feature = 0; feature < features; ++feature) weights_[feature] -= learningRate_ * (gradient[feature] * inverseCount + l2_ * weights_[feature]);
        bias_ -= learningRate_ * biasGradient * inverseCount;
    }
}

float LogisticRegression::predict_probability(const std::vector<float>& sample) const
{
    if (weights_.empty() || sample.size() != weights_.size()) throw std::invalid_argument("LogisticRegression is unfitted or feature count differs.");
    float score = bias_;
    for (std::size_t feature = 0; feature < sample.size(); ++feature) score += sample[feature] * weights_[feature];
    return score >= 0.0f ? 1.0f / (1.0f + std::exp(-score)) : std::exp(score) / (1.0f + std::exp(score));
}

int LogisticRegression::predict(const std::vector<float>& sample, float threshold) const
{
    if (!(threshold > 0.0f && threshold < 1.0f)) throw std::invalid_argument("Prediction threshold must be between zero and one.");
    return predict_probability(sample) >= threshold ? 1 : 0;
}
