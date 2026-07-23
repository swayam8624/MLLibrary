#include "classical.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <random>
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

PCA::PCA(std::size_t components, std::size_t maxIterations, float tolerance)
    : componentCount_(components), maxIterations_(maxIterations), tolerance_(tolerance)
{
    if (components == 0 || maxIterations == 0 || tolerance <= 0.0f) throw std::invalid_argument("PCA parameters are invalid.");
}

void PCA::fit(const DenseTable& samples)
{
    const std::size_t features = validate_table(samples);
    if (componentCount_ > features) throw std::invalid_argument("PCA component count exceeds feature count.");
    mean_.assign(features, 0.0f);
    for (const auto& sample : samples) for (std::size_t feature = 0; feature < features; ++feature) mean_[feature] += sample[feature];
    for (float& value : mean_) value /= static_cast<float>(samples.size());
    DenseTable covariance(features, std::vector<float>(features, 0.0f));
    for (const auto& sample : samples)
    {
        for (std::size_t row = 0; row < features; ++row)
            for (std::size_t column = 0; column < features; ++column)
                covariance[row][column] += (sample[row] - mean_[row]) * (sample[column] - mean_[column]);
    }
    const float inverseCount = 1.0f / static_cast<float>(samples.size());
    for (auto& row : covariance) for (float& value : row) value *= inverseCount;
    components_.clear();
    for (std::size_t component = 0; component < componentCount_; ++component)
    {
        std::vector<float> vector(features, 1.0f / std::sqrt(static_cast<float>(features)));
        for (std::size_t iteration = 0; iteration < maxIterations_; ++iteration)
        {
            std::vector<float> next(features, 0.0f);
            for (std::size_t row = 0; row < features; ++row)
                for (std::size_t column = 0; column < features; ++column) next[row] += covariance[row][column] * vector[column];
            float norm = 0.0f;
            for (float value : next) norm += value * value;
            norm = std::sqrt(norm);
            if (norm <= std::numeric_limits<float>::epsilon()) throw std::logic_error("PCA covariance has insufficient remaining variance.");
            for (float& value : next) value /= norm;
            float difference = 0.0f;
            for (std::size_t index = 0; index < features; ++index) { const float delta = next[index] - vector[index]; difference += delta * delta; }
            vector = std::move(next);
            if (difference <= tolerance_ * tolerance_) break;
        }
        float eigenvalue = 0.0f;
        for (std::size_t row = 0; row < features; ++row)
            for (std::size_t column = 0; column < features; ++column) eigenvalue += vector[row] * covariance[row][column] * vector[column];
        components_.push_back(vector);
        for (std::size_t row = 0; row < features; ++row)
            for (std::size_t column = 0; column < features; ++column) covariance[row][column] -= eigenvalue * vector[row] * vector[column];
    }
}

DenseTable PCA::transform(const DenseTable& samples) const
{
    const std::size_t features = validate_table(samples);
    if (components_.empty() || features != mean_.size()) throw std::invalid_argument("PCA is unfitted or feature count differs.");
    DenseTable result(samples.size(), std::vector<float>(components_.size(), 0.0f));
    for (std::size_t row = 0; row < samples.size(); ++row)
        for (std::size_t component = 0; component < components_.size(); ++component)
            for (std::size_t feature = 0; feature < features; ++feature) result[row][component] += (samples[row][feature] - mean_[feature]) * components_[component][feature];
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

LinearRegression::LinearRegression(float learningRate, std::size_t iterations, float l2)
    : learningRate_(learningRate), iterations_(iterations), l2_(l2)
{
    if (!(learningRate > 0.0f) || iterations == 0 || l2 < 0.0f) throw std::invalid_argument("LinearRegression parameters are invalid.");
}

void LinearRegression::fit(const DenseTable& samples, const std::vector<float>& targets)
{
    const std::size_t features = validate_table(samples);
    if (targets.size() != samples.size()) throw std::invalid_argument("LinearRegression targets must match samples.");
    weights_.assign(features, 0.0f);
    bias_ = 0.0f;
    const float inverseCount = 1.0f / static_cast<float>(samples.size());
    for (std::size_t iteration = 0; iteration < iterations_; ++iteration)
    {
        std::vector<float> gradient(features, 0.0f);
        float biasGradient = 0.0f;
        for (std::size_t row = 0; row < samples.size(); ++row)
        {
            const float error = predict(samples[row]) - targets[row];
            for (std::size_t feature = 0; feature < features; ++feature) gradient[feature] += error * samples[row][feature];
            biasGradient += error;
        }
        for (std::size_t feature = 0; feature < features; ++feature) weights_[feature] -= learningRate_ * (gradient[feature] * inverseCount + l2_ * weights_[feature]);
        bias_ -= learningRate_ * biasGradient * inverseCount;
    }
}

float LinearRegression::predict(const std::vector<float>& sample) const
{
    if (weights_.empty() || sample.size() != weights_.size()) throw std::invalid_argument("LinearRegression is unfitted or feature count differs.");
    float result = bias_;
    for (std::size_t feature = 0; feature < sample.size(); ++feature) result += sample[feature] * weights_[feature];
    return result;
}

KNearestNeighbors::KNearestNeighbors(std::size_t neighbors) : neighbors_(neighbors)
{
    if (neighbors == 0) throw std::invalid_argument("KNearestNeighbors requires at least one neighbor.");
}

void KNearestNeighbors::fit(DenseTable samples, std::vector<int> labels)
{
    validate_table(samples);
    if (labels.size() != samples.size()) throw std::invalid_argument("KNearestNeighbors labels must match samples.");
    samples_ = std::move(samples);
    labels_ = std::move(labels);
}

int KNearestNeighbors::predict(const std::vector<float>& sample) const
{
    if (samples_.empty()) throw std::logic_error("KNearestNeighbors must be fitted before predict.");
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(samples_.size());
    for (std::size_t row = 0; row < samples_.size(); ++row) ranked.emplace_back(squared_distance(sample, samples_[row]), labels_[row]);
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first ? lhs.second < rhs.second : lhs.first < rhs.first; });
    std::vector<std::pair<int, std::size_t>> votes;
    for (std::size_t index = 0; index < std::min(neighbors_, ranked.size()); ++index)
    {
        const int label = ranked[index].second;
        auto found = std::find_if(votes.begin(), votes.end(), [label](const auto& vote) { return vote.first == label; });
        if (found == votes.end()) votes.emplace_back(label, 1); else ++found->second;
    }
    return std::max_element(votes.begin(), votes.end(), [](const auto& lhs, const auto& rhs) { return lhs.second == rhs.second ? lhs.first > rhs.first : lhs.second < rhs.second; })->first;
}

GaussianNaiveBayes::GaussianNaiveBayes(float varianceSmoothing) : varianceSmoothing_(varianceSmoothing)
{
    if (!(varianceSmoothing > 0.0f)) throw std::invalid_argument("GaussianNaiveBayes smoothing must be positive.");
}

void GaussianNaiveBayes::fit(const DenseTable& samples, const std::vector<int>& labels)
{
    const std::size_t features = validate_table(samples);
    if (labels.size() != samples.size()) throw std::invalid_argument("GaussianNaiveBayes labels must match samples.");
    classes_ = labels;
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    means_.assign(classes_.size(), std::vector<float>(features, 0.0f));
    variances_.assign(classes_.size(), std::vector<float>(features, 0.0f));
    logPriors_.assign(classes_.size(), 0.0f);
    std::vector<std::size_t> counts(classes_.size(), 0);
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        const std::size_t category = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), labels[row]) - classes_.begin());
        ++counts[category];
        for (std::size_t feature = 0; feature < features; ++feature) means_[category][feature] += samples[row][feature];
    }
    for (std::size_t category = 0; category < classes_.size(); ++category) for (float& value : means_[category]) value /= static_cast<float>(counts[category]);
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        const std::size_t category = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), labels[row]) - classes_.begin());
        for (std::size_t feature = 0; feature < features; ++feature) { const float delta = samples[row][feature] - means_[category][feature]; variances_[category][feature] += delta * delta; }
    }
    for (std::size_t category = 0; category < classes_.size(); ++category)
    {
        logPriors_[category] = std::log(static_cast<float>(counts[category]) / static_cast<float>(samples.size()));
        for (float& value : variances_[category]) value = value / static_cast<float>(counts[category]) + varianceSmoothing_;
    }
}

int GaussianNaiveBayes::predict(const std::vector<float>& sample) const
{
    if (classes_.empty() || sample.size() != means_.front().size()) throw std::invalid_argument("GaussianNaiveBayes is unfitted or feature count differs.");
    std::size_t best = 0;
    float bestLogProbability = -std::numeric_limits<float>::infinity();
    constexpr float kLogTwoPi = 1.8378770664f;
    for (std::size_t category = 0; category < classes_.size(); ++category)
    {
        float logProbability = logPriors_[category];
        for (std::size_t feature = 0; feature < sample.size(); ++feature)
        {
            const float variance = variances_[category][feature];
            const float delta = sample[feature] - means_[category][feature];
            logProbability += -0.5f * (kLogTwoPi + std::log(variance) + delta * delta / variance);
        }
        if (logProbability > bestLogProbability) { best = category; bestLogProbability = logProbability; }
    }
    return classes_[best];
}

namespace {

int majority_label(const std::vector<int>& labels, const std::vector<std::size_t>& rows)
{
    std::map<int, std::size_t> counts;
    for (std::size_t row : rows) ++counts[labels[row]];
    int bestLabel = counts.begin()->first;
    std::size_t bestCount = 0;
    for (const auto& [label, count] : counts)
    {
        if (count > bestCount)
        {
            bestLabel = label;
            bestCount = count;
        }
    }
    return bestLabel;
}

float gini_impurity(const std::map<int, std::size_t>& counts, std::size_t total)
{
    if (total == 0) return 0.0f;
    float squaredProbability = 0.0f;
    for (const auto& [label, count] : counts)
    {
        (void)label;
        const float probability = static_cast<float>(count) / static_cast<float>(total);
        squaredProbability += probability * probability;
    }
    return 1.0f - squaredProbability;
}

} // namespace

DecisionTreeClassifier::DecisionTreeClassifier(
    std::size_t maxDepth,
    std::size_t minSamplesSplit,
    std::size_t minSamplesLeaf,
    std::size_t maxFeatures,
    std::uint32_t seed)
    : maxDepth_(maxDepth)
    , minSamplesSplit_(minSamplesSplit)
    , minSamplesLeaf_(minSamplesLeaf)
    , maxFeatures_(maxFeatures)
    , seed_(seed)
{
    if (maxDepth == 0 || minSamplesSplit < 2 || minSamplesLeaf == 0)
        throw std::invalid_argument("DecisionTreeClassifier parameters are invalid.");
}

void DecisionTreeClassifier::fit(const DenseTable& samples, const std::vector<int>& labels)
{
    featureCount_ = validate_table(samples);
    if (labels.size() != samples.size()) throw std::invalid_argument("Decision-tree labels must match samples.");
    if (maxFeatures_ > featureCount_) throw std::invalid_argument("Decision-tree maxFeatures exceeds feature count.");
    nodes_.clear();
    std::vector<std::size_t> rows(samples.size());
    std::iota(rows.begin(), rows.end(), 0);
    build_node(samples, labels, rows, 0);
}

std::size_t DecisionTreeClassifier::build_node(
    const DenseTable& samples,
    const std::vector<int>& labels,
    const std::vector<std::size_t>& rows,
    std::size_t depth)
{
    Node node;
    node.prediction = majority_label(labels, rows);
    const std::size_t nodeIndex = nodes_.size();
    nodes_.push_back(node);

    std::map<int, std::size_t> parentCounts;
    for (std::size_t row : rows) ++parentCounts[labels[row]];
    if (depth >= maxDepth_ || rows.size() < minSamplesSplit_ || parentCounts.size() == 1)
        return nodeIndex;

    std::vector<std::size_t> features(featureCount_);
    std::iota(features.begin(), features.end(), 0);
    const std::size_t candidateFeatureCount = maxFeatures_ == 0 ? featureCount_ : maxFeatures_;
    if (candidateFeatureCount < featureCount_)
    {
        std::mt19937 generator(seed_ ^ static_cast<std::uint32_t>(nodeIndex * 0x9e3779b9u));
        std::shuffle(features.begin(), features.end(), generator);
        features.resize(candidateFeatureCount);
    }

    const float parentGini = gini_impurity(parentCounts, rows.size());
    float bestGain = 0.0f;
    std::size_t bestFeature = 0;
    float bestThreshold = 0.0f;
    bool splitFound = false;
    for (std::size_t feature : features)
    {
        std::vector<std::pair<float, int>> ordered;
        ordered.reserve(rows.size());
        for (std::size_t row : rows) ordered.emplace_back(samples[row][feature], labels[row]);
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs)
        {
            return lhs.first < rhs.first;
        });

        std::map<int, std::size_t> leftCounts;
        std::map<int, std::size_t> rightCounts = parentCounts;
        for (std::size_t position = 0; position + 1 < ordered.size(); ++position)
        {
            ++leftCounts[ordered[position].second];
            auto right = rightCounts.find(ordered[position].second);
            if (--right->second == 0) rightCounts.erase(right);
            const std::size_t leftSize = position + 1;
            const std::size_t rightSize = ordered.size() - leftSize;
            if (leftSize < minSamplesLeaf_ || rightSize < minSamplesLeaf_
                || ordered[position].first == ordered[position + 1].first) continue;

            const float threshold = ordered[position].first
                + (ordered[position + 1].first - ordered[position].first) * 0.5f;
            const float weightedGini =
                static_cast<float>(leftSize) / rows.size() * gini_impurity(leftCounts, leftSize)
                + static_cast<float>(rightSize) / rows.size() * gini_impurity(rightCounts, rightSize);
            const float gain = parentGini - weightedGini;
            constexpr float kTieTolerance = 1e-7f;
            if (!splitFound || gain > bestGain + kTieTolerance
                || (std::abs(gain - bestGain) <= kTieTolerance
                    && (feature < bestFeature || (feature == bestFeature && threshold < bestThreshold))))
            {
                splitFound = true;
                bestGain = gain;
                bestFeature = feature;
                bestThreshold = threshold;
            }
        }
    }
    if (!splitFound || bestGain < -1e-7f) return nodeIndex;

    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;
    leftRows.reserve(rows.size());
    rightRows.reserve(rows.size());
    for (std::size_t row : rows)
    {
        (samples[row][bestFeature] <= bestThreshold ? leftRows : rightRows).push_back(row);
    }
    if (leftRows.size() < minSamplesLeaf_ || rightRows.size() < minSamplesLeaf_) return nodeIndex;

    const std::size_t left = build_node(samples, labels, leftRows, depth + 1);
    const std::size_t right = build_node(samples, labels, rightRows, depth + 1);
    nodes_[nodeIndex].leaf = false;
    nodes_[nodeIndex].feature = bestFeature;
    nodes_[nodeIndex].threshold = bestThreshold;
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    return nodeIndex;
}

int DecisionTreeClassifier::predict(const std::vector<float>& sample) const
{
    if (nodes_.empty() || sample.size() != featureCount_)
        throw std::invalid_argument("DecisionTreeClassifier is unfitted or feature count differs.");
    std::size_t nodeIndex = 0;
    while (!nodes_[nodeIndex].leaf)
    {
        const Node& node = nodes_[nodeIndex];
        nodeIndex = sample[node.feature] <= node.threshold ? node.left : node.right;
    }
    return nodes_[nodeIndex].prediction;
}

RandomForestClassifier::RandomForestClassifier(
    std::size_t trees,
    std::size_t maxDepth,
    std::size_t minSamplesSplit,
    std::size_t minSamplesLeaf,
    std::size_t maxFeatures,
    std::uint32_t seed)
    : treeCount_(trees)
    , maxDepth_(maxDepth)
    , minSamplesSplit_(minSamplesSplit)
    , minSamplesLeaf_(minSamplesLeaf)
    , maxFeatures_(maxFeatures)
    , seed_(seed)
{
    if (trees == 0 || maxDepth == 0 || minSamplesSplit < 2 || minSamplesLeaf == 0)
        throw std::invalid_argument("RandomForestClassifier parameters are invalid.");
}

void RandomForestClassifier::fit(const DenseTable& samples, const std::vector<int>& labels)
{
    const std::size_t features = validate_table(samples);
    if (labels.size() != samples.size()) throw std::invalid_argument("Random-forest labels must match samples.");
    const std::size_t featureSubset = maxFeatures_ == 0
        ? std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(features))))
        : maxFeatures_;
    if (featureSubset > features) throw std::invalid_argument("Random-forest maxFeatures exceeds feature count.");

    forest_.clear();
    forest_.reserve(treeCount_);
    std::mt19937 generator(seed_);
    std::uniform_int_distribution<std::size_t> sampleDistribution(0, samples.size() - 1);
    for (std::size_t treeIndex = 0; treeIndex < treeCount_; ++treeIndex)
    {
        DenseTable bootstrapSamples;
        std::vector<int> bootstrapLabels;
        bootstrapSamples.reserve(samples.size());
        bootstrapLabels.reserve(labels.size());
        for (std::size_t row = 0; row < samples.size(); ++row)
        {
            const std::size_t selected = sampleDistribution(generator);
            bootstrapSamples.push_back(samples[selected]);
            bootstrapLabels.push_back(labels[selected]);
        }
        forest_.emplace_back(
            maxDepth_,
            minSamplesSplit_,
            minSamplesLeaf_,
            featureSubset,
            seed_ ^ static_cast<std::uint32_t>((treeIndex + 1) * 0x85ebca6bu));
        forest_.back().fit(bootstrapSamples, bootstrapLabels);
    }
}

int RandomForestClassifier::predict(const std::vector<float>& sample) const
{
    if (forest_.empty()) throw std::logic_error("RandomForestClassifier must be fitted before predict.");
    std::map<int, std::size_t> votes;
    for (const DecisionTreeClassifier& tree : forest_) ++votes[tree.predict(sample)];
    int bestLabel = votes.begin()->first;
    std::size_t bestCount = 0;
    for (const auto& [label, count] : votes)
    {
        if (count > bestCount)
        {
            bestLabel = label;
            bestCount = count;
        }
    }
    return bestLabel;
}
