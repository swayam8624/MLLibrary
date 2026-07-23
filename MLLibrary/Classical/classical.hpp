#pragma once

#include <cstddef>
#include <cstdint>
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

/// Principal component analysis using centered covariance, deterministic power
/// iteration, and rank-one deflation. Components are stored in descending
/// extracted-variance order and transform rows into component coordinates.
class PCA final {
public:
    PCA(std::size_t components, std::size_t maxIterations = 500, float tolerance = 1e-6f);
    void fit(const DenseTable& samples);
    [[nodiscard]] DenseTable transform(const DenseTable& samples) const;
    [[nodiscard]] const DenseTable& components() const noexcept { return components_; }
    [[nodiscard]] const std::vector<float>& mean() const noexcept { return mean_; }

private:
    std::size_t componentCount_;
    std::size_t maxIterations_;
    float tolerance_;
    std::vector<float> mean_;
    DenseTable components_;
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

/// Single-output linear regression optimized by deterministic full-batch
/// gradient descent. Targets are continuous scalar values.
class LinearRegression final {
public:
    LinearRegression(float learningRate = 0.05f, std::size_t iterations = 1000, float l2 = 0.0f);
    void fit(const DenseTable& samples, const std::vector<float>& targets);
    [[nodiscard]] float predict(const std::vector<float>& sample) const;
    [[nodiscard]] const std::vector<float>& weights() const noexcept { return weights_; }
    [[nodiscard]] float bias() const noexcept { return bias_; }

private:
    float learningRate_;
    std::size_t iterations_;
    float l2_;
    std::vector<float> weights_;
    float bias_ = 0.0f;
};

/// Stores a labelled tabular set and predicts by deterministic Euclidean
/// nearest-neighbor voting. Equal vote counts resolve to the smaller label.
class KNearestNeighbors final {
public:
    explicit KNearestNeighbors(std::size_t neighbors = 5);
    void fit(DenseTable samples, std::vector<int> labels);
    [[nodiscard]] int predict(const std::vector<float>& sample) const;

private:
    std::size_t neighbors_;
    DenseTable samples_;
    std::vector<int> labels_;
};

/// Gaussian naive Bayes for integer class labels. Variance smoothing prevents
/// a constant feature in one class from producing division by zero.
class GaussianNaiveBayes final {
public:
    explicit GaussianNaiveBayes(float varianceSmoothing = 1e-6f);
    void fit(const DenseTable& samples, const std::vector<int>& labels);
    [[nodiscard]] int predict(const std::vector<float>& sample) const;

private:
    float varianceSmoothing_;
    std::vector<int> classes_;
    DenseTable means_;
    DenseTable variances_;
    std::vector<float> logPriors_;
};

/// CART-style classification tree using exhaustive midpoint candidates and
/// Gini impurity. Labels may be any integers. Equal-quality splits resolve by
/// feature index then threshold so a fixed seed is reproducible.
class DecisionTreeClassifier final {
public:
    DecisionTreeClassifier(
        std::size_t maxDepth = 16,
        std::size_t minSamplesSplit = 2,
        std::size_t minSamplesLeaf = 1,
        std::size_t maxFeatures = 0,
        std::uint32_t seed = 0);
    void fit(const DenseTable& samples, const std::vector<int>& labels);
    [[nodiscard]] int predict(const std::vector<float>& sample) const;
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t feature_count() const noexcept { return featureCount_; }

private:
    struct Node final {
        std::size_t feature = 0;
        float threshold = 0.0f;
        std::size_t left = 0;
        std::size_t right = 0;
        int prediction = 0;
        bool leaf = true;
    };

    std::size_t build_node(
        const DenseTable& samples,
        const std::vector<int>& labels,
        const std::vector<std::size_t>& rows,
        std::size_t depth);

    std::size_t maxDepth_;
    std::size_t minSamplesSplit_;
    std::size_t minSamplesLeaf_;
    std::size_t maxFeatures_;
    std::uint32_t seed_;
    std::size_t featureCount_ = 0;
    std::vector<Node> nodes_;
};

/// Deterministic bootstrap-aggregated classification forest. Each tree uses a
/// seeded bootstrap sample and seeded per-node feature subset. Vote ties resolve
/// to the smaller integer class label.
class RandomForestClassifier final {
public:
    RandomForestClassifier(
        std::size_t trees = 100,
        std::size_t maxDepth = 16,
        std::size_t minSamplesSplit = 2,
        std::size_t minSamplesLeaf = 1,
        std::size_t maxFeatures = 0,
        std::uint32_t seed = 5489u);
    void fit(const DenseTable& samples, const std::vector<int>& labels);
    [[nodiscard]] int predict(const std::vector<float>& sample) const;
    [[nodiscard]] std::size_t tree_count() const noexcept { return forest_.size(); }

private:
    std::size_t treeCount_;
    std::size_t maxDepth_;
    std::size_t minSamplesSplit_;
    std::size_t minSamplesLeaf_;
    std::size_t maxFeatures_;
    std::uint32_t seed_;
    std::vector<DecisionTreeClassifier> forest_;
};
