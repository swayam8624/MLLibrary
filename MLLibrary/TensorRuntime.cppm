module;

#include <cstddef>
#include <stdexcept>
#include <utility>

export module MLLibrary.TensorRuntime;

import Kairo.Foundation.Math.Tensor;

export namespace mllibrary::tensor_runtime
{
    using kairo::foundation::math::CrossEntropyMean;
    using kairo::foundation::math::MatMul;
    using kairo::foundation::math::SoftmaxLastDim;
    using kairo::foundation::math::Tensor;

    /// A dense, one-hot classification head trained directly on batch tensors.
    /// Inputs are [batch, features] and labels are [batch, classes]. Weights
    /// are [features, classes], which keeps MatMul contiguous and backend-ready.
    class TensorLinearClassifier final
    {
    public:
        TensorLinearClassifier(std::size_t features, std::size_t classes)
            : m_weights({ features, classes })
            , m_bias({ classes })
        {
            if (features == 0 || classes == 0)
            {
                throw std::invalid_argument("TensorLinearClassifier requires non-zero feature and class counts.");
            }
        }

        [[nodiscard]]
        std::size_t FeatureCount() const noexcept { return m_weights.Dim(0); }

        [[nodiscard]]
        std::size_t ClassCount() const noexcept { return m_weights.Dim(1); }

        [[nodiscard]]
        Tensor<float>& Weights() noexcept { return m_weights; }

        [[nodiscard]]
        const Tensor<float>& Weights() const noexcept { return m_weights; }

        [[nodiscard]]
        Tensor<float>& Bias() noexcept { return m_bias; }

        [[nodiscard]]
        const Tensor<float>& Bias() const noexcept { return m_bias; }

        [[nodiscard]]
        Tensor<float> Forward(const Tensor<float>& inputs) const
        {
            ValidateInputs(inputs);
            Tensor<float> logits = MatMul(inputs, m_weights);
            for (std::size_t row = 0; row < logits.Dim(0); ++row)
            {
                for (std::size_t column = 0; column < logits.Dim(1); ++column)
                {
                    logits(row, column) += m_bias[column];
                }
            }
            return logits;
        }

        /// Runs one full-batch softmax cross-entropy SGD update and returns the
        /// loss measured before the update. No mutable input or label tensor is
        /// retained, making batch ownership explicit for data pipelines.
        float TrainBatchSGD(const Tensor<float>& inputs, const Tensor<float>& labels, float learningRate)
        {
            ValidateInputs(inputs);
            ValidateLabels(inputs, labels);
            if (!(learningRate > 0.0f))
            {
                throw std::invalid_argument("TrainBatchSGD requires a positive learning rate.");
            }

            const Tensor<float> probabilities = SoftmaxLastDim(Forward(inputs));
            const float loss = CrossEntropyMean(labels, probabilities);
            Tensor<float> gradient = probabilities - labels;
            const float inverseBatch = 1.0f / static_cast<float>(inputs.Dim(0));
            gradient = gradient * inverseBatch;

            Tensor<float> weightGradient({ FeatureCount(), ClassCount() });
            Tensor<float> biasGradient({ ClassCount() });
            for (std::size_t feature = 0; feature < FeatureCount(); ++feature)
            {
                for (std::size_t category = 0; category < ClassCount(); ++category)
                {
                    float sum = 0.0f;
                    for (std::size_t sample = 0; sample < inputs.Dim(0); ++sample)
                    {
                        sum += inputs(sample, feature) * gradient(sample, category);
                    }
                    weightGradient(feature, category) = sum;
                }
            }
            for (std::size_t category = 0; category < ClassCount(); ++category)
            {
                float sum = 0.0f;
                for (std::size_t sample = 0; sample < inputs.Dim(0); ++sample)
                {
                    sum += gradient(sample, category);
                }
                biasGradient[category] = sum;
            }

            for (std::size_t index = 0; index < m_weights.Size(); ++index)
            {
                m_weights[index] -= learningRate * weightGradient[index];
            }
            for (std::size_t index = 0; index < m_bias.Size(); ++index)
            {
                m_bias[index] -= learningRate * biasGradient[index];
            }
            return loss;
        }

        [[nodiscard]]
        Tensor<std::size_t> Predict(const Tensor<float>& inputs) const
        {
            const Tensor<float> logits = Forward(inputs);
            Tensor<std::size_t> predictions({ logits.Dim(0) });
            for (std::size_t row = 0; row < logits.Dim(0); ++row)
            {
                std::size_t best = 0;
                for (std::size_t category = 1; category < ClassCount(); ++category)
                {
                    if (logits(row, category) > logits(row, best)) best = category;
                }
                predictions[row] = best;
            }
            return predictions;
        }

        [[nodiscard]]
        float Accuracy(const Tensor<float>& inputs, const Tensor<float>& labels) const
        {
            ValidateLabels(inputs, labels);
            const Tensor<std::size_t> predictions = Predict(inputs);
            std::size_t correct = 0;
            for (std::size_t row = 0; row < labels.Dim(0); ++row)
            {
                std::size_t expected = 0;
                for (std::size_t category = 1; category < ClassCount(); ++category)
                {
                    if (labels(row, category) > labels(row, expected)) expected = category;
                }
                correct += predictions[row] == expected;
            }
            return static_cast<float>(correct) / static_cast<float>(labels.Dim(0));
        }

    private:
        void ValidateInputs(const Tensor<float>& inputs) const
        {
            if (inputs.Rank() != 2 || inputs.Dim(0) == 0 || inputs.Dim(1) != FeatureCount())
            {
                throw std::invalid_argument("TensorLinearClassifier inputs must be [non-zero batch, feature count].");
            }
        }

        void ValidateLabels(const Tensor<float>& inputs, const Tensor<float>& labels) const
        {
            ValidateInputs(inputs);
            if (labels.Rank() != 2 || labels.Dim(0) != inputs.Dim(0) || labels.Dim(1) != ClassCount())
            {
                throw std::invalid_argument("TensorLinearClassifier labels must be one-hot [batch, class count].");
            }
        }

        Tensor<float> m_weights;
        Tensor<float> m_bias;
    };
}
