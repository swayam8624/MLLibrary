import Kairo.Foundation.Math.Tensor;
import MLLibrary.TensorRuntime;

using kairo::foundation::math::CrossEntropyMean;
using kairo::foundation::math::MatMul;
using kairo::foundation::math::SoftmaxLastDim;
using kairo::foundation::math::Tensor;
using mllibrary::tensor_runtime::TensorLinearClassifier;

int main()
{
    Tensor<float> lhs({ 2, 3 }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    Tensor<float> rhs({ 3, 2 }, { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f });
    const auto product = MatMul(lhs, rhs);
    if (product(0, 0) != 4.0f || product(1, 1) != 11.0f) {
        return 1;
    }

    Tensor<float> labels({ 1, 2 }, { 1.0f, 0.0f });
    Tensor<float> logits({ 1, 2 }, { 2.0f, 0.0f });
    const auto probabilities = SoftmaxLastDim(logits);
    const float loss = CrossEntropyMean(labels, probabilities);
    if (!(probabilities(0, 0) > probabilities(0, 1)) || !(loss > 0.0f && loss < 1.0f)) {
        return 1;
    }

    Tensor<float> trainInputs({ 4, 2 }, {
        1.0f, 0.0f,
        0.8f, 0.1f,
        0.0f, 1.0f,
        0.1f, 0.8f,
    });
    Tensor<float> trainLabels({ 4, 2 }, {
        1.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 1.0f,
    });
    TensorLinearClassifier classifier(2, 2);
    const float initialLoss = classifier.TrainBatchSGD(trainInputs, trainLabels, 0.25f);
    float finalLoss = initialLoss;
    for (int iteration = 0; iteration < 80; ++iteration) {
        finalLoss = classifier.TrainBatchSGD(trainInputs, trainLabels, 0.25f);
    }
    if (!(finalLoss < initialLoss && classifier.Accuracy(trainInputs, trainLabels) == 1.0f)) {
        return 1;
    }
    return 0;
}
