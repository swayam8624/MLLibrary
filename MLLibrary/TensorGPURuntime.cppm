module;

#include <cstddef>
#include <stdexcept>

export module MLLibrary.TensorGPURuntime;

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorGPU;

export namespace mllibrary::tensor_runtime
{
    using kairo::foundation::math::Tensor;
    using kairo::foundation::math::TensorGPUExecutor;

    /// Executes the dense matrix product on GPU, then applies the classifier
    /// bias on the synchronized host result. The explicit split preserves
    /// numerical equivalence while KairoGPU grows a broadcast-add kernel and
    /// persistent Tensor device storage.
    [[nodiscard]]
    inline Tensor<float> LinearForwardGPU(
        TensorGPUExecutor& executor,
        const Tensor<float>& inputs,
        const Tensor<float>& weights,
        const Tensor<float>& bias)
    {
        if (inputs.Rank() != 2 || weights.Rank() != 2 || bias.Rank() != 1
            || inputs.Dim(0) == 0 || inputs.Dim(1) != weights.Dim(0)
            || weights.Dim(1) != bias.Dim(0))
        {
            throw std::invalid_argument(
                "LinearForwardGPU expects [batch,features], [features,outputs], and [outputs].");
        }
        Tensor<float> logits = executor.MatMul(inputs, weights);
        for (std::size_t row = 0; row < logits.Dim(0); ++row)
        {
            for (std::size_t column = 0; column < logits.Dim(1); ++column)
            {
                logits(row, column) += bias[column];
            }
        }
        return logits;
    }
}
