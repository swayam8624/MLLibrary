#include <cmath>
#include <cstddef>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorGPU;
import Kairo.GPU;
import MLLibrary.TensorGPURuntime;

int main()
{
    using kairo::foundation::math::Tensor;
    kairo::gpu::Device device({ .backend = kairo::gpu::Backend::Metal, .debugName = "mllibrary-gpu-test" });
    kairo::foundation::math::TensorGPUExecutor executor(device);
    const Tensor<float> inputs({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
    const Tensor<float> weights({ 2, 2 }, { 2, -1, 0.5f, 3 });
    const Tensor<float> bias({ 2 }, { 0.25f, -0.5f });

    const auto gpu = mllibrary::tensor_runtime::LinearForwardGPU(executor, inputs, weights, bias);
    auto cpu = kairo::foundation::math::MatMul(inputs, weights);
    for (std::size_t row = 0; row < cpu.Dim(0); ++row)
    {
        for (std::size_t column = 0; column < cpu.Dim(1); ++column)
        {
            cpu(row, column) += bias[column];
        }
    }
    for (std::size_t index = 0; index < gpu.Size(); ++index)
    {
        if (std::abs(gpu[index] - cpu[index]) > 1e-5f) return 1;
    }
    return 0;
}
