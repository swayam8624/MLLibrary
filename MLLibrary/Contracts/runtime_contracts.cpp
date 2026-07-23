#if !defined(MLLIB_CONTRACTS_MODULE_BODY)
#include "runtime_contracts.hpp"
#include <algorithm>
#include <limits>
#include <utility>
#endif

namespace mllibrary::contracts
{
    ContractError::ContractError(ContractErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code)
    {
    }

    std::size_t data_type_size(DataType type)
    {
        switch (type)
        {
        case DataType::Float32:
        case DataType::Int32: return 4;
        case DataType::Float16:
        case DataType::BFloat16: return 2;
        case DataType::Int8:
        case DataType::UInt8: return 1;
        case DataType::Int64: return 8;
        }
        throw ContractError(ContractErrorCode::UnsupportedDataType, "Unknown Tensor data type.");
    }

    std::string_view data_type_name(DataType type) noexcept
    {
        switch (type)
        {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::BFloat16: return "bfloat16";
        case DataType::Int8: return "int8";
        case DataType::UInt8: return "uint8";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        }
        return "unknown";
    }

    std::string_view device_name(DeviceType device) noexcept
    {
        switch (device)
        {
        case DeviceType::CPUScalar: return "cpu-scalar";
        case DeviceType::CPUThreaded: return "cpu-threaded";
        case DeviceType::CPUSIMD: return "cpu-simd";
        case DeviceType::Metal: return "metal";
        case DeviceType::Vulkan: return "vulkan";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::WebGPU: return "webgpu";
        }
        return "unknown";
    }

    std::size_t TensorDescriptor::element_count() const
    {
        if (shape.empty()) throw ContractError(ContractErrorCode::InvalidShape, "Tensor shape must have at least one dimension.");
        std::size_t elements = 1;
        for (std::size_t dimension : shape)
        {
            if (dimension == 0) throw ContractError(ContractErrorCode::InvalidShape, "Tensor dimensions must be non-zero.");
            if (elements > std::numeric_limits<std::size_t>::max() / dimension)
                throw ContractError(ContractErrorCode::ShapeOverflow, "Tensor shape product overflows size_t.");
            elements *= dimension;
        }
        return elements;
    }

    std::size_t TensorDescriptor::byte_size() const
    {
        const std::size_t elements = element_count();
        const std::size_t width = data_type_size(dataType);
        if (elements > std::numeric_limits<std::size_t>::max() / width)
            throw ContractError(ContractErrorCode::ShapeOverflow, "Tensor byte size overflows size_t.");
        return elements * width;
    }

    bool TensorDescriptor::contiguous() const
    {
        if (shape.empty()) return false;
        if (strides.empty()) return layout != TensorLayout::Strided;
        if (strides.size() != shape.size()) return false;
        std::size_t expected = 1;
        for (std::size_t axis = shape.size(); axis-- > 0;)
        {
            if (strides[axis] != expected) return false;
            if (expected > std::numeric_limits<std::size_t>::max() / shape[axis]) return false;
            expected *= shape[axis];
        }
        return true;
    }

    void TensorDescriptor::validate() const
    {
        (void)byte_size();
        if (!strides.empty() && strides.size() != shape.size())
            throw ContractError(ContractErrorCode::InvalidStride, "Tensor strides must be empty or match rank.");
        if (layout == TensorLayout::Strided && strides.empty())
            throw ContractError(ContractErrorCode::InvalidStride, "Strided Tensor layout requires explicit strides.");
        if (ownership == StorageOwnership::DeviceOwned
            && device != DeviceType::Metal && device != DeviceType::Vulkan
            && device != DeviceType::CUDA && device != DeviceType::WebGPU)
            throw ContractError(ContractErrorCode::UnsupportedDevice, "Device-owned Tensor storage requires a GPU device.");
        if ((layout == TensorLayout::NCHW || layout == TensorLayout::NHWC || layout == TensorLayout::OHWI)
            && shape.size() != 4)
            throw ContractError(ContractErrorCode::UnsupportedLayout, "Image and filter Tensor layouts require rank four.");
    }

    void ModelCapabilityRegistry::register_model(ModelCapability capability)
    {
        if (capability.modelID.empty() || capability.dataTypes.empty() || capability.devices.empty()
            || capability.maximumRank == 0 || capability.maximumInputBytes == 0
            || (!capability.supportsInference && !capability.supportsTraining))
            throw std::invalid_argument("Model capability is incomplete.");
        const auto duplicate = std::find_if(capabilities_.begin(), capabilities_.end(),
            [&capability](const ModelCapability& existing) { return existing.modelID == capability.modelID; });
        if (duplicate != capabilities_.end())
            throw ContractError(ContractErrorCode::DuplicateCapability, "Model capability id is already registered.");
        capabilities_.push_back(std::move(capability));
        std::sort(capabilities_.begin(), capabilities_.end(),
            [](const ModelCapability& lhs, const ModelCapability& rhs) { return lhs.modelID < rhs.modelID; });
    }

    const ModelCapability& ModelCapabilityRegistry::require(std::string_view modelID) const
    {
        const auto found = std::lower_bound(capabilities_.begin(), capabilities_.end(), modelID,
            [](const ModelCapability& capability, std::string_view id) { return capability.modelID < id; });
        if (found == capabilities_.end() || found->modelID != modelID)
            throw ContractError(ContractErrorCode::CapabilityNotFound, "Requested model capability is not registered.");
        return *found;
    }

    const ModelCapability& ModelCapabilityRegistry::select(
        ModelTask task,
        DataType dataType,
        DeviceType device,
        bool requireTraining,
        std::size_t inputRank,
        std::size_t inputBytes) const
    {
        const auto found = std::find_if(capabilities_.begin(), capabilities_.end(),
            [&](const ModelCapability& capability)
            {
                return capability.task == task
                    && (!requireTraining ? capability.supportsInference : capability.supportsTraining)
                    && capability.maximumRank >= inputRank
                    && capability.maximumInputBytes >= inputBytes
                    && std::find(capability.dataTypes.begin(), capability.dataTypes.end(), dataType) != capability.dataTypes.end()
                    && std::find(capability.devices.begin(), capability.devices.end(), device) != capability.devices.end();
            });
        if (found == capabilities_.end())
            throw ContractError(ContractErrorCode::CapabilityNotFound, "No model satisfies the requested capability contract.");
        return *found;
    }

    void DatasetFormat::validate() const
    {
        if (version != CurrentVersion || identifier.empty() || sampleCount == 0)
            throw ContractError(ContractErrorCode::IncompatibleFormat, "Dataset format metadata is invalid or unsupported.");
        samples.validate();
        if (samples.shape.front() != sampleCount)
            throw ContractError(ContractErrorCode::InvalidShape, "Dataset sample count must equal the leading sample dimension.");
        if (labels)
        {
            labels->validate();
            if (labels->shape.front() != sampleCount)
                throw ContractError(ContractErrorCode::InvalidShape, "Dataset label count must equal the sample count.");
        }
    }

    void CheckpointFormat::validate() const
    {
        if (version != CurrentVersion || !includesParameters)
            throw ContractError(ContractErrorCode::IncompatibleFormat, "Checkpoint metadata is unsupported or omits parameters.");
    }
}
