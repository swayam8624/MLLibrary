#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mllibrary::contracts
{
    enum class DataType : std::uint8_t
    {
        Float32,
        Float16,
        BFloat16,
        Int8,
        UInt8,
        Int32,
        Int64
    };

    enum class TensorLayout : std::uint8_t
    {
        RowMajor,
        NCHW,
        NHWC,
        OHWI,
        Strided
    };

    enum class DeviceType : std::uint8_t
    {
        CPUScalar,
        CPUThreaded,
        CPUSIMD,
        Metal,
        Vulkan,
        CUDA,
        WebGPU
    };

    enum class StorageOwnership : std::uint8_t
    {
        Owned,
        SharedView,
        Borrowed,
        MemoryMapped,
        DeviceOwned
    };

    enum class ContractErrorCode : std::uint8_t
    {
        InvalidShape,
        ShapeOverflow,
        InvalidStride,
        UnsupportedDataType,
        UnsupportedDevice,
        UnsupportedLayout,
        CapabilityNotFound,
        DuplicateCapability,
        IncompatibleFormat
    };

    class ContractError final : public std::runtime_error
    {
    public:
        ContractError(ContractErrorCode code, std::string message);
        [[nodiscard]] ContractErrorCode code() const noexcept { return code_; }

    private:
        ContractErrorCode code_;
    };

    [[nodiscard]] std::size_t data_type_size(DataType type);
    [[nodiscard]] std::string_view data_type_name(DataType type) noexcept;
    [[nodiscard]] std::string_view device_name(DeviceType device) noexcept;

    /// Canonical logical and physical Tensor description shared by model,
    /// dataset, serialization, benchmark, and backend boundaries.
    struct TensorDescriptor final
    {
        std::vector<std::size_t> shape;
        std::vector<std::size_t> strides;
        DataType dataType = DataType::Float32;
        TensorLayout layout = TensorLayout::RowMajor;
        DeviceType device = DeviceType::CPUScalar;
        StorageOwnership ownership = StorageOwnership::Owned;

        void validate() const;
        [[nodiscard]] std::size_t element_count() const;
        [[nodiscard]] std::size_t byte_size() const;
        [[nodiscard]] bool contiguous() const;
    };

    enum class ModelTask : std::uint8_t
    {
        Classification,
        Regression,
        Clustering,
        Embedding,
        Generation,
        Vision,
        Audio
    };

    struct ModelCapability final
    {
        std::string modelID;
        ModelTask task = ModelTask::Classification;
        std::vector<DataType> dataTypes;
        std::vector<DeviceType> devices;
        std::size_t maximumRank = 0;
        std::size_t maximumInputBytes = 0;
        bool supportsInference = true;
        bool supportsTraining = false;
    };

    class ModelCapabilityRegistry final
    {
    public:
        void register_model(ModelCapability capability);
        [[nodiscard]] const ModelCapability& require(std::string_view modelID) const;
        [[nodiscard]] const ModelCapability& select(
            ModelTask task,
            DataType dataType,
            DeviceType device,
            bool requireTraining,
            std::size_t inputRank,
            std::size_t inputBytes) const;
        [[nodiscard]] std::size_t size() const noexcept { return capabilities_.size(); }

    private:
        std::vector<ModelCapability> capabilities_;
    };

    struct DatasetFormat final
    {
        static constexpr std::uint32_t CurrentVersion = 1u;
        std::uint32_t version = CurrentVersion;
        std::string identifier;
        TensorDescriptor samples;
        std::optional<TensorDescriptor> labels;
        std::size_t sampleCount = 0;

        void validate() const;
    };

    struct CheckpointFormat final
    {
        static constexpr std::uint32_t CurrentVersion = 1u;
        std::uint32_t version = CurrentVersion;
        bool includesParameters = true;
        bool includesOptimizerState = false;
        bool includesRandomState = false;

        void validate() const;
    };
}
