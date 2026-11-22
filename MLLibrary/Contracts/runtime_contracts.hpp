// TEACHING CONTRACT API: dtype, logical shape, strides, ownership and
// device/algorithm capability metadata used at model/backend boundaries.
// Enums describe possible choices; they do not prove that a GPU backend or
// kernel exists on the current host. See OBSERVABILITY_AND_BENCHMARK_LAB.
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
    // Logical shape and physical storage contract. validate/element_count/
    // byte_size reject overflow; contiguous checks effective dense strides.
    // Descriptor validation does not allocate the backing memory.
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

    // Store named capabilities and select by task/dtype/device/shape budget.
    // Matching metadata is not a promise of successful native execution.
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

    // Metadata contract for sample and optional label tensors. Version and
    // shape checks do not verify the actual bytes on disk.
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

    // Format feature flags are declarative. In particular classic KMLCP01
    // files serialize weights only, regardless of broader metadata concepts.
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
