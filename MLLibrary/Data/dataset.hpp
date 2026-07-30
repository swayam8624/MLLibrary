#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mllibrary::data {

enum class ColumnType : std::uint8_t {
    Numeric,
    Boolean,
    Categorical,
    Text,
};

struct ColumnSchema final {
    std::string name;
    ColumnType type = ColumnType::Numeric;
    bool nullable = false;
    bool target = false;
};

struct DatasetSchema final {
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    std::vector<ColumnSchema> columns;

    void validate() const;
    [[nodiscard]] std::size_t find_column(std::string_view name) const noexcept;
    [[nodiscard]] std::optional<std::size_t> target_column() const noexcept;
};

enum class DatasetErrorCode : std::uint8_t {
    Io,
    InvalidSchema,
    MalformedCsv,
    HeaderMismatch,
    ColumnCountMismatch,
    MissingValue,
    InvalidNumber,
    InvalidBoolean,
    TypeMismatch,
    OutOfRange,
    InvalidSplit,
};

class DatasetError final : public std::runtime_error {
public:
    DatasetError(
        DatasetErrorCode code,
        std::string message,
        std::size_t row = 0,
        std::size_t column = 0);

    [[nodiscard]] DatasetErrorCode code() const noexcept { return code_; }
    // One-based physical CSV row/column when available; zero means not applicable.
    [[nodiscard]] std::size_t row() const noexcept { return row_; }
    [[nodiscard]] std::size_t column() const noexcept { return column_; }

private:
    DatasetErrorCode code_;
    std::size_t row_;
    std::size_t column_;
};

using DataValue = std::variant<std::monostate, double, bool, std::string>;
using DataRow = std::vector<DataValue>;

struct CsvReadOptions final {
    char delimiter = ',';
    bool has_header = true;
    bool trim_unquoted_fields = true;
    bool skip_blank_records = true;
};

struct DatasetSplitOptions final {
    double training_fraction = 0.70;
    double validation_fraction = 0.15;
    double test_fraction = 0.15;
    std::uint64_t seed = 5489u;
    bool shuffle = true;
};

struct DatasetSplit;

class Dataset final {
public:
    Dataset() = default;
    Dataset(DatasetSchema schema, std::vector<DataRow> rows, std::string source = {});

    static Dataset from_csv(
        const std::string& path,
        DatasetSchema schema,
        CsvReadOptions options = {});

    [[nodiscard]] const DatasetSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const std::vector<DataRow>& rows() const noexcept { return rows_; }
    [[nodiscard]] const std::string& source() const noexcept { return source_; }
    [[nodiscard]] std::size_t row_count() const noexcept { return rows_.size(); }
    [[nodiscard]] std::size_t column_count() const noexcept { return schema_.columns.size(); }
    [[nodiscard]] bool empty() const noexcept { return rows_.empty(); }

    [[nodiscard]] const DataRow& row(std::size_t row_index) const;
    [[nodiscard]] const DataValue& value(std::size_t row_index, std::size_t column_index) const;
    [[nodiscard]] bool is_missing(std::size_t row_index, std::size_t column_index) const;
    [[nodiscard]] std::optional<double> numeric(
        std::size_t row_index, std::size_t column_index) const;
    [[nodiscard]] std::optional<bool> boolean(
        std::size_t row_index, std::size_t column_index) const;
    [[nodiscard]] std::optional<std::string_view> text(
        std::size_t row_index, std::size_t column_index) const;

    // Stable FNV-1a content fingerprint for reproducibility manifests. This is
    // intentionally not a cryptographic integrity hash.
    [[nodiscard]] std::uint64_t fingerprint64() const noexcept;
    [[nodiscard]] std::string fingerprint() const;

    // Preserves schema and source metadata while selecting rows in exactly the
    // supplied order. Duplicate indices are permitted intentionally for future
    // bootstrap and resampling workflows.
    [[nodiscard]] Dataset select_rows(
        const std::vector<std::size_t>& indices) const;
    [[nodiscard]] DatasetSplit split(const DatasetSplitOptions& options = {}) const;

private:
    void validate_rows() const;

    DatasetSchema schema_;
    std::vector<DataRow> rows_;
    std::string source_;
};

struct DatasetSplit final {
    Dataset training;
    Dataset validation;
    Dataset test;
};

[[nodiscard]] const char* column_type_name(ColumnType type) noexcept;

} // namespace mllibrary::data
