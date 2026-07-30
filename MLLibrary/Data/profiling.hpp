#pragma once
#include "dataset.hpp"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace mllibrary::data {

struct SchemaInferenceOptions final {
    CsvReadOptions csv{};
    std::size_t maximum_rows = 10000;
    std::size_t categorical_unique_threshold = 64;
    double categorical_unique_ratio = 0.20;
    std::size_t text_length_threshold = 96;
    std::optional<std::string> target_column;
};

struct ValueFrequency final {
    std::string value;
    std::size_t count = 0;
};

struct NumericProfile final {
    bool available = false;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double standard_deviation = 0.0;
};

struct ColumnProfile final {
    std::string name;
    ColumnType type = ColumnType::Numeric;
    bool nullable = false;
    bool target = false;
    std::size_t row_count = 0;
    std::size_t non_missing_count = 0;
    std::size_t missing_count = 0;
    std::size_t invalid_count = 0;
    std::size_t unique_count = 0;
    NumericProfile numeric{};
    std::vector<ValueFrequency> top_values;
};

struct DatasetProfileOptions final {
    std::size_t top_values_per_column = 10;
    bool detect_duplicate_rows = true;
};

struct DatasetProfile final {
    std::string source;
    std::string fingerprint;
    std::size_t row_count = 0;
    std::size_t column_count = 0;
    std::size_t duplicate_rows = 0;
    std::vector<ColumnProfile> columns;
    std::vector<ValueFrequency> target_distribution;
    double target_majority_fraction = 0.0;
    std::vector<std::string> warnings;
};

[[nodiscard]] DatasetSchema infer_csv_schema(
    const std::string& path,
    const SchemaInferenceOptions& options = {});

[[nodiscard]] Dataset load_inferred_csv(
    const std::string& path,
    const SchemaInferenceOptions& options = {});

[[nodiscard]] DatasetProfile profile_dataset(
    const Dataset& dataset,
    const DatasetProfileOptions& options = {});

[[nodiscard]] std::string dataset_profile_to_json(const DatasetProfile& profile);
[[nodiscard]] std::string dataset_profile_to_html(const DatasetProfile& profile);
void write_dataset_profile_json(const DatasetProfile& profile, const std::string& path);
void write_dataset_profile_html(const DatasetProfile& profile, const std::string& path);

} // namespace mllibrary::data
