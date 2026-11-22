// TEACHING PUBLIC API: fit a preprocessing state ONLY on training data,
// then use transform() with frozen statistics and category vocabulary on
// validation/test rows. CrossValidator constructs a fresh pipeline per fold.
// Implementation: Preprocessing/pipeline.cpp; see DATA_TO_AUTOML_LAB.md.
#pragma once

#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mllibrary::preprocessing {

using mllibrary::data::ColumnSchema;
using mllibrary::data::ColumnType;
using mllibrary::data::DataValue;
using mllibrary::data::Dataset;

class PreprocessingError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericImputation : std::uint8_t {
    Error,
    Mean,
    Median,
    Constant,
};

enum class NumericScaling : std::uint8_t {
    None,
    Standard,
    MinMax,
};

enum class UnknownCategoryPolicy : std::uint8_t {
    Ignore,
    Error,
};

struct PipelineOptions final {
    NumericImputation numeric_imputation = NumericImputation::Mean;
    NumericScaling numeric_scaling = NumericScaling::Standard;
    double numeric_constant = 0.0;
    UnknownCategoryPolicy unknown_category = UnknownCategoryPolicy::Ignore;
    bool include_boolean_columns = true;
    bool include_categorical_columns = true;
    bool drop_text_columns = true;
};

// Non-owning semantics are NOT implied: this struct owns a dense row-major
// double feature vector and separate typed targets, plus feature names.
// ExperimentRunner converts features to Float32 for classical estimators.
struct PreparedDataset final {
    std::size_t row_count = 0;
    std::size_t column_count = 0;
    std::vector<double> values;
    std::vector<std::string> feature_names;
    std::vector<DataValue> targets;
    std::optional<ColumnSchema> target_schema;

    [[nodiscard]] double feature(std::size_t row, std::size_t column) const;
};

// Stateful transformer: fit() learns numerical imputation/scaling and a
// sorted one-hot vocabulary, excludes the target, and caches schema lineage.
// Loaded pipelines must still check compatible schema on transform().
class FittedPipeline final {
public:
    explicit FittedPipeline(PipelineOptions options = {});

    // Mutates fitted state based only on the caller's training subset.
    // Never call using a validation/test dataset during model selection.
    void fit(const Dataset& training);
    // Read-only application of the learned state, with a declared policy
    // for unknown categories and missing values. Does not relearn anything.
    [[nodiscard]] PreparedDataset transform(const Dataset& dataset) const;
    [[nodiscard]] PreparedDataset fit_transform(const Dataset& training);

    [[nodiscard]] bool fitted() const noexcept { return fitted_; }
    [[nodiscard]] const PipelineOptions& options() const noexcept { return options_; }
    [[nodiscard]] const std::vector<std::string>& feature_names() const noexcept { return feature_names_; }
    [[nodiscard]] const std::string& training_fingerprint() const noexcept { return training_fingerprint_; }
    [[nodiscard]] const std::string& schema_signature() const noexcept { return schema_signature_; }

    void save(const std::string& path) const;
    [[nodiscard]] static FittedPipeline load(const std::string& path);

private:
    enum class StateKind : std::uint8_t {
        Numeric,
        Boolean,
        Categorical,
    };

    struct ColumnState final {
        StateKind kind = StateKind::Numeric;
        std::size_t source_index = 0;
        std::size_t feature_offset = 0;
        std::string name;
        double impute_value = 0.0;
        double offset = 0.0;
        double scale = 1.0;
        bool boolean_impute = false;
        bool has_missing_category = false;
        std::vector<std::string> categories;
    };

    [[nodiscard]] static std::string compute_schema_signature(const Dataset& dataset);
    void require_compatible(const Dataset& dataset) const;

    PipelineOptions options_{};
    bool fitted_ = false;
    std::string training_fingerprint_;
    std::string schema_signature_;
    std::vector<std::string> feature_names_;
    std::vector<ColumnState> states_;
};

} // namespace mllibrary::preprocessing
