#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace mllibrary::preprocessing {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_byte(std::uint64_t& hash, std::uint8_t value) noexcept
{
    hash ^= value;
    hash *= fnv_prime;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        hash_byte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffu));
}

void hash_string(std::uint64_t& hash, const std::string& value) noexcept
{
    hash_u64(hash, value.size());
    for (unsigned char character : value) hash_byte(hash, character);
}

std::string hex64(std::uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

std::string category_feature_name(const std::string& column, const std::string& value)
{
    return column + "=" + value;
}

void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw PreprocessingError(std::string(name) + " must be finite.");
}

int enum_value(NumericImputation value) { return static_cast<int>(value); }
int enum_value(NumericScaling value) { return static_cast<int>(value); }
int enum_value(UnknownCategoryPolicy value) { return static_cast<int>(value); }

} // namespace

double PreparedDataset::feature(std::size_t row, std::size_t column) const
{
    if (row >= row_count || column >= column_count)
        throw PreprocessingError("Prepared feature index is out of range.");
    return values[row * column_count + column];
}

FittedPipeline::FittedPipeline(PipelineOptions options)
    : options_(options)
{
    require_finite(options_.numeric_constant, "Numeric imputation constant");
}

std::string FittedPipeline::compute_schema_signature(const Dataset& dataset)
{
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, dataset.schema().columns.size());
    for (const ColumnSchema& column : dataset.schema().columns)
    {
        hash_string(hash, column.name);
        hash_byte(hash, static_cast<std::uint8_t>(column.type));
        hash_byte(hash, column.nullable ? 1u : 0u);
        hash_byte(hash, column.target ? 1u : 0u);
    }
    return hex64(hash);
}

void FittedPipeline::fit(const Dataset& training)
{
    if (training.empty())
        throw PreprocessingError("A preprocessing pipeline cannot fit an empty dataset.");
    require_finite(options_.numeric_constant, "Numeric imputation constant");

    std::vector<ColumnState> next_states;
    std::vector<std::string> next_feature_names;

    for (std::size_t column_index = 0; column_index < training.column_count(); ++column_index)
    {
        const ColumnSchema& schema = training.schema().columns[column_index];
        if (schema.target) continue;

        if (schema.type == ColumnType::Numeric)
        {
            std::vector<double> observed;
            observed.reserve(training.row_count());
            std::size_t missing = 0;
            for (std::size_t row = 0; row < training.row_count(); ++row)
            {
                const std::optional<double> value = training.numeric(row, column_index);
                if (value) observed.push_back(*value); else ++missing;
            }

            if (missing > 0 && options_.numeric_imputation == NumericImputation::Error)
                throw PreprocessingError("Numeric column '" + schema.name + "' contains missing values.");
            if (observed.empty() && options_.numeric_imputation != NumericImputation::Constant)
                throw PreprocessingError("Numeric column '" + schema.name + "' has no values to fit.");

            ColumnState state;
            state.kind = StateKind::Numeric;
            state.source_index = column_index;
            state.feature_offset = next_feature_names.size();
            state.name = schema.name;

            if (options_.numeric_imputation == NumericImputation::Constant)
            {
                state.impute_value = options_.numeric_constant;
            }
            else if (options_.numeric_imputation == NumericImputation::Median)
            {
                std::vector<double> sorted = observed;
                std::sort(sorted.begin(), sorted.end());
                const std::size_t middle = sorted.size() / 2;
                state.impute_value = sorted.size() % 2 == 0
                    ? (sorted[middle - 1] + sorted[middle]) * 0.5
                    : sorted[middle];
            }
            else
            {
                state.impute_value = std::accumulate(observed.begin(), observed.end(), 0.0)
                    / static_cast<double>(observed.size());
            }

            if (options_.numeric_scaling == NumericScaling::Standard)
            {
                const double mean = observed.empty() ? state.impute_value
                    : std::accumulate(observed.begin(), observed.end(), 0.0)
                        / static_cast<double>(observed.size());
                double variance = 0.0;
                for (double value : observed)
                {
                    const double delta = value - mean;
                    variance += delta * delta;
                }
                if (!observed.empty()) variance /= static_cast<double>(observed.size());
                state.offset = mean;
                state.scale = std::sqrt(variance);
                if (!(state.scale > 0.0) || !std::isfinite(state.scale)) state.scale = 1.0;
            }
            else if (options_.numeric_scaling == NumericScaling::MinMax)
            {
                const auto bounds = std::minmax_element(observed.begin(), observed.end());
                state.offset = observed.empty() ? state.impute_value : *bounds.first;
                const double maximum = observed.empty() ? state.impute_value : *bounds.second;
                state.scale = maximum - state.offset;
                if (!(state.scale > 0.0) || !std::isfinite(state.scale)) state.scale = 1.0;
            }

            next_feature_names.push_back(schema.name);
            next_states.push_back(std::move(state));
            continue;
        }

        if (schema.type == ColumnType::Boolean)
        {
            if (!options_.include_boolean_columns) continue;
            std::size_t true_count = 0;
            std::size_t false_count = 0;
            for (std::size_t row = 0; row < training.row_count(); ++row)
            {
                const std::optional<bool> value = training.boolean(row, column_index);
                if (!value) continue;
                if (*value) ++true_count; else ++false_count;
            }
            ColumnState state;
            state.kind = StateKind::Boolean;
            state.source_index = column_index;
            state.feature_offset = next_feature_names.size();
            state.name = schema.name;
            state.boolean_impute = true_count > false_count;
            next_feature_names.push_back(schema.name);
            next_states.push_back(std::move(state));
            continue;
        }

        if (schema.type == ColumnType::Categorical)
        {
            if (!options_.include_categorical_columns) continue;
            std::unordered_set<std::string> unique;
            bool missing = false;
            for (std::size_t row = 0; row < training.row_count(); ++row)
            {
                const std::optional<std::string_view> value = training.text(row, column_index);
                if (value) unique.emplace(*value); else missing = true;
            }
            if (unique.empty() && !missing)
                throw PreprocessingError("Categorical column '" + schema.name + "' has no values to fit.");

            ColumnState state;
            state.kind = StateKind::Categorical;
            state.source_index = column_index;
            state.feature_offset = next_feature_names.size();
            state.name = schema.name;
            state.has_missing_category = missing;
            state.categories.assign(unique.begin(), unique.end());
            std::sort(state.categories.begin(), state.categories.end());
            for (const std::string& category : state.categories)
                next_feature_names.push_back(category_feature_name(schema.name, category));
            if (missing)
                next_feature_names.push_back(category_feature_name(schema.name, "<missing>"));
            next_states.push_back(std::move(state));
            continue;
        }

        if (schema.type == ColumnType::Text && !options_.drop_text_columns)
            throw PreprocessingError("Text column '" + schema.name + "' requires an explicit text transform.");
    }

    if (next_feature_names.empty())
        throw PreprocessingError("The fitted pipeline produced no feature columns.");

    states_ = std::move(next_states);
    feature_names_ = std::move(next_feature_names);
    training_fingerprint_ = training.fingerprint();
    schema_signature_ = compute_schema_signature(training);
    fitted_ = true;
}

void FittedPipeline::require_compatible(const Dataset& dataset) const
{
    if (!fitted_) throw PreprocessingError("Preprocessing pipeline must be fitted before transform.");
    if (compute_schema_signature(dataset) != schema_signature_)
        throw PreprocessingError("Dataset schema does not match the fitted preprocessing pipeline.");
}

PreparedDataset FittedPipeline::transform(const Dataset& dataset) const
{
    require_compatible(dataset);
    PreparedDataset result;
    result.row_count = dataset.row_count();
    result.column_count = feature_names_.size();
    result.feature_names = feature_names_;
    result.values.assign(result.row_count * result.column_count, 0.0);

    if (const std::optional<std::size_t> target = dataset.schema().target_column())
    {
        result.target_schema = dataset.schema().columns[*target];
        result.targets.reserve(dataset.row_count());
        for (std::size_t row = 0; row < dataset.row_count(); ++row)
            result.targets.push_back(dataset.value(row, *target));
    }

    for (std::size_t row = 0; row < dataset.row_count(); ++row)
    {
        for (const ColumnState& state : states_)
        {
            if (state.kind == StateKind::Numeric)
            {
                const std::optional<double> stored = dataset.numeric(row, state.source_index);
                if (!stored && options_.numeric_imputation == NumericImputation::Error)
                    throw PreprocessingError("Transform encountered a missing value in numeric column '" + state.name + "'.");
                const double value = stored.value_or(state.impute_value);
                result.values[row * result.column_count + state.feature_offset]
                    = (value - state.offset) / state.scale;
            }
            else if (state.kind == StateKind::Boolean)
            {
                const std::optional<bool> stored = dataset.boolean(row, state.source_index);
                result.values[row * result.column_count + state.feature_offset]
                    = stored.value_or(state.boolean_impute) ? 1.0 : 0.0;
            }
            else
            {
                const std::optional<std::string_view> stored = dataset.text(row, state.source_index);
                if (!stored)
                {
                    if (state.has_missing_category)
                    {
                        const std::size_t feature = state.feature_offset + state.categories.size();
                        result.values[row * result.column_count + feature] = 1.0;
                    }
                    else if (options_.unknown_category == UnknownCategoryPolicy::Error)
                    {
                        throw PreprocessingError("Transform encountered an unfitted missing category in column '" + state.name + "'.");
                    }
                    continue;
                }

                const auto found = std::lower_bound(state.categories.begin(), state.categories.end(), *stored);
                if (found == state.categories.end() || *found != *stored)
                {
                    if (options_.unknown_category == UnknownCategoryPolicy::Error)
                        throw PreprocessingError("Unknown category '" + std::string(*stored)
                            + "' in column '" + state.name + "'.");
                    continue;
                }
                const std::size_t category = static_cast<std::size_t>(found - state.categories.begin());
                result.values[row * result.column_count + state.feature_offset + category] = 1.0;
            }
        }
    }
    return result;
}

PreparedDataset FittedPipeline::fit_transform(const Dataset& training)
{
    fit(training);
    return transform(training);
}

void FittedPipeline::save(const std::string& path) const
{
    if (!fitted_) throw PreprocessingError("Cannot save an unfitted preprocessing pipeline.");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw PreprocessingError("Cannot open preprocessing pipeline file '" + path + "'.");

    output << "KAIROML_PIPELINE 1\n";
    output << std::quoted(schema_signature_) << ' ' << std::quoted(training_fingerprint_) << '\n';
    output << enum_value(options_.numeric_imputation) << ' '
           << enum_value(options_.numeric_scaling) << ' '
           << std::setprecision(17) << options_.numeric_constant << ' '
           << enum_value(options_.unknown_category) << ' '
           << options_.include_boolean_columns << ' '
           << options_.include_categorical_columns << ' '
           << options_.drop_text_columns << '\n';
    output << feature_names_.size();
    for (const std::string& name : feature_names_) output << ' ' << std::quoted(name);
    output << '\n' << states_.size() << '\n';

    for (const ColumnState& state : states_)
    {
        output << static_cast<int>(state.kind) << ' ' << state.source_index << ' '
               << state.feature_offset << ' ' << std::quoted(state.name) << ' '
               << std::setprecision(17) << state.impute_value << ' '
               << state.offset << ' ' << state.scale << ' '
               << state.boolean_impute << ' ' << state.has_missing_category << ' '
               << state.categories.size();
        for (const std::string& category : state.categories) output << ' ' << std::quoted(category);
        output << '\n';
    }
    if (!output) throw PreprocessingError("Failed while writing preprocessing pipeline file '" + path + "'.");
}

FittedPipeline FittedPipeline::load(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw PreprocessingError("Cannot open preprocessing pipeline file '" + path + "'.");

    std::string magic;
    int version = 0;
    input >> magic >> version;
    if (magic != "KAIROML_PIPELINE" || version != 1)
        throw PreprocessingError("Unsupported preprocessing pipeline format.");

    FittedPipeline pipeline;
    input >> std::quoted(pipeline.schema_signature_)
          >> std::quoted(pipeline.training_fingerprint_);

    int imputation = 0;
    int scaling = 0;
    int unknown = 0;
    input >> imputation >> scaling >> pipeline.options_.numeric_constant >> unknown
          >> pipeline.options_.include_boolean_columns
          >> pipeline.options_.include_categorical_columns
          >> pipeline.options_.drop_text_columns;
    if (imputation < enum_value(NumericImputation::Error)
        || imputation > enum_value(NumericImputation::Constant)
        || scaling < enum_value(NumericScaling::None)
        || scaling > enum_value(NumericScaling::MinMax)
        || unknown < enum_value(UnknownCategoryPolicy::Ignore)
        || unknown > enum_value(UnknownCategoryPolicy::Error))
    {
        throw PreprocessingError("Preprocessing pipeline contains an invalid option value.");
    }
    pipeline.options_.numeric_imputation = static_cast<NumericImputation>(imputation);
    pipeline.options_.numeric_scaling = static_cast<NumericScaling>(scaling);
    pipeline.options_.unknown_category = static_cast<UnknownCategoryPolicy>(unknown);
    require_finite(pipeline.options_.numeric_constant, "Numeric imputation constant");

    std::size_t feature_count = 0;
    input >> feature_count;
    pipeline.feature_names_.resize(feature_count);
    for (std::string& name : pipeline.feature_names_) input >> std::quoted(name);

    std::size_t state_count = 0;
    input >> state_count;
    pipeline.states_.resize(state_count);
    for (ColumnState& state : pipeline.states_)
    {
        int kind = 0;
        std::size_t category_count = 0;
        input >> kind >> state.source_index >> state.feature_offset >> std::quoted(state.name)
              >> state.impute_value >> state.offset >> state.scale
              >> state.boolean_impute >> state.has_missing_category >> category_count;
        if (kind < static_cast<int>(StateKind::Numeric)
            || kind > static_cast<int>(StateKind::Categorical)
            || !(state.scale > 0.0) || !std::isfinite(state.scale)
            || !std::isfinite(state.impute_value) || !std::isfinite(state.offset))
        {
            throw PreprocessingError("Preprocessing pipeline contains invalid fitted state.");
        }
        state.kind = static_cast<StateKind>(kind);
        state.categories.resize(category_count);
        for (std::string& category : state.categories) input >> std::quoted(category);
    }

    if (!input || pipeline.feature_names_.empty() || pipeline.states_.empty())
        throw PreprocessingError("Preprocessing pipeline file is truncated or empty.");
    input >> std::ws;
    if (!input.eof()) throw PreprocessingError("Preprocessing pipeline file contains trailing data.");
    pipeline.fitted_ = true;
    return pipeline;
}

} // namespace mllibrary::preprocessing
