#include "cross_validation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mllibrary::evaluation {
namespace {

using mllibrary::data::ColumnType;
using mllibrary::data::DataRow;
using mllibrary::data::DataValue;
using mllibrary::data::DatasetSchema;
using mllibrary::experiment::ExperimentRunner;
using mllibrary::preprocessing::FittedPipeline;
using mllibrary::preprocessing::PreparedDataset;

void validate_fold_request(std::size_t row_count, std::size_t fold_count)
{
    if (fold_count < 2)
        throw std::invalid_argument("Cross-validation requires at least two folds.");
    if (row_count < fold_count)
        throw std::invalid_argument("Cross-validation requires at least one validation row per fold.");
}

std::uint64_t bounded_random(std::mt19937_64& generator, std::uint64_t bound)
{
    const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
    for (;;)
    {
        const std::uint64_t value = generator();
        if (value >= threshold)
            return value % bound;
    }
}

void deterministic_shuffle(
    std::vector<std::size_t>& values,
    std::mt19937_64& generator)
{
    for (std::size_t remaining = values.size(); remaining > 1; --remaining)
    {
        const std::size_t selected = static_cast<std::size_t>(
            bounded_random(generator, remaining));
        std::swap(values[remaining - 1], values[selected]);
    }
}

std::vector<FoldIndices> folds_from_validation_buckets(
    std::size_t row_count,
    const std::vector<std::vector<std::size_t>>& validation_buckets)
{
    std::vector<FoldIndices> result;
    result.reserve(validation_buckets.size());
    for (const auto& validation : validation_buckets)
    {
        if (validation.empty())
            throw std::invalid_argument("Every cross-validation fold must contain validation rows.");

        std::vector<bool> held_out(row_count, false);
        for (std::size_t index : validation)
        {
            if (index >= row_count || held_out[index])
                throw std::invalid_argument("Fold validation indices must be unique and in range.");
            held_out[index] = true;
        }

        FoldIndices fold;
        fold.validation = validation;
        fold.training.reserve(row_count - validation.size());
        for (std::size_t index = 0; index < row_count; ++index)
        {
            if (!held_out[index])
                fold.training.push_back(index);
        }
        if (fold.training.empty())
            throw std::invalid_argument("Every cross-validation fold must contain training rows.");
        result.push_back(std::move(fold));
    }
    return result;
}

std::string value_key(const DataValue& value)
{
    if (const auto* number = std::get_if<double>(&value))
    {
        if (!std::isfinite(*number))
            throw std::invalid_argument("Split columns and targets must contain finite values.");
        std::ostringstream output;
        output << "n:" << std::setprecision(std::numeric_limits<double>::max_digits10)
               << *number;
        return output.str();
    }
    if (const auto* boolean_value = std::get_if<bool>(&value))
        return *boolean_value ? "b:1" : "b:0";
    if (const auto* text = std::get_if<std::string>(&value))
        return "s:" + *text;
    throw std::invalid_argument("Split columns and targets must not contain missing values.");
}

std::vector<std::string> column_keys(
    const Dataset& dataset,
    std::size_t column_index)
{
    std::vector<std::string> keys;
    keys.reserve(dataset.row_count());
    for (std::size_t row = 0; row < dataset.row_count(); ++row)
        keys.push_back(value_key(dataset.value(row, column_index)));
    return keys;
}

bool value_less(
    const DataValue& left,
    const DataValue& right,
    ColumnType type)
{
    if (std::holds_alternative<std::monostate>(left)
        || std::holds_alternative<std::monostate>(right))
    {
        throw std::invalid_argument("Time-order columns must not contain missing values.");
    }

    switch (type)
    {
    case ColumnType::Numeric:
        return std::get<double>(left) < std::get<double>(right);
    case ColumnType::Boolean:
        return std::get<bool>(left) < std::get<bool>(right);
    case ColumnType::Categorical:
    case ColumnType::Text:
        return std::get<std::string>(left) < std::get<std::string>(right);
    }
    return false;
}

Dataset drop_column(const Dataset& dataset, std::size_t drop_index)
{
    if (drop_index >= dataset.column_count())
        throw std::invalid_argument("Cannot exclude an out-of-range split column.");
    if (dataset.schema().columns[drop_index].target)
        throw std::invalid_argument("The target column cannot also be excluded as a split column.");

    DatasetSchema schema;
    schema.columns.reserve(dataset.column_count() - 1);
    for (std::size_t column = 0; column < dataset.column_count(); ++column)
    {
        if (column != drop_index)
            schema.columns.push_back(dataset.schema().columns[column]);
    }

    std::vector<DataRow> rows;
    rows.reserve(dataset.row_count());
    for (const DataRow& source : dataset.rows())
    {
        DataRow row;
        row.reserve(source.size() - 1);
        for (std::size_t column = 0; column < source.size(); ++column)
        {
            if (column != drop_index)
                row.push_back(source[column]);
        }
        rows.push_back(std::move(row));
    }
    return Dataset(
        std::move(schema),
        std::move(rows),
        dataset.source() + "#without:" + dataset.schema().columns[drop_index].name);
}

bool is_classification_metric(PrimaryMetric metric)
{
    return metric == PrimaryMetric::Accuracy
        || metric == PrimaryMetric::MacroF1
        || metric == PrimaryMetric::WeightedF1
        || metric == PrimaryMetric::LogLoss
        || metric == PrimaryMetric::RocAuc;
}

bool is_regression_metric(PrimaryMetric metric)
{
    return metric == PrimaryMetric::MeanAbsoluteError
        || metric == PrimaryMetric::RootMeanSquaredError
        || metric == PrimaryMetric::R2;
}

double primary_value(
    PrimaryMetric metric,
    const ClassificationMetrics* classification,
    const RegressionMetrics* regression)
{
    switch (metric)
    {
    case PrimaryMetric::Accuracy:
        return classification->accuracy;
    case PrimaryMetric::MacroF1:
        return classification->macro_f1;
    case PrimaryMetric::WeightedF1:
        return classification->weighted_f1;
    case PrimaryMetric::LogLoss:
        if (!classification->log_loss)
            throw std::invalid_argument("The selected estimator did not produce binary scores for log loss.");
        return *classification->log_loss;
    case PrimaryMetric::RocAuc:
        if (!classification->roc_auc)
            throw std::invalid_argument("ROC-AUC requires both binary classes in the validation fold and probability scores.");
        return *classification->roc_auc;
    case PrimaryMetric::MeanAbsoluteError:
        return regression->mean_absolute_error;
    case PrimaryMetric::RootMeanSquaredError:
        return regression->root_mean_squared_error;
    case PrimaryMetric::R2:
        return regression->r2;
    }
    throw std::invalid_argument("Unsupported primary metric.");
}

MetricAggregate aggregate_values(const std::vector<double>& values)
{
    if (values.empty())
        throw std::invalid_argument("Cannot aggregate an empty metric vector.");
    MetricAggregate result;
    result.minimum = *std::min_element(values.begin(), values.end());
    result.maximum = *std::max_element(values.begin(), values.end());
    result.mean = std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
    double variance = 0.0;
    for (double value : values)
    {
        const double difference = value - result.mean;
        variance += difference * difference;
    }
    result.standard_deviation = std::sqrt(
        variance / static_cast<double>(values.size()));
    return result;
}

std::string json_escape(std::string_view value)
{
    std::ostringstream output;
    for (unsigned char character : value)
    {
        switch (character)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << static_cast<char>(character); break;
        }
    }
    return output.str();
}

} // namespace

std::vector<FoldIndices> make_k_folds(
    std::size_t row_count,
    std::size_t fold_count,
    std::uint64_t seed,
    bool shuffle)
{
    validate_fold_request(row_count, fold_count);
    std::vector<std::size_t> order(row_count);
    std::iota(order.begin(), order.end(), 0);
    if (shuffle)
    {
        std::mt19937_64 generator(seed);
        deterministic_shuffle(order, generator);
    }

    std::vector<std::vector<std::size_t>> validation(fold_count);
    for (std::size_t position = 0; position < order.size(); ++position)
        validation[position % fold_count].push_back(order[position]);
    return folds_from_validation_buckets(row_count, validation);
}

std::vector<FoldIndices> make_stratified_folds(
    const std::vector<std::string>& labels,
    std::size_t fold_count,
    std::uint64_t seed,
    bool shuffle)
{
    validate_fold_request(labels.size(), fold_count);
    std::map<std::string, std::vector<std::size_t>> by_label;
    for (std::size_t index = 0; index < labels.size(); ++index)
        by_label[labels[index]].push_back(index);
    if (by_label.size() < 2)
        throw std::invalid_argument("Stratified classification requires at least two classes.");
    for (const auto& [label, rows] : by_label)
    {
        (void)label;
        if (rows.size() < 2)
        {
            throw std::invalid_argument(
                "Every class requires at least two rows so each training fold retains that class.");
        }
    }

    std::mt19937_64 generator(seed);
    std::vector<std::vector<std::size_t>> validation(fold_count);
    std::size_t offset = 0;
    for (auto& [label, rows] : by_label)
    {
        (void)label;
        if (shuffle)
            deterministic_shuffle(rows, generator);
        for (std::size_t position = 0; position < rows.size(); ++position)
            validation[(offset + position) % fold_count].push_back(rows[position]);
        offset = (offset + rows.size()) % fold_count;
    }
    return folds_from_validation_buckets(labels.size(), validation);
}

std::vector<FoldIndices> make_group_folds(
    const std::vector<std::string>& groups,
    std::size_t fold_count)
{
    validate_fold_request(groups.size(), fold_count);
    std::map<std::string, std::vector<std::size_t>> rows_by_group;
    for (std::size_t index = 0; index < groups.size(); ++index)
        rows_by_group[groups[index]].push_back(index);
    if (rows_by_group.size() < fold_count)
        throw std::invalid_argument("Group cross-validation requires at least one distinct group per fold.");

    std::vector<std::pair<std::string, std::vector<std::size_t>>> ordered(
        rows_by_group.begin(), rows_by_group.end());
    std::sort(
        ordered.begin(), ordered.end(),
        [](const auto& left, const auto& right)
        {
            if (left.second.size() != right.second.size())
                return left.second.size() > right.second.size();
            return left.first < right.first;
        });

    std::vector<std::vector<std::size_t>> validation(fold_count);
    std::vector<std::size_t> fold_sizes(fold_count, 0);
    for (const auto& [group, rows] : ordered)
    {
        (void)group;
        const std::size_t destination = static_cast<std::size_t>(
            std::min_element(fold_sizes.begin(), fold_sizes.end()) - fold_sizes.begin());
        validation[destination].insert(
            validation[destination].end(), rows.begin(), rows.end());
        fold_sizes[destination] += rows.size();
    }
    return folds_from_validation_buckets(groups.size(), validation);
}

std::vector<FoldIndices> make_time_series_folds(
    const std::vector<std::size_t>& ordered_rows,
    std::size_t fold_count)
{
    validate_fold_request(ordered_rows.size(), fold_count + 1);
    std::set<std::size_t> unique(ordered_rows.begin(), ordered_rows.end());
    if (unique.size() != ordered_rows.size())
        throw std::invalid_argument("Time-series row order must contain unique indices.");

    const std::size_t block = ordered_rows.size() / (fold_count + 1);
    if (block == 0)
        throw std::invalid_argument("Time-series cross-validation requires more rows than folds.");

    std::vector<FoldIndices> folds;
    folds.reserve(fold_count);
    for (std::size_t fold_index = 0; fold_index < fold_count; ++fold_index)
    {
        const std::size_t training_end = block * (fold_index + 1);
        const std::size_t validation_end = fold_index + 1 == fold_count
            ? ordered_rows.size()
            : block * (fold_index + 2);
        FoldIndices fold;
        fold.training.assign(
            ordered_rows.begin(),
            ordered_rows.begin() + static_cast<std::ptrdiff_t>(training_end));
        fold.validation.assign(
            ordered_rows.begin() + static_cast<std::ptrdiff_t>(training_end),
            ordered_rows.begin() + static_cast<std::ptrdiff_t>(validation_end));
        if (fold.validation.empty())
            throw std::invalid_argument("Every time-series fold must contain future validation rows.");
        folds.push_back(std::move(fold));
    }
    return folds;
}

CrossValidator::CrossValidator(CrossValidationConfig config)
    : config_(std::move(config))
{
}

CrossValidationResult CrossValidator::run(const Dataset& dataset) const
{
    CrossValidationResult result;
    result.strategy = config_.strategy;
    result.primary_metric = config_.primary_metric;

    try
    {
        if (dataset.empty())
            throw std::invalid_argument("Cross-validation requires a non-empty dataset.");
        const auto target_column = dataset.schema().target_column();
        if (!target_column)
            throw std::invalid_argument("Cross-validation requires one declared target column.");
        if (config_.task == TaskKind::Classification
            && !is_classification_metric(config_.primary_metric))
        {
            throw std::invalid_argument("The selected primary metric is not a classification metric.");
        }
        if (config_.task == TaskKind::Regression
            && !is_regression_metric(config_.primary_metric))
        {
            throw std::invalid_argument("The selected primary metric is not a regression metric.");
        }

        std::vector<FoldIndices> folds;
        std::optional<std::size_t> split_column;
        switch (config_.strategy)
        {
        case FoldStrategy::KFold:
            folds = make_k_folds(
                dataset.row_count(), config_.folds, config_.seed, config_.shuffle);
            break;
        case FoldStrategy::Stratified:
            if (config_.task != TaskKind::Classification)
                throw std::invalid_argument("Stratified folds currently require a classification task.");
            folds = make_stratified_folds(
                column_keys(dataset, *target_column),
                config_.folds,
                config_.seed,
                config_.shuffle);
            break;
        case FoldStrategy::Group:
        {
            if (config_.group_column.empty())
                throw std::invalid_argument("Group cross-validation requires a group column name.");
            const std::size_t column = dataset.schema().find_column(config_.group_column);
            if (column == DatasetSchema::npos)
                throw std::invalid_argument("Group column '" + config_.group_column + "' was not found.");
            if (column == *target_column)
                throw std::invalid_argument("The target column cannot be used as the group column.");
            split_column = column;
            folds = make_group_folds(column_keys(dataset, column), config_.folds);
            break;
        }
        case FoldStrategy::TimeSeries:
        {
            std::vector<std::size_t> order(dataset.row_count());
            std::iota(order.begin(), order.end(), 0);
            if (!config_.time_column.empty())
            {
                const std::size_t column = dataset.schema().find_column(config_.time_column);
                if (column == DatasetSchema::npos)
                    throw std::invalid_argument("Time column '" + config_.time_column + "' was not found.");
                if (column == *target_column)
                    throw std::invalid_argument("The target column cannot be used as the time column.");
                split_column = column;
                const ColumnType type = dataset.schema().columns[column].type;
                std::stable_sort(
                    order.begin(), order.end(),
                    [&](std::size_t left, std::size_t right)
                    {
                        const DataValue& left_value = dataset.value(left, column);
                        const DataValue& right_value = dataset.value(right, column);
                        if (value_less(left_value, right_value, type)) return true;
                        if (value_less(right_value, left_value, type)) return false;
                        return left < right;
                    });
            }
            folds = make_time_series_folds(order, config_.folds);
            break;
        }
        }

        Dataset model_dataset = dataset;
        if (config_.exclude_split_column && split_column)
            model_dataset = drop_column(dataset, *split_column);

        std::vector<double> primary_values;
        primary_values.reserve(folds.size());
        result.folds.reserve(folds.size());
        for (std::size_t fold_index = 0; fold_index < folds.size(); ++fold_index)
        {
            const Dataset training_raw = model_dataset.select_rows(folds[fold_index].training);
            const Dataset validation_raw = model_dataset.select_rows(folds[fold_index].validation);

            FittedPipeline pipeline(config_.pipeline);
            const PreparedDataset training = pipeline.fit_transform(training_raw);
            const PreparedDataset validation = pipeline.transform(validation_raw);

            FoldEvaluation fold;
            fold.fold_index = fold_index;
            fold.training_rows = training.row_count;
            fold.validation_rows = validation.row_count;
            fold.experiment = ExperimentRunner(
                config_.task, config_.estimator).run(
                    training,
                    &validation,
                    nullptr,
                    training_raw.fingerprint(),
                    pipeline.schema_signature() + "@" + pipeline.training_fingerprint());
            if (!fold.experiment.success || !fold.experiment.validation)
            {
                throw std::runtime_error(
                    "Fold " + std::to_string(fold_index)
                    + " failed: " + fold.experiment.error);
            }

            const auto& evaluation = *fold.experiment.validation;
            if (config_.task == TaskKind::Classification)
            {
                fold.classification = calculate_classification_metrics(
                    evaluation.actual,
                    evaluation.predicted,
                    evaluation.scores);
                fold.primary_metric_value = primary_value(
                    config_.primary_metric, &*fold.classification, nullptr);
            }
            else
            {
                fold.regression = calculate_regression_metrics(
                    evaluation.actual,
                    evaluation.predicted);
                fold.primary_metric_value = primary_value(
                    config_.primary_metric, nullptr, &*fold.regression);
            }
            primary_values.push_back(fold.primary_metric_value);
            result.folds.push_back(std::move(fold));
        }

        result.aggregate = aggregate_values(primary_values);
        result.success = true;
    }
    catch (const std::exception& error)
    {
        result.success = false;
        result.error = error.what();
    }
    return result;
}

std::string CrossValidationResult::to_json() const
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n"
           << "  \"success\": " << (success ? "true" : "false") << ",\n"
           << "  \"error\": \"" << json_escape(error) << "\",\n"
           << "  \"strategy\": \"" << fold_strategy_name(strategy) << "\",\n"
           << "  \"primary_metric\": \"" << primary_metric_name(primary_metric) << "\",\n"
           << "  \"aggregate\": {\n"
           << "    \"mean\": " << aggregate.mean << ",\n"
           << "    \"standard_deviation\": " << aggregate.standard_deviation << ",\n"
           << "    \"minimum\": " << aggregate.minimum << ",\n"
           << "    \"maximum\": " << aggregate.maximum << "\n"
           << "  },\n"
           << "  \"folds\": [\n";
    for (std::size_t index = 0; index < folds.size(); ++index)
    {
        const FoldEvaluation& fold = folds[index];
        output << "    {\n"
               << "      \"fold\": " << fold.fold_index << ",\n"
               << "      \"training_rows\": " << fold.training_rows << ",\n"
               << "      \"validation_rows\": " << fold.validation_rows << ",\n"
               << "      \"primary_metric_value\": " << fold.primary_metric_value;
        if (fold.classification)
        {
            output << ",\n      \"accuracy\": " << fold.classification->accuracy
                   << ",\n      \"macro_f1\": " << fold.classification->macro_f1
                   << ",\n      \"weighted_f1\": " << fold.classification->weighted_f1;
            if (fold.classification->log_loss)
                output << ",\n      \"log_loss\": " << *fold.classification->log_loss;
            if (fold.classification->roc_auc)
                output << ",\n      \"roc_auc\": " << *fold.classification->roc_auc;
        }
        if (fold.regression)
        {
            output << ",\n      \"mae\": " << fold.regression->mean_absolute_error
                   << ",\n      \"rmse\": " << fold.regression->root_mean_squared_error
                   << ",\n      \"r2\": " << fold.regression->r2;
        }
        output << "\n    }";
        if (index + 1 != folds.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
    return output.str();
}

void CrossValidationResult::write_json(const std::string& path) const
{
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty())
    {
        std::error_code error;
        std::filesystem::create_directories(output_path.parent_path(), error);
        if (error)
            throw std::runtime_error("Cannot create cross-validation output directory: " + error.message());
    }
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot create cross-validation report '" + path + "'.");
    output << to_json();
    if (!output)
        throw std::runtime_error("Writing cross-validation report failed for '" + path + "'.");
}

const char* fold_strategy_name(FoldStrategy strategy) noexcept
{
    switch (strategy)
    {
    case FoldStrategy::KFold: return "k_fold";
    case FoldStrategy::Stratified: return "stratified";
    case FoldStrategy::Group: return "group";
    case FoldStrategy::TimeSeries: return "time_series";
    }
    return "unknown";
}

const char* primary_metric_name(PrimaryMetric metric) noexcept
{
    switch (metric)
    {
    case PrimaryMetric::Accuracy: return "accuracy";
    case PrimaryMetric::MacroF1: return "macro_f1";
    case PrimaryMetric::WeightedF1: return "weighted_f1";
    case PrimaryMetric::LogLoss: return "log_loss";
    case PrimaryMetric::RocAuc: return "roc_auc";
    case PrimaryMetric::MeanAbsoluteError: return "mean_absolute_error";
    case PrimaryMetric::RootMeanSquaredError: return "root_mean_squared_error";
    case PrimaryMetric::R2: return "r2";
    }
    return "unknown";
}

} // namespace mllibrary::evaluation
