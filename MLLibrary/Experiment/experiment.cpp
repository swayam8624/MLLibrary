#include "experiment.hpp"

#include "classical.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mllibrary::experiment {
namespace {

using Clock = std::chrono::steady_clock;
using mllibrary::data::DataValue;

void validate_prepared(
    const PreparedDataset& dataset,
    bool require_targets,
    const char* role)
{
    if (dataset.row_count == 0)
        throw std::invalid_argument(std::string(role) + " dataset must contain rows.");
    if (dataset.column_count == 0)
        throw std::invalid_argument(std::string(role) + " dataset must contain features.");
    if (dataset.values.size() != dataset.row_count * dataset.column_count)
        throw std::invalid_argument(std::string(role) + " feature storage does not match its shape.");
    if (dataset.feature_names.size() != dataset.column_count)
        throw std::invalid_argument(std::string(role) + " feature-name count does not match its shape.");
    if (require_targets && dataset.targets.size() != dataset.row_count)
        throw std::invalid_argument(std::string(role) + " target count does not match its row count.");
    if (!dataset.targets.empty() && dataset.targets.size() != dataset.row_count)
        throw std::invalid_argument(std::string(role) + " target count is incomplete.");

    for (double value : dataset.values)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(role) + " dataset contains a non-finite feature.");
        if (value > std::numeric_limits<float>::max()
            || value < -std::numeric_limits<float>::max())
        {
            throw std::invalid_argument(std::string(role) + " feature exceeds the classical float range.");
        }
    }
}

void require_same_features(
    const PreparedDataset& training,
    const PreparedDataset& other,
    const char* role)
{
    validate_prepared(other, true, role);
    if (training.column_count != other.column_count
        || training.feature_names != other.feature_names)
    {
        throw std::invalid_argument(
            std::string(role) + " features do not match the training feature contract.");
    }
}

DenseTable to_dense_table(const PreparedDataset& dataset)
{
    DenseTable rows(dataset.row_count, std::vector<float>(dataset.column_count));
    for (std::size_t row = 0; row < dataset.row_count; ++row)
    {
        for (std::size_t column = 0; column < dataset.column_count; ++column)
        {
            rows[row][column] = static_cast<float>(
                dataset.values[row * dataset.column_count + column]);
        }
    }
    return rows;
}

std::string target_display(const DataValue& value)
{
    if (const auto* boolean_value = std::get_if<bool>(&value))
        return *boolean_value ? "true" : "false";
    if (const auto* number = std::get_if<double>(&value))
    {
        if (!std::isfinite(*number))
            throw std::invalid_argument("Classification targets must be finite.");
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << *number;
        return output.str();
    }
    if (const auto* text = std::get_if<std::string>(&value))
        return *text;
    throw std::invalid_argument("Classification targets must not be missing.");
}

std::string target_key(const DataValue& value)
{
    if (const auto* boolean_value = std::get_if<bool>(&value))
        return *boolean_value ? "b:1" : "b:0";
    if (const auto* number = std::get_if<double>(&value))
        return "n:" + target_display(*number);
    if (const auto* text = std::get_if<std::string>(&value))
        return "s:" + *text;
    throw std::invalid_argument("Classification targets must not be missing.");
}

class LabelEncoder final {
public:
    void fit(const std::vector<DataValue>& targets)
    {
        if (targets.empty())
            throw std::invalid_argument("Classification training requires targets.");

        std::map<std::string, std::string> ordered;
        for (const DataValue& value : targets)
            ordered.emplace(target_key(value), target_display(value));
        if (ordered.size() < 2)
            throw std::invalid_argument("Classification training requires at least two classes.");

        labels_.clear();
        index_by_key_.clear();
        labels_.reserve(ordered.size());
        for (const auto& [key, display] : ordered)
        {
            const int index = static_cast<int>(labels_.size());
            index_by_key_.emplace(key, index);
            labels_.push_back(display);
        }
    }

    [[nodiscard]] std::vector<int> encode_int(
        const std::vector<DataValue>& targets) const
    {
        std::vector<int> encoded;
        encoded.reserve(targets.size());
        for (const DataValue& value : targets)
        {
            const auto found = index_by_key_.find(target_key(value));
            if (found == index_by_key_.end())
            {
                throw std::invalid_argument(
                    "Evaluation contains a target class absent from training: '"
                    + target_display(value) + "'.");
            }
            encoded.push_back(found->second);
        }
        return encoded;
    }

    [[nodiscard]] std::vector<double> encode_double(
        const std::vector<DataValue>& targets) const
    {
        const std::vector<int> integers = encode_int(targets);
        return std::vector<double>(integers.begin(), integers.end());
    }

    [[nodiscard]] const std::vector<std::string>& labels() const noexcept
    {
        return labels_;
    }

private:
    std::vector<std::string> labels_;
    std::unordered_map<std::string, int> index_by_key_;
};

class ClassificationAdapterBase {
public:
    void prepare_fit(
        const PreparedDataset& training,
        const PreparedDataset* validation)
    {
        validate_prepared(training, true, "Training");
        if (validation)
            require_same_features(training, *validation, "Validation");
        feature_count_ = training.column_count;
        encoder_.fit(training.targets);
    }

    void require_predictable(const PreparedDataset& dataset) const
    {
        validate_prepared(dataset, false, "Prediction");
        if (dataset.column_count != feature_count_)
            throw std::invalid_argument("Prediction feature width does not match the fitted estimator.");
    }

    [[nodiscard]] std::vector<double> encode_targets(
        const PreparedDataset& dataset) const
    {
        if (dataset.targets.size() != dataset.row_count)
            throw std::invalid_argument("Evaluation requires one target per row.");
        return encoder_.encode_double(dataset.targets);
    }

    [[nodiscard]] const std::vector<std::string>& class_labels() const noexcept
    {
        return encoder_.labels();
    }

protected:
    LabelEncoder encoder_;
    std::size_t feature_count_ = 0;
};

class LogisticEstimator final : public Estimator, private ClassificationAdapterBase {
public:
    explicit LogisticEstimator(const EstimatorConfig& config)
        : model_(
            static_cast<float>(config.learning_rate),
            config.iterations,
            static_cast<float>(config.l2))
    {
    }

    const char* name() const noexcept override { return "logistic_regression"; }
    TaskKind task() const noexcept override { return TaskKind::Classification; }

    void fit(const PreparedDataset& training, const PreparedDataset* validation) override
    {
        prepare_fit(training, validation);
        if (class_labels().size() != 2)
            throw std::invalid_argument("Logistic regression currently supports exactly two classes.");
        model_.fit(to_dense_table(training), encoder_.encode_int(training.targets));
    }

    std::vector<double> predict(const PreparedDataset& dataset) const override
    {
        require_predictable(dataset);
        const DenseTable rows = to_dense_table(dataset);
        std::vector<double> output;
        output.reserve(rows.size());
        for (const auto& row : rows)
            output.push_back(model_.predict(row));
        return output;
    }

    std::vector<double> predict_scores(const PreparedDataset& dataset) const override
    {
        require_predictable(dataset);
        const DenseTable rows = to_dense_table(dataset);
        std::vector<double> output;
        output.reserve(rows.size());
        for (const auto& row : rows)
            output.push_back(model_.predict_probability(row));
        return output;
    }

    std::vector<double> encode_targets(const PreparedDataset& dataset) const override
    {
        return ClassificationAdapterBase::encode_targets(dataset);
    }

    const std::vector<std::string>& class_labels() const noexcept override
    {
        return ClassificationAdapterBase::class_labels();
    }

    std::size_t model_state_units() const noexcept override
    {
        return model_.weights().size() + 1;
    }

private:
    ::LogisticRegression model_;
};

class LinearEstimator final : public Estimator {
public:
    explicit LinearEstimator(const EstimatorConfig& config)
        : model_(
            static_cast<float>(config.learning_rate),
            config.iterations,
            static_cast<float>(config.l2))
    {
    }

    const char* name() const noexcept override { return "linear_regression"; }
    TaskKind task() const noexcept override { return TaskKind::Regression; }

    void fit(const PreparedDataset& training, const PreparedDataset* validation) override
    {
        validate_prepared(training, true, "Training");
        if (validation)
            require_same_features(training, *validation, "Validation");
        feature_count_ = training.column_count;
        std::vector<float> targets;
        targets.reserve(training.targets.size());
        for (const DataValue& value : training.targets)
        {
            const auto* number = std::get_if<double>(&value);
            if (!number || !std::isfinite(*number)
                || *number > std::numeric_limits<float>::max()
                || *number < -std::numeric_limits<float>::max())
            {
                throw std::invalid_argument(
                    "Linear regression requires finite numeric targets in float range.");
            }
            targets.push_back(static_cast<float>(*number));
        }
        model_.fit(to_dense_table(training), targets);
    }

    std::vector<double> predict(const PreparedDataset& dataset) const override
    {
        validate_prepared(dataset, false, "Prediction");
        if (dataset.column_count != feature_count_)
            throw std::invalid_argument("Prediction feature width does not match the fitted estimator.");
        const DenseTable rows = to_dense_table(dataset);
        std::vector<double> output;
        output.reserve(rows.size());
        for (const auto& row : rows)
            output.push_back(model_.predict(row));
        return output;
    }

    std::vector<double> encode_targets(const PreparedDataset& dataset) const override
    {
        if (dataset.targets.size() != dataset.row_count)
            throw std::invalid_argument("Regression evaluation requires one target per row.");
        std::vector<double> output;
        output.reserve(dataset.targets.size());
        for (const DataValue& value : dataset.targets)
        {
            const auto* number = std::get_if<double>(&value);
            if (!number || !std::isfinite(*number))
                throw std::invalid_argument("Regression evaluation requires finite numeric targets.");
            output.push_back(*number);
        }
        return output;
    }

    const std::vector<std::string>& class_labels() const noexcept override
    {
        return empty_labels_;
    }

    std::size_t model_state_units() const noexcept override
    {
        return model_.weights().size() + 1;
    }

private:
    ::LinearRegression model_;
    std::size_t feature_count_ = 0;
    std::vector<std::string> empty_labels_;
};

template <typename Model>
class GenericClassifierEstimator final : public Estimator, private ClassificationAdapterBase {
public:
    GenericClassifierEstimator(Model model, const char* name)
        : model_(std::move(model)), name_(name)
    {
    }

    const char* name() const noexcept override { return name_; }
    TaskKind task() const noexcept override { return TaskKind::Classification; }

    void fit(const PreparedDataset& training, const PreparedDataset* validation) override
    {
        prepare_fit(training, validation);
        training_rows_ = training.row_count;
        model_.fit(to_dense_table(training), encoder_.encode_int(training.targets));
    }

    std::vector<double> predict(const PreparedDataset& dataset) const override
    {
        require_predictable(dataset);
        const DenseTable rows = to_dense_table(dataset);
        std::vector<double> output;
        output.reserve(rows.size());
        for (const auto& row : rows)
            output.push_back(model_.predict(row));
        return output;
    }

    std::vector<double> encode_targets(const PreparedDataset& dataset) const override
    {
        return ClassificationAdapterBase::encode_targets(dataset);
    }

    const std::vector<std::string>& class_labels() const noexcept override
    {
        return ClassificationAdapterBase::class_labels();
    }

    std::size_t model_state_units() const noexcept override
    {
        if constexpr (requires(const Model& model) { model.node_count(); })
            return model_.node_count();
        else if constexpr (requires(const Model& model) { model.tree_count(); })
            return model_.tree_count();
        else
            return training_rows_ * (feature_count_ + 1);
    }

private:
    Model model_;
    const char* name_;
    std::size_t training_rows_ = 0;
};

std::string json_escape(std::string_view value)
{
    std::ostringstream output;
    for (unsigned char character : value)
    {
        switch (character)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20)
            {
                output << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned>(character)
                       << std::dec << std::setfill(' ');
            }
            else
            {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

void write_predictions(
    const std::filesystem::path& path,
    const DatasetEvaluation& evaluation)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot create prediction artifact '" + path.string() + "'.");
    output << "row,actual,predicted,score\n";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t index = 0; index < evaluation.row_count; ++index)
    {
        output << index << ',' << evaluation.actual[index] << ','
               << evaluation.predicted[index] << ',';
        if (!evaluation.scores.empty())
            output << evaluation.scores[index];
        output << '\n';
    }
    if (!output)
        throw std::runtime_error("Writing prediction artifact failed for '" + path.string() + "'.");
}

DatasetEvaluation evaluate_dataset(
    const Estimator& estimator,
    TaskKind task,
    const PreparedDataset& dataset)
{
    DatasetEvaluation result;
    result.row_count = dataset.row_count;
    result.actual = estimator.encode_targets(dataset);

    const auto start = Clock::now();
    result.predicted = estimator.predict(dataset);
    result.scores = estimator.predict_scores(dataset);
    const auto finish = Clock::now();
    result.inference_seconds = std::chrono::duration<double>(finish - start).count();

    if (result.predicted.size() != result.actual.size())
        throw std::runtime_error("Estimator returned a prediction count that does not match the dataset.");
    if (!result.scores.empty() && result.scores.size() != result.actual.size())
        throw std::runtime_error("Estimator returned a score count that does not match the dataset.");

    if (task == TaskKind::Classification)
    {
        std::size_t correct = 0;
        for (std::size_t index = 0; index < result.actual.size(); ++index)
            correct += result.actual[index] == result.predicted[index] ? 1u : 0u;
        result.accuracy = static_cast<double>(correct)
            / static_cast<double>(result.actual.size());
    }
    else
    {
        double squared_error = 0.0;
        for (std::size_t index = 0; index < result.actual.size(); ++index)
        {
            const double error = result.predicted[index] - result.actual[index];
            squared_error += error * error;
        }
        result.mean_squared_error = squared_error
            / static_cast<double>(result.actual.size());
    }
    return result;
}

} // namespace

std::vector<double> Estimator::predict_scores(const PreparedDataset&) const
{
    return {};
}

std::unique_ptr<Estimator> make_estimator(const EstimatorConfig& config)
{
    if (!(config.learning_rate > 0.0) || !std::isfinite(config.learning_rate))
        throw std::invalid_argument("Estimator learning rate must be positive and finite.");
    if (!std::isfinite(config.l2) || config.l2 < 0.0)
        throw std::invalid_argument("Estimator L2 penalty must be non-negative and finite.");

    switch (config.kind)
    {
    case EstimatorKind::LogisticRegression:
        return std::make_unique<LogisticEstimator>(config);
    case EstimatorKind::LinearRegression:
        return std::make_unique<LinearEstimator>(config);
    case EstimatorKind::KNearestNeighbors:
        return std::make_unique<GenericClassifierEstimator<::KNearestNeighbors>>(
            ::KNearestNeighbors(config.neighbors), "k_nearest_neighbors");
    case EstimatorKind::GaussianNaiveBayes:
        return std::make_unique<GenericClassifierEstimator<::GaussianNaiveBayes>>(
            ::GaussianNaiveBayes(static_cast<float>(config.variance_smoothing)),
            "gaussian_naive_bayes");
    case EstimatorKind::DecisionTreeClassifier:
        return std::make_unique<GenericClassifierEstimator<::DecisionTreeClassifier>>(
            ::DecisionTreeClassifier(
                config.max_depth,
                config.min_samples_split,
                config.min_samples_leaf,
                config.max_features,
                config.seed),
            "decision_tree_classifier");
    case EstimatorKind::RandomForestClassifier:
        return std::make_unique<GenericClassifierEstimator<::RandomForestClassifier>>(
            ::RandomForestClassifier(
                config.trees,
                config.max_depth,
                config.min_samples_split,
                config.min_samples_leaf,
                config.max_features,
                config.seed),
            "random_forest_classifier");
    }
    throw std::invalid_argument("Unsupported estimator kind.");
}

ExperimentRunner::ExperimentRunner(TaskKind task, EstimatorConfig estimator)
    : task_(task), estimator_(estimator)
{
}

ExperimentResult ExperimentRunner::run(
    const PreparedDataset& training,
    const PreparedDataset* validation,
    const PreparedDataset* test,
    std::string dataset_fingerprint,
    std::string pipeline_fingerprint,
    const std::string& output_directory) const
{
    ExperimentResult result;
    result.task = task_;
    result.estimator = estimator_;
    result.dataset_fingerprint = std::move(dataset_fingerprint);
    result.pipeline_fingerprint = std::move(pipeline_fingerprint);

    try
    {
        std::unique_ptr<Estimator> estimator = make_estimator(estimator_);
        result.estimator_name = estimator->name();
        if (estimator->task() != task_)
        {
            throw std::invalid_argument(
                "Estimator '" + result.estimator_name
                + "' is incompatible with task '" + task_kind_name(task_) + "'.");
        }

        const auto fit_start = Clock::now();
        estimator->fit(training, validation);
        const auto fit_finish = Clock::now();
        result.fit_seconds = std::chrono::duration<double>(fit_finish - fit_start).count();
        result.model_state_units = estimator->model_state_units();
        result.class_labels = estimator->class_labels();

        if (validation)
            result.validation = evaluate_dataset(*estimator, task_, *validation);
        if (test)
            result.test = evaluate_dataset(*estimator, task_, *test);

        result.success = true;
        if (!output_directory.empty())
            result.write_artifacts(output_directory);
    }
    catch (const std::exception& error)
    {
        result.success = false;
        result.error = error.what();
    }
    return result;
}

std::string ExperimentResult::to_json() const
{
    auto write_evaluation = [](std::ostringstream& output, const DatasetEvaluation& value)
    {
        output << "{\n"
               << "      \"row_count\": " << value.row_count << ",\n"
               << "      \"accuracy\": " << value.accuracy << ",\n"
               << "      \"mean_squared_error\": " << value.mean_squared_error << ",\n"
               << "      \"inference_seconds\": " << value.inference_seconds << "\n"
               << "    }";
    };

    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n"
           << "  \"success\": " << (success ? "true" : "false") << ",\n"
           << "  \"error\": \"" << json_escape(error) << "\",\n"
           << "  \"task\": \"" << task_kind_name(task) << "\",\n"
           << "  \"estimator\": {\n"
           << "    \"kind\": \"" << estimator_kind_name(estimator.kind) << "\",\n"
           << "    \"name\": \"" << json_escape(estimator_name) << "\",\n"
           << "    \"learning_rate\": " << estimator.learning_rate << ",\n"
           << "    \"iterations\": " << estimator.iterations << ",\n"
           << "    \"l2\": " << estimator.l2 << ",\n"
           << "    \"neighbors\": " << estimator.neighbors << ",\n"
           << "    \"variance_smoothing\": " << estimator.variance_smoothing << ",\n"
           << "    \"trees\": " << estimator.trees << ",\n"
           << "    \"max_depth\": " << estimator.max_depth << ",\n"
           << "    \"min_samples_split\": " << estimator.min_samples_split << ",\n"
           << "    \"min_samples_leaf\": " << estimator.min_samples_leaf << ",\n"
           << "    \"max_features\": " << estimator.max_features << ",\n"
           << "    \"seed\": " << estimator.seed << "\n"
           << "  },\n"
           << "  \"dataset_fingerprint\": \"" << json_escape(dataset_fingerprint) << "\",\n"
           << "  \"pipeline_fingerprint\": \"" << json_escape(pipeline_fingerprint) << "\",\n"
           << "  \"fit_seconds\": " << fit_seconds << ",\n"
           << "  \"model_state_units\": " << model_state_units << ",\n"
           << "  \"class_labels\": [";
    for (std::size_t index = 0; index < class_labels.size(); ++index)
    {
        if (index != 0) output << ", ";
        output << '"' << json_escape(class_labels[index]) << '"';
    }
    output << "],\n  \"validation\": ";
    if (validation) write_evaluation(output, *validation); else output << "null";
    output << ",\n  \"test\": ";
    if (test) write_evaluation(output, *test); else output << "null";
    output << "\n}\n";
    return output.str();
}

void ExperimentResult::write_artifacts(const std::string& output_directory) const
{
    const std::filesystem::path directory(output_directory);
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
        throw std::runtime_error("Cannot create experiment output directory: " + error.message());

    const std::filesystem::path manifest = directory / "experiment.json";
    std::ofstream output(manifest, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot create experiment manifest '" + manifest.string() + "'.");
    output << to_json();
    if (!output)
        throw std::runtime_error("Writing experiment manifest failed for '" + manifest.string() + "'.");
    output.close();

    if (validation)
        write_predictions(directory / "validation_predictions.csv", *validation);
    if (test)
        write_predictions(directory / "test_predictions.csv", *test);
}

const char* task_kind_name(TaskKind task) noexcept
{
    switch (task)
    {
    case TaskKind::Classification: return "classification";
    case TaskKind::Regression: return "regression";
    }
    return "unknown";
}

const char* estimator_kind_name(EstimatorKind estimator) noexcept
{
    switch (estimator)
    {
    case EstimatorKind::LogisticRegression: return "logistic_regression";
    case EstimatorKind::LinearRegression: return "linear_regression";
    case EstimatorKind::KNearestNeighbors: return "k_nearest_neighbors";
    case EstimatorKind::GaussianNaiveBayes: return "gaussian_naive_bayes";
    case EstimatorKind::DecisionTreeClassifier: return "decision_tree_classifier";
    case EstimatorKind::RandomForestClassifier: return "random_forest_classifier";
    }
    return "unknown";
}

} // namespace mllibrary::experiment
