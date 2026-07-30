#include "pipeline.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace mllibrary::data;
using namespace mllibrary::preprocessing;

namespace {

DatasetSchema schema()
{
    return DatasetSchema{{
        {"age", ColumnType::Numeric, true, false},
        {"income", ColumnType::Numeric, false, false},
        {"active", ColumnType::Boolean, true, false},
        {"city", ColumnType::Categorical, true, false},
        {"notes", ColumnType::Text, false, false},
        {"label", ColumnType::Boolean, false, true}
    }};
}

Dataset training_dataset()
{
    return Dataset(schema(), {
        {10.0, 5.0, true, std::string("A"), std::string("first"), false},
        {20.0, 5.0, false, std::string("B"), std::string("second"), true},
        {std::monostate{}, 5.0, std::monostate{}, std::monostate{}, std::string("third"), false}
    }, "memory://training");
}

bool nearly_equal(double a, double b, double tolerance = 1e-9)
{
    return std::fabs(a - b) <= tolerance;
}

bool test_pipeline_values_and_validation_transform()
{
    FittedPipeline pipeline;
    const PreparedDataset train = pipeline.fit_transform(training_dataset());
    if (train.column_count != 6 || train.feature_names.size() != 6) return false;
    if (train.feature_names != std::vector<std::string>{
            "age", "income", "active", "city=A", "city=B", "city=<missing>" }) return false;
    if (!nearly_equal(train.feature(0, 0), -1.0)) return false;
    if (!nearly_equal(train.feature(1, 0), 1.0)) return false;
    if (!nearly_equal(train.feature(2, 0), 0.0)) return false;
    if (!nearly_equal(train.feature(0, 1), 0.0)) return false;
    if (!nearly_equal(train.feature(0, 2), 1.0)) return false;
    if (!nearly_equal(train.feature(2, 2), 0.0)) return false;
    if (!nearly_equal(train.feature(0, 3), 1.0)
        || !nearly_equal(train.feature(1, 4), 1.0)
        || !nearly_equal(train.feature(2, 5), 1.0)) return false;
    if (train.targets.size() != 3 || !std::holds_alternative<bool>(train.targets[1])
        || !std::get<bool>(train.targets[1])) return false;

    Dataset validation(schema(), {
        {100.0, 5.0, true, std::string("C"), std::string("validation"), true}
    }, "memory://validation");
    const PreparedDataset transformed = pipeline.transform(validation);
    if (!nearly_equal(transformed.feature(0, 0), 17.0)) return false;
    for (std::size_t column = 3; column < 6; ++column)
        if (!nearly_equal(transformed.feature(0, column), 0.0)) return false;
    return true;
}

bool test_unknown_category_error_policy()
{
    PipelineOptions options;
    options.unknown_category = UnknownCategoryPolicy::Error;
    FittedPipeline pipeline(options);
    pipeline.fit(training_dataset());
    Dataset validation(schema(), {
        {10.0, 5.0, true, std::string("C"), std::string("validation"), true}
    });
    try
    {
        (void)pipeline.transform(validation);
    }
    catch (const PreprocessingError&)
    {
        return true;
    }
    return false;
}

bool test_serialization_round_trip()
{
    FittedPipeline pipeline;
    pipeline.fit(training_dataset());
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "mllibrary-pipeline.kmlp";
    pipeline.save(path.string());
    const FittedPipeline loaded = FittedPipeline::load(path.string());
    std::filesystem::remove(path);

    const PreparedDataset first = pipeline.transform(training_dataset());
    const PreparedDataset second = loaded.transform(training_dataset());
    return first.feature_names == second.feature_names
        && first.values == second.values
        && first.targets == second.targets
        && loaded.training_fingerprint() == pipeline.training_fingerprint();
}

bool test_schema_mismatch_is_rejected()
{
    FittedPipeline pipeline;
    pipeline.fit(training_dataset());
    DatasetSchema changed = schema();
    changed.columns[0].name = "age_years";
    Dataset other(changed, {
        {10.0, 5.0, true, std::string("A"), std::string("x"), false}
    });
    try
    {
        (void)pipeline.transform(other);
    }
    catch (const PreprocessingError&)
    {
        return true;
    }
    return false;
}

bool test_median_and_minmax()
{
    DatasetSchema numeric_schema{{
        {"value", ColumnType::Numeric, true, false},
        {"target", ColumnType::Numeric, false, true}
    }};
    Dataset data(numeric_schema, {
        {1.0, 0.0},
        {3.0, 1.0},
        {9.0, 2.0},
        {std::monostate{}, 3.0}
    });
    PipelineOptions options;
    options.numeric_imputation = NumericImputation::Median;
    options.numeric_scaling = NumericScaling::MinMax;
    FittedPipeline pipeline(options);
    const PreparedDataset prepared = pipeline.fit_transform(data);
    return prepared.column_count == 1
        && nearly_equal(prepared.feature(0, 0), 0.0)
        && nearly_equal(prepared.feature(2, 0), 1.0)
        && nearly_equal(prepared.feature(3, 0), 0.25);
}

} // namespace

int main()
{
    if (!test_pipeline_values_and_validation_transform()) {
        std::fputs("pipeline transform test failed\n", stderr); return 1;
    }
    if (!test_unknown_category_error_policy()) {
        std::fputs("unknown category policy test failed\n", stderr); return 1;
    }
    if (!test_serialization_round_trip()) {
        std::fputs("pipeline serialization test failed\n", stderr); return 1;
    }
    if (!test_schema_mismatch_is_rejected()) {
        std::fputs("pipeline schema test failed\n", stderr); return 1;
    }
    if (!test_median_and_minmax()) {
        std::fputs("median/minmax test failed\n", stderr); return 1;
    }
    return 0;
}
