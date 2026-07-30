#include "profiling.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mllibrary::data;

namespace {

std::filesystem::path temp_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

void write_file(const std::filesystem::path& path, const std::string& content)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
}

bool test_schema_inference_and_loading()
{
    const auto path = temp_path("mllibrary-profile-inference.csv");
    write_file(path,
        "age,active,city,notes,churn\n"
        "20,true,Kolkata,short,false\n"
        "30,false,Delhi,another long free-form note,true\n"
        ",true,Kolkata,third note,false\n"
        "40,false,Delhi,fourth note,false\n");

    SchemaInferenceOptions options;
    options.target_column = "churn";
    options.categorical_unique_ratio = 0.60;
    options.text_length_threshold = 12;
    const DatasetSchema schema = infer_csv_schema(path.string(), options);
    const bool schema_ok = schema.columns.size() == 5
        && schema.columns[0].type == ColumnType::Numeric
        && schema.columns[0].nullable
        && schema.columns[1].type == ColumnType::Boolean
        && schema.columns[2].type == ColumnType::Categorical
        && schema.columns[3].type == ColumnType::Text
        && schema.columns[4].type == ColumnType::Boolean
        && schema.columns[4].target;

    const Dataset dataset = load_inferred_csv(path.string(), options);
    const bool loaded = dataset.row_count() == 4
        && !dataset.numeric(2, 0).has_value()
        && dataset.text(0, 2).value() == "Kolkata";
    std::filesystem::remove(path);
    return schema_ok && loaded;
}

bool test_profile_statistics_and_reports()
{
    DatasetSchema schema{{
        {"age", ColumnType::Numeric, true, false},
        {"city", ColumnType::Categorical, false, false},
        {"churn", ColumnType::Boolean, false, true}
    }};
    Dataset dataset(schema, {
        {20.0, std::string("Kolkata"), false},
        {30.0, std::string("Delhi"), false},
        {std::monostate{}, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), false},
        {20.0, std::string("Kolkata"), true},
        {20.0, std::string("Kolkata"), false}
    }, "memory://profile");

    const DatasetProfile profile = profile_dataset(dataset);
    if (profile.row_count != 11 || profile.column_count != 3) return false;
    if (profile.duplicate_rows != 6) return false;
    if (profile.columns[0].missing_count != 1 || profile.columns[0].unique_count != 2) return false;
    if (!profile.columns[0].numeric.available) return false;
    if (std::fabs(profile.columns[0].numeric.minimum - 20.0) > 1e-9) return false;
    if (std::fabs(profile.columns[0].numeric.maximum - 30.0) > 1e-9) return false;
    if (profile.target_distribution.size() != 2 || profile.target_distribution[0].value != "false") return false;
    if (profile.target_majority_fraction < 0.90) return false;

    const std::string json = dataset_profile_to_json(profile);
    const std::string html = dataset_profile_to_html(profile);
    if (json.find("\"duplicate_rows\": 6") == std::string::npos) return false;
    if (json.find("\"target_majority_fraction\"") == std::string::npos) return false;
    if (html.find("Dataset profile") == std::string::npos || html.find("Kolkata") == std::string::npos) return false;

    const auto json_path = temp_path("mllibrary-profile.json");
    const auto html_path = temp_path("mllibrary-profile.html");
    write_dataset_profile_json(profile, json_path.string());
    write_dataset_profile_html(profile, html_path.string());
    const bool files_ok = std::filesystem::file_size(json_path) > 0
        && std::filesystem::file_size(html_path) > 0;
    std::filesystem::remove(json_path);
    std::filesystem::remove(html_path);
    return files_ok;
}

bool test_missing_target_is_rejected()
{
    const auto path = temp_path("mllibrary-profile-missing-target.csv");
    write_file(path, "a,b\n1,x\n2,y\n");
    SchemaInferenceOptions options;
    options.target_column = "missing";
    bool rejected = false;
    try
    {
        (void)infer_csv_schema(path.string(), options);
    }
    catch (const DatasetError& error)
    {
        rejected = error.code() == DatasetErrorCode::InvalidSchema;
    }
    std::filesystem::remove(path);
    return rejected;
}

} // namespace

int main()
{
    if (!test_schema_inference_and_loading())
    {
        std::fputs("schema inference test failed\n", stderr);
        return 1;
    }
    if (!test_profile_statistics_and_reports())
    {
        std::fputs("dataset profile test failed\n", stderr);
        return 1;
    }
    if (!test_missing_target_is_rejected())
    {
        std::fputs("missing target test failed\n", stderr);
        return 1;
    }
    return 0;
}
