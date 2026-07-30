#include "profiling.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mllibrary::data {
namespace {

struct RawField final {
    std::string text;
    bool quoted = false;
};
using RawRecord = std::vector<RawField>;

std::string trim_copy(std::string_view input)
{
    std::size_t first = 0;
    while (first < input.size() && (input[first] == ' ' || input[first] == '\t')) ++first;
    std::size_t last = input.size();
    while (last > first && (input[last - 1] == ' ' || input[last - 1] == '\t')) --last;
    return std::string(input.substr(first, last - first));
}

std::string read_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw DatasetError(DatasetErrorCode::Io, "Cannot open dataset file '" + path + "'.");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof())
        throw DatasetError(DatasetErrorCode::Io, "Failed while reading dataset file '" + path + "'.");
    return buffer.str();
}

std::vector<RawRecord> parse_records(const std::string& content, const CsvReadOptions& options)
{
    if (options.delimiter == '\0' || options.delimiter == '"'
        || options.delimiter == '\n' || options.delimiter == '\r')
    {
        throw DatasetError(DatasetErrorCode::MalformedCsv,
            "CSV delimiter must be a non-newline character other than a quote.");
    }

    std::vector<RawRecord> records;
    RawRecord record;
    std::string field;
    bool quoted = false;
    bool inside_quotes = false;
    bool after_quote = false;
    bool active = false;
    std::size_t physical_line = 1;

    auto finish_field = [&]() {
        RawField value;
        value.quoted = quoted;
        value.text = quoted || !options.trim_unquoted_fields ? field : trim_copy(field);
        record.push_back(std::move(value));
        field.clear();
        quoted = false;
        after_quote = false;
    };
    auto finish_record = [&]() {
        finish_field();
        const bool blank = record.size() == 1 && !record[0].quoted && record[0].text.empty();
        if (!(options.skip_blank_records && blank)) records.push_back(std::move(record));
        record.clear();
        active = false;
    };

    for (std::size_t index = 0; index < content.size(); ++index)
    {
        const char character = content[index];
        if (inside_quotes)
        {
            if (character == '"')
            {
                if (index + 1 < content.size() && content[index + 1] == '"')
                {
                    field.push_back('"');
                    ++index;
                }
                else
                {
                    inside_quotes = false;
                    after_quote = true;
                }
            }
            else if (character == '\r' || character == '\n')
            {
                if (character == '\r' && index + 1 < content.size() && content[index + 1] == '\n') ++index;
                field.push_back('\n');
                ++physical_line;
            }
            else
            {
                field.push_back(character);
            }
            continue;
        }

        if (after_quote)
        {
            if (character == options.delimiter)
            {
                finish_field();
                active = true;
                continue;
            }
            if (character == ' ' || character == '\t') continue;
            if (character == '\r' || character == '\n')
            {
                if (character == '\r' && index + 1 < content.size() && content[index + 1] == '\n') ++index;
                finish_record();
                ++physical_line;
                continue;
            }
            throw DatasetError(DatasetErrorCode::MalformedCsv,
                "Unexpected character after a closing quote.", physical_line, record.size() + 1);
        }

        if (character == options.delimiter)
        {
            finish_field();
            active = true;
        }
        else if (character == '\r' || character == '\n')
        {
            if (character == '\r' && index + 1 < content.size() && content[index + 1] == '\n') ++index;
            finish_record();
            ++physical_line;
        }
        else if (character == '"')
        {
            const bool leading_space = options.trim_unquoted_fields && trim_copy(field).empty();
            if (!field.empty() && !leading_space)
                throw DatasetError(DatasetErrorCode::MalformedCsv,
                    "A quoted field must begin at the start of a column.", physical_line, record.size() + 1);
            field.clear();
            quoted = true;
            inside_quotes = true;
            active = true;
        }
        else
        {
            field.push_back(character);
            active = true;
        }
    }

    if (inside_quotes)
        throw DatasetError(DatasetErrorCode::MalformedCsv, "CSV ended inside a quoted field.");
    if (active || after_quote || !record.empty() || !field.empty()) finish_record();
    return records;
}

bool is_missing(const RawField& field) noexcept
{
    return field.text.empty() && !field.quoted;
}

bool parse_boolean(std::string value, bool& output)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "true" || value == "1" || value == "yes") { output = true; return true; }
    if (value == "false" || value == "0" || value == "no") { output = false; return true; }
    return false;
}

bool parse_number(const std::string& text, double& output)
{
    errno = 0;
    char* end = nullptr;
    output = std::strtod(text.c_str(), &end);
    return end != text.c_str() && end && *end == '\0' && errno != ERANGE && std::isfinite(output);
}

std::string value_string(const DataValue& value)
{
    if (std::holds_alternative<std::monostate>(value)) return "<missing>";
    if (const auto* number = std::get_if<double>(&value))
    {
        std::ostringstream output;
        output << std::setprecision(17) << *number;
        return output.str();
    }
    if (const auto* boolean_value = std::get_if<bool>(&value)) return *boolean_value ? "true" : "false";
    return std::get<std::string>(value);
}

std::string row_key(const DataRow& row)
{
    std::string key;
    for (const DataValue& value : row)
    {
        key += std::to_string(value.index());
        key.push_back(':');
        const std::string rendered = value_string(value);
        key += std::to_string(rendered.size());
        key.push_back(':');
        key += rendered;
        key.push_back('|');
    }
    return key;
}

std::vector<ValueFrequency> top_frequencies(
    const std::unordered_map<std::string, std::size_t>& frequencies,
    std::size_t limit)
{
    std::vector<ValueFrequency> result;
    result.reserve(frequencies.size());
    for (const auto& [value, count] : frequencies) result.push_back({ value, count });
    std::sort(result.begin(), result.end(), [](const ValueFrequency& a, const ValueFrequency& b) {
        return a.count != b.count ? a.count > b.count : a.value < b.value;
    });
    if (result.size() > limit) result.resize(limit);
    return result;
}

std::string json_escape(std::string_view value)
{
    std::ostringstream output;
    for (unsigned char c : value)
    {
        switch (c)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20)
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            else
                output << static_cast<char>(c);
        }
    }
    return output.str();
}

std::string html_escape(std::string_view value)
{
    std::string output;
    for (char c : value)
    {
        switch (c)
        {
        case '&': output += "&amp;"; break;
        case '<': output += "&lt;"; break;
        case '>': output += "&gt;"; break;
        case '"': output += "&quot;"; break;
        case '\'': output += "&#39;"; break;
        default: output.push_back(c); break;
        }
    }
    return output;
}

void write_text(const std::string& path, const std::string& content)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw DatasetError(DatasetErrorCode::Io, "Cannot open report file '" + path + "'.");
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output)
        throw DatasetError(DatasetErrorCode::Io, "Failed while writing report file '" + path + "'.");
}

} // namespace

DatasetSchema infer_csv_schema(const std::string& path, const SchemaInferenceOptions& options)
{
    if (options.maximum_rows == 0)
        throw DatasetError(DatasetErrorCode::InvalidSchema, "Schema inference requires at least one sample row.");
    if (!std::isfinite(options.categorical_unique_ratio)
        || options.categorical_unique_ratio < 0.0 || options.categorical_unique_ratio > 1.0)
    {
        throw DatasetError(DatasetErrorCode::InvalidSchema,
            "Categorical unique ratio must be finite and in [0, 1].");
    }

    std::vector<RawRecord> records = parse_records(read_file(path), options.csv);
    if (records.empty())
        throw DatasetError(DatasetErrorCode::MalformedCsv, "Dataset file contains no records.");

    const std::size_t width = records.front().size();
    if (width == 0)
        throw DatasetError(DatasetErrorCode::MalformedCsv, "Dataset file contains no columns.");

    std::vector<std::string> names(width);
    std::size_t data_begin = 0;
    if (options.csv.has_header)
    {
        for (std::size_t column = 0; column < width; ++column)
        {
            names[column] = records.front()[column].text;
            if (trim_copy(names[column]).empty())
                throw DatasetError(DatasetErrorCode::InvalidSchema,
                    "CSV header names must not be empty.", 1, column + 1);
        }
        data_begin = 1;
    }
    else
    {
        for (std::size_t column = 0; column < width; ++column)
            names[column] = "column_" + std::to_string(column + 1);
    }

    const std::size_t available = records.size() - data_begin;
    if (available == 0)
        throw DatasetError(DatasetErrorCode::MalformedCsv, "Dataset file contains no data rows.");
    const std::size_t sample_count = std::min(available, options.maximum_rows);

    struct Candidate final {
        bool boolean = true;
        bool numeric = true;
        bool nullable = false;
        std::size_t non_missing = 0;
        std::size_t maximum_length = 0;
        std::unordered_set<std::string> unique;
    };
    std::vector<Candidate> candidates(width);

    for (std::size_t row = 0; row < sample_count; ++row)
    {
        const RawRecord& record = records[data_begin + row];
        if (record.size() != width)
            throw DatasetError(DatasetErrorCode::ColumnCountMismatch,
                "CSV records must have a consistent number of columns.", data_begin + row + 1);
        for (std::size_t column = 0; column < width; ++column)
        {
            const RawField& field = record[column];
            Candidate& candidate = candidates[column];
            if (is_missing(field)) { candidate.nullable = true; continue; }
            ++candidate.non_missing;
            candidate.maximum_length = std::max(candidate.maximum_length, field.text.size());
            candidate.unique.insert(field.text);
            bool boolean_value = false;
            double number = 0.0;
            candidate.boolean = candidate.boolean && parse_boolean(field.text, boolean_value);
            candidate.numeric = candidate.numeric && parse_number(field.text, number);
        }
    }

    DatasetSchema schema;
    schema.columns.reserve(width);
    for (std::size_t column = 0; column < width; ++column)
    {
        const Candidate& candidate = candidates[column];
        ColumnType type = ColumnType::Text;
        if (candidate.non_missing > 0 && candidate.boolean)
            type = ColumnType::Boolean;
        else if (candidate.non_missing > 0 && candidate.numeric)
            type = ColumnType::Numeric;
        else
        {
            const double ratio = candidate.non_missing == 0 ? 0.0
                : static_cast<double>(candidate.unique.size()) / static_cast<double>(candidate.non_missing);
            if (candidate.unique.size() <= options.categorical_unique_threshold
                && ratio <= options.categorical_unique_ratio
                && candidate.maximum_length <= options.text_length_threshold)
                type = ColumnType::Categorical;
        }
        schema.columns.push_back({ names[column], type, candidate.nullable, false });
    }

    if (options.target_column)
    {
        const std::size_t target = schema.find_column(*options.target_column);
        if (target == DatasetSchema::npos)
            throw DatasetError(DatasetErrorCode::InvalidSchema,
                "Target column '" + *options.target_column + "' was not found in the CSV header.");
        schema.columns[target].target = true;
    }
    schema.validate();
    return schema;
}

Dataset load_inferred_csv(const std::string& path, const SchemaInferenceOptions& options)
{
    return Dataset::from_csv(path, infer_csv_schema(path, options), options.csv);
}

DatasetProfile profile_dataset(const Dataset& dataset, const DatasetProfileOptions& options)
{
    DatasetProfile profile;
    profile.source = dataset.source();
    profile.fingerprint = dataset.fingerprint();
    profile.row_count = dataset.row_count();
    profile.column_count = dataset.column_count();
    profile.columns.reserve(profile.column_count);

    std::unordered_map<std::string, std::size_t> duplicate_frequencies;
    if (options.detect_duplicate_rows)
    {
        for (const DataRow& row : dataset.rows()) ++duplicate_frequencies[row_key(row)];
        for (const auto& [key, count] : duplicate_frequencies)
            if (count > 1) profile.duplicate_rows += count - 1;
    }

    for (std::size_t column_index = 0; column_index < dataset.column_count(); ++column_index)
    {
        const ColumnSchema& schema = dataset.schema().columns[column_index];
        ColumnProfile column;
        column.name = schema.name;
        column.type = schema.type;
        column.nullable = schema.nullable;
        column.target = schema.target;
        column.row_count = dataset.row_count();

        std::unordered_map<std::string, std::size_t> frequencies;
        double sum = 0.0;
        double squared_sum = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();

        for (std::size_t row = 0; row < dataset.row_count(); ++row)
        {
            const DataValue& value = dataset.value(row, column_index);
            if (std::holds_alternative<std::monostate>(value))
            {
                ++column.missing_count;
                continue;
            }
            ++column.non_missing_count;
            ++frequencies[value_string(value)];
            if (const auto* number = std::get_if<double>(&value))
            {
                if (!std::isfinite(*number)) { ++column.invalid_count; continue; }
                minimum = std::min(minimum, *number);
                maximum = std::max(maximum, *number);
                sum += *number;
                squared_sum += *number * *number;
            }
        }

        column.unique_count = frequencies.size();
        column.top_values = top_frequencies(frequencies, options.top_values_per_column);
        if (schema.type == ColumnType::Numeric && column.non_missing_count > column.invalid_count)
        {
            const std::size_t count = column.non_missing_count - column.invalid_count;
            column.numeric.available = true;
            column.numeric.minimum = minimum;
            column.numeric.maximum = maximum;
            column.numeric.mean = sum / static_cast<double>(count);
            const double variance = std::max(0.0,
                squared_sum / static_cast<double>(count) - column.numeric.mean * column.numeric.mean);
            column.numeric.standard_deviation = std::sqrt(variance);
        }

        if (schema.target)
        {
            profile.target_distribution = top_frequencies(frequencies, frequencies.size());
            if (column.non_missing_count > 0 && !profile.target_distribution.empty())
                profile.target_majority_fraction = static_cast<double>(profile.target_distribution.front().count)
                    / static_cast<double>(column.non_missing_count);
        }
        profile.columns.push_back(std::move(column));
    }

    if (profile.row_count == 0) profile.warnings.push_back("Dataset has no rows.");
    if (profile.duplicate_rows > 0)
        profile.warnings.push_back(std::to_string(profile.duplicate_rows) + " duplicate rows detected.");
    for (const ColumnProfile& column : profile.columns)
    {
        if (column.missing_count > 0)
            profile.warnings.push_back("Column '" + column.name + "' contains "
                + std::to_string(column.missing_count) + " missing values.");
        if (column.non_missing_count > 0 && column.unique_count <= 1)
            profile.warnings.push_back("Column '" + column.name + "' is constant.");
    }
    if (!profile.target_distribution.empty() && profile.target_majority_fraction >= 0.80)
        profile.warnings.push_back("Target distribution is imbalanced; majority class is at least 80%.");
    return profile;
}

std::string dataset_profile_to_json(const DatasetProfile& profile)
{
    std::ostringstream output;
    output << std::setprecision(12);
    output << "{\n  \"source\": \"" << json_escape(profile.source) << "\",\n"
           << "  \"fingerprint\": \"" << json_escape(profile.fingerprint) << "\",\n"
           << "  \"row_count\": " << profile.row_count << ",\n"
           << "  \"column_count\": " << profile.column_count << ",\n"
           << "  \"duplicate_rows\": " << profile.duplicate_rows << ",\n"
           << "  \"target_majority_fraction\": " << profile.target_majority_fraction << ",\n"
           << "  \"columns\": [\n";
    for (std::size_t index = 0; index < profile.columns.size(); ++index)
    {
        const ColumnProfile& column = profile.columns[index];
        output << "    {\"name\": \"" << json_escape(column.name)
               << "\", \"type\": \"" << column_type_name(column.type)
               << "\", \"nullable\": " << (column.nullable ? "true" : "false")
               << ", \"target\": " << (column.target ? "true" : "false")
               << ", \"missing_count\": " << column.missing_count
               << ", \"invalid_count\": " << column.invalid_count
               << ", \"unique_count\": " << column.unique_count;
        if (column.numeric.available)
            output << ", \"numeric\": {\"minimum\": " << column.numeric.minimum
                   << ", \"maximum\": " << column.numeric.maximum
                   << ", \"mean\": " << column.numeric.mean
                   << ", \"standard_deviation\": " << column.numeric.standard_deviation << '}';
        output << ", \"top_values\": [";
        for (std::size_t value_index = 0; value_index < column.top_values.size(); ++value_index)
        {
            if (value_index) output << ',';
            output << "{\"value\": \"" << json_escape(column.top_values[value_index].value)
                   << "\", \"count\": " << column.top_values[value_index].count << '}';
        }
        output << "]}" << (index + 1 == profile.columns.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"target_distribution\": [";
    for (std::size_t index = 0; index < profile.target_distribution.size(); ++index)
    {
        if (index) output << ',';
        output << "{\"value\": \"" << json_escape(profile.target_distribution[index].value)
               << "\", \"count\": " << profile.target_distribution[index].count << '}';
    }
    output << "],\n  \"warnings\": [";
    for (std::size_t index = 0; index < profile.warnings.size(); ++index)
    {
        if (index) output << ',';
        output << '\"' << json_escape(profile.warnings[index]) << '\"';
    }
    output << "]\n}\n";
    return output.str();
}

std::string dataset_profile_to_html(const DatasetProfile& profile)
{
    std::ostringstream output;
    output << "<!doctype html><html><head><meta charset=\"utf-8\"><title>Dataset profile</title>"
           << "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
           << "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.55rem;text-align:left}"
           << "th{background:#f4f4f4}.warn{padding:.6rem;background:#fff4d6;margin:.35rem 0}</style></head><body>";
    output << "<h1>Dataset profile</h1><p><strong>Source:</strong> " << html_escape(profile.source)
           << "</p><p><strong>Fingerprint:</strong> <code>" << html_escape(profile.fingerprint)
           << "</code></p><p>Rows: " << profile.row_count << " &middot; Columns: " << profile.column_count
           << " &middot; Duplicate rows: " << profile.duplicate_rows << "</p>";
    if (!profile.warnings.empty())
    {
        output << "<h2>Warnings</h2>";
        for (const std::string& warning : profile.warnings)
            output << "<div class=\"warn\">" << html_escape(warning) << "</div>";
    }
    output << "<h2>Columns</h2><table><thead><tr><th>Name</th><th>Type</th><th>Target</th>"
           << "<th>Missing</th><th>Unique</th><th>Numeric summary</th><th>Most common</th></tr></thead><tbody>";
    for (const ColumnProfile& column : profile.columns)
    {
        output << "<tr><td>" << html_escape(column.name) << "</td><td>" << column_type_name(column.type)
               << "</td><td>" << (column.target ? "yes" : "no") << "</td><td>" << column.missing_count
               << "</td><td>" << column.unique_count << "</td><td>";
        if (column.numeric.available)
            output << "min " << column.numeric.minimum << ", max " << column.numeric.maximum
                   << ", mean " << column.numeric.mean << ", sd " << column.numeric.standard_deviation;
        else
            output << "&mdash;";
        output << "</td><td>";
        for (std::size_t index = 0; index < column.top_values.size(); ++index)
        {
            if (index) output << "<br>";
            output << html_escape(column.top_values[index].value) << " (" << column.top_values[index].count << ')';
        }
        output << "</td></tr>";
    }
    output << "</tbody></table>";
    if (!profile.target_distribution.empty())
    {
        output << "<h2>Target distribution</h2><ul>";
        for (const ValueFrequency& value : profile.target_distribution)
            output << "<li>" << html_escape(value.value) << ": " << value.count << "</li>";
        output << "</ul><p>Majority fraction: " << profile.target_majority_fraction << "</p>";
    }
    output << "</body></html>\n";
    return output.str();
}

void write_dataset_profile_json(const DatasetProfile& profile, const std::string& path)
{
    write_text(path, dataset_profile_to_json(profile));
}

void write_dataset_profile_html(const DatasetProfile& profile, const std::string& path)
{
    write_text(path, dataset_profile_to_html(profile));
}

} // namespace mllibrary::data
