#include "dataset.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace mllibrary::data {
namespace {

struct ParsedField final {
    std::string text;
    bool quoted = false;
};

struct ParsedRecord final {
    std::vector<ParsedField> fields;
    std::size_t physical_line = 0;
};

std::string trim_copy(std::string_view input)
{
    std::size_t first = 0;
    while (first < input.size()
        && (input[first] == ' ' || input[first] == '\t'))
    {
        ++first;
    }

    std::size_t last = input.size();
    while (last > first
        && (input[last - 1] == ' ' || input[last - 1] == '\t'))
    {
        --last;
    }
    return std::string(input.substr(first, last - first));
}

bool blank_record(const ParsedRecord& record)
{
    return record.fields.size() == 1
        && !record.fields[0].quoted
        && record.fields[0].text.empty();
}

std::string contextual_message(
    std::string message,
    std::size_t row,
    std::size_t column)
{
    if (row != 0)
    {
        message += " (row " + std::to_string(row);
        if (column != 0)
            message += ", column " + std::to_string(column);
        message += ')';
    }
    return message;
}

std::vector<ParsedRecord> parse_csv_records(
    const std::string& content,
    const CsvReadOptions& options)
{
    if (options.delimiter == '\0' || options.delimiter == '"'
        || options.delimiter == '\n' || options.delimiter == '\r')
    {
        throw DatasetError(
            DatasetErrorCode::MalformedCsv,
            "CSV delimiter must be a non-newline character other than a quote.");
    }

    std::vector<ParsedRecord> records;
    std::vector<ParsedField> fields;
    std::string field;
    bool field_quoted = false;
    bool inside_quotes = false;
    bool after_quote = false;
    bool record_active = false;
    std::size_t physical_line = 1;
    std::size_t record_line = 1;

    auto finish_field = [&]()
    {
        ParsedField parsed;
        parsed.quoted = field_quoted;
        parsed.text = field_quoted || !options.trim_unquoted_fields
            ? field
            : trim_copy(field);
        fields.push_back(std::move(parsed));
        field.clear();
        field_quoted = false;
        after_quote = false;
    };

    auto finish_record = [&]()
    {
        finish_field();
        ParsedRecord record{ std::move(fields), record_line };
        fields.clear();
        if (!(options.skip_blank_records && blank_record(record)))
            records.push_back(std::move(record));
        record_active = false;
    };

    for (std::size_t index = 0; index < content.size(); ++index)
    {
        char character = content[index];

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
                continue;
            }

            if (character == '\r')
            {
                if (index + 1 < content.size() && content[index + 1] == '\n')
                    ++index;
                field.push_back('\n');
                ++physical_line;
                continue;
            }
            if (character == '\n')
            {
                field.push_back('\n');
                ++physical_line;
                continue;
            }

            field.push_back(character);
            continue;
        }

        if (after_quote)
        {
            if (character == options.delimiter)
            {
                finish_field();
                record_active = true;
                continue;
            }
            if (character == ' ' || character == '\t')
                continue;
            if (character == '\r' || character == '\n')
            {
                if (character == '\r'
                    && index + 1 < content.size()
                    && content[index + 1] == '\n')
                {
                    ++index;
                }
                finish_record();
                ++physical_line;
                record_line = physical_line;
                continue;
            }

            throw DatasetError(
                DatasetErrorCode::MalformedCsv,
                "Unexpected character after a closing quote.",
                physical_line,
                fields.size() + 1);
        }

        if (character == options.delimiter)
        {
            finish_field();
            record_active = true;
            continue;
        }

        if (character == '\r' || character == '\n')
        {
            if (character == '\r'
                && index + 1 < content.size()
                && content[index + 1] == '\n')
            {
                ++index;
            }
            finish_record();
            ++physical_line;
            record_line = physical_line;
            continue;
        }

        if (character == '"')
        {
            const bool only_leading_whitespace = options.trim_unquoted_fields
                && trim_copy(field).empty();
            if (!field.empty() && !only_leading_whitespace)
            {
                throw DatasetError(
                    DatasetErrorCode::MalformedCsv,
                    "A quoted field must begin at the start of a column.",
                    physical_line,
                    fields.size() + 1);
            }
            field.clear();
            field_quoted = true;
            inside_quotes = true;
            record_active = true;
            continue;
        }

        field.push_back(character);
        record_active = true;
    }

    if (inside_quotes)
    {
        throw DatasetError(
            DatasetErrorCode::MalformedCsv,
            "CSV ended inside a quoted field.",
            record_line,
            fields.size() + 1);
    }

    if (record_active || after_quote || !fields.empty() || !field.empty())
        finish_record();

    return records;
}

DataValue parse_field(
    const ParsedField& field,
    const ColumnSchema& schema,
    std::size_t row,
    std::size_t column)
{
    const bool missing = field.text.empty() && !field.quoted;
    if (missing)
    {
        if (!schema.nullable)
        {
            throw DatasetError(
                DatasetErrorCode::MissingValue,
                "Column '" + schema.name + "' does not allow missing values.",
                row,
                column);
        }
        return std::monostate{};
    }

    switch (schema.type)
    {
    case ColumnType::Numeric:
    {
        errno = 0;
        char* end = nullptr;
        const double value = std::strtod(field.text.c_str(), &end);
        if (end == field.text.c_str() || !end || *end != '\0'
            || errno == ERANGE || !std::isfinite(value))
        {
            throw DatasetError(
                DatasetErrorCode::InvalidNumber,
                "Invalid numeric value '" + field.text + "' for column '"
                    + schema.name + "'.",
                row,
                column);
        }
        return value;
    }

    case ColumnType::Boolean:
    {
        std::string normalized = field.text;
        std::transform(
            normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (normalized == "true" || normalized == "1" || normalized == "yes")
            return true;
        if (normalized == "false" || normalized == "0" || normalized == "no")
            return false;
        throw DatasetError(
            DatasetErrorCode::InvalidBoolean,
            "Invalid boolean value '" + field.text + "' for column '"
                + schema.name + "'.",
            row,
            column);
    }

    case ColumnType::Categorical:
    case ColumnType::Text:
        return field.text;
    }

    throw DatasetError(
        DatasetErrorCode::InvalidSchema,
        "Unsupported column type for '" + schema.name + "'.",
        row,
        column);
}

bool value_matches(ColumnType type, const DataValue& value)
{
    switch (type)
    {
    case ColumnType::Numeric:
        return std::holds_alternative<double>(value);
    case ColumnType::Boolean:
        return std::holds_alternative<bool>(value);
    case ColumnType::Categorical:
    case ColumnType::Text:
        return std::holds_alternative<std::string>(value);
    }
    return false;
}

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

void hash_string(std::uint64_t& hash, std::string_view value) noexcept
{
    hash_u64(hash, value.size());
    for (unsigned char character : value)
        hash_byte(hash, character);
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

} // namespace

DatasetError::DatasetError(
    DatasetErrorCode code,
    std::string message,
    std::size_t row,
    std::size_t column)
    : std::runtime_error(contextual_message(std::move(message), row, column)),
      code_(code),
      row_(row),
      column_(column)
{
}

void DatasetSchema::validate() const
{
    if (columns.empty())
    {
        throw DatasetError(
            DatasetErrorCode::InvalidSchema,
            "A dataset schema requires at least one column.");
    }

    std::unordered_set<std::string> names;
    std::size_t target_count = 0;
    for (std::size_t index = 0; index < columns.size(); ++index)
    {
        const ColumnSchema& column = columns[index];
        if (trim_copy(column.name).empty())
        {
            throw DatasetError(
                DatasetErrorCode::InvalidSchema,
                "Dataset column names must not be empty.",
                0,
                index + 1);
        }
        if (!names.insert(column.name).second)
        {
            throw DatasetError(
                DatasetErrorCode::InvalidSchema,
                "Duplicate dataset column name '" + column.name + "'.",
                0,
                index + 1);
        }
        if (column.target)
            ++target_count;
    }

    if (target_count > 1)
    {
        throw DatasetError(
            DatasetErrorCode::InvalidSchema,
            "A dataset schema may declare at most one target column.");
    }
}

std::size_t DatasetSchema::find_column(std::string_view name) const noexcept
{
    for (std::size_t index = 0; index < columns.size(); ++index)
    {
        if (columns[index].name == name)
            return index;
    }
    return npos;
}

std::optional<std::size_t> DatasetSchema::target_column() const noexcept
{
    for (std::size_t index = 0; index < columns.size(); ++index)
    {
        if (columns[index].target)
            return index;
    }
    return std::nullopt;
}

Dataset::Dataset(
    DatasetSchema schema,
    std::vector<DataRow> rows,
    std::string source)
    : schema_(std::move(schema)),
      rows_(std::move(rows)),
      source_(std::move(source))
{
    schema_.validate();
    validate_rows();
}

Dataset Dataset::from_csv(
    const std::string& path,
    DatasetSchema schema,
    CsvReadOptions options)
{
    schema.validate();

    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw DatasetError(
            DatasetErrorCode::Io,
            "Cannot open dataset file '" + path + "'.");
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof())
    {
        throw DatasetError(
            DatasetErrorCode::Io,
            "Failed while reading dataset file '" + path + "'.");
    }

    std::vector<ParsedRecord> records = parse_csv_records(buffer.str(), options);
    if (records.empty())
    {
        throw DatasetError(
            DatasetErrorCode::MalformedCsv,
            "Dataset file contains no records.");
    }

    std::size_t data_begin = 0;
    if (options.has_header)
    {
        const ParsedRecord& header = records.front();
        if (header.fields.size() != schema.columns.size())
        {
            throw DatasetError(
                DatasetErrorCode::HeaderMismatch,
                "CSV header width does not match the schema.",
                header.physical_line);
        }
        for (std::size_t column = 0; column < schema.columns.size(); ++column)
        {
            if (header.fields[column].text != schema.columns[column].name)
            {
                throw DatasetError(
                    DatasetErrorCode::HeaderMismatch,
                    "Expected header '" + schema.columns[column].name
                        + "' but found '" + header.fields[column].text + "'.",
                    header.physical_line,
                    column + 1);
            }
        }
        data_begin = 1;
    }

    std::vector<DataRow> rows;
    rows.reserve(records.size() - data_begin);
    for (std::size_t record_index = data_begin; record_index < records.size(); ++record_index)
    {
        const ParsedRecord& record = records[record_index];
        if (record.fields.size() != schema.columns.size())
        {
            throw DatasetError(
                DatasetErrorCode::ColumnCountMismatch,
                "CSV record has " + std::to_string(record.fields.size())
                    + " columns; the schema requires "
                    + std::to_string(schema.columns.size()) + ".",
                record.physical_line);
        }

        DataRow row;
        row.reserve(schema.columns.size());
        for (std::size_t column = 0; column < schema.columns.size(); ++column)
        {
            row.push_back(parse_field(
                record.fields[column],
                schema.columns[column],
                record.physical_line,
                column + 1));
        }
        rows.push_back(std::move(row));
    }

    return Dataset(std::move(schema), std::move(rows), path);
}

void Dataset::validate_rows() const
{
    for (std::size_t row_index = 0; row_index < rows_.size(); ++row_index)
    {
        const DataRow& current = rows_[row_index];
        if (current.size() != schema_.columns.size())
        {
            throw DatasetError(
                DatasetErrorCode::ColumnCountMismatch,
                "Dataset row width does not match the schema.",
                row_index + 1);
        }

        for (std::size_t column_index = 0;
             column_index < schema_.columns.size();
             ++column_index)
        {
            const ColumnSchema& column = schema_.columns[column_index];
            const DataValue& value = current[column_index];
            if (std::holds_alternative<std::monostate>(value))
            {
                if (!column.nullable)
                {
                    throw DatasetError(
                        DatasetErrorCode::MissingValue,
                        "Column '" + column.name + "' does not allow missing values.",
                        row_index + 1,
                        column_index + 1);
                }
                continue;
            }

            if (!value_matches(column.type, value))
            {
                throw DatasetError(
                    DatasetErrorCode::TypeMismatch,
                    "Stored value does not match the declared type of column '"
                        + column.name + "'.",
                    row_index + 1,
                    column_index + 1);
            }

            if (column.type == ColumnType::Numeric
                && !std::isfinite(std::get<double>(value)))
            {
                throw DatasetError(
                    DatasetErrorCode::InvalidNumber,
                    "Numeric dataset values must be finite.",
                    row_index + 1,
                    column_index + 1);
            }
        }
    }
}

const DataRow& Dataset::row(std::size_t row_index) const
{
    if (row_index >= rows_.size())
    {
        throw DatasetError(
            DatasetErrorCode::OutOfRange,
            "Dataset row index is out of range.",
            row_index + 1);
    }
    return rows_[row_index];
}

const DataValue& Dataset::value(
    std::size_t row_index,
    std::size_t column_index) const
{
    if (column_index >= schema_.columns.size())
    {
        throw DatasetError(
            DatasetErrorCode::OutOfRange,
            "Dataset column index is out of range.",
            row_index + 1,
            column_index + 1);
    }
    return row(row_index)[column_index];
}

bool Dataset::is_missing(
    std::size_t row_index,
    std::size_t column_index) const
{
    return std::holds_alternative<std::monostate>(
        value(row_index, column_index));
}

std::optional<double> Dataset::numeric(
    std::size_t row_index,
    std::size_t column_index) const
{
    const DataValue& stored = value(row_index, column_index);
    if (std::holds_alternative<std::monostate>(stored))
        return std::nullopt;
    if (!std::holds_alternative<double>(stored))
    {
        throw DatasetError(
            DatasetErrorCode::TypeMismatch,
            "Dataset value is not numeric.",
            row_index + 1,
            column_index + 1);
    }
    return std::get<double>(stored);
}

std::optional<bool> Dataset::boolean(
    std::size_t row_index,
    std::size_t column_index) const
{
    const DataValue& stored = value(row_index, column_index);
    if (std::holds_alternative<std::monostate>(stored))
        return std::nullopt;
    if (!std::holds_alternative<bool>(stored))
    {
        throw DatasetError(
            DatasetErrorCode::TypeMismatch,
            "Dataset value is not boolean.",
            row_index + 1,
            column_index + 1);
    }
    return std::get<bool>(stored);
}

std::optional<std::string_view> Dataset::text(
    std::size_t row_index,
    std::size_t column_index) const
{
    const DataValue& stored = value(row_index, column_index);
    if (std::holds_alternative<std::monostate>(stored))
        return std::nullopt;
    if (!std::holds_alternative<std::string>(stored))
    {
        throw DatasetError(
            DatasetErrorCode::TypeMismatch,
            "Dataset value is not text or categorical data.",
            row_index + 1,
            column_index + 1);
    }
    return std::string_view(std::get<std::string>(stored));
}

std::uint64_t Dataset::fingerprint64() const noexcept
{
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, schema_.columns.size());
    for (const ColumnSchema& column : schema_.columns)
    {
        hash_string(hash, column.name);
        hash_byte(hash, static_cast<std::uint8_t>(column.type));
        hash_byte(hash, column.nullable ? 1u : 0u);
        hash_byte(hash, column.target ? 1u : 0u);
    }

    hash_u64(hash, rows_.size());
    for (const DataRow& current : rows_)
    {
        for (const DataValue& value : current)
        {
            hash_byte(hash, static_cast<std::uint8_t>(value.index()));
            if (const auto* number = std::get_if<double>(&value))
            {
                hash_u64(hash, std::bit_cast<std::uint64_t>(*number));
            }
            else if (const auto* boolean_value = std::get_if<bool>(&value))
            {
                hash_byte(hash, *boolean_value ? 1u : 0u);
            }
            else if (const auto* string_value = std::get_if<std::string>(&value))
            {
                hash_string(hash, *string_value);
            }
        }
    }
    return hash;
}

std::string Dataset::fingerprint() const
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << fingerprint64();
    return output.str();
}

Dataset Dataset::select_rows(const std::vector<std::size_t>& indices) const
{
    std::vector<DataRow> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices)
    {
        if (index >= rows_.size())
        {
            throw DatasetError(
                DatasetErrorCode::OutOfRange,
                "Dataset split selected an invalid row index.",
                index + 1);
        }
        selected.push_back(rows_[index]);
    }
    return Dataset(schema_, std::move(selected), source_);
}

DatasetSplit Dataset::split(const DatasetSplitOptions& options) const
{
    const bool finite = std::isfinite(options.training_fraction)
        && std::isfinite(options.validation_fraction)
        && std::isfinite(options.test_fraction);
    const double sum = options.training_fraction
        + options.validation_fraction
        + options.test_fraction;
    if (!finite
        || options.training_fraction < 0.0
        || options.validation_fraction < 0.0
        || options.test_fraction < 0.0
        || std::fabs(sum - 1.0) > 1e-9)
    {
        throw DatasetError(
            DatasetErrorCode::InvalidSplit,
            "Dataset split fractions must be finite, non-negative, and sum to one.");
    }

    std::vector<std::size_t> order(rows_.size());
    std::iota(order.begin(), order.end(), 0);
    if (options.shuffle)
    {
        std::mt19937_64 generator(options.seed);
        for (std::size_t remaining = order.size(); remaining > 1; --remaining)
        {
            const std::size_t selected = static_cast<std::size_t>(
                bounded_random(generator, remaining));
            std::swap(order[remaining - 1], order[selected]);
        }
    }

    const std::size_t training_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(order.size()) * options.training_fraction));
    const std::size_t validation_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(order.size()) * options.validation_fraction));
    const std::size_t test_count = order.size() - training_count - validation_count;

    auto make_indices = [&](std::size_t begin, std::size_t count)
    {
        return std::vector<std::size_t>(
            order.begin() + static_cast<std::ptrdiff_t>(begin),
            order.begin() + static_cast<std::ptrdiff_t>(begin + count));
    };

    DatasetSplit result{
        select_rows(make_indices(0, training_count)),
        select_rows(make_indices(training_count, validation_count)),
        select_rows(make_indices(training_count + validation_count, test_count))
    };
    result.training.source_ = source_ + "#training";
    result.validation.source_ = source_ + "#validation";
    result.test.source_ = source_ + "#test";
    return result;
}

const char* column_type_name(ColumnType type) noexcept
{
    switch (type)
    {
    case ColumnType::Numeric: return "numeric";
    case ColumnType::Boolean: return "boolean";
    case ColumnType::Categorical: return "categorical";
    case ColumnType::Text: return "text";
    }
    return "unknown";
}

} // namespace mllibrary::data
