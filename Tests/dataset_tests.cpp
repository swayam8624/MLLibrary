#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "arena.h"
#include "base.h"
#include "data.hpp"
#include "dataset.hpp"

namespace {

using namespace mllibrary::data;

std::filesystem::path temporary_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

bool write_text(const std::filesystem::path& path, const std::string& content)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    return output.good();
}

bool write_floats(const std::filesystem::path& path, const std::vector<f32>& values)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(f32)));
    return output.good();
}

DatasetSchema customer_schema()
{
    return DatasetSchema{
        .columns = {
            { .name = "age", .type = ColumnType::Numeric },
            { .name = "subscribed", .type = ColumnType::Boolean },
            { .name = "city", .type = ColumnType::Categorical, .nullable = true },
            { .name = "note", .type = ColumnType::Text, .nullable = true },
        }
    };
}

bool test_typed_csv_parsing_and_fingerprint()
{
    const std::filesystem::path path = temporary_path("mllibrary-dataset-parse.csv");
    const std::string content =
        "age,subscribed,city,note\r\n"
        "31,true,Kolkata,\"likes, commas\"\r\n"
        "27,0,,\"said \"\"hello\"\"\"\r\n"
        "44,yes,Delhi,\"line one\nline two\"\r\n";
    if (!write_text(path, content)) return false;

    bool passed = false;
    try
    {
        const Dataset first = Dataset::from_csv(path.string(), customer_schema());
        const Dataset second = Dataset::from_csv(path.string(), customer_schema());
        const auto first_note = first.text(0, 3);
        const auto second_note = first.text(1, 3);
        const auto multiline_note = first.text(2, 3);

        passed = first.row_count() == 3
            && first.column_count() == 4
            && first.numeric(0, 0).value_or(0.0) == 31.0
            && first.boolean(0, 1).value_or(false)
            && first.text(0, 2).value_or("") == "Kolkata"
            && first_note.value_or("") == "likes, commas"
            && first.is_missing(1, 2)
            && second_note.value_or("") == "said \"hello\""
            && multiline_note.value_or("") == "line one\nline two"
            && first.fingerprint() == second.fingerprint()
            && first.fingerprint().size() == 16;
    }
    catch (...)
    {
        passed = false;
    }

    std::filesystem::remove(path);
    return passed;
}

bool test_parse_errors_include_row_and_column()
{
    const std::filesystem::path path = temporary_path("mllibrary-dataset-invalid.csv");
    if (!write_text(
            path,
            "age,subscribed,city,note\n"
            "not-a-number,true,Kolkata,hello\n"))
        return false;

    bool passed = false;
    try
    {
        (void)Dataset::from_csv(path.string(), customer_schema());
    }
    catch (const DatasetError& error)
    {
        passed = error.code() == DatasetErrorCode::InvalidNumber
            && error.row() == 2
            && error.column() == 1;
    }

    std::filesystem::remove(path);
    return passed;
}

bool test_header_mismatch_is_rejected()
{
    const std::filesystem::path path = temporary_path("mllibrary-dataset-header.csv");
    if (!write_text(
            path,
            "age,active,city,note\n"
            "31,true,Kolkata,hello\n"))
        return false;

    bool passed = false;
    try
    {
        (void)Dataset::from_csv(path.string(), customer_schema());
    }
    catch (const DatasetError& error)
    {
        passed = error.code() == DatasetErrorCode::HeaderMismatch
            && error.row() == 1
            && error.column() == 2;
    }

    std::filesystem::remove(path);
    return passed;
}

Dataset make_split_dataset()
{
    DatasetSchema schema{
        .columns = {
            { .name = "id", .type = ColumnType::Numeric },
            { .name = "label", .type = ColumnType::Categorical, .target = true },
        }
    };

    std::vector<DataRow> rows;
    for (std::size_t index = 0; index < 10; ++index)
    {
        rows.push_back({
            static_cast<double>(index),
            std::string(index % 2 == 0 ? "even" : "odd")
        });
    }
    return Dataset(std::move(schema), std::move(rows), "memory://split-fixture");
}

bool test_deterministic_dataset_split()
{
    const Dataset dataset = make_split_dataset();
    const DatasetSplitOptions options{
        .training_fraction = 0.60,
        .validation_fraction = 0.20,
        .test_fraction = 0.20,
        .seed = 42,
        .shuffle = true
    };

    const DatasetSplit first = dataset.split(options);
    const DatasetSplit second = dataset.split(options);
    if (first.training.row_count() != 6
        || first.validation.row_count() != 2
        || first.test.row_count() != 2
        || first.training.fingerprint() != second.training.fingerprint()
        || first.validation.fingerprint() != second.validation.fingerprint()
        || first.test.fingerprint() != second.test.fingerprint())
    {
        return false;
    }

    std::unordered_set<int> identifiers;
    auto collect = [&](const Dataset& part)
    {
        for (std::size_t row = 0; row < part.row_count(); ++row)
        {
            const auto identifier = part.numeric(row, 0);
            if (!identifier) return false;
            identifiers.insert(static_cast<int>(*identifier));
        }
        return true;
    };

    return collect(first.training)
        && collect(first.validation)
        && collect(first.test)
        && identifiers.size() == 10
        && first.training.source() == "memory://split-fixture#training"
        && first.validation.source() == "memory://split-fixture#validation"
        && first.test.source() == "memory://split-fixture#test";
}

bool test_invalid_split_is_rejected()
{
    try
    {
        (void)make_split_dataset().split({
            .training_fraction = 0.80,
            .validation_fraction = 0.30,
            .test_fraction = 0.10,
            .seed = 1,
            .shuffle = true
        });
    }
    catch (const DatasetError& error)
    {
        return error.code() == DatasetErrorCode::InvalidSplit;
    }
    return false;
}

bool test_raw_matrix_loader_requires_exact_size()
{
    const std::filesystem::path exact_path = temporary_path("mllibrary-matrix-exact.bin");
    const std::filesystem::path short_path = temporary_path("mllibrary-matrix-short.bin");
    const std::filesystem::path long_path = temporary_path("mllibrary-matrix-long.bin");
    if (!write_floats(exact_path, { 1.5f, -2.0f })
        || !write_floats(short_path, { 1.5f })
        || !write_floats(long_path, { 1.5f, -2.0f, 7.0f }))
    {
        return false;
    }

    MemArena* arena = MemArena::create(MiB(8), KiB(64));
    if (!arena) return false;
    Matrix* exact = mat_load(arena, 1, 2, exact_path.string().c_str());
    Matrix* short_matrix = mat_load(arena, 1, 2, short_path.string().c_str());
    Matrix* long_matrix = mat_load(arena, 1, 2, long_path.string().c_str());

    const bool passed = exact
        && exact->data[0] == 1.5f
        && exact->data[1] == -2.0f
        && !short_matrix
        && !long_matrix;

    MemArena::destroy(arena);
    std::filesystem::remove(exact_path);
    std::filesystem::remove(short_path);
    std::filesystem::remove(long_path);
    return passed;
}

} // namespace

int main()
{
    if (!test_typed_csv_parsing_and_fingerprint())
    {
        std::fputs("typed CSV parsing test failed\n", stderr);
        return 1;
    }
    if (!test_parse_errors_include_row_and_column())
    {
        std::fputs("dataset error context test failed\n", stderr);
        return 1;
    }
    if (!test_header_mismatch_is_rejected())
    {
        std::fputs("dataset header mismatch test failed\n", stderr);
        return 1;
    }
    if (!test_deterministic_dataset_split())
    {
        std::fputs("deterministic dataset split test failed\n", stderr);
        return 1;
    }
    if (!test_invalid_split_is_rejected())
    {
        std::fputs("invalid dataset split test failed\n", stderr);
        return 1;
    }
    if (!test_raw_matrix_loader_requires_exact_size())
    {
        std::fputs("strict raw matrix loader test failed\n", stderr);
        return 1;
    }
    return 0;
}
