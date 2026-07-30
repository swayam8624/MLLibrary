#include "profiling.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::fprintf(stderr,
            "Usage: MLLibraryDatasetInspector <dataset.csv> <profile.json> <profile.html> [target-column]\n");
        return 2;
    }

    try
    {
        mllibrary::data::SchemaInferenceOptions inference;
        if (argc == 5) inference.target_column = argv[4];
        const mllibrary::data::Dataset dataset =
            mllibrary::data::load_inferred_csv(argv[1], inference);
        const mllibrary::data::DatasetProfile profile =
            mllibrary::data::profile_dataset(dataset);
        mllibrary::data::write_dataset_profile_json(profile, argv[2]);
        mllibrary::data::write_dataset_profile_html(profile, argv[3]);
        std::printf(
            "Profiled %zu rows, %zu columns, %zu duplicate rows. Fingerprint: %s\n",
            profile.row_count,
            profile.column_count,
            profile.duplicate_rows,
            profile.fingerprint.c_str());
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Dataset inspection failed: %s\n", error.what());
        return 1;
    }
}
