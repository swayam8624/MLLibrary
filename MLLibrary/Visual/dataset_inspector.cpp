// TEACHING TOOL: CSV typed inference -> Dataset -> profile -> JSON and HTML.
// It calls Data/profiling.cpp rather than the raw MNIST Float32 reader.
// Full-dataset profiles are descriptive; protect an untouched final test set
// from model-selection decisions. See DATA_TO_AUTOML_LAB.md.

#include "profiling.hpp"

#include <cstdio>
#include <exception>

// Accept dataset, JSON and HTML output paths plus optional declared target.
// Catch parser/IO exceptions as CLI failures instead of emitting a false
// success summary. The source data is not uploaded to a remote service.
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
