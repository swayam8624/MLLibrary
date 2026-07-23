#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mllibrary::benchmark
{
    struct ExperimentManifest final
    {
        static constexpr std::uint32_t SchemaVersion = 1u;
        std::string runID;
        std::string benchmarkName;
        std::uint64_t seed = 0;
        std::string compiler;
        std::string platform;
        std::vector<std::string> arguments;

        void validate() const;
    };

    struct BenchmarkResult final
    {
        std::size_t warmupIterations = 0;
        std::size_t measuredIterations = 0;
        double meanLatencyMilliseconds = 0.0;
        double p50LatencyMilliseconds = 0.0;
        double p95LatencyMilliseconds = 0.0;
        double throughputOperationsPerSecond = 0.0;
        std::size_t peakResidentBytes = 0;
        double maximumAbsoluteError = 0.0;

        void validate() const;
    };

    bool write_json_report(
        const ExperimentManifest& manifest,
        const BenchmarkResult& result,
        const char* path,
        std::string* error = nullptr);
}
