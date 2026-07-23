#include "benchmark.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string_view>

namespace
{
    void write_json_string(std::ostream& output, std::string_view value)
    {
        output << '"';
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
                if (character < 0x20u)
                    output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                else
                    output << static_cast<char>(character);
            }
        }
        output << '"';
    }
}

namespace mllibrary::benchmark
{
    void ExperimentManifest::validate() const
    {
        if (runID.empty() || benchmarkName.empty() || compiler.empty() || platform.empty())
            throw std::invalid_argument("Experiment manifest identity and environment fields are required.");
    }

    void BenchmarkResult::validate() const
    {
        if (measuredIterations == 0
            || !std::isfinite(meanLatencyMilliseconds) || meanLatencyMilliseconds < 0.0
            || !std::isfinite(p50LatencyMilliseconds) || p50LatencyMilliseconds < 0.0
            || !std::isfinite(p95LatencyMilliseconds) || p95LatencyMilliseconds < 0.0
            || !std::isfinite(throughputOperationsPerSecond) || throughputOperationsPerSecond < 0.0
            || !std::isfinite(maximumAbsoluteError) || maximumAbsoluteError < 0.0)
            throw std::invalid_argument("Benchmark result contains invalid measurements.");
    }

    bool write_json_report(
        const ExperimentManifest& manifest,
        const BenchmarkResult& result,
        const char* path,
        std::string* error)
    {
        try
        {
            manifest.validate();
            result.validate();
        }
        catch (const std::exception& exception)
        {
            if (error) *error = exception.what();
            return false;
        }
        if (!path || !*path)
        {
            if (error) *error = "Benchmark report path is required.";
            return false;
        }
        std::ofstream output(path);
        if (!output)
        {
            if (error) *error = "Cannot open benchmark report.";
            return false;
        }
        output << "{\n  \"schema_version\":" << ExperimentManifest::SchemaVersion << ",\n  \"manifest\":{\n"
               << "    \"run_id\":";
        write_json_string(output, manifest.runID);
        output << ",\n    \"benchmark\":";
        write_json_string(output, manifest.benchmarkName);
        output << ",\n    \"seed\":" << manifest.seed << ",\n    \"compiler\":";
        write_json_string(output, manifest.compiler);
        output << ",\n    \"platform\":";
        write_json_string(output, manifest.platform);
        output << ",\n    \"arguments\":[";
        for (std::size_t index = 0; index < manifest.arguments.size(); ++index)
        {
            if (index != 0) output << ',';
            write_json_string(output, manifest.arguments[index]);
        }
        output << "]\n  },\n  \"metrics\":{\n"
               << "    \"warmup_iterations\":" << result.warmupIterations << ",\n"
               << "    \"measured_iterations\":" << result.measuredIterations << ",\n"
               << "    \"mean_latency_ms\":" << std::setprecision(12) << result.meanLatencyMilliseconds << ",\n"
               << "    \"p50_latency_ms\":" << result.p50LatencyMilliseconds << ",\n"
               << "    \"p95_latency_ms\":" << result.p95LatencyMilliseconds << ",\n"
               << "    \"throughput_operations_per_second\":" << result.throughputOperationsPerSecond << ",\n"
               << "    \"peak_resident_bytes\":" << result.peakResidentBytes << ",\n"
               << "    \"maximum_absolute_error\":" << result.maximumAbsoluteError << "\n"
               << "  }\n}\n";
        if (!output)
        {
            if (error) *error = "Benchmark report write failed.";
            return false;
        }
        return true;
    }
}
