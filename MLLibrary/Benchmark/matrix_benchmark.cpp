#include "benchmark.hpp"

#include "arena.h"
#include "base.h"
#include "matrix.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace
{
    struct Options final
    {
        const char* output = nullptr;
        std::size_t size = 128;
        std::size_t iterations = 10;
        std::uint64_t seed = 5489u;
    };

    template<typename T>
    bool parse_integer(std::string_view text, T& value)
    {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc{} && end == text.data() + text.size();
    }

    bool parse_options(int count, char** values, Options& options)
    {
        for (int index = 1; index < count; index += 2)
        {
            if (index + 1 >= count) return false;
            const std::string_view option(values[index]);
            if (option == "--output") options.output = values[index + 1];
            else if (option == "--size")
            {
                if (!parse_integer(std::string_view(values[index + 1]), options.size)) return false;
            }
            else if (option == "--iterations")
            {
                if (!parse_integer(std::string_view(values[index + 1]), options.iterations)) return false;
            }
            else if (option == "--seed")
            {
                if (!parse_integer(std::string_view(values[index + 1]), options.seed)) return false;
            }
            else return false;
        }
        return options.output && options.size > 0 && options.size <= 2048 && options.iterations > 0;
    }

    bool multiply(Matrix* output, const Matrix* lhs, const Matrix* rhs)
    {
        return mat_mul(output, lhs, rhs, true, false, false);
    }

    std::size_t peak_resident_bytes()
    {
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
        return static_cast<std::size_t>(usage.ru_maxrss);
#else
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024u;
#endif
    }

    double percentile(std::vector<double> values, double quantile)
    {
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
        return values[std::min(index, values.size() - 1)];
    }
}

int main(int argumentCount, char** arguments)
{
    Options options;
    if (!parse_options(argumentCount, arguments, options))
    {
        std::cerr << "Usage: " << arguments[0]
                  << " --output report.json --size N --iterations N --seed N\n";
        return 2;
    }

    const std::size_t elements = options.size * options.size;
    const u64 matrix_bytes = static_cast<u64>(elements) * sizeof(f32);
    const u64 reserve_bytes = std::max<u64>(MiB(64), matrix_bytes * 4 + MiB(8));
    MemArena* arena = MemArena::create(reserve_bytes, MiB(1));
    if (!arena)
    {
        std::cerr << "Cannot allocate benchmark arena.\n";
        return 1;
    }

    const u32 dimension = static_cast<u32>(options.size);
    Matrix* lhs = mat_create(arena, dimension, dimension);
    Matrix* rhs = mat_create(arena, dimension, dimension);
    Matrix* output = mat_create(arena, dimension, dimension);
    if (!lhs || !rhs || !output)
    {
        std::cerr << "Cannot allocate benchmark matrices.\n";
        MemArena::destroy(arena);
        return 1;
    }

    std::mt19937_64 generator(options.seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::size_t index = 0; index < elements; ++index)
    {
        lhs->data[index] = distribution(generator);
        rhs->data[index] = distribution(generator);
    }

    constexpr std::size_t warmups = 2;
    for (std::size_t iteration = 0; iteration < warmups; ++iteration)
    {
        if (!multiply(output, lhs, rhs))
        {
            std::cerr << "Production matrix multiplication rejected the benchmark shape.\n";
            MemArena::destroy(arena);
            return 1;
        }
    }

    std::vector<double> latencies;
    latencies.reserve(options.iterations);
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration)
    {
        const auto start = std::chrono::steady_clock::now();
        const bool multiplied = multiply(output, lhs, rhs);
        const auto stop = std::chrono::steady_clock::now();
        if (!multiplied)
        {
            std::cerr << "Production matrix multiplication failed.\n";
            MemArena::destroy(arena);
            return 1;
        }
        latencies.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    double maximumError = 0.0;
    for (std::size_t row = 0; row < options.size; ++row)
        for (std::size_t column = 0; column < options.size; ++column)
        {
            double reference = 0.0;
            for (std::size_t inner = 0; inner < options.size; ++inner)
                reference += static_cast<double>(lhs->data[row * options.size + inner])
                    * static_cast<double>(rhs->data[inner * options.size + column]);
            maximumError = std::max(maximumError,
                std::abs(reference - static_cast<double>(output->data[row * options.size + column])));
        }

    const double totalMilliseconds = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double operationsPerIteration = 2.0 * static_cast<double>(options.size)
        * static_cast<double>(options.size) * static_cast<double>(options.size);
#if defined(__clang__)
    const std::string compiler = "clang-" __clang_version__;
#elif defined(__GNUC__)
    const std::string compiler = "gcc-" __VERSION__;
#else
    const std::string compiler = "unknown";
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    const std::string platform = "macos-arm64";
#elif defined(__APPLE__)
    const std::string platform = "macos";
#elif defined(__linux__) && defined(__x86_64__)
    const std::string platform = "linux-x86_64";
#else
    const std::string platform = "unknown";
#endif

    const mllibrary::benchmark::ExperimentManifest manifest{
        .runID = "matrix-square-seed-" + std::to_string(options.seed),
        .benchmarkName = "cpu.scalar.mat_mul",
        .seed = options.seed,
        .compiler = compiler,
        .platform = platform,
        .arguments = {
            "size=" + std::to_string(options.size),
            "iterations=" + std::to_string(options.iterations)
        }
    };
    const mllibrary::benchmark::BenchmarkResult result{
        .warmupIterations = warmups,
        .measuredIterations = options.iterations,
        .meanLatencyMilliseconds = totalMilliseconds / static_cast<double>(options.iterations),
        .p50LatencyMilliseconds = percentile(latencies, 0.50),
        .p95LatencyMilliseconds = percentile(latencies, 0.95),
        .throughputOperationsPerSecond = operationsPerIteration * options.iterations
            / (totalMilliseconds / 1000.0),
        .peakResidentBytes = peak_resident_bytes(),
        .maximumAbsoluteError = maximumError
    };
    std::string error;
    if (!mllibrary::benchmark::write_json_report(manifest, result, options.output, &error))
    {
        std::cerr << error << '\n';
        MemArena::destroy(arena);
        return 1;
    }

    MemArena::destroy(arena);
    return 0;
}
