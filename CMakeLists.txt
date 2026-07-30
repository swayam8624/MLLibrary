cmake_minimum_required(VERSION 3.28)

project(MLLibrary VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MLLIB_USE_KAIRO_MATH "Build and link the KairoMath module target when using an upstream module-capable compiler." OFF)
option(MLLIB_USE_KAIRO_GPU "Build the optional Metal Tensor execution integration." OFF)
set(MLLIB_KAIRO_FOUNDATION_DIR
    "/Users/swayamsingal/Desktop/Programming/Kairo/Foundation"
    CACHE PATH "Path to the Kairo Foundation checkout.")
set(MLLIB_KAIRO_GPU_DIR
    "/Users/swayamsingal/Desktop/Programming/Kairo/KairoGPU"
    CACHE PATH "Path to the KairoGPU checkout.")

if(MLLIB_USE_KAIRO_GPU AND NOT MLLIB_USE_KAIRO_MATH)
    message(FATAL_ERROR "MLLIB_USE_KAIRO_GPU requires MLLIB_USE_KAIRO_MATH.")
endif()

set(MLLIBRARY_CORE_SOURCES
    MLLibrary/Core/arena.cpp
    MLLibrary/Core/prng.cpp
    MLLibrary/Math/matrix.cpp
    MLLibrary/Model/model.cpp
    MLLibrary/Training/training.cpp
    MLLibrary/Training/checkpoint.cpp
    MLLibrary/Data/data.cpp
    MLLibrary/Data/dataset.cpp
    MLLibrary/Data/profiling.cpp
    MLLibrary/Preprocessing/pipeline.cpp
    MLLibrary/Experiment/experiment.cpp
    MLLibrary/Evaluation/metrics.cpp
    MLLibrary/Evaluation/cross_validation.cpp
    MLLibrary/Search/search.cpp
    MLLibrary/Classical/classical.cpp
    MLLibrary/Contracts/runtime_contracts.cpp
)

add_library(MLLibraryCore STATIC ${MLLIBRARY_CORE_SOURCES})
add_library(MLLibrary::Core ALIAS MLLibraryCore)
set_target_properties(MLLibraryCore PROPERTIES EXPORT_NAME Core)

target_include_directories(MLLibraryCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Core>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Math>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Model>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Training>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Data>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Preprocessing>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Experiment>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Evaluation>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Search>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Classical>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Contracts>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Core>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Math>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Model>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Training>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Data>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Preprocessing>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Experiment>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Evaluation>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Search>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Classical>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary/Contracts>
)
target_compile_features(MLLibraryCore PUBLIC cxx_std_23)

add_executable(MLLibrary MLLibrary/main.cpp)
target_link_libraries(MLLibrary PRIVATE MLLibrary::Core)
target_compile_features(MLLibrary PRIVATE cxx_std_23)

add_executable(MLLibraryTrainingReport
    MLLibrary/Visual/training_report.cpp
)
add_executable(MLLibraryDatasetReport
    MLLibrary/Visual/dataset_report.cpp
)
add_executable(MLLibraryDatasetInspector
    MLLibrary/Visual/dataset_inspector.cpp
)
target_link_libraries(MLLibraryDatasetInspector PRIVATE MLLibrary::Core)
add_executable(MLLibraryBenchmark
    MLLibrary/Benchmark/benchmark.cpp
    MLLibrary/Benchmark/matrix_benchmark.cpp
)
target_link_libraries(MLLibraryBenchmark PRIVATE MLLibrary::Core)
target_compile_features(MLLibraryTrainingReport PRIVATE cxx_std_23)
target_compile_features(MLLibraryDatasetReport PRIVATE cxx_std_23)
target_compile_features(MLLibraryDatasetInspector PRIVATE cxx_std_23)
target_compile_features(MLLibraryBenchmark PRIVATE cxx_std_23)
target_include_directories(MLLibraryBenchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(MLLibraryCore PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    target_compile_options(MLLibrary PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    target_compile_options(MLLibraryTrainingReport PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MLLibraryDatasetReport PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MLLibraryDatasetInspector PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MLLibraryBenchmark PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(MLLibraryCore PRIVATE /W4)
    target_compile_options(MLLibrary PRIVATE /W4)
    target_compile_options(MLLibraryTrainingReport PRIVATE /W4)
    target_compile_options(MLLibraryDatasetReport PRIVATE /W4)
    target_compile_options(MLLibraryDatasetInspector PRIVATE /W4)
    target_compile_options(MLLibraryBenchmark PRIVATE /W4)
endif()

if(MLLIB_USE_KAIRO_MATH)
    set(KAIRO_MATH_DIR "${MLLIB_KAIRO_FOUNDATION_DIR}/KairoMath")
    if(EXISTS "${KAIRO_MATH_DIR}/CMakeLists.txt")
        if(MLLIB_USE_KAIRO_GPU)
            set(KAIRO_MATH_USE_GPU ON CACHE BOOL "Enable KairoMath GPU execution." FORCE)
            set(KAIRO_GPU_DIR "${MLLIB_KAIRO_GPU_DIR}" CACHE PATH "Path to KairoGPU." FORCE)
        endif()
        add_subdirectory("${KAIRO_MATH_DIR}" "${CMAKE_BINARY_DIR}/KairoMath")
        target_link_libraries(MLLibrary PRIVATE KairoMath)
        target_compile_definitions(MLLibrary PRIVATE MLLIB_HAS_KAIRO_MATH=1)
        add_library(MLLibraryTensorRuntime)
        target_sources(MLLibraryTensorRuntime
            PUBLIC
                FILE_SET CXX_MODULES
                BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary
                FILES MLLibrary/TensorRuntime.cppm
        )
        target_link_libraries(MLLibraryTensorRuntime PUBLIC KairoMath)
        target_compile_features(MLLibraryTensorRuntime PUBLIC cxx_std_23)
        add_library(MLLibraryContractsModule
            ModuleInterfaces/contracts_impl.cpp
        )
        target_sources(MLLibraryContractsModule
            PUBLIC
                FILE_SET CXX_MODULES
                BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterfaces
                FILES ModuleInterfaces/contracts.cppm
        )
        target_include_directories(MLLibraryContractsModule PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Contracts)
        target_compile_features(MLLibraryContractsModule PUBLIC cxx_std_23)
        if(MLLIB_USE_KAIRO_GPU)
            target_sources(MLLibraryTensorRuntime
                PUBLIC
                    FILE_SET CXX_MODULES
                    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary
                    FILES MLLibrary/TensorGPURuntime.cppm
            )
            target_link_libraries(MLLibraryTensorRuntime PUBLIC KairoGPU)
        endif()
        if(BUILD_TESTING)
            add_executable(MLLibraryTensorIntegrationTests Tests/tensor_integration.cpp)
            target_link_libraries(MLLibraryTensorIntegrationTests PRIVATE KairoMath MLLibraryTensorRuntime)
            target_compile_features(MLLibraryTensorIntegrationTests PRIVATE cxx_std_23)
            add_test(NAME MLLibraryTensorIntegrationTests COMMAND MLLibraryTensorIntegrationTests)
            add_executable(MLLibraryContractsModuleTests Tests/contracts_module.cpp)
            target_link_libraries(MLLibraryContractsModuleTests PRIVATE MLLibraryContractsModule)
            target_compile_features(MLLibraryContractsModuleTests PUBLIC cxx_std_23)
            add_test(NAME MLLibraryContractsModuleTests COMMAND MLLibraryContractsModuleTests)
            if(MLLIB_USE_KAIRO_GPU)
                add_executable(MLLibraryTensorGPUIntegrationTests Tests/tensor_gpu_integration.cpp)
                target_link_libraries(MLLibraryTensorGPUIntegrationTests PRIVATE KairoMath KairoGPU MLLibraryTensorRuntime)
                target_compile_features(MLLibraryTensorGPUIntegrationTests PRIVATE cxx_std_23)
                add_test(NAME MLLibraryTensorGPUIntegrationTests COMMAND MLLibraryTensorGPUIntegrationTests)
            endif()
        endif()
    else()
        message(FATAL_ERROR "MLLIB_USE_KAIRO_MATH is ON, but KairoMath was not found at: ${KAIRO_MATH_DIR}")
    endif()
endif()

foreach(data_file
    train_images.mat
    train_labels.mat
    test_images.mat
    test_labels.mat)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/${data_file}")
        add_custom_command(TARGET MLLibrary POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/${data_file}"
                "$<TARGET_FILE_DIR:MLLibrary>/${data_file}")
    endif()
endforeach()

configure_package_config_file(
    cmake/MLLibraryConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MLLibraryConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MLLibrary"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MLLibraryConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS MLLibraryCore
    EXPORT MLLibraryTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY MLLibrary/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/MLLibrary
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)
install(EXPORT MLLibraryTargets
    FILE MLLibraryTargets.cmake
    NAMESPACE MLLibrary::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MLLibrary
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MLLibraryConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MLLibraryConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MLLibrary
)
install(TARGETS MLLibrary RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS MLLibraryTrainingReport RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS MLLibraryDatasetReport RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS MLLibraryDatasetInspector RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS MLLibraryBenchmark RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTING)
    add_executable(MLLibraryCoreTests Tests/core_tests.cpp)
    target_link_libraries(MLLibraryCoreTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryCoreTests PRIVATE cxx_std_23)

    add_executable(MLLibraryArenaTests Tests/arena_tests.cpp)
    target_link_libraries(MLLibraryArenaTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryArenaTests PRIVATE cxx_std_23)

    add_executable(MLLibraryTrainingTests Tests/training_tests.cpp)
    target_link_libraries(MLLibraryTrainingTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryTrainingTests PRIVATE cxx_std_23)

    add_executable(MLLibraryAutogradTests Tests/autograd_tests.cpp)
    target_link_libraries(MLLibraryAutogradTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryAutogradTests PRIVATE cxx_std_23)

    add_executable(MLLibraryDatasetTests Tests/dataset_tests.cpp)
    target_link_libraries(MLLibraryDatasetTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryDatasetTests PRIVATE cxx_std_23)

    add_executable(MLLibraryProfilingTests Tests/profiling_tests.cpp)
    target_link_libraries(MLLibraryProfilingTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryProfilingTests PRIVATE cxx_std_23)

    add_executable(MLLibraryPipelineTests Tests/pipeline_tests.cpp)
    target_link_libraries(MLLibraryPipelineTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryPipelineTests PRIVATE cxx_std_23)

    add_executable(MLLibraryExperimentTests Tests/experiment_tests.cpp)
    target_link_libraries(MLLibraryExperimentTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryExperimentTests PRIVATE cxx_std_23)

    add_executable(MLLibraryEvaluationTests Tests/evaluation_tests.cpp)
    target_link_libraries(MLLibraryEvaluationTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibraryEvaluationTests PRIVATE cxx_std_23)

    add_executable(MLLibrarySearchTests Tests/search_tests.cpp)
    target_link_libraries(MLLibrarySearchTests PRIVATE MLLibrary::Core)
    target_compile_features(MLLibrarySearchTests PRIVATE cxx_std_23)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(MLLibraryCoreTests PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
        target_compile_options(MLLibraryArenaTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryTrainingTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryAutogradTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryDatasetTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryProfilingTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryPipelineTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryExperimentTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibraryEvaluationTests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(MLLibrarySearchTests PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(MLLibraryCoreTests PRIVATE /W4)
        target_compile_options(MLLibraryArenaTests PRIVATE /W4)
        target_compile_options(MLLibraryTrainingTests PRIVATE /W4)
        target_compile_options(MLLibraryAutogradTests PRIVATE /W4)
        target_compile_options(MLLibraryDatasetTests PRIVATE /W4)
        target_compile_options(MLLibraryProfilingTests PRIVATE /W4)
        target_compile_options(MLLibraryPipelineTests PRIVATE /W4)
        target_compile_options(MLLibraryExperimentTests PRIVATE /W4)
        target_compile_options(MLLibraryEvaluationTests PRIVATE /W4)
        target_compile_options(MLLibrarySearchTests PRIVATE /W4)
    endif()

    add_test(NAME MLLibraryCoreTests COMMAND MLLibraryCoreTests)
    add_test(NAME MLLibraryArenaTests COMMAND MLLibraryArenaTests)
    add_test(NAME MLLibraryTrainingTests COMMAND MLLibraryTrainingTests)
    add_test(NAME MLLibraryAutogradTests COMMAND MLLibraryAutogradTests)
    add_test(NAME MLLibraryDatasetTests COMMAND MLLibraryDatasetTests)
    add_test(NAME MLLibraryProfilingTests COMMAND MLLibraryProfilingTests)
    add_test(NAME MLLibraryPipelineTests COMMAND MLLibraryPipelineTests)
    add_test(NAME MLLibraryExperimentTests COMMAND MLLibraryExperimentTests)
    add_test(NAME MLLibraryEvaluationTests COMMAND MLLibraryEvaluationTests)
    add_test(NAME MLLibrarySearchTests COMMAND MLLibrarySearchTests)
    add_test(
        NAME MLLibraryBenchmarkTest
        COMMAND MLLibraryBenchmark
            --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_test.json"
            --size 32
            --iterations 3
            --seed 42)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/test_images.mat"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/test_labels.mat")
        add_test(
            NAME MLLibraryDatasetReportTest
            COMMAND MLLibraryDatasetReport
                "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/test_images.mat"
                "${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/test_labels.mat"
                "${CMAKE_CURRENT_BINARY_DIR}/dataset_report_test.html"
                16)
    endif()
endif()
