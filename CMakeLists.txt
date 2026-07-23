cmake_minimum_required(VERSION 3.28)

project(MLLibrary LANGUAGES CXX)

include(CTest)

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

add_executable(MLLibrary
    MLLibrary/main.cpp
    MLLibrary/Core/arena.cpp
    MLLibrary/Core/prng.cpp
    MLLibrary/Math/matrix.cpp
    MLLibrary/Model/model.cpp
    MLLibrary/Training/training.cpp
    MLLibrary/Training/checkpoint.cpp
    MLLibrary/Data/data.cpp
    MLLibrary/Classical/classical.cpp
    MLLibrary/Contracts/runtime_contracts.cpp
)

add_executable(MLLibraryTrainingReport
    MLLibrary/Visual/training_report.cpp
)
add_executable(MLLibraryDatasetReport
    MLLibrary/Visual/dataset_report.cpp
)
add_executable(MLLibraryBenchmark
    MLLibrary/Benchmark/benchmark.cpp
    MLLibrary/Benchmark/matrix_benchmark.cpp
)
target_compile_features(MLLibraryTrainingReport PRIVATE cxx_std_23)
target_compile_features(MLLibraryDatasetReport PRIVATE cxx_std_23)
target_compile_features(MLLibraryBenchmark PRIVATE cxx_std_23)
target_include_directories(MLLibraryBenchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Benchmark)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(MLLibraryTrainingReport PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MLLibraryDatasetReport PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MLLibraryBenchmark PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(MLLibraryTrainingReport PRIVATE /W4)
    target_compile_options(MLLibraryDatasetReport PRIVATE /W4)
    target_compile_options(MLLibraryBenchmark PRIVATE /W4)
endif()

target_include_directories(MLLibrary
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Core
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Math
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Model
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Training
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Training
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Data
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Contracts
)

target_compile_features(MLLibrary PRIVATE cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(MLLibrary PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
elseif(MSVC)
    target_compile_options(MLLibrary PRIVATE /W4)
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
            target_compile_features(MLLibraryContractsModuleTests PRIVATE cxx_std_23)
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

install(TARGETS MLLibrary RUNTIME DESTINATION bin)
install(TARGETS MLLibraryTrainingReport RUNTIME DESTINATION bin)
install(TARGETS MLLibraryDatasetReport RUNTIME DESTINATION bin)
install(TARGETS MLLibraryBenchmark RUNTIME DESTINATION bin)
if(BUILD_TESTING)
    add_executable(MLLibraryCoreTests
        Tests/core_tests.cpp
        MLLibrary/Core/arena.cpp
        MLLibrary/Core/prng.cpp
        MLLibrary/Math/matrix.cpp
        MLLibrary/Model/model.cpp
        MLLibrary/Training/training.cpp
        MLLibrary/Training/checkpoint.cpp
        MLLibrary/Classical/classical.cpp
        MLLibrary/Contracts/runtime_contracts.cpp
    )
    target_include_directories(MLLibraryCoreTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Core
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Math
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Model
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Classical
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Contracts
    )
    target_compile_features(MLLibraryCoreTests PRIVATE cxx_std_23)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(MLLibraryCoreTests PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    elseif(MSVC)
        target_compile_options(MLLibraryCoreTests PRIVATE /W4)
    endif()
    add_test(NAME MLLibraryCoreTests COMMAND MLLibraryCoreTests)
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
