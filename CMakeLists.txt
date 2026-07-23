cmake_minimum_required(VERSION 3.28)

project(MLLibrary LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MLLIB_USE_KAIRO_MATH "Build and link the KairoMath module target when using an upstream module-capable compiler." OFF)
set(MLLIB_KAIRO_FOUNDATION_DIR
    "/Users/swayamsingal/Desktop/Programming/Kairo/Foundation"
    CACHE PATH "Path to the Kairo Foundation checkout.")

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
)

add_executable(MLLibraryTrainingReport
    MLLibrary/Visual/training_report.cpp
)
target_compile_features(MLLibraryTrainingReport PRIVATE cxx_std_23)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(MLLibraryTrainingReport PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(MLLibraryTrainingReport PRIVATE /W4)
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
        if(BUILD_TESTING)
            add_executable(MLLibraryTensorIntegrationTests Tests/tensor_integration.cpp)
            target_link_libraries(MLLibraryTensorIntegrationTests PRIVATE KairoMath MLLibraryTensorRuntime)
            target_compile_features(MLLibraryTensorIntegrationTests PRIVATE cxx_std_23)
            add_test(NAME MLLibraryTensorIntegrationTests COMMAND MLLibraryTensorIntegrationTests)
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
    )
    target_include_directories(MLLibraryCoreTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Core
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Math
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Model
        ${CMAKE_CURRENT_SOURCE_DIR}/MLLibrary/Classical
    )
    target_compile_features(MLLibraryCoreTests PRIVATE cxx_std_23)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(MLLibraryCoreTests PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    elseif(MSVC)
        target_compile_options(MLLibraryCoreTests PRIVATE /W4)
    endif()
    add_test(NAME MLLibraryCoreTests COMMAND MLLibraryCoreTests)
endif()
