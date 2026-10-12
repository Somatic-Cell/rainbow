# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME "Rainbow")
set(TARGET_NAME ${PROJECT_NAME})

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(
        CMAKE_CUDA_ARCHITECTURES
        120                              # JP: 適時自身の環境に合わせて変更する
        CACHE STRING
        "CUDA architectures to build for"
    )
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY None)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
project(
    ${PROJECT_NAME}
    VERSION 0.1
    LANGUAGES CXX CUDA                # JP: CUDA をリストに登録することで， FindCUDA module の呼び出しが不要
)

# -----------------------------------------------------
# JP: ビルドの設定
# -----------------------------------------------------

### JP: C++ の設定
set(CMAKE_CXX_STANDARD 20)              # JP: 使用する C++ のバージョン
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # JP: 上記の標準を必須として使用することを要求
set(CMAKE_CXX_EXTENSIONS OFF)           # JP: ベンダー拡張は OFF

### JP: CUDA の設定
set(CMAKE_CUDA_STANDARD 20)             # JP: 使用する C++ のバージョン
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # JP: 上記の標準を必須として使用することを要求
set(CMAKE_CUDA_EXTENSIONS OFF)          # JP: ベンダー拡張は OFF


### テストするかどうか
option(BUILD_TESTING "Build the existing regression tests" OFF)
option(RAINBOW_ENABLE_DIAGNOSTICS "Enable detailed patch-witness JSON in the CLI" OFF)
include(CTest)

# -----------------------------------------------------
# Dependencies
# -----------------------------------------------------
get_property(rainbow_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT rainbow_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

list(
    APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" 

)

### JP: CUDA / OptiX の読み込み
find_package(CUDAToolkit 13.0 REQUIRED) 
find_package(OptiX91 REQUIRED)


if(MSVC)
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
endif()

function(rainbow_configure_host target)
    if(WIN32)
        target_compile_definitions(${target} PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /W4 /Zc:__cplusplus /fp:precise)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
endfunction()

# Preserve the real target name for the existing tests. New code can use the alias.
# The numerical implementation is compiled once and shared, not copied into apps.
add_library(rainbow_drop_trace_host STATIC
    src/cuda_driver.cpp
    src/cuda_module.cpp
    src/cuda_error.cpp
    src/optix_context.cpp
    src/optix_error.cpp
    src/read_binary_file.cpp
    src/raindrop_tracer.cpp
    src/patch_accel.cpp
    src/patch_query.cpp
    src/patch_optics.cpp
)
add_library(rainbow::solver ALIAS rainbow_drop_trace_host)
target_include_directories(rainbow_drop_trace_host PUBLIC
    "${PROJECT_SOURCE_DIR}/include" "${OPTIX91_INCLUDE_DIR}")
target_link_libraries(rainbow_drop_trace_host PUBLIC CUDA::cuda_driver)
if(UNIX)
    target_link_libraries(rainbow_drop_trace_host PRIVATE ${CMAKE_DL_LIBS})
endif()
rainbow_configure_host(rainbow_drop_trace_host)
if(MSVC)
    set_target_properties(rainbow_drop_trace_host PROPERTIES
        COMPILE_PDB_NAME rainbow_drop_trace_host
        COMPILE_PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/pdb")
endif()

include("${PROJECT_SOURCE_DIR}/cmake/rainbow_modules.cmake")

# This is still the existing reference/CSV CLI, NOT the proposed dataset generator.
add_executable(rainbow_trace apps/trace_raindrop.cpp)
target_link_libraries(rainbow_trace PRIVATE rainbow::solver)
rainbow_configure_host(rainbow_trace)
target_compile_definitions(rainbow_trace PRIVATE
    RAINBOW_ENABLE_DIAGNOSTICS=$<BOOL:${RAINBOW_ENABLE_DIAGNOSTICS}>)
rainbow_stage_solver_modules(rainbow_trace rainbow_stage_modules)

# Detailed readback/reporting is not a dependency of the numerical solver library.
# Runtime failure detection in the solver is NOT disabled by this option.
if(RAINBOW_ENABLE_DIAGNOSTICS OR BUILD_TESTING)
    add_library(rainbow_diagnostics STATIC
        src/patch_failure_report.cpp
        src/patch_failure_capture.cpp)
    target_link_libraries(rainbow_diagnostics PUBLIC rainbow::solver)
    rainbow_configure_host(rainbow_diagnostics)
    if(MSVC)
        set_target_properties(rainbow_diagnostics PROPERTIES
            COMPILE_PDB_NAME rainbow_diagnostics
            COMPILE_PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/pdb")
        set_property(SOURCE src/patch_failure_report.cpp APPEND PROPERTY
            COMPILE_OPTIONS /fp:strict)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_property(SOURCE src/patch_failure_report.cpp APPEND PROPERTY
            COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
    endif()
endif()
if(RAINBOW_ENABLE_DIAGNOSTICS)
    target_link_libraries(rainbow_trace PRIVATE rainbow_diagnostics)
endif()

if(BUILD_TESTING)
    include("${PROJECT_SOURCE_DIR}/cmake/rainbow_smoke.cmake")
    add_subdirectory(tests)

    # 診断APIを直接使用するテストにだけ，診断ライブラリをリンクする．
    foreach(report_test IN ITEMS
        rainbow_patch_failure_capture_gpu_tests
        rainbow_unpolarized_cuda_tests
    )
        target_link_libraries(
            ${report_test}
            PRIVATE
                rainbow_diagnostics
        )
    endforeach()
endif()
include("${PROJECT_SOURCE_DIR}/cmake/phase_dataset.cmake")

message(STATUS "Rainbow: CUDA architectures=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Rainbow: BUILD_TESTING=${BUILD_TESTING}")
message(STATUS "Rainbow: detailed diagnostics=${RAINBOW_ENABLE_DIAGNOSTICS}")
message(STATUS "Rainbow: current executable=rainbow_trace (reference/CSV CLI)")

