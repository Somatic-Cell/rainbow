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
        86                              # JP: 適時自身の環境に合わせて変更する
        CACHE STRING
        "CUDA architectures to build for"
    )
endif()

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

# -----------------------------------------------------
# Dependencies
# -----------------------------------------------------


### JP: OptiX の読み込み
list(
    APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" 

)

### JP: CUDA / OptiX の読み込み
find_package(CUDAToolkit 13.0 REQUIRED) 
find_package(OptiX91 REQUIRED)


# -----------------------------------------------------
# JP: 実行ファイルの定義
# -----------------------------------------------------

add_executable(
    ${PROJECT_NAME}
    apps/main.cpp 
    src/cuda_driver.cpp 
    src/optix.cpp 
    src/patch_accel.cpp 
)

# JP: インクルードパスの追加
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPTIX91_INCLUDE_DIR}"
)

# JP: ライブラリやターゲットをリンク
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        CUDA::cuda_driver   # JP: FindCudaToolkit の成果物 (cuCtxGetCurrent() はここにいる)
)

# JP: コンパイル時のマクロ定義の追加


# JP: CMake 上の変数を C++ に渡す
if (WIN32)
    target_compile_definitions(
        ${PROJECT_NAME} 
        PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
endif()

if (MSVC)
    target_compile_options(
        ${PROJECT_NAME}
        PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>
    )
endif()

# -----------------------------------------------------
# JP: CUDA Device Module
# OutgoingVertex から OutgoingPatch と OptixAABB を構築
# -----------------------------------------------------

add_library(
    rainbow_patch_build_module
    OBJECT
        device/patch_build.cu
)

set_target_properties(
    rainbow_patch_build_module
    PROPERTIES
        CUDA_FATBIN_COMPILATION ON
)

target_include_directories(
    rainbow_patch_build_module
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"   
        "${CMAKE_CURRENT_SOURCE_DIR}/device"   
)

# -----------------------------------------------------
# JP: OptiX Module
# 入射パッチ群を雨粒に交差させ，経路ごとに OutgoingVertex を作成
# -----------------------------------------------------

add_library(
    rainbow_raindrop_trace_module
    OBJECT
        shaders/entry/raindrop_trace.cu
)

set_target_properties(
    rainbow_raindrop_trace_module
    PROPERTIES
        CUDA_OPTIX_COMPILATION ON   
)

target_include_directories(
    rainbow_raindrop_trace_module
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"   
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${OPTIX91_INCLUDE_DIR}"
)

# -----------------------------------------------------
# JP: OptiX Module
# 出射パッチを問い合わせ，位相関数の LUT を作成
# -----------------------------------------------------

add_library(
    rainbow_patch_query_module
    OBJECT
        shaders/entry/patch_query.cu
)

set_target_properties(
    rainbow_patch_query_module
    PROPERTIES
        CUDA_OPTIX_COMPILATION ON   
)

target_include_directories(
    rainbow_patch_query_module
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"   
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${OPTIX91_INCLUDE_DIR}"
)

# -----------------------------------------------------------------------------
# Build ordering
# -----------------------------------------------------------------------------

foreach(
    cuda_module_target
    IN ITEMS
        rainbow_patch_build_module
        rainbow_raindrop_trace_module
        rainbow_patch_query_module
)
    target_compile_options(
        ${cuda_module_target}
        PRIVATE
            $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/utf-8>
    )
endforeach()

# .fatbin と 2 個の .optixir を Rainbow.exe より先に生成する．
# これらは実行時にロードする module であり，実行ファイルへリンクしない．
add_dependencies(
    Rainbow
    rainbow_patch_build_module
    rainbow_raindrop_trace_module
    rainbow_patch_query_module
)

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD

    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules"
    
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_OBJECTS:rainbow_patch_build_module>"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules/patch_build.fatbin"

    COMMAND_EXPAND_LISTS
    VERBATIM
)


# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------


message(STATUS "----------------------------------------")
message(STATUS "Info :")
message(STATUS "Project name = ${PROJECT_NAME}")
message(STATUS "Project version = ${PROJECT_VERSION}")
message(STATUS "CMAKE_VERSION = ${CMAKE_VERSION}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_LIBDIR = ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "OPTIX = ${optix_LIBRARY}")

message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")