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

### JP: CUDA の読み込み
find_package(CUDAToolkit 13.1 REQUIRED) 

### JP: OptiX の読み込み
list(APPEND
    CMAKE_MODULE_PATH
)

set(
    CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" 
)

find_package(OptiX91 REQUIRED)

# -----------------------------------------------------
# JP: サブモジュールの読み込み
# -----------------------------------------------------

# -----------------------------------------------------
# JP: CUDA Device Module
# -----------------------------------------------------
set(USE_OPTIX_IR FALSE)
set(CUDA_CC ${CMAKE_CUDA_ARCHITECTURES})

### JP: cu ファイルを ptx ファイルへコンパイル


add_library(
    rainbow_patch_build
    OBJECT
        device/patch_build.cu
)

set_target_properties(
    rainbow_patch_build
    PROPERTIES
        CUDA_FATBIN_COMPILATION ON
)

target_include_directories(
    rainbow_patch_build
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"   
        "${CMAKE_CURRENT_SOURCE_DIR}/device"   
)

add_library(
    rainbow_optix_module
    OBJECT
        device/optix_programs.cu
)

set_target_properties(
    rainbow_optix_module
    PROPERTIES
        CUDA_OPTIX_COMPILATION ON   
)

target_include_directories(
    rainbow_optix_module
    PRIVATE
        "${OPTIX91_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"   
        "${CMAKE_CURRENT_SOURCE_DIR}/device"   
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        "${OPTIX91_INCLUDE_DIR}"   
)


# JP: ビルドした結果をまとめる
set(BUILD_CUDA_KERNELS
  ${CUDA_PTXES} 
  ${OPTIX_PTXES}
)


# -----------------------------------------------------
# JP: 実行ファイルの定義
# -----------------------------------------------------
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

add_executable(
    ${PROJECT_NAME}
    apps/main.cpp 
    src/cuda_driver.cpp 
    "${BUILD_CUDA_KERNELS}"
)

add_dependencies(
    ${PROJECT_NAME}
    rainbow_cuda_module
    rainbow_optix_module
)

# JP: インクルードパスの追加
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    # "${OPTIX_INCLUDE_DIR}"
)

# JP: ライブラリやターゲットをリンク
target_link_libraries(
    ${PROJECT_NAME}
    CUDA::cuda_driver   # JP: FindCudaToolkit の成果物 (cuCtxGetCurrent() はここにいる)
)

# JP: コンパイル時のマクロ定義の追加
target_compile_definitions(
    ${PROJECT_NAME} 
    PRIVATE
)

# JP: CMake 上の変数を C++ に渡す
if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(USE_OPTIX_IR)
    add_definitions("-DUSE_OPTIX_IR")
endif()

if (MSVC)
    add_compile_options(
        ${PROJECT_NAME}
        PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>
    )
endif()


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