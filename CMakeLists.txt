# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

cmake_minimum_required(VERSION 4.2)

set(CMAKE_CUDA_ARCHITECTURES 86)        # 適時自身の環境に合わせて変更する

set(PROJECT_NAME "Rainbow")
set(TARGET_NAME ${PROJECT_NAME})

project(${PROJECT_NAME}
    VERSION 0.1
    DESCRIPTION "This is a simple project to develop a renderer"
    LANGUAGES C CXX CUDA                # JP: CUDA をリストに登録することで， FindCUDA module の呼び出しが不要
)

# -----------------------------------------------------
# ビルドの設定
# -----------------------------------------------------

### JP: C++ 標準の設定
set(CMAKE_CXX_STANDARD 20)              # JP: 使用する C++ のバージョン
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # JP: 上記の標準を必須として使用することを要求
set(CMAKE_CXX_EXTENSIONS OFF)           # JP: ベンダー拡張は OFF

set(
    CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit 13.1 REQUIRED) 
find_package(OptiX91)

if(OptiX9_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX9_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 9.1.0 not found.")
endif()

# -----------------------------------------------------
# サブモジュールの読み込み
# -----------------------------------------------------

# -----------------------------------------------------
# .cu ファイルのコンパイル
# -----------------------------------------------------
set(CPP_VER_CUDA c++20)
set(USE_OPTIX_IR FALSE)
set(CUDA_CC ${CMAKE_CUDA_ARCHITECTURES})

### コンパイルするファイルを列挙

set(DEVICE_PROGRAMS_PATH ${PROJECT_SOURCE_DIR}/device)  # プログラムの所在

# ptx ファイルに変換したいファイル群
# ${DEVICE_PROGRAMS_PATH}/.../.cu
set(
    CUDA_KERNELS
)

# optixir に変換したいファイル群
# ${DEVICE_PROGRAMS_PATH}/.../.cu
set(
    OPTIX_KERNELS
)

### cu ファイルを ptx ファイルへコンパイル

# optix が見つかっていれば，optixir に変換する．
# そうでなければ，ptx に変換するプログラム群に optixir に変換しようとしていたものも含める．
if(OptiX9_FOUND)
    set(USE_OPTIX_IR TRUE)
else()
    set(
        ${CUDA_KERNELS}
        ${CUDA_KERNELS}
        ${OPTIX_KERNELS}
    )
    set(OPTIX_KERNELS)
endif()

# .cu ファイルをコンパイルするのに必要なヘッダファイルたち
set(
    GPU_KERNEL_DEPENDENCIES
)

# .cu ファイルを .ptx ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内)
set(CUDA_NVCC_OPTIONS
    "-Xcompiler" "\"/utf-8 /Zc:__cplusplus\""   # 警告無視，標準準拠オプション
    "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
    "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
    "--gpu-code=sm_${CUDA_CC}" 
    "-std=${CPP_VER_CUDA}"                  
    "-cudart" "shared"
    # "--use_fast_math"                           # 精度を犠牲に高速な近似計算を実行
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")   # Debug ビルドのときだけ
    list(APPEND 
        CUDA_NVCC_OPTIONS
        "-D_DEBUG=1"                    # _DEBUG を定義
        "-G"                            # GPU のデバッグ情報を含める                    
    )
endif()


nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    # 例えば build/Debug/ptxes　以下に .ptx ファイルが生成
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS ${CUDA_NVCC_OPTIONS}
)

message(STATUS "CUDA_KERNELS=${CUDA_KERNELS}")


### cu ファイルを .optixir ファイルにコンパイル

# .cu ファイルを .optixir ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内) 
if(USE_OPTIX_IR)
    set(OPTIX_NVCC_OPTIONS
        "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
        "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
        "-std=${CPP_VER_CUDA}"
        "-cudart" "shared"
        # "--use_fast_math"                           # 精度を犠牲に高速な近似計算を実行
        "--relocatable-device-code=true"
        "-I${OPTIX_INCLUDE_DIR}"
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug") # Debug ビルドのときだけ
        list(APPEND 
            OPTIX_NVCC_OPTIONS
            "-D_DEBUG=1"             # _DEBUG を定義
            "-G"                     # GPU のデバッグ情報を含める                    
            "-O0"                    # 最適化を無効化 
        )
    endif()
    
    nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS ${OPTIX_NVCC_OPTIONS}
)
endif()


# ビルドした結果をまとめる
set(BUILD_CUDA_KERNELS
  ${CUDA_PTXES} 
  ${OPTIX_PTXES}
)


# -----------------------------------------------------
# 実行ファイルの定義
# -----------------------------------------------------
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

add_executable(${PROJECT_NAME}
    "${BUILD_CUDA_KERNELS}"
)

# -----------------------------------------------------
# ターゲットの設定
# -----------------------------------------------------
# コンパイル時のマクロ定義の追加
target_compile_definitions(
    ${PROJECT_NAME} PRIVATE
    /utf-8
    /fsanitize=address
    /O2
)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/utf-8>
                      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# CMake 上の変数を C++ に渡す
if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(USE_OPTIX_IR)
    add_definitions("-DUSE_OPTIX_IR")
endif()


# インクルードパスの追加
target_include_directories(
    ${PROJECT_NAME} PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}"
)

# ライブラリやターゲットをリンク
target_link_libraries(
    ${PROJECT_NAME}
    CUDA::cudart        # FindCudaToolkit の成果物
    CUDA::cuda_driver   # FindCudaToolkit の成果物 (cuCtxGetCurrent() はここにいる)
)


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