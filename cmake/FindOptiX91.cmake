# =========================================
# File Name : FindOptiX91.cmake
# Encoding  : UTF-8
# =========================================
# OptiX ライブラリを読み込むために，頑張って optix.h　を検出する
#
# OptiX SDK がどこにインストールされているかを探す
# プロジェクトの一つ上の階層に OptiX があるかどうか探す（GUI などから設定できるようにもする）

include(FindPackageHandleStandardArgs)  # 標準で搭載されている便利なモジュール：https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

set(
    OptiX91_ROOT
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
    CACHE PATH
    "Root directory of the OptiX 9.1 SDK"
)

# optix.h の検出
# まず，ユーザ指定の OPTIX91_INSTALL_DIR 配下の include/ に optix.h があるかどうかを探す
find_path(
    OPTIX91_INCLUDE_DIR 
    NAMES 
        optix.h
    HINTS 
        "${OptiX91_ROOT}" 
        "$ENV{OptiX91_ROOT}" 
        "$ENV{OPTIX_ROOT}"
    PATH_SUFFIXES
        include
)

find_package_handle_standard_args(
    OptiX91                  # パッケージ名
    REQUIRED_VARS
        OPTIX91_INCLUDE_DIR      # 結果として使う関数
)

mark_as_advanced(
    OPTIX91_INCLUDE_DIR
) # GUI 上には表示しない