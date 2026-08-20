# =========================================
# File Name : FindOptiX91.cmake
# Encoding  : UTF-8
# =========================================
# OptiX ライブラリを読み込むために，頑張って optix.h　を検出する

# OptiX SDK がどこにインストールされているかを探す
# プロジェクトの一つ上の階層に OptiX があるかどうか探す（GUI などから設定できるようにもする）
if (WIN32) 
    # Windows 環境だったら． 
    # WIN32 は Windows 環境だったら 32bit OS でも 64bit OS でも True を返すことに注意．　 True を返す 
    set(OPTIX91_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" CACHE PATH "Path to OptiX installed location.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

# optix.h の検出
# まず，ユーザ指定の OPTIX91_INSTALL_DIR 配下の include/ に optix.h があるかどうかを探す
find_path(
    OPTIX91_INCLUDE_DIR 
    NAMES 
        optix.h 
    PATHS 
        ${OPTIX91_INSTALL_DIR}/include
    NO_DEFAULT_PATH                 # システム標準のパスは検索しない
)

# 標準パスも探してみる
find_path(
    OPTIX91_INCLUDE_DIR 
    NAMES optix.h 
)

# エラーメッセージ
include(FindPackageHandleStandardArgs)  # 標準で搭載されている便利なモジュール：https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html
find_package_handle_standard_args(
    OPTIX91                  # パッケージ名
    DEFAULT_MSG             # エラーメッセージの表示方法
    OPTIX91_INCLUDE_DIR      # 結果として使う関数
    )
mark_as_advanced(OPTIX91_INCLUDE_DIR) # GUI 上には表示しない