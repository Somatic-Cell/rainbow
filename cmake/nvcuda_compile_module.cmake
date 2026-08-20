# Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, 
# with or without modification, 
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice, 
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.
#
# - Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to 
#   endorse or promote products derived from this software without specific prior written permission.
#
# - THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY fEXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
#   BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
#   IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
#   OR CONSEQUENTIAL DAMAGES 
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
#  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Parallel Thread Execution (PTX) 
# CUDA のための中間表現
# 実行時に JIT コンパイルされる？
# NVCUDA_COMPILE_PTX(
#   SOURCES         コンパイル対象の .cu ファイルのリスト：     file1.cu file2.cu                                   
#   FILENAME_SUFFIX 出力ファイル名に付け加えるサフィックス：    "_optix8"
#   DEPENDENCIES    変更を監視する追加ファイル：                header1.h header2.h
#   TARGET_PATH     生成された .ptx を出力するディレクトリ：    <path where output files should be stored>
#   GENERATED_FILES 呼び出し元に .ptx ファイルのリストを返すための変数名    program_modules
#   NVCC_OPTIONS    nvcc に渡す追加オプション                  -arch=sm_50 -use_fast_math -lineinfo, ... 
# )



function(NVCUDA_COMPILE_PTX)
    # 64 ビット以外は拒否
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
    endif()

    set(options "")
    set(oneValueArgs TARGET_PATH GENERATED_FILES FILENAME_SUFFIX)
    set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

    # この関数に渡された引数リストを扱いやすく分解
    CMAKE_PARSE_ARGUMENTS(
        NVCUDA_COMPILE_PTX      # 出力される変数のプレフィックス (この関数内では，以降 NVCUDA_COMPILE_PTX_OPTIONS のような変数として扱う)
        "${options}"            # True / False 
        "${oneValueArgs}"       # 値を一つだけとる引数
        "${multiValueArgs}"     # 値を複数とる引数
        ${ARGN}                 # 残りのすべての引数
    )

    # if(NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
    #     # Under Linux make sure the target directory exists.
    #     file(MAKE_DIRECTORY ${NVCUDA_COMPILE_PTX_TARGET_PATH})
    # endif()
    file(MAKE_DIRECTORY ${NVCUDA_COMPILE_PTX_TARGET_PATH})


    # Custom build rule to generate ptx files from cuda files
    foreach(input ${NVCUDA_COMPILE_PTX_SOURCES})
        get_filename_component(input_we "${input}" NAME_WE) # Without extension (ディレクトリ，拡張子なしのファイル名)
        get_filename_component(ABS_PATH "${input}" ABSOLUTE)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${ABS_PATH}")
        
        # 出力パスの組み立て
        set(output "${NVCUDA_COMPILE_PTX_TARGET_PATH}/${input_we}${NVCUDA_COMPILE_PTX_FILENAME_SUFFIX}.ptx")

        list(APPEND PTX_FILES "${output}")

        # .cu ファイルを .ptx ファイルに変換するためのビルドルールを定義
        add_custom_command(
        OUTPUT  "${output}"
        DEPENDS "${input}" ${NVCUDA_COMPILE_PTX_DEPENDENCIES}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --machine=64 --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} "${input}" -o "${output}"  > "${output}.log" 2>&1
    )
    endforeach()

    # 呼び出しもとで指定した GENERATED_FILES に .ptx の出力ファイル名リストを代入
    # 呼び出し元のスコープの変数として使用可能
    set(${NVCUDA_COMPILE_PTX_GENERATED_FILES} ${PTX_FILES} PARENT_SCOPE)
endfunction()



# 
function(NVCUDA_COMPILE_OPTIX_IR)
    # 64 ビットのビルドでなければ中止
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
    endif()

    set(options "")
    set(oneValueArgs TARGET_PATH GENERATED_FILES FILENAME_SUFFIX)
    set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

    CMAKE_PARSE_ARGUMENTS(
        NVCUDA_COMPILE_OPTIX_IR 
        "${options}" 
        "${oneValueArgs}" 
        "${multiValueArgs}" 
        ${ARGN}
    )


    if(NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
        # Under Linux make sure the target directory exists.
        file(MAKE_DIRECTORY ${NVCUDA_COMPILE_OPTIX_IR_TARGET_PATH})
    endif()
    # file(MAKE_DIRECTORY ${NVCUDA_COMPILE_OPTIX_IR_TARGET_PATH})

    # Custom build rule to generate optixir files from cuda files
    foreach(input ${NVCUDA_COMPILE_OPTIX_IR_SOURCES})
        get_filename_component(input_we "${input}" NAME_WE)
        get_filename_component(ABS_PATH "${input}" ABSOLUTE)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${ABS_PATH}")

        set(output "${NVCUDA_COMPILE_OPTIX_IR_TARGET_PATH}/${input_we}${NVCUDA_COMPILE_OPTIX_IR_FILENAME_SUFFIX}.optixir")

        list(APPEND OPTIXIR_FILES "${output}")

        add_custom_command(
            OUTPUT  "${output}"
            DEPENDS "${input}" ${NVCUDA_COMPILE_OPTIX_IR_DEPENDENCIES}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --machine=64 --optix-ir ${NVCUDA_COMPILE_OPTIX_IR_NVCC_OPTIONS} "${input}" -o "${output}" > "${output}.log" 2>&1
            # WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM
        )
    endforeach()
    set(${NVCUDA_COMPILE_OPTIX_IR_GENERATED_FILES} ${OPTIXIR_FILES} PARENT_SCOPE)
endfunction()