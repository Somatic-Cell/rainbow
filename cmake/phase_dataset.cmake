# Include once from the root AFTER rainbow_modules.cmake has defined its helper.
include_guard(GLOBAL)

rainbow_add_module(rainbow_phase_cdf_module device/phase_cdf.cu FATBIN)
add_library(rainbow_phase_dataset STATIC
    "${PROJECT_SOURCE_DIR}/src/phase_cdf.cpp"
    "${PROJECT_SOURCE_DIR}/src/npy_writer.cpp")
target_link_libraries(rainbow_phase_dataset PUBLIC rainbow::solver)
rainbow_configure_host(rainbow_phase_dataset)

add_executable(rainbow_generate "${PROJECT_SOURCE_DIR}/apps/generate_phase.cpp")
target_link_libraries(rainbow_generate PRIVATE rainbow_phase_dataset)
rainbow_configure_host(rainbow_generate)
target_compile_definitions(rainbow_generate PRIVATE
    RAINBOW_ENABLE_DIAGNOSTICS=$<BOOL:${RAINBOW_ENABLE_DIAGNOSTICS}>)
if(RAINBOW_ENABLE_DIAGNOSTICS)
    target_link_libraries(rainbow_generate PRIVATE rainbow_diagnostics)
endif()

# Provenance at configure time, including untracked source files. Not a claim
# that a dirty build is identical to the named commit.
find_package(Git QUIET)
set(rainbow_dataset_commit unknown)
set(rainbow_dataset_dirty 1)
if(Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE _git_result OUTPUT_VARIABLE _git_commit
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_git_result EQUAL 0)
        set(rainbow_dataset_commit "${_git_commit}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE _status_result OUTPUT_VARIABLE _status
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_status_result EQUAL 0 AND _status STREQUAL "")
            set(rainbow_dataset_dirty 0)
        endif()
    endif()
endif()
target_compile_definitions(rainbow_generate PRIVATE
    RAINBOW_SOURCE_COMMIT="${rainbow_dataset_commit}"
    RAINBOW_SOURCE_DIRTY=${rainbow_dataset_dirty})

rainbow_stage_solver_modules(rainbow_generate rainbow_generate_stage_solver)
add_custom_target(rainbow_generate_stage_cdf
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:rainbow_generate>/modules"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_OBJECTS:rainbow_phase_cdf_module>"
        "$<TARGET_FILE_DIR:rainbow_generate>/modules/phase_cdf.fatbin"
    COMMAND_EXPAND_LISTS VERBATIM)
add_dependencies(rainbow_generate_stage_cdf rainbow_phase_cdf_module)
add_dependencies(rainbow_generate rainbow_generate_stage_cdf)

if(BUILD_TESTING)
    add_executable(rainbow_phase_numpy_cpu_tests
        "${PROJECT_SOURCE_DIR}/tests/test_phase_numpy_cpu.cpp"
        "${PROJECT_SOURCE_DIR}/src/npy_writer.cpp")
    target_include_directories(rainbow_phase_numpy_cpu_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(rainbow_phase_numpy_cpu_tests PRIVATE cxx_std_20)
    rainbow_configure_host(rainbow_phase_numpy_cpu_tests)
    add_test(NAME rainbow_phase_numpy_cpu COMMAND rainbow_phase_numpy_cpu_tests)
    set_tests_properties(rainbow_phase_numpy_cpu PROPERTIES LABELS "cpu;cdf;io")

    add_executable(rainbow_phase_cdf_cuda_tests "${PROJECT_SOURCE_DIR}/tests/test_phase_cdf_cuda.cpp")
    target_link_libraries(rainbow_phase_cdf_cuda_tests PRIVATE rainbow_phase_dataset)
    rainbow_configure_host(rainbow_phase_cdf_cuda_tests)
    add_dependencies(rainbow_phase_cdf_cuda_tests rainbow_phase_cdf_module)
    add_test(NAME rainbow_phase_cdf_cuda COMMAND rainbow_phase_cdf_cuda_tests
        "$<TARGET_OBJECTS:rainbow_phase_cdf_module>" COMMAND_EXPAND_LISTS)
    set_tests_properties(rainbow_phase_cdf_cuda PROPERTIES LABELS "gpu;cuda;cdf")

    add_executable(rainbow_phase_storage_cpu_tests
        "${PROJECT_SOURCE_DIR}/tests/test_phase_storage_cpu.cpp")
    target_include_directories(rainbow_phase_storage_cpu_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(rainbow_phase_storage_cpu_tests PRIVATE cxx_std_20)
    rainbow_configure_host(rainbow_phase_storage_cpu_tests)
    add_test(NAME rainbow_phase_storage_cpu COMMAND rainbow_phase_storage_cpu_tests)
    set_tests_properties(rainbow_phase_storage_cpu PROPERTIES LABELS "cpu;cdf;storage")

    add_executable(rainbow_phase_storage_cuda_tests
        "${PROJECT_SOURCE_DIR}/tests/test_phase_storage_cuda.cpp")
    target_link_libraries(rainbow_phase_storage_cuda_tests PRIVATE rainbow_phase_dataset)
    rainbow_configure_host(rainbow_phase_storage_cuda_tests)
    add_dependencies(rainbow_phase_storage_cuda_tests rainbow_phase_cdf_module)
    add_test(NAME rainbow_phase_storage_cuda COMMAND rainbow_phase_storage_cuda_tests
        "$<TARGET_OBJECTS:rainbow_phase_cdf_module>" COMMAND_EXPAND_LISTS)
    set_tests_properties(rainbow_phase_storage_cuda PROPERTIES LABELS "gpu;cuda;cdf;storage")

    # Optional Python discovery is confined to the explicit testing build.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
            RESULT_VARIABLE _numpy_found OUTPUT_QUIET ERROR_QUIET)
        if(_numpy_found EQUAL 0)
            add_test(NAME rainbow_phase_numpy_python
                COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                    -s "${PROJECT_SOURCE_DIR}/tests" -p test_phase_numpy.py -v)
            set_tests_properties(rainbow_phase_numpy_python PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                LABELS "python;cdf;io")
            add_test(NAME rainbow_phase_numpy_cpp_interop
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/tests/check_phase_numpy_cpp.py"
                    "$<TARGET_FILE:rainbow_phase_numpy_cpu_tests>")
            set_tests_properties(rainbow_phase_numpy_cpp_interop PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                LABELS "python;cpp;cdf;io")
        endif()
    endif()
endif()

target_sources(rainbow_generate PRIVATE
    "${PROJECT_SOURCE_DIR}/src/water_refractive_index.cpp")

if(BUILD_TESTING)
    add_executable(rainbow_water_optics_cpu_tests
        "${PROJECT_SOURCE_DIR}/tests/test_water_optics_cpu.cpp"
        "${PROJECT_SOURCE_DIR}/src/water_refractive_index.cpp")

    target_include_directories(rainbow_water_optics_cpu_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(rainbow_water_optics_cpu_tests PRIVATE cxx_std_20)
    rainbow_configure_host(rainbow_water_optics_cpu_tests)

    add_test(
        NAME rainbow_water_optics_cpu
        COMMAND rainbow_water_optics_cpu_tests)
    set_tests_properties(
        rainbow_water_optics_cpu
        PROPERTIES LABELS "cpu;material")
endif()

target_sources(rainbow_generate PRIVATE
    "${PROJECT_SOURCE_DIR}/src/projected_area.cpp")

# 投影面積の凸性検査で使用する，区間演算の計算規約を維持する．
if(MSVC)
    set_source_files_properties(
        "${PROJECT_SOURCE_DIR}/src/projected_area.cpp"
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(
        "${PROJECT_SOURCE_DIR}/src/projected_area.cpp"
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()

if(BUILD_TESTING)
    add_executable(rainbow_projected_area_cpu_tests
        "${PROJECT_SOURCE_DIR}/tests/test_projected_area_cpu.cpp"
        "${PROJECT_SOURCE_DIR}/src/projected_area.cpp")

    target_include_directories(rainbow_projected_area_cpu_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(rainbow_projected_area_cpu_tests PRIVATE cxx_std_20)
    rainbow_configure_host(rainbow_projected_area_cpu_tests)

    add_test(
        NAME rainbow_projected_area_cpu
        COMMAND rainbow_projected_area_cpu_tests)
    set_tests_properties(
        rainbow_projected_area_cpu
        PROPERTIES LABELS "cpu;geometry")

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME rainbow_dataset_batch_python
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests"
                -p test_generate_phase_dataset.py -v)
        set_tests_properties(
            rainbow_dataset_batch_python
            PROPERTIES LABELS "python;dataset")
    endif()
endif()
