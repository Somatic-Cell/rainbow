#include <rainbow/cuda_driver.hpp>

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace 
{

[[noreturn]]
void throw_cuda_error(
    const CUresult result,
    const std::string_view expression,
    const std::string_view file,
    const int line)
{
    const char* error_name = nullptr;
    const char* error_description = nullptr;

    static_cast<void>(
        cuGetErrorName(result, &error_name));

    static_cast<void>(
        cuGetErrorString(result, &error_description));

    std::ostringstream message;

    message
        << expression
        << " failed at "
        << file
        << ':'
        << line
        << '\n'
        << "CUDA error: "
        << (error_name != nullptr
                ? error_name
                : "unknown")
        << '\n'
        << "Description: "
        << (error_description != nullptr
                ? error_description
                : "unavailable");

    throw std::runtime_error(message.str());
}

#define RAINBOW_CUDA_CHECK(expression)                          \
    do                                                          \
    {                                                           \
        const CUresult cuda_result = (expression);              \
                                                                \
        if(cuda_result != CUDA_SUCCESS)                         \
        {                                                       \
            throw_cuda_error(                                   \
                cuda_result,                                    \
                #expression,                                    \
                __FILE__,                                       \
                __LINE__);                                      \
        }                                                       \
    } while(false)


std::vector<char> read_binary_file(
    const std::filesystem::path& path)
{
    std::ifstream stream(
        path,
        std::ios::binary | std::ios::ate);

    if(!stream)
    {
        throw std::runtime_error(
            "Failed to open CUDA module: "
            + path.string());
    }

    const std::streampos end_position =
        stream.tellg();

    if(end_position == std::streampos(-1)
       || end_position == std::streampos(0))
    {
        throw std::runtime_error(
            "CUDA module is empty or unreadable: "
            + path.string());
    }

    const auto size =
        static_cast<std::size_t>(end_position);

    std::vector<char> data(size);

    stream.seekg(0, std::ios::beg);

    if(!stream.read(
           data.data(),
           static_cast<std::streamsize>(data.size())))
    {
        throw std::runtime_error(
            "Failed to read CUDA module: "
            + path.string());
    }

    return data;
}


struct CudaResources
{
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUdeviceptr output = 0;

    bool primary_context_retained = false;

    ~CudaResources() noexcept
    {
        if(context != nullptr)
        {
            static_cast<void>(
                cuCtxSetCurrent(context));
        }

        if(output != 0)
        {
            static_cast<void>(
                cuMemFree(output));
        }

        if(module != nullptr)
        {
            static_cast<void>(
                cuModuleUnload(module));
        }

        if(stream != nullptr)
        {
            static_cast<void>(
                cuStreamDestroy(stream));
        }

        if(context != nullptr)
        {
            static_cast<void>(
                cuCtxSetCurrent(nullptr));
        }

        if(primary_context_retained)
        {
            static_cast<void>(
                cuDevicePrimaryCtxRelease(device));
        }
    }
};

}


namespace rainbow
{

void run_cuda_driver_smoke_test(
    const std::filesystem::path& fatbin_path)
{
    const std::vector<char> fatbin =
        read_binary_file(fatbin_path);

    CudaResources resources;

    RAINBOW_CUDA_CHECK(
        cuInit(0));

    int device_count = 0;

    RAINBOW_CUDA_CHECK(
        cuDeviceGetCount(&device_count));

    if(device_count <= 0)
    {
        throw std::runtime_error(
            "No CUDA-capable device was found.");
    }

    RAINBOW_CUDA_CHECK(
        cuDeviceGet(
            &resources.device,
            0));

    char device_name[256] = {};

    RAINBOW_CUDA_CHECK(
        cuDeviceGetName(
            device_name,
            static_cast<int>(sizeof(device_name)),
            resources.device));

    RAINBOW_CUDA_CHECK(
        cuDevicePrimaryCtxRetain(
            &resources.context,
            resources.device));

    resources.primary_context_retained = true;

    RAINBOW_CUDA_CHECK(
        cuCtxSetCurrent(resources.context));

    RAINBOW_CUDA_CHECK(
        cuStreamCreate(
            &resources.stream,
            CU_STREAM_NON_BLOCKING));

    RAINBOW_CUDA_CHECK(
        cuModuleLoadData(
            &resources.module,
            fatbin.data()));

    CUfunction function = nullptr;

    RAINBOW_CUDA_CHECK(
        cuModuleGetFunction(
            &function,
            resources.module,
            "write_test_pattern"));

    std::uint32_t count = 1024;

    constexpr unsigned int block_size = 256;

    const unsigned int grid_size =
        (count + block_size - 1)
        / block_size;

    const std::size_t byte_size =
        static_cast<std::size_t>(count)
        * sizeof(std::uint32_t);

    RAINBOW_CUDA_CHECK(
        cuMemAlloc(
            &resources.output,
            byte_size));

    void* kernel_parameters[] =
    {
        &resources.output,
        &count
    };

    RAINBOW_CUDA_CHECK(
        cuLaunchKernel(
            function,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            resources.stream,
            kernel_parameters,
            nullptr));

    RAINBOW_CUDA_CHECK(
        cuStreamSynchronize(
            resources.stream));

    std::vector<std::uint32_t> output(count);

    RAINBOW_CUDA_CHECK(
        cuMemcpyDtoH(
            output.data(),
            resources.output,
            byte_size));

    for(std::uint32_t index = 0;
        index < count;
        ++index)
    {
        const std::uint32_t expected =
            index ^ 0x5a5a5a5au;

        if(output[index] != expected)
        {
            std::ostringstream message;

            message
                << "CUDA output mismatch at index "
                << index
                << ": expected "
                << expected
                << ", got "
                << output[index];

            throw std::runtime_error(
                message.str());
        }
    }

    std::cout
        << "CUDA device: "
        << device_name
        << '\n';

    std::cout
        << "CUDA Driver API smoke test: passed"
        << '\n';
}

}

#undef RAINBOW_CUDA_CHECK