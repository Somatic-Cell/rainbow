#include <rainbow/cuda_driver.hpp>

#include <exception>
#include <filesystem>
#include <iostream>

int main(
    const int argc,
    char* argv[])
{
    try
    {
        const std::filesystem::path executable_path =
            std::filesystem::absolute(argv[0]);

        const std::filesystem::path fatbin_path =
            argc >= 2
                ? std::filesystem::path(argv[1])
                : executable_path.parent_path()
                    / "modules"
                    / "patch_build.fatbin";

        rainbow::run_cuda_driver_smoke_test(
            fatbin_path);

        return 0;
    }
    catch(const std::exception& exception)
    {
        std::cerr
            << "Error: "
            << exception.what()
            << '\n';

        return 1;
    }
}