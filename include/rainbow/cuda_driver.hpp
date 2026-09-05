#pragma once

#include <filesystem>

namespace rainbow
{
    void run_cuda_driver_smoke_test(
        const std::filesystem::path& fatbin_path
    );
}