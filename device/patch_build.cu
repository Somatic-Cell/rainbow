#include <cstdint>

extern "C" __global__
void write_test_pattern(
    std::uint32_t* output,
    std::uint32_t count
)
{
    const std::int32_t index = 
        blockIdx.x * blockDim.x + threadIdx.x;

    if(index < count)
    {
        output[index] = index  ^ 0x5a5a5a5au;
    }
}