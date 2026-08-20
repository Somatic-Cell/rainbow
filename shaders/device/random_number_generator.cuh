#ifndef RANDOM_NUMBER_GENERATOR_CUH_
#define RANDOM_NUMBER_GENERATOR_CUH_

template<unsigned int N=16>
class LCG {
public:
    inline __host__ __device__ LCG()
    {

    }

    inline __host__ __device__ LCG(unsigned int val0, unsigned int val1)
    {
        init(val0, val1);
    }

    inline __host__ __device__ float operator() ()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return (state >> 8) / (float) 0x01000000;
    }

    inline __host__ __device__ void init(const unsigned int val0, const unsigned int val1)
    {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for(unsigned int n = 0; n < N; n++){
            s0 += 0x9e3779b9;
            v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
            v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
        }
        state = v0;
    }

protected:

    uint32_t state;
};

__device__ __forceinline__ uint32_t hash(uint32_t x) {
    x ^= x >> 17; x *= 0xed5ad4bb;
    x ^= x >> 11; x *= 0xac4c1b51;
    x ^= x >> 15; x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

#endif // RANDOM_NUMBER_GENERATOR_CUH_