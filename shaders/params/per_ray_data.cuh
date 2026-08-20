#ifndef PER_RAY_DATA_CUH_
#define PER_RAY_DATA_CUH_

#include "../config.cuh"
#include <optix_device.h>

struct PRD {
    Random random;

    float3 position;
    float3 wo;          // outgoing direction (カメラ方向，ワールド空間)
    float3 wi;          // incoming direction (光源方向，ワールド空間)

    float3 albedo;
    float3 primaryNormal;
    float3 primaryAlbedo;
    float3 contribution;
    
    struct{
        float light;
        float bxdf;
    } pdf;
    
    int bounce;
    int lastHitMaterialType;
    bool continueTrace;

    uint instanceID;
};

struct ShadowPRD {
    Random random;
    bool visible    {false};
};


static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr; 
}

static __forceinline__ __device__
void packPointer( void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer(u0, u1));
}

#endif // PER_RAY_DATA_CUH_