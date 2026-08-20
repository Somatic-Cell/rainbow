#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"

extern "C" __global__ void __anyhit__radiance()
{
    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();
    
    return;
}