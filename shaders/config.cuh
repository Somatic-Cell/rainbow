#ifndef CONFIG_CUH_
#define CONFIG_CUH_

#include "device/random_number_generator.cuh"
#include "../include/launch_params.h"

typedef LCG<16> Random;

extern "C" __constant__ LaunchParams optixLaunchParams;

#endif // CONFIG_CUH_