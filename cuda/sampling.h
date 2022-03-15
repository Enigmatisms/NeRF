#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

/**
 * @param imgs flatten img tensor
 * @param params P(extrinsics) * K^{-1} (cam intrinsics)
 * @param cam_num, width, height are all related to the original shape of imgs
*/
__global__ void getSampledPoints(
    const float* const imgs, const Eigen::Matrix3f* const params, float* output, 
    float* lengths, curandState* r_state, int width, int height, float near_t = 0.0, float far_t = 0.0
);