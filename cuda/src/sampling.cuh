/**
 * @file sampling.cuh
 * @author Enigmatisms
 * @brief Randomized sampler, used in training phase to conserve GPU memory
 * @date 2022-03-17
 * @copyright Copyright (c) 2022
 */
#pragma once
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include <torch/script.h>

/**
 * @param imgs flatten img tensor
 * @param params P(extrinsics) * K^{-1} (cam intrinsics)
 * @param cam_num, width, height are all related to the original shape of imgs
*/
__global__ void getSampledPoints(
    const float *const imgs, const float* const params, float *output, float *lengths,
    curandState *r_state, int cam_num, int width, int height, int offset, float near_t = 0.0, float resolution = 0.0
);

/// Main kernel function for sampler
__host__ void cudaSampler(
    at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int sample_ray_num, int sample_bin_num, float near_t = 0.0, float resolution = 0.0
);

__global__ void easySamplerKernel(
    const float *const imgs, const float* const params, float *output, float *lengths,
    curandState *r_state, int cam_num, int width, int height, int offset, int r_offset, float focal, float near_t = 0.0, float resolution = 0.0
);

__host__ void easySampler(
    at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int sample_ray_num, int sample_bin_num, float focal, float near_t = 0.0, float far_t = 0.0
);

// can not sample the whole image space, since synthetic data is 800 * 800
// 90 images costs 27G GPU memory