/**
 * @file deterministic_sampler.h
 * @author Enigmatisms
 * @brief Samples pixels of which alpha is greater than 1e-3 for training
 * @date 2022-03-26
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

__global__ void validSamplerKernel(
    const float *const valid_rgb, const int* const coords, const float* const params, float *output, float *lengths,
    curandState *r_state, int valid_num, int width, int height, int offset, int seed_offset, float focal, float near_t = 0.0, float resolution = 0.0
);

/// rgb shape (valid num, 3), coords shape (valid num, 3) (3 --> x, y, index)
__host__ void validSampler(
    at::Tensor rgb, at::Tensor coords, at::Tensor tfs, at::Tensor output, at::Tensor lengths, int img_w,
    int img_h, int sample_ray_num, int sample_bin_num, float focal, float near_t = 0.0, float far_t = 0.0
);
