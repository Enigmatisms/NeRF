/**
 * @file deterministic_sampler.h
 * @author Enigmatisms
 * @brief Non-random sampling, usually involves in the testing phase, in which the whole image plane should be sampled
 * @date 2022-03-23
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

/// for testing, 4-8 images being sampled at the same time is recommended
__global__ void imageSampler(
    const float *const imgs, const float* const params, float *output, float *lengths,
    curandState *r_state, int cam_num, int width, int height, int offset, float focal, float near_t = 0.0, float resolution = 0.0
);

__host__ void imageSampler(
    at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int sample_ray_num, int sample_bin_num, float focal, float near_t = 0.0, float far_t = 0.0
);