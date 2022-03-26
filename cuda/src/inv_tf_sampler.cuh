/**
 * @file inv_tf_sampler.cuh
 * @author Enigmatisms
 * @brief Inverse transform sampling used before fine network
 * @date 2022-03-20
 * @copyright Copyright (c) 2022
 */
#pragma once
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void inverseTransformSamplePtKernel(
    const float* const weights, const float* const rays, float* output, 
    curandState* r_states, int coarse_bins, int offset, float near, float resolution
);

__host__ void inverseTransformSamplePt(
    at::Tensor weight, at::Tensor rays, at::Tensor output, int sampled_pnum, float near, float far
);

__global__ void inverseTransformSampleKernel(
    const float* const weights, float* output, curandState* r_states,
    int coarse_bins, int offset, int r_offset, float near, float resolution
);

__host__ void inverseTransformSample(
    at::Tensor weight, at::Tensor output, int sampled_pnum, float near, float far
);