#pragma once
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>
#include <torch/script.h>

/// Shape of input is (pnum, 3)
__global__ void positionalEncode(
    const float* const input, float* output, int pnum, bool normalize
);

/// Main kernel function for positional encoding
__host__ void peKernel(
    at::Tensor input, at::Tensor output, int pnum, bool normalize = true
);
