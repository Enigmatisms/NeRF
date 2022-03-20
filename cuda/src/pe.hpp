#pragma once
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>
#include <torch/script.h>

/// Shape of input is (pnum, 3)
template <bool USE_GLOBAL>
__global__ void peKernel(
    const float* const input, float* output, int pnum, int offset, bool normalize
);

/// Main kernel function for positional encoding
__host__ void positionalEncode(
    at::Tensor input, at::Tensor output, int flevel_num, bool normalize = true
);
