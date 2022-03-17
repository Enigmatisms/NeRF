#pragma once
#include "sampling.h"
#include "sample_extension.h"
#include <torch/extension.h>

void cudaSampler(at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths, int sample_ray_num, int sample_bin_num, float near_t, float resolution) {
    cudaSamplerKernel(imgs, tfs, output, lengths, sample_ray_num, sample_bin_num, near_t, resolution);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, sampler)
{
  sampler.def ("sampling", &cudaSampler, "NeRF sampling function");
}
