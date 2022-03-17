#pragma once
#include <torch/torch.h>

void cudaSampler(at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths, int sample_ray_num, int sample_bin_num, float near_t, float resolution);
