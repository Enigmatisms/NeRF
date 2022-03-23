#include "deterministic_sampler.h"

__global__ void imageSampler(
    const float *const imgs, const float* const params, float *output, float *lengths,
    curandState *r_state, int cam_num, int width, int height, int offset, float focal, float near_t = 0.0, float resolution = 0.0
) {

}

__host__ void imageSampler(
    at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int sample_ray_num, int sample_bin_num, float focal, float near_t = 0.0, float far_t = 0.0
) {

}
