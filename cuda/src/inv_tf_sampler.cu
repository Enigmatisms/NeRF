#include <Eigen/Core>
#include "error_check.hpp"
#include "inv_tf_sampler.h"

/// shape of weights: (ray_num, point_num(64)), information about rays is not needed
/// shape of output (ray_num, point_num(128)) length from point to camera origin
__global__ void inverseTransformSampleKernel(
    const float* const weights, float* output, curandState* r_states,
    int coarse_bins, int offset, float near, float resolution
) {
    extern __shared__ float data[];         /// length: output_pnum + 6 
    const int ray_id = offset + blockIdx.x, pt_id = threadIdx.x, output_pnum = blockDim.x;
    const int state_id = ray_id * output_pnum + pt_id;
    curand_init(state_id, 0, 0, &r_states[state_id]);
    const float weight = curand_uniform(&r_states[state_id]);
    /// e.g: 128 pts for fine network and 64 pts for the coarse, data is the same as concatenating two identical copy of weight
    data[pt_id] = weights[ray_id * coarse_bins + (pt_id % coarse_bins)];
    __syncthreads();
    // the last value of CDF is 1.0, since weight < 1.0, there's no need to compare with the last value
    int bin_id = 0, max_comp = coarse_bins - 1;         
    for (int i = 0; i < max_comp; i++) {
        bin_id += int(weight > data[i]);
    }
    // since weight itself is a random float, we resue weight as ramdomized sampling perturbation
    const float sampled_depth = near + float(bin_id) * resolution + weight * resolution;
    output[ray_id * output_pnum + pt_id] = sampled_depth;
    __syncthreads();
}

__host__ void inverseTransformSample(
    at::Tensor weights, at::Tensor output, int sampled_pnum, float near, float far
) {
    curandState *rand_states;
    const int ray_num = weights.size(0), coarse_pnum = weights.size(1);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&rand_states, ray_num * sampled_pnum * sizeof(curandState)));
    const float* const weight_data = weights.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    const float resolution = (far - near) / float(sampled_pnum);
    cudaStream_t streams[8];
    if (ray_num > 16) {
        for (int i = 0; i < 8; i++)
            cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
        /// make sure that number of rays to sample is the multiple of 16
        int cascade_num = ray_num >> 4;      // sample_ray_num / 16
        for (int i = 0; i < cascade_num; i++) {
            inverseTransformSampleKernel <<< 16, sampled_pnum, sampled_pnum * sizeof(float), streams[i % 8]>>> (
                weight_data, output_data, rand_states, coarse_pnum, i << 4, near, resolution
            );
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        for (int i = 0; i < 8; i++)
            cudaStreamDestroy(streams[i]);
    } else {
        inverseTransformSampleKernel <<< 16, sampled_pnum, sampled_pnum * sizeof(float)>>> (
            weight_data, output_data, rand_states, coarse_pnum, 0, near, resolution
        );
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    CUDA_CHECK_RETURN(cudaFree(rand_states));
}
