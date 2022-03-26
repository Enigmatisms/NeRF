#include <Eigen/Core>
#include "error_check.cuh"
#include "inv_tf_sampler.cuh"
/// ============================== @deprecated module ==============================
/// shape of weights: (ray_num, point_num(64))
/// shape of pts (ray_num, 6) 6---> (相机t（相机光心原点），光线方向)
/// shape of output (ray_num, point_num(128), 6) 6---> (点位置，光线方向)
/// @deprecated for test only
__global__ void inverseTransformSamplePtKernel(
    const float* const weights, const float* const rays, float* output,
    curandState* r_states, int coarse_bins, int offset, float near, float resolution
) {
    extern __shared__ float data[];         /// length: output_pnum + 6 
    const int ray_id = offset + blockIdx.x, pt_id = threadIdx.x, output_pnum = blockDim.x;
    const int state_id = ray_id * output_pnum + pt_id;
    curand_init(state_id, 0, 0, &r_states[state_id]);
    const float weight = curand_uniform(&r_states[state_id]);
    /// TODO: why I think pts is not easy to obtain?
    const int weight_base = ray_id * coarse_bins, ray_info_base = ray_id * 6;
    const float* const ray_info_global = &rays[ray_info_base];
    float* ray_info_shared = &data[output_pnum];
    /// e.g: 128 pts for fine network and 64 pts for the coarse, data is the same as concatenating two identical copy of weight
    data[pt_id] = weights[ray_id * coarse_bins + (pt_id % coarse_bins)];
    /// TODO I can't find a way to get around warp divergence for loading data from global memory to shared memory
    if (pt_id == 0) {
        for (int i = 0; i < 6; i++)
            ray_info_shared[i] = ray_info_global[i];
    }
    __syncthreads();
    // the last value of CDF is 1.0, since weight < 1.0, there's no need to compare with the last value
    /// TODO: check if the statement above holds
    int bin_id = 0, max_comp = coarse_bins - 1;         
    for (int i = 0; i < max_comp; i++) {
        bin_id += int(weight > data[i]);
    }
    Eigen::Vector3f origin, direct;
    origin << ray_info_shared[0], ray_info_shared[1], ray_info_shared[2];
    direct << ray_info_shared[3], ray_info_shared[4], ray_info_shared[5];
    // since weight itself is a random float, we resue weight as ramdomized sampling perturbation
    const float sampled_depth = near + float(bin_id) * resolution + weight * resolution;
    Eigen::Vector3f sampled = origin + sampled_depth * direct;
    const int output_base = ray_id * output_pnum * 6 + pt_id * 6;
    for (int i = 0; i < 3; i++) {
        output[output_base + i] = sampled(i);
        output[output_base + i + 3] = direct(i);
    }
    __syncthreads();
}

/// @deprecated
__host__ void inverseTransformSamplePt(
    at::Tensor weights, at::Tensor rays, at::Tensor output, int sampled_pnum, float near, float far
) {
    curandState *rand_states;
    const int ray_num = weights.size(0), coarse_pnum = weights.size(1);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&rand_states, ray_num * sampled_pnum * sizeof(curandState)));
    const float* const weight_data = weights.data_ptr<float>();
    const float* const ray_data = rays.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    const float resolution = (far - near) / float(sampled_pnum);
    cudaStream_t streams[8];
    for (int i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    /// make sure that number of rays to sample is the multiple of 16
    int cascade_num = ray_num >> 4;      // sample_ray_num / 16
    for (int i = 0; i < cascade_num; i++) {
        inverseTransformSamplePtKernel <<< 16, sampled_pnum, (sampled_pnum + 6) * sizeof(float), streams[i % 8]>>> (
            weight_data, ray_data, output_data, rand_states, coarse_pnum, i << 4, near, resolution
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaFree(rand_states));
}
