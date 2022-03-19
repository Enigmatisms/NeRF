#include "pe.hpp"
#include "error_check.hpp"

constexpr float PI = 3.14159265358979f;

/// input number of points (pnum) mod 16 = 0, since block size is 16, thread size is up to freq_n
/// output: concatenated (pnum, 2 * L * 3)
template <bool USE_GLOBAL>
__global__ void positionalEncode(
    const float* const input, float* output, int pnum, int offset, bool normalize
) {
    extern __shared__ float pt_val[];
    const int freq_id = threadIdx.x, point_id = blockIdx.x + offset, freq_num = blockDim.x;
    const int input_base = point_id * 3, output_base = (input_base * freq_num) << 1;
    float normalize_sum = 0.0;
    if (USE_GLOBAL == false) {
        if (freq_id == 0) {
            for (int i = 0; i < 3; i++) {
                pt_val[i] = input[input_base + i];
                if (normalize)
                    normalize_sum += powf(pt_val[i], 2);
            }
        }
        __syncthreads();
    }
    const float freq = powf(2, freq_id);
    if (normalize)
        normalize_sum = 1.0 / sqrtf(normalize_sum);
    const int output_addr = output_base + 6 * freq_id;
    for (int i = 0; i < 3; i++) {
        float v = 0.0;                     // bank conflict, yet I think this is still faster than direct global memory access
        if (USE_GLOBAL)
            v = input[input_base + i] * freq;
        else
            v = pt_val[i] * freq;
        if (normalize)
            v *= normalize_sum;
        output[output_addr + i]     = sinf(v);
        output[output_addr + 3 + i] = cosf(v);
    }
    __syncthreads();
}

void peKernel(
    at::Tensor input, at::Tensor output, int flevel_num, bool normalize
) {
    /// pnum should better be the multiple of 16, since block size is 16
    int pnum = input.size(0);
    const float* const input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    
    /// make sure that number of rays to sample is the multiple of 16
    int cascade_num = (pnum >= 16) ? (pnum >> 4) : 1;      // sample_ray_num / 16
    if (pnum >= 16) {
        /// GPU stream concurrency
        cudaStream_t streams[8];
        for (int i = 0; i < 8; i++)
            cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
        for (int i = 0; i < cascade_num; i++) {
            positionalEncode<false><<< 16, flevel_num, 4 * sizeof(float), streams[i % 8]>>> (
                input_data, output_data, pnum, i << 4, normalize
            );
        }
        for (int i = 0; i < 8; i++)
            cudaStreamDestroy(streams[i]);
    } else {
        positionalEncode<false><<< pnum, flevel_num, 4 * sizeof(float)>>> (
            input_data, output_data, pnum, 0, normalize
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
