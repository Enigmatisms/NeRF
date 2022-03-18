#include "pe.h"

constexpr float PI = 3.14159265358979f;

/// input number of points (pnum) mod 16 = 0, since block size is 16, thread size is up to freq_n
/// output: concatenated (pnum, 2 * L * 3)
__global__ void positionalEncode(
    const float* const input, float* output, int pnum, bool normalize
) {
    const int freq_id = threadIdx.x, point_id = blockIdx.x, freq_num = blockDim.x;
    const float freq = powf(2, freq_id) * PI;
    const int input_base = point_id * 3, output_base = (input_base * freq_num) << 1;
    for (int i = 0; i < 3; i++) {
        const float v = input[input_base + i] * freq;
        const int output_addr = output_base + freq_id << 1 + i * 2 * freq_num;
        output[output_addr]     = sinf(v);
        output[output_addr + 1] = cosf(v);
    }
    __syncthreads();
}

__host__ void peKernel(
    at::Tensor input, at::Tensor output, int pnum, bool normalize = true
) {
    ;
}
