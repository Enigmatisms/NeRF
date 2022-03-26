#include <Eigen/Core>
#include "valid_sampler.cuh"
#include "error_check.cuh"

__global__ void validSamplerKernel(
    const float *const valid_rgb, const int* const coords, const float* const params, float *output, float *lengths,
    curandState *r_state, int valid_num, int width, int height, int offset, int seed_offset, float focal, float near_t, float resolution
) {
    extern __shared__ float transforms[];           /// 9 floats for RK^{-1}, 3 floats for t, 3 floats for RGB value, 2 int for image coords, 1 int for camera index (18) 
    int* coord_ptr = (int*)(transforms + 15);       // tf[15] --> x, tf[16] --> y, tf[17] --> index
    const int ray_id = blockIdx.x + offset, bin_id = threadIdx.x, bin_num = blockDim.x;
    const int state_id = ray_id * bin_num + bin_id;
    int cam_id = 0, row_id = 0, col_id = 0;
    curand_init(state_id, 0, seed_offset, &r_state[state_id]);
    if (bin_id == 0) {
        int pixel_id = (curand(&r_state[state_id]) % (uint32_t)valid_num), pixel_base = pixel_id * 3;
        for (int i = 0; i < 3; i++)
            coord_ptr[i] = coords[pixel_base + i];
        cam_id = coord_ptr[2];
        const float* const ptr = params + 12 * cam_id;
        for (int i = 0; i < 3; i++)
            transforms[12 + i] = valid_rgb[pixel_base + i];
        for (int i = 0; i < 12; i++)
            transforms[i] = ptr[i];
    }
    __syncthreads();
    row_id = coord_ptr[0];
    col_id = coord_ptr[1];
    cam_id = coord_ptr[2];
    Eigen::Matrix3f T;       // A is equal to PK^-1, these are ex(in)trinsics respectively
    Eigen::Vector3f t;
    T << transforms[0], transforms[1], transforms[2], transforms[4], transforms[5], transforms[6], transforms[8], transforms[9], transforms[10];
    t << transforms[3], transforms[7], transforms[11];
    Eigen::Vector3f raw_dir = T * Eigen::Vector3f(float(col_id - width >> 1) / focal, float((height >> 1) - row_id) / focal, -1.0);
    raw_dir = (raw_dir / raw_dir.norm()).eval();            // normalized direction in world frame
    float sample_depth = near_t + resolution * bin_id + curand_uniform(&r_state[state_id]) * resolution;
    // output shape (ray_num, point num + 1, 9) (9 dims per point), length is of shape (ray_num, point_num) (no plus 1)
    const int ray_base = ray_id * bin_num, total_base = (ray_base + ray_id + bin_id) * 9;
    lengths[ray_base + bin_id] = sample_depth;
    // sampled point origin
    Eigen::Vector3f p = raw_dir * sample_depth + t;
    // 此处需要额外输出t以及
    for (int i = 0; i < 3; i++) {
        output[total_base + i] = p(i);                      // point origin
        output[total_base + i + 3] = raw_dir(i);            // normalized direction
        output[total_base + i + 6] = transforms[12 + i];    // rgb value
    }
    if (bin_id == 0) {
        const int end_base = (ray_base + ray_id + bin_num) * 9;
         for (int i = 0; i < 3; i++) {
            output[end_base + i] = t(i);                      // point origin
            output[end_base + i + 3] = raw_dir(i);            // normalized direction
            output[end_base + i + 6] = 0;                     // rgb value
        }
    }
    __syncthreads();
}


/// rgb shape (valid num, 3), coords shape (valid num, 3) (3 --> x, y, index)
__host__ void validSampler(
    at::Tensor rgb, at::Tensor coords, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int img_w, int img_h, int sample_ray_num, int sample_bin_num, float focal, float near_t, float far_t
) {
    static int r_state_offset = 0;
    curandState *rand_states;
    const int valid_num = rgb.size(0);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&rand_states, sample_ray_num * sample_bin_num * sizeof(curandState)));
    const float* const rgb_data = rgb.data_ptr<float>();
    const int* const coords_data = coords.data_ptr<int>();
    const float* const param_data = tfs.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    float* length_data = lengths.data_ptr<float>();
    const float resolution = (far_t - near_t) / float(sample_bin_num);
    cudaStream_t streams[16];
    for (int i = 0; i < 16; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    int cascade_num = sample_ray_num >> 6;
    for (int i = 0; i < cascade_num; i++) {
        validSamplerKernel <<< 64, sample_bin_num, 18 * sizeof(float), streams[i % 16]>>> (
            rgb_data, coords_data, param_data, output_data, length_data, rand_states, valid_num, img_w, img_h, i << 6, r_state_offset, focal, near_t, resolution
        );
    }
    r_state_offset++;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 16; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaFree(rand_states));
}
