#include <Eigen/Core>
#include "error_check.hpp"
#include "sampling.h"

/**
 * @todo
 * 1. output不够完整，应该输出9维向量，（位置，方向，RGB），RGB直接在此处输出是为了减少外部变形concatenate的时间
 * 2. 写两个接口，一个是RGB放在最后（6维，最后一个向量前三维度是RGB，后三为0），另一个是RGB在每一个向量中
 * 3. 后一个接口的RGB也需要放在shared memory中
 * 4. [... python是什么意思？] （run_nerf.py 202）
 * 5. 个人感觉，只有方向是需要normalized的，一时因为origin normalize之后就没有意义了，二是官方实现中除了NDC之外貌似也没有对origin进行显式normalize
 * 6. 方向的normalize是l2 norm
 */

/// offset is used in multi-stream concurrency
/// params is (ray_num, 3, 4)
__global__ void getSampledPoints(
    const float *const imgs, const float* const params, float *output, float *lengths,
    curandState *r_state, int cam_num, int width, int height, int offset, float near_t, float resolution)
{
    extern __shared__ float transforms[];           /// 9 floats for RK^{-1}, 3 floats for t, 1 int for sampled_id 
    int* sample_id = (int*)(transforms + 12);
    const int ray_id = blockIdx.x + offset, bin_id = threadIdx.x, bin_num = blockDim.x, image_size = width * height;
    const int state_id = ray_id * bin_num + bin_id;
    int cam_id = 0, row_id = 0, col_id = 0, id_in_img = 0;
    /// copy PK^-1 from global memory to shared local memory, enabling faster accessing
    curand_init(state_id, 0, 0, &r_state[state_id]);
    if (bin_id == 0) {
        *sample_id = curand(&r_state[state_id]) % (cam_num * image_size);
        cam_id = *sample_id / (image_size);
        const float* const ptr = params + 12 * cam_id;
        for (int i = 0; i < 12; i++)
            transforms[i] = ptr[i];
        // printf("%d, %d, [%d, %d]\n", id_in_img / width, id_in_img % width, *sample_id, ray_id);
    }
    __syncthreads();
    id_in_img = (*sample_id % image_size);
    cam_id = *sample_id / (image_size), row_id = id_in_img / width, col_id = id_in_img % width;
    Eigen::Matrix3f T;       // A is equal to PK^-1, these are ex(in)trinsics respectively
    Eigen::Vector3f t;
    T << transforms[0], transforms[1], transforms[2], transforms[4], transforms[5], transforms[6], transforms[8], transforms[9], transforms[10];
    t << transforms[3], transforms[7], transforms[11];
    Eigen::Vector3f raw_dir = T * Eigen::Vector3f(col_id, row_id, 1.0);
    raw_dir = (raw_dir / raw_dir.norm()).eval();            // normalized direction in world frame
    float sample_depth = near_t + resolution * bin_id + curand_uniform(&r_state[state_id]) * resolution;
    const int ray_base = ray_id * bin_num, total_base = (ray_base + ray_id + bin_id) * 3;
    lengths[ray_base + bin_id] = sample_depth;
    Eigen::Vector3f p = raw_dir * sample_depth + t;
    output[total_base] = p.x();
    output[total_base + 1] = p.y();
    output[total_base + 2] = p.z();
    if (bin_id == 0) {
        const int image_offset = row_id * width + col_id, batch_base = 3 * image_size * cam_id, rgb_base = (ray_base + ray_id + bin_num) * 3;
        output[rgb_base] = imgs[batch_base + image_offset];
        output[rgb_base + 1] = imgs[batch_base + image_size + image_offset];
        output[rgb_base + 2] = imgs[batch_base + image_size << 1 + image_offset];
    }
    __syncthreads();
}

/// input tensor imgs (N, 3, H, W),
/// camera poses, which should be convert to Eigen, the shape is (batch (number of cams), 3, 4)
/// output: 1. (sample_ray_num, sample_bin_num + 1, 3), points sampled and the gt color 2. length (sample_ray_num, sample_bin_num)
/// both of the output is a tensor
__host__ void cudaSamplerKernel(
    at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths,
    int sample_ray_num, int sample_bin_num, float near_t, float resolution
) {
    curandState *rand_states;
    const int batch_size = imgs.size(0), width = imgs.size(3), height = imgs.size(2);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&rand_states, sample_ray_num * sample_bin_num * sizeof(curandState)));
    const float* const img_data = imgs.data_ptr<float>();
    const float* const param_data = tfs.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    float* length_data = lengths.data_ptr<float>();

    /// GPU stream concurrency
    cudaStream_t streams[8];
    for (int i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    /// make sure that number of rays to sample is the multiple of 16
    int cascade_num = sample_ray_num >> 4;      // sample_ray_num / 16
    for (int i = 0; i < cascade_num; i++) {
        getSampledPoints <<< 16, sample_bin_num, 13, streams[i % 8]>>> (
            img_data, param_data, output_data, length_data, rand_states, batch_size, width, height, i << 4, near_t, resolution
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaFree(rand_states));
}
