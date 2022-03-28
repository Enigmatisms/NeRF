#include <Eigen/Core>
#include "error_check.cuh"
#include "deterministic_sampler.cuh"

/// 先不用考虑太复杂的情况，实现上不同的地方在于：
/// 全图采样不需要RGB输入，输出是(H, W, point_num, 6)，个人不准备实现更大的batch操作，一是没必要，GPU本身就没有那么大的并行度，二是更加麻烦
/// R,t的存储与原来一致，开大小为12的共享内存
/// 由于不需要RGB输入，其实就不需要图像输入，也不需要访问图像，图像坐标X，Y仅仅用于计算方向
/// 输入的param是大小为(3, 4)的矩阵
/// output (H, W, P, 3), length (H, W, P)
/// 需要大量allocate random state吗？可以测试一下，由于每次运行kernel，只会运行block大小，那么randomState在开始前创建，执行完销毁
__global__ void imageSamplerKernel(
    const float* const params, float *output, float *lengths, curandState *r_state,
    int width, int height, int offset_x, int offset_y, int r_offset, float focal, float near_t, float resolution
) {
    extern __shared__ float transform[];
    /// TODO: offsets in both dimensions are needed
    const int col_id = blockIdx.x + offset_x, row_id = blockIdx.y + offset_y, point_id = threadIdx.x, pnum = blockDim.x;
    const int state_id = (blockIdx.y * gridDim.x + blockIdx.x) * pnum + point_id;
    curand_init(state_id, 0, r_offset, &r_state[state_id]);
    if (point_id < 12) {
        transform[point_id] = params[point_id];
    }
    __syncthreads();
    Eigen::Matrix3f T;       // A is equal to PK^-1, these are ex(in)trinsics respectively
    Eigen::Vector3f t;
    T << transform[0], transform[1], transform[2], transform[4], transform[5], transform[6], transform[8], transform[9], transform[10];
    t << transform[3], transform[7], transform[11];
    const Eigen::Vector3f raw_dir = T * Eigen::Vector3f(float(col_id - (width >> 1)) / focal, float((height >> 1) - row_id) / focal, -1.0);
    // raw_dir = (raw_dir / raw_dir.norm()).eval();            // normalized direction in world frame
    float sample_depth = near_t + resolution * point_id + curand_uniform(&r_state[state_id]) * resolution;
    // output shape (ray_num, point num, 9) (9 dims per point)
    const int point_base = (row_id * width + col_id) * pnum + point_id;
    lengths[point_base] = sample_depth;
    // sampled point origin
    Eigen::Vector3f p = raw_dir * sample_depth + t;
    const int output_base = point_base * 6;
    for (int i = 0; i < 3; i++) {
        output[output_base + i] = p(i);                      // point origin
        output[output_base + i + 3] = raw_dir(i);            // normalized direction
    }
    __syncthreads();
}

#define BLOCK_SHAPE_X 50                // how many point position in x direction
#define BLOCK_SHAPE_Y 50
__host__ void imageSampler(
    at::Tensor tf, at::Tensor output, at::Tensor lengths, int img_w, int img_h,
    int sample_point_num, float focal, float near_t, float far_t
) {
    static int r_state_offset = 0;
    const float* const tf_data = tf.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    float* length_data = lengths.data_ptr<float>();
    const float resolution = (far_t - near_t) / float(sample_point_num);
    cudaStream_t streams[16];
    for (int i = 0; i < 16; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const int block_num_x = img_w / BLOCK_SHAPE_X, block_num_y = img_h / BLOCK_SHAPE_Y;
    for (int i = 0; i < block_num_y; i++) {
        for (int j = 0; j < block_num_x; j++) {
            curandState *rand_states;
            CUDA_CHECK_RETURN(cudaMalloc((void **)&rand_states, BLOCK_SHAPE_X * BLOCK_SHAPE_Y * sample_point_num * sizeof(curandState)));
            dim3 block_grid(BLOCK_SHAPE_X, BLOCK_SHAPE_Y);
            imageSamplerKernel <<< block_grid, sample_point_num, 12 * sizeof(float), streams[j % 16]>>> (
                tf_data, output_data, length_data, rand_states, img_w, img_h, j * BLOCK_SHAPE_X, i * BLOCK_SHAPE_Y, r_state_offset, focal, near_t, resolution
            );
            CUDA_CHECK_RETURN(cudaFree(rand_states));       /// TODO: 这是一种提高GPU吞吐率的方法吗
        }
        r_state_offset++;
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 16; i++)
        cudaStreamDestroy(streams[i]);
}
