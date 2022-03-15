

#include "sampling.h"

/// grid中有 sample number 个 block(M)，每个block有 number of samples per ray(N)个线程
/// output已经初始化为(M, N+1, 3), lengths为(M, N)
__global__ void getSampledPoints(
    const float* const imgs, const Eigen::Matrix3f* const params, float* output, 
    float* lengths, curandState* r_state, int width, int height, float near_t, float far_t
) {
    const int ray_id = blockIdx.x, bin_id = threadIdx.x, bin_num = blockDim.x;
    curand_init(ray_id, ray_id, 0, &r_state[ray_id]);
    const uint32_t sample_id = curand(&r_state[ray_id]);
    const int image_size = width * height, id_in_img = (sample_id % image_size);
    const int cam_id = sample_id / (image_size), row_id = id_in_img / width, col_id = id_in_img % width;
    const Eigen::Matrix3f A = params[cam_id];
    Eigen::Vector3f raw_dir = A * Eigen::Vector3f(col_id, row_id, 1.0);
    raw_dir = (raw_dir / raw_dir.norm()).eval();            // normalized direction in world frame
    const float bin_resolution = float(far_t - near_t) / float(bin_num);
    float sample_depth = near_t + bin_resolution * bin_id + curand_uniform(&r_state[ray_id]) * bin_resolution;
    const int ray_base = ray_id * bin_num, total_base = (ray_base + ray_id + bin_id) * 3, rgb_base = (ray_base + ray_id + bin_num - 1) * 3;
    lengths[ray_base + bin_id] = sample_depth;
    Eigen::Vector3f p = raw_dir * sample_depth;
    output[total_base] = p.x();
    output[total_base + 1] = p.y();
    output[total_base + 2] = p.z();
    if (bin_id == 0) {
        const int image_offset = row_id * width + col_id, batch_base = 3 * image_size * cam_id;
        output[rgb_base] = imgs[batch_base + image_offset];
        output[rgb_base + 1] = imgs[batch_base + image_size + image_offset];
        output[rgb_base + 2] = imgs[batch_base + image_size << 1 + image_offset];
    }
}
