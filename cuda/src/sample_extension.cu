#include "sampling.h"
#include "sample_extension.h"
#include <torch/extension.h>

void cudaSampler(at::Tensor imgs, at::Tensor tfs, at::Tensor output, at::Tensor lengths, int sample_ray_num, int sample_bin_num, float near_t, float resolution) {
	cudaSamplerKernel(imgs, tfs, output, lengths, sample_ray_num, sample_bin_num, near_t, resolution);
}

const std::string sampler_docs = 
	"NeRF sampling function for coarse network implemented in CUDA\n\n"
	"imgs:torch.Tensor 		input multi-view images, shape (N, C, H, W)\n"
	"tfs:torch.Tensor  		transformation. Extrinsics (P) multiples inverse of intrinsic (K^-1) and translation -> (T, t), shape (cam_num, 3, 4)\n"
	"output:torch.Tensor  	sampled points for each ray, with ground truth color, shape (ray_num, sample per ray + 1, 3), the last 3 elems are RGB vals\n"
	"lengths:torch.Tensor  	the length from camera origin to each sampled points, shape (ray_num, sample per ray)\n"
	"sample_ray_num:int  	number of rays to sample\n"
	"sample_bin_num:int  	number of samples per ray\n"
	"near_t:int  			minimum distance from camera origin\n"
	"resolution:float  		(max_distance - min_distance) / sample_bin_num"
;

PYBIND11_MODULE (TORCH_EXTENSION_NAME, sampler)
{
  	sampler.def ("sampling", &cudaSampler, sampler_docs.c_str());
}
