#include "sampling.h"
#include "pe.hpp"
#include "inv_tf_sampler.h"
#include "deterministic_sampler.h"
#include <torch/extension.h>

const std::string conservative_sampler_docs = 
	"NeRF sampling function for coarse network implemented in CUDA\n\n"
	"This function is abandoned, since it doesn't output direction information\n\n"
	"imgs:torch.Tensor 		input multi-view images, shape (N, C, H, W)\n"
	"tfs:torch.Tensor  		transformation. Extrinsics (P) multiples inverse of intrinsic (K^-1) and translation -> (T, t), shape (cam_num, 3, 4)\n"
	"output:torch.Tensor  	sampled points for each ray, with ground truth color, shape (ray_num, sample per ray + 1, 3), the last 3 elems are RGB vals\n"
	"lengths:torch.Tensor  	the length from camera origin to each sampled points, shape (ray_num, sample per ray)\n"
	"sample_ray_num:int  	number of rays to sample\n"
	"sample_bin_num:int  	number of samples per ray\n"
	"near_t:float  			minimum distance from camera origin\n"
	"resolution:float  		(max_distance - min_distance) / sample_bin_num"
;

const std::string sampler_docs = 
	"NeRF sampling function for coarse network implemented in CUDA\n\n"
	"imgs:torch.Tensor 		input multi-view images, shape (N, C, H, W)\n"
	"tfs:torch.Tensor  		transformation. Extrinsics (P) multiples inverse of intrinsic (K^-1) and translation -> (T, t), shape (cam_num, 3, 4)\n"
	"output:torch.Tensor  	sampled points for each ray, with ground truth color, shape (ray_num, sample per ray + 1, 3), the last 3 elems are RGB vals\n"
	"lengths:torch.Tensor  	the length from camera origin to each sampled points, shape (ray_num, sample per ray)\n"
	"sample_ray_num:int  	number of rays to sample\n"
	"sample_bin_num:int  	number of samples per ray\n"
	"focal:float			Focal length camera intrinsic\n"
	"near_t:float  			minimum distance from camera origin (meter)\n"
	"far_t:float  			maximum distance from camera origin (meter)\n"
;

const std::string pe_docs = 
	"Parallel sinusoidal positional encoding\n\n"
	"input:torch.Tensor		values that needs to be encoded, the shape is normally (batch_num, 3), output (batch_num, 2 * 3 * flevel_num)\n"
	"output:torch.Tensor	Sinusoidal encoded input (batch_num, 2 * 3 * flevel_num)\n"
	"flevel_num:int			Number of different frequency\n"
	"normalize:bool			Whether using L2 norm to normalize input\n"
;

const std::string inv_tf_docs = 
	"Inverse transform sampler implemented in CUDA\n\n"
	"weights:torch.Tensor	weights output by coarse network (sums to 1, per ray), shape (ray_num, sample point num per ray in coarse network)\n"
	"output:torch.Tensor	inverse transform sampled points depth (from camera origin to point), shape (ray_num, fine points per ray)\n"
	"sampled_pnum:int		fine points per ray to sample\n"
	"near:float				minimum distance from camera origin (meter)\n"
	"far:float				maximum distance from camera origin (meter)\n"
;

const std::string inv_tf_pt_docs = 
	"(Deprecated, only for testing) Inverse transform sampler implemented in CUDA\n\n"
	"weights:torch.Tensor	weights output by coarse network (sums to 1, per ray), shape (ray_num, sample point num per ray in coarse network)\n"
	"rays:torch.Tensor		Information about rays (camera translation and ray orientation), shape (ray_num, 6)\n"
	"output:torch.Tensor	inverse transform sampled points, shape (ray_num, fine points per ray, 6)\n"
	"sampled_pnum:int		fine points per ray to sample\n"
	"near:float				minimum distance from camera origin (meter)\n"
	"far:float				maximum distance from camera origin (meter)\n"
;

const std::string image_sampler_doc = 
	"Deterministic sampler for the whole image, usually used in rendering and evaluation\n\n"
	"tf:torch.Tensor		(3, 4) float Tensor, R(3 * 3) and t(3 * 1), the pose of the camera\n"
	"output:torch.Tensor	Sampled points in shape (H, W, pnum, 6) \n"
	"lengths:torch.Tensor	Sampled z values of each point in shape (H, W, pnum)\n"
	"img_w:int				image width in pixel\n"
	"img_h:int				image height in pixel\n"
	"pnum:int				points per pixel to sample\n"
	"focal:float			Focal length camera intrinsic\n"
	"near:float				minimum distance from camera origin (meter)\n"
	"far:float				maximum distance from camera origin (meter)\n"
;

PYBIND11_MODULE (TORCH_EXTENSION_NAME, nerf_helper)
{
  	nerf_helper.def ("comservativeSampling", &cudaSampler, conservative_sampler_docs.c_str());
  	nerf_helper.def ("sampling", &easySampler, sampler_docs.c_str());
	nerf_helper.def ("encoding", &positionalEncode, pe_docs.c_str());
	nerf_helper.def ("invTransformSample", &inverseTransformSample, inv_tf_docs.c_str());
	nerf_helper.def ("invTransformSamplePt", &inverseTransformSamplePt, inv_tf_pt_docs.c_str());
	nerf_helper.def ("imageSampling", &imageSampler, image_sampler_doc.c_str());
}
