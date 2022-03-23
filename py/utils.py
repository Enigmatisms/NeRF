#-*-coding:utf-8-*-

"""
    Utility function for NeRF
    @author Enigmatisms @date 2022.3.22
"""
import torch
import numpy as np
from nerf_helper import invTransformSample

# input weights shape: (ray_num, point_num), prefix sum should be performed in dim1
def weightPrefixSum(weights:torch.Tensor) -> torch.Tensor:
    result = weights.clone()
    print(result.shape, result[:, 0].shape)
    for i in range(1, weights.shape[1]):
        result[:, i] += result[:, i-1]
    return result

def generateTestSamples(ray_num:int, coarse_pnum:int, sigma_factor:float = 0.01):
    def gaussian(x:float, mean:float, std:float):
        return 1./(np.sqrt(2 * np.pi) * std) * np.exp(-((x - mean)**2) / (2 * std ** 2))
    result = []
    for _ in range(ray_num):
        gauss = gaussian(np.linspace(2, 6, coarse_pnum), 4, 4 * sigma_factor)
        gauss += np.random.uniform(0, np.mean(gauss) * 0.1, size = gauss.shape)
        gauss /= np.sum(gauss)
        result.append(torch.from_numpy(gauss).view(1, -1))
    return torch.cat(result, dim = 0).float().cuda()

# float32. Shape of ray: (ray_num, 6) --> (origin, direction)
def inverseSample(weights:torch.Tensor, rays:torch.Tensor, sample_pnum:int, near:float=2., far:float=2.) -> torch.Tensor:
    cdf = weightPrefixSum(weights)
    sample_depth = torch.zeros(rays.shape[0], sample_pnum).float().cuda()
    # start_time = time()
    invTransformSample(cdf, sample_depth, sample_pnum, near, far)
    # end_time = time()
    # print("Process completed within %.6lf s"%(end_time - start_time))
    sort_depth, _ = torch.sort(sample_depth, dim = -1)          # shape (ray_num, sample_pnum)
    # Use sort depth to calculate sampled points
    raw_pts = rays.repeat(repeats = (1, 1, sample_pnum)).view(rays.shape[0], sample_pnum, -1)
    # depth * ray_direction + origin (this should be further tested)
    raw_pts[:, :, :3] += sort_depth[:, :, None] * raw_pts[:, :, 3:]
    return raw_pts, sort_depth          # depth is used for rendering

def fov2Focal(fov:float, img_width:float) -> float:
    return (img_width / 2) / np.tan(fov * 0.5)
