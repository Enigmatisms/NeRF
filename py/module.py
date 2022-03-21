#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from nerf_helper import invTransformSample
from utils import *


# float32. Shape of ray: (ray_num, 6) --> (origin, direction)
def inverseSample(weights:torch.Tensor, rays:torch.Tensor, sample_pnum:int) -> torch.Tensor:
    cdf = weightPrefixSum(weights)
    sample_depth = torch.zeros(rays.shape[0], sample_pnum).cuda()
    invTransformSample(cdf, sample_depth, sample_pnum, 2., 6.)
    sort_depth, _ = torch.sort(sample_depth, dim = -1)          # shape (ray_num, sample_pnum)
    # Use sort depth to calculate sampled points
    raw_pts = rays.repeat(repeats = (1, 1, sample_pnum)).view(rays.shape[0], sample_pnum, -1)
    # depth * ray_direction + origin (this should be further tested)
    raw_pts[:, :, :3] += sort_depth[:, :, None] * raw_pts[:, :, 3:]
    return raw_pts

class NeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # Positional Encoding is implemented through CUDA

"""
TODO:
- 数据集已经可以加载了，也即coarse网络是完全可以跑起来的，现在还有一个主要问题就是
    - 只差最后一部分：光线render以及全图render
"""