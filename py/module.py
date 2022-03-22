#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from torch.nn import functional as F
from nerf_helper import invTransformSample
from utils import *

class NeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # Positional Encoding is implemented through CUDA

    def render(rgbo:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        rgb:torch.Tensor = rgbo[..., :3] # shape (ray_num, pnum, 3)
        # RGB passed through sigmoid
        rgb_normed:torch.Tensor = F.sigmoid(rgb)

        opacity:torch.Tensor = rgbo[..., -1]
        
        delta:torch.Tensor = torch.hstack((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e9]).repeat((depth.shape[0], 1)).cuda()))
        mult:torch.Tensor = -opacity * delta

        ts:torch.Tensor = torch.exp(torch.hstack((torch.zeros(mult.shape[0], 1, dtype = torch.float32).cuda(), torch.cumprod(mult)[:, :-1])))
        alpha:torch.Tensor = 1. - torch.exp(mult)       # shape (ray_num, point_num)
        # fusion requires normalization, rgb output should be passed through sigmoid
        weights:torch.Tensor = ts * alpha               # shape (ray_num, point_num)
        weights_normed:torch.Tensor = torch.sum(weights, dim = -1, keepdim = True)

        weighted_rgb:torch.Tensor = weights_normed[:, :, None] * rgb_normed
        return torch.sum(weighted_rgb, dim = 1)

"""
TODO:
- 数据集已经可以加载了，也即coarse网络是完全可以跑起来的，现在还有一个主要问题就是
    - 只差最后一部分：光线render以及全图render
"""