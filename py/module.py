#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from torch.nn import functional as F
from nerf_helper import positionalEncode
from utils import inverseSample

def makeMLP(in_chan, out_chan, act = nn.ReLU, batch_norm = False):
    modules = [nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(nn.BatchNorm1d(out_chan))
    if act == True:
        modules.append(nn.ReLU(inplace = True))
    return modules

class NeRF(nn.Module):
    def __init__(self, position_flevel, direction_flevel) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.direction_flevel = direction_flevel

        module_list = makeMLP(60, 256)
        # for i in range(7):

        self.lin_body = nn.Sequential(
            
        )

    # Positional Encoding is implemented through CUDA

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    def forward(self, pts:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        flat_batch = pts.shape[0] * pts.shape[1]
        position_dim, direction_dim = 6 * self.position_flevel, 6 * self.direction_flevel
        encoded_x:torch.Tensor = torch.zeros(flat_batch, position_dim)
        encoded_r:torch.Tensor = torch.zeros(flat_batch, direction_dim)
        positionalEncode(pts[:, :, :3].view(-1, 3), encoded_x, self.position_flevel, False)
        positionalEncode(pts[:, :, 3:6].view(-1, 3), encoded_r, self.direction_flevel, False)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], direction_dim)


        pass

    def render(self, rgbo:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
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