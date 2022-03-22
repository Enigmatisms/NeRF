#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

from re import M
import torch
from torch import nn
from torch.nn import functional as F
from nerf_helper import positionalEncode
from utils import inverseSample

def makeMLP(in_chan, out_chan, act = nn.ReLU, batch_norm = False):
    modules = [nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(nn.BatchNorm1d(out_chan))
    if not act is None:
        modules.append(act)
    return modules

# This module is shared by coarse and fine network, with no need to modify
class NeRF(nn.Module):
    def __init__(self, position_flevel, direction_flevel) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.direction_flevel = direction_flevel

        module_list = makeMLP(60, 256)
        for _ in range(4):
            module_list.extend(makeMLP(256, 256))

        self.lin_block1 = nn.Sequential(*module_list)       # MLP before skip connection
        self.lin_block2 = nn.Sequential(
            *makeMLP(316, 256),
            *makeMLP(256, 256), *makeMLP(256, 256),
            *makeMLP(256, 256, None)
        )

        self.opacity_head = nn.Sequential(                  # authors said that ReLU is used here
            *makeMLP(256, 1)
        )
        self.rgb_layer = nn.Sequential(
            *makeMLP(280, 128),
            *makeMLP(128, 3, nn.Sigmoid)
        )

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    # TODO: fine-network输入的point_num是192，会产生影响吗？
    def forward(self, pts:torch.Tensor) -> torch.Tensor:
        flat_batch = pts.shape[0] * pts.shape[1]
        position_dim, direction_dim = 6 * self.position_flevel, 6 * self.direction_flevel
        encoded_x:torch.Tensor = torch.zeros(flat_batch, position_dim)
        encoded_r:torch.Tensor = torch.zeros(flat_batch, direction_dim)
        positionalEncode(pts[:, :, :3].view(-1, 3), encoded_x, self.position_flevel, False)
        positionalEncode(pts[:, :, 3:6].view(-1, 3), encoded_r, self.direction_flevel, False)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], direction_dim)

        tmp = self.lin_block1(encoded_x)
        encoded_x = torch.cat((tmp, encoded_x), dim = -1)
        encoded_x = self.lin_block2(encoded_x)
        opacity = self.opacity_head(encoded_x)
        rgb = self.rgb_layer(torch.cat((encoded_x, encoded_r), dim = -1))
        return torch.cat((rgb, opacity), dim = -1)

    """
        This function is important for inverse transform sampling, since for every ray
        we will have 64 normalized weights (summing to 1.) for inverse sampling
    """
    @staticmethod
    def getNormedWeight(opacity:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        delta:torch.Tensor = torch.hstack((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e9]).repeat((depth.shape[0], 1)).cuda()))
        mult:torch.Tensor = -opacity * delta

        ts:torch.Tensor = torch.exp(torch.hstack((torch.zeros(mult.shape[0], 1, dtype = torch.float32).cuda(), torch.cumprod(mult)[:, :-1])))
        alpha:torch.Tensor = 1. - torch.exp(mult)       # shape (ray_num, point_num)
        # fusion requires normalization, rgb output should be passed through sigmoid
        weights:torch.Tensor = ts * alpha               # shape (ray_num, point_num)
        return torch.sum(weights, dim = -1, keepdim = True)

    @staticmethod
    def render(rgbo:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        rgb:torch.Tensor = rgbo[..., :3] # shape (ray_num, pnum, 3)
        # RGB passed through sigmoid
        rgb_normed:torch.Tensor = F.sigmoid(rgb)

        opacity:torch.Tensor = rgbo[..., -1]
        weights_normed:torch.Tensor = NeRF.getNormedWeight(opacity, depth)

        weighted_rgb:torch.Tensor = weights_normed[:, :, None] * rgb_normed
        return torch.sum(weighted_rgb, dim = 1)

"""
Latest TODO:
- 数据集加载与网络runablity测试
- test模块（全图sample）
- 训练可执行文件
"""