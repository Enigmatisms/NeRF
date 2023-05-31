#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from nerf.nerf_base import NeRF
from nerf.nerf_helper import makeMLP, positional_encoding
# import tinycudann as tcnn

# This module is shared by coarse and fine network, with no need to modify
class MipNeRF(NeRF):
    def __init__(self, position_flevel, direction_flevel, hidden_unit = 256, cat_origin = True) -> None:
        super().__init__(position_flevel, cat_origin)
        self.direction_flevel = direction_flevel
        extra_width = 3 if cat_origin else 0
        module_list = makeMLP(6 * position_flevel + extra_width, hidden_unit)
        for _ in range(3):
            module_list.extend(makeMLP(hidden_unit, hidden_unit))

        self.lin_block1 = nn.Sequential(*module_list)       # MLP before skip connection
        self.lin_block2 = nn.Sequential(
            *makeMLP(hidden_unit + 6 * position_flevel + extra_width, hidden_unit),
            *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, 256)
        )

        self.bottle_neck = nn.Sequential(*makeMLP(256, 256, None))

        self.opacity_head = nn.Sequential(                  # authors said that ReLU is used here
            *makeMLP(256, 1, None)
        )
        self.rgb_layer = nn.Sequential(
            *makeMLP(280 + extra_width, 128),
            *makeMLP(128, 3, nn.Sigmoid())
        )
        self.apply(self.init_weight)

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    def forward(self, pts:torch.Tensor) -> torch.Tensor:
        position_dim, direction_dim = 6 * self.position_flevel, 6 * self.direction_flevel
        encoded_x = positional_encoding(pts[:, :, :3], self.position_flevel)
        rotation = pts[:, :, 3:6].reshape(-1, 3)
        rotation = rotation / rotation.norm(dim = -1, keepdim = True)
        encoded_r = positional_encoding(rotation, self.direction_flevel)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)
        encoded_r = encoded_r.view(pts.shape[0], pts.shape[1], direction_dim)

        if self.cat_origin:
            encoded_x = torch.cat((pts[:, :, :3], encoded_x), -1)
            encoded_r = torch.cat((rotation.view(pts.shape[0], pts.shape[1], -1), encoded_r), -1)

        tmp = self.lin_block1(encoded_x)
        encoded_x = torch.cat((encoded_x, tmp), dim = -1)
        encoded_x = self.lin_block2(encoded_x)
        opacity = self.opacity_head(encoded_x)
        encoded_x = self.bottle_neck(encoded_x)
        rgb = self.rgb_layer(torch.cat((encoded_x, encoded_r), dim = -1))
        return torch.cat((rgb, opacity), dim = -1)      # output (ray_num, point_num, 4)
    