#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from torch.nn import functional as F
from py.nerf_helper import makeMLP, positional_encoding
# import tinycudann as tcnn

# This module is shared by coarse and fine network, with no need to modify
class NeRF(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(self, position_flevel, direction_flevel, hidden_unit = 256, cat_origin = True) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.direction_flevel = direction_flevel
        extra_width = 3 if cat_origin else 0
        module_list = makeMLP(60 + extra_width, hidden_unit)
        for _ in range(3):
            module_list.extend(makeMLP(hidden_unit, hidden_unit))

        self.lin_block1 = nn.Sequential(*module_list)       # MLP before skip connection
        self.lin_block2 = nn.Sequential(
            *makeMLP(hidden_unit + 60 + extra_width, hidden_unit),
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
        self.cat_origin = cat_origin
        self.apply(self.init_weight)

    def loadFromFile(self, load_path:str, use_amp = False, opt = None, other_stuff = None):
        save = torch.load(load_path)   
        save_model = save['model']                  
        state_dict = {k:save_model[k] for k in self.state_dict().keys()}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        if not opt is None:
            opt.load_state_dict(save['optimizer'])
        if use_amp:
            from apex import amp
            amp.load_state_dict(save['amp'])
        print("NeRF Model loaded from '%s'"%(load_path))
        if not other_stuff is None:
            return [save[k] for k in other_stuff]


    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    def forward(self, pts:torch.Tensor, encoded_pt:torch.Tensor = None) -> torch.Tensor:
        position_dim, direction_dim = 6 * self.position_flevel, 6 * self.direction_flevel
        if not encoded_pt is None:
            encoded_x = encoded_pt
        else:
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

    @staticmethod
    def length2pts(rays:torch.Tensor, f_zvals:torch.Tensor) -> torch.Tensor:
        sample_pnum = f_zvals.shape[1]
        pts = rays[...,None,:3] + rays[...,None,3:] * f_zvals[...,:,None] 
        return torch.cat((pts, rays[:, 3:].unsqueeze(-2).repeat(1, sample_pnum, 1)), dim = -1)                 # output is (ray_num, coarse_pts num + fine pts num, 6)

    # rays is of shape (ray_num, 6)
    @staticmethod
    def coarseFineMerge(rays:torch.Tensor, c_zvals:torch.Tensor, f_zvals:torch.Tensor) -> torch.Tensor:
        zvals = torch.cat((f_zvals, c_zvals), dim = -1)
        zvals, _ = torch.sort(zvals, dim = -1)
        sample_pnum = f_zvals.shape[1] + c_zvals.shape[1]
        # Use sort depth to calculate sampled points
        pts = rays[...,None,:3] + rays[...,None,3:] * zvals[...,:,None] 
        # depth * ray_direction + origin (this should be further tested)
        return torch.cat((pts, rays[:, 3:].unsqueeze(-2).repeat(1, sample_pnum, 1)), dim = -1), zvals          # output is (ray_num, coarse_pts num + fine pts num, 6)

    """
        This function is important for inverse transform sampling, since for every ray
        we will have 64 normalized weights (summing to 1.) for inverse sampling
    """
    @staticmethod
    def getNormedWeight(opacity:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        delta:torch.Tensor = torch.cat((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e10]).repeat((depth.shape[0], 1)).cuda()), dim = -1)
        # print(opacity.shape, depth[:, 1:].shape, raw_delta.shape, delta.shape)
        mult:torch.Tensor = torch.exp(-F.relu(opacity) * delta)
        alpha:torch.Tensor = 1. - mult
        # fusion requires normalization, rgb output should be passed through sigmoid
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), mult + 1e-10], -1), -1)[:, :-1]
        return weights

    # depth shape: (ray_num, point_num)
    # need the norm of rays, shape: (ray_num, point_num)
    @staticmethod
    def render(rgbo:torch.Tensor, depth:torch.Tensor, ray_dirs:torch.Tensor, mul_norm:bool = True, white_bkg:bool = False) -> torch.Tensor:
        if mul_norm == True:
            depth = depth * (ray_dirs.norm(dim = -1, keepdim = True))
        rgb:torch.Tensor = rgbo[..., :3] # shape (ray_num, pnum, 3)
        opacity:torch.Tensor = rgbo[..., -1]             # 1e-5 is used for eliminating numerical instability
        weights = NeRF.getNormedWeight(opacity, depth)
        rgb:torch.Tensor = torch.sum(weights[:, :, None] * rgb, dim = -2)
        if white_bkg:
            acc_map = torch.sum(weights, -1)
            rgb = rgb + (1.-acc_map[...,None])
        return rgb, weights     # output (ray_num, 3) and (ray_num, point_num)

class DecayLrScheduler:
    def __init__(self, min_r, decay_r, step, lr):
        self.min_ratio = min_r
        self.decay_rate = decay_r
        self.decay_step = step
        self.lr = lr

    def update_opt_lr(self, train_cnt, opt: torch.optim.Optimizer = None):
        new_lrate = self.lr * max((self.decay_rate ** (train_cnt / self.decay_step)), self.min_ratio)
        if opt is not None:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lrate
        return opt, new_lrate

if __name__ == "__main__":
    print("Hello NeRF world!")
    