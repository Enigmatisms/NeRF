#-*-coding:utf-8-*-

"""
    NeRF network Base Class (to be inherited from)
"""

import torch
from torch import nn
from torch.nn import functional as F

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

    def __init__(self, position_flevel, cat_origin = True, density_act = F.relu) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.cat_origin = cat_origin
        self.density_act = density_act

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

    @staticmethod
    def length2pts(rays:torch.Tensor, f_zvals:torch.Tensor) -> torch.Tensor:
        sample_pnum = f_zvals.shape[1]
        pts = rays[...,None,:3] + rays[...,None,3:] * f_zvals[...,:,None] 
        return torch.cat((pts, rays[:, 3:].unsqueeze(-2).repeat(1, sample_pnum, 1)), dim = -1)                 # output is (ray_num, coarse_pts num + fine pts num, 6)

    """
        This function is important for inverse transform sampling, since for every ray
        we will have 64 normalized weights (summing to 1.) for inverse sampling
    """
    @staticmethod
    def getNormedWeight(opacity:torch.Tensor, depth:torch.Tensor, density_act = F.relu) -> torch.Tensor:
        delta:torch.Tensor = torch.cat((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e10]).repeat((depth.shape[0], 1)).cuda()), dim = -1)
        # print(opacity.shape, depth[:, 1:].shape, raw_delta.shape, delta.shape)
        mult:torch.Tensor = torch.exp(-density_act(opacity) * delta)
        alpha:torch.Tensor = 1. - mult
        # fusion requires normalization, rgb output should be passed through sigmoid
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), mult + 1e-10], -1), -1)[:, :-1]
        return weights

    # depth shape: (ray_num, point_num)
    # need the norm of rays, shape: (ray_num, point_num)
    @staticmethod
    def render(rgbo:torch.Tensor, depth:torch.Tensor, ray_dirs:torch.Tensor, mul_norm:bool = True, white_bkg:bool = False, density_act = F.relu) -> torch.Tensor:
        if mul_norm == True:
            depth = depth * (ray_dirs.norm(dim = -1, keepdim = True))
        rgb:torch.Tensor = rgbo[..., :3] # shape (ray_num, pnum, 3)
        opacity:torch.Tensor = rgbo[..., -1]             # 1e-5 is used for eliminating numerical instability
        weights = NeRF.getNormedWeight(opacity, depth, density_act)
        rgb:torch.Tensor = torch.sum(weights[:, :, None] * rgb, dim = -2)
        if white_bkg:
            acc_map = torch.sum(weights, -1)
            rgb = rgb + (1.-acc_map[...,None])
        return rgb, weights     # output (ray_num, 3) and (ray_num, point_num)

class DecayLrScheduler:
    def __init__(self, min_r, decay_r, step, lr, warmup_step = 0):
        self.min_ratio = min_r
        self.decay_rate = decay_r
        self.decay_step = step
        self.warmup_step = warmup_step
        self.lr = lr
        if warmup_step > 0:
            print("Warming up step: %d"%(warmup_step))

    def update_opt_lr(self, train_cnt, opt: torch.optim.Optimizer = None):
        if train_cnt < self.warmup_step:
            ratio = train_cnt / self.warmup_step
            new_lrate = self.lr * (self.min_ratio * (1. - ratio) + ratio)
        else:
            new_lrate = self.lr * max((self.decay_rate ** ((train_cnt - self.warmup_step) / self.decay_step)), self.min_ratio)
        if opt is not None:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lrate
        return opt, new_lrate

if __name__ == "__main__":
    print("Hello NeRF world!")
    