#-*-coding:utf-8-*-

"""
    Implementation related with mip NeRF 360
"""

import torch
from torch import nn
from torch.nn import functional as F
from py.nerf_helper import makeMLP, positional_encoding

# according to calculated weights (of proposal net) and indices of inverse sampling, calculate the bounds required for loss computation
# input weights (from proposal net) shape: (ray_num, num of proposal interval), inds shape (ray_num, fine_sample num + 1? TODO, 2)
# 输入的inds应该是sample_pdf中的 below，每个点将有两个值。考虑到sample_pdf得到的点数量为(cone_num + 1)
def getBounds(weights:torch.Tensor, inds:torch.Tensor, sort_inds:torch.Tensor):
    ray_num, target_device = weights.shape[0], weights.device
    inds = torch.gather(inds, -1, sort_inds)
    starts, ends = inds[:, :-1], inds[:, 1:] + 1
    sat:torch.Tensor = torch.cat((torch.zeros(ray_num, 1, device = target_device), torch.cumsum(weights, dim = -1)), dim = -1)                  # proposal net 的weights
    return torch.gather(sat, -1, ends) - torch.gather(sat, -1, starts)

class ProposalLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prop_bounds:torch.Tensor, nerf_weights:torch.Tensor) -> torch.Tensor:
        bound_diff = (F.relu(nerf_weights - prop_bounds)) ** 2
        return torch.sum(bound_diff / (nerf_weights + 1e-8))

class Regularizer(nn.Module):
    def __init__(self) -> None: super().__init__()
    # inputs are of the same shape (ray_num, num of cones)
    def forward(self, weights:torch.Tensor, fine_ts:torch.Tensor):
        center = (fine_ts[..., :-1] + fine_ts[..., 1:]) / 2.
        dists = torch.abs(center[:, None, :] - center[..., None])
        dists = dists / dists.norm(dim = -1, keepdim = True)
        avg_weights = (weights[..., :-1] + weights[..., 1:]) / 2.
        mult_ws =  avg_weights[:, None, :] * avg_weights[..., None]
        delta = fine_ts[..., 1:] - fine_ts[..., :-1]
        return torch.mean(mult_ws * dists) + torch.mean(delta * (avg_weights ** 2)) / 3.

class SoftL1Loss(nn.Module):
    def __init__(self, epsilon = 0.001) -> None: 
        super().__init__()
        self.eps = epsilon
    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        return torch.mean(torch.sqrt(self.eps ** 2 + (pred - target) ** 2))

class LossPSNR(nn.Module):
    __LOG_10__ = 2.3025851249694824
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return -10. * torch.log(x) / LossPSNR.__LOG_10__

class ProposalNetwork(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(self, position_flevel, hidden_unit = 128, cat_origin = True) -> None:
        super().__init__()
        self.position_dims = position_flevel * 6
        self.cat_origin = cat_origin
        extra_dims = 3 if cat_origin else 0
        self.layers = nn.Sequential(
            *makeMLP(self.position_dims + extra_dims, hidden_unit),
            *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, hidden_unit),
            *makeMLP(hidden_unit, 1, nn.Softplus())
        )
        self.apply(self.init_weight)

    def loadFromFile(self, load_path:str, use_amp = False, other_stuff = None):
        save = torch.load(load_path)   
        save_model = save['model']                  
        state_dict = {k:save_model[k] for k in self.state_dict().keys()}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        if use_amp:
            from apex import amp
            amp.load_state_dict(save['amp'])
        print("NeRF Model loaded from '%s'"%(load_path))
        if not other_stuff is None:
            return [save[k] for k in other_stuff]

    def forward(self, pts:torch.Tensor, encoded_pt:torch.Tensor = None) -> torch.Tensor:
        if not encoded_pt is None:
            encoded_x = encoded_pt.view(pts.shape[0], pts.shape[1], self.position_dims)
        else:
            encoded_x = positional_encoding(pts, 10)
        if self.cat_origin:
            encoded_x = torch.cat((pts, encoded_x), -1)
        # output shape (ray_num, coarse_sample_num, 1) --> (ray_num, coarse_sample_num)
        return self.layers(encoded_x).squeeze(-1)

    # zvals from refractive sampler are already weighed by ray_dir norms
    @staticmethod
    def get_weights(density:torch.Tensor, zvals:torch.Tensor, ray_dirs:torch.Tensor = None) -> torch.Tensor:
        if not ray_dirs is None:
            zvals = zvals * (ray_dirs.norm(dim = -1, keepdim = True))
        delta:torch.Tensor = torch.cat((zvals[:, 1:] - zvals[:, :-1], torch.FloatTensor([1e10]).repeat((zvals.shape[0], 1)).cuda()), dim = -1)
        mult:torch.Tensor = torch.exp(-F.relu(density) * delta)
        alpha:torch.Tensor = 1. - mult
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), mult + 1e-10], -1), -1)[:, :-1]
        return weights     # output (ray_num, num of coarse sample (proposal interval number))

if __name__ == "__main__":
    print("Hello NeRF world!")
