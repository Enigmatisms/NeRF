#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

from numpy import log
import torch
from torch import nn
from torch.nn import functional as F
from py.ref_func import generate_ide_fn
from py.nerf_helper import makeMLP, positional_encoding, linear_to_srgb

# This module is shared by coarse and fine network, with no need to modify
class RefNeRF(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(self, 
        position_flevel, sh_max_level, 
        bottle_neck_dim = 128,
        hidden_unit = 256, 
        output_dim = 256, 
        use_srgb = False,
        cat_origin = True,
        perturb_bottle_neck_w = 0.1,
    ) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.sh_max_level = sh_max_level
        self.bottle_neck_dim = bottle_neck_dim
        self.dir_enc_dim = (1 << sh_max_level) - 1 + sh_max_level

        extra_width = 3 if cat_origin else 0
        spatial_module_list = makeMLP(60 + extra_width, hidden_unit)
        for _ in range(3):
            spatial_module_list.extend(makeMLP(hidden_unit, hidden_unit))

        # spatial MLP part (spa_xxxx)
        self.spa_block1 = nn.Sequential(*spatial_module_list)       # MLP before skip connection
        self.spa_block2 = nn.Sequential(
            *makeMLP(hidden_unit + 60 + extra_width, hidden_unit),
            *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, hidden_unit),
            *makeMLP(hidden_unit, output_dim)
        )

        self.rho_tau_head = nn.Linear(output_dim, 2)
        self.norm_col_tint_head = nn.Linear(output_dim, 9)  # output normal prediction, color, tint (all 3)
        self.bottle_neck = nn.Linear(output_dim, bottle_neck_dim)
        self.spec_rgb_head = nn.Sequential(*makeMLP(output_dim, 3, nn.Sigmoid()))

        dir_input_dim = 1 + bottle_neck_dim + self.dir_enc_dim
        directional_module_list = makeMLP(dir_input_dim, hidden_unit)
        for _ in range(3):
            directional_module_list.extend(makeMLP(hidden_unit, hidden_unit))

        self.dir_block1 = nn.Sequential(*directional_module_list)
        self.dir_block2 = nn.Sequential(                   # skip connection ()
            *makeMLP(hidden_unit + dir_input_dim, hidden_unit),
            *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, output_dim),
            *makeMLP(hidden_unit, output_dim)
        )
        # \rho is roughness coefficient, \tau is density

        self.use_srgb = use_srgb
        self.cat_origin = cat_origin
        self.perturb_bottle_neck_w = perturb_bottle_neck_w
        self.integrated_dir_enc = generate_ide_fn(sh_max_level)
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
        print("Ref NeRF Model loaded from '%s'"%(load_path))
        if not other_stuff is None:
            return [save[k] for k in other_stuff]

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    def forward(self, pts:torch.Tensor) -> torch.Tensor:
        position_dim = 6 * self.position_flevel
        encoded_x = positional_encoding(pts[:, :, :3], self.position_flevel)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)

        if self.cat_origin:
            encoded_x = torch.cat((pts[:, :, :3], encoded_x), -1)

        x_tmp = self.spa_block1(encoded_x)
        encoded_x = torch.cat((encoded_x, x_tmp), dim = -1)
        intermediate = self.spa_block2(encoded_x)               # output of spatial network

        [normal, diffuse_rgb, spec_tint]: list[torch.Tensor, torch.Tensor, torch.Tensor] = self.norm_col_tint_head(intermediate).split((3, 3, 3), dim = -1)
        roughness, density = self.rho_tau_head(intermediate).split((1, 1), dim = -1)
        roughness = F.softplus(roughness - 1.)
        spa_info_b = self.bottle_neck(intermediate)
        spa_info_b = spa_info_b + torch.normal(0, self.perturb_bottle_neck_w, spa_info_b.shape)

        normal = normal / (normal.norm(dim = -1, keepdim = True) + 1e-6)
        # needs further validation
        reflect_r = pts[..., 3:] - 2. * torch.sum(pts[..., 3:] * normal, dim = -1, keepdim = True) * normal
        wr_ide = self.integrated_dir_enc(reflect_r, roughness)
        nv_dot = -torch.sum(normal * pts[..., 3:], dim = -1, keepdim = True)     # normal dot (-view_dir)

        all_inputs = torch.cat((spa_info_b, wr_ide, nv_dot), dim = -1)
        r_tmp = self.dir_block1(all_inputs)
        all_inputs = torch.cat((all_inputs, r_tmp), dim = -1)

        specular_rgb = self.spec_rgb_head(self.dir_block2(all_inputs)) * F.sigmoid(spec_tint) 
        if self.use_srgb == True:
            diffuse_rgb = F.sigmoid(diffuse_rgb - log(3.))
            rgb = torch.clip(linear_to_srgb(specular_rgb + diffuse_rgb), 0.0, 1.0)
        else:
            diffuse_rgb = F.sigmoid(diffuse_rgb)
            rgb = torch.clip(specular_rgb + diffuse_rgb, 0.0, 1.0)

        return torch.cat((rgb, density), dim = -1)      # output (ray_num, point_num, 4)

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
    def getNormedWeight(opacity:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        delta:torch.Tensor = torch.cat((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e10]).repeat((depth.shape[0], 1)).cuda()), dim = -1)
        # print(opacity.shape, depth[:, 1:].shape, raw_delta.shape, delta.shape)
        mult:torch.Tensor = torch.exp(-F.softplus(opacity) * delta)
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
        weights = RefNeRF.getNormedWeight(opacity, depth)
        rgb:torch.Tensor = torch.sum(weights[:, :, None] * rgb, dim = -2)
        if white_bkg:
            acc_map = torch.sum(weights, -1)
            rgb = rgb + (1.-acc_map[...,None])
        return rgb, weights     # output (ray_num, 3) and (ray_num, point_num)