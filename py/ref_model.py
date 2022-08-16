#-*-coding:utf-8-*-

"""
    Ref NeRF network details. To be finished ...
"""
import torch
from torch import nn
from numpy import log
from py.nerf_base import NeRF
from typing import Optional, Tuple
from torch.nn import functional as F
from py.ref_func import generate_ide_fn
from py.nerf_helper import makeMLP, positional_encoding, linear_to_srgb

# Inherited from NeRF (base class)
class RefNeRF(NeRF):
    def __init__(self, 
        position_flevel, sh_max_level, 
        bottle_neck_dim = 128,
        hidden_unit = 256, 
        output_dim = 256, 
        use_srgb = False,
        cat_origin = True,
        perturb_bottle_neck_w = 0.1,
    ) -> None:
        super().__init__(position_flevel, cat_origin, lambda x: x)          # density is not activated during render
        self.sh_max_level = sh_max_level
        self.bottle_neck_dim = bottle_neck_dim
        self.dir_enc_dim = ((1 << sh_max_level) - 1 + sh_max_level) << 1

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
        self.perturb_bottle_neck_w = perturb_bottle_neck_w
        self.integrated_dir_enc = generate_ide_fn(sh_max_level)
        self.apply(self.init_weight)

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    def forward(self, pts:torch.Tensor, ray_d: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        position_dim = 6 * self.position_flevel
        encoded_x = positional_encoding(pts[:, :, :3], self.position_flevel)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)

        if self.cat_origin:
            encoded_x = torch.cat((pts[:, :, :3], encoded_x), -1)

        x_tmp = self.spa_block1(encoded_x)
        encoded_x = torch.cat((encoded_x, x_tmp), dim = -1)
        intermediate = self.spa_block2(encoded_x)               # output of spatial network

        [normal, diffuse_rgb, spec_tint] = self.norm_col_tint_head(intermediate).split((3, 3, 3), dim = -1)
        roughness, density = self.rho_tau_head(intermediate).split((1, 1), dim = -1)
        roughness = F.softplus(roughness - 1.)
        spa_info_b = self.bottle_neck(intermediate)
        spa_info_b = spa_info_b + torch.normal(0, self.perturb_bottle_neck_w, spa_info_b.shape, device = spa_info_b.device)

        normal = -normal / (normal.norm(dim = -1, keepdim = True) + 1e-6)
        # needs further validation
        ray_d = pts[..., 3:] if ray_d is None else ray_d
        reflect_r = ray_d - 2. * torch.sum(ray_d * normal, dim = -1, keepdim = True) * normal
        wr_ide = self.integrated_dir_enc(reflect_r, roughness)
        nv_dot = torch.sum(normal * ray_d, dim = -1, keepdim = True)     # normal dot (-view_dir)

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

        return torch.cat((rgb, density), dim = -1), normal      # output (ray_num, point_num, 4) + (ray_num, point_num, 3)

class WeightedNormalLoss(nn.Module):
    def __init__(self, size_average = False):
        super().__init__()
        self.size_average = size_average        # average (per point, not per ray)
    
    # weight (ray_num, point_num)
    def forward(self, weight:torch.Tensor, d_norm: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        dot_diff = 1. - torch.sum(d_norm * p_norm, dim = -1)
        # norm_diff = torch.pow((d_norm - p_norm), 2)
        return torch.mean(weight * dot_diff) if self.size_average == True else torch.sum(weight * dot_diff)

class BackFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # 注意，可以使用pts[..., 3:] 作为输入
    def forward(self, weight:torch.Tensor, normal: torch.Tensor, ray_d: torch.Tensor) -> torch.Tensor:
        return torch.sum(weight * F.relu(torch.sum(normal * ray_d, dim = -1)))
