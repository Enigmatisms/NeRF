#-*-coding:utf-8-*-

"""
    Mip-NeRF and Mip NeRF 360 related methods implementation
    @author Enigmatisms
    @date 2022.4.22 @copyright
"""

""" 
TODO
(1) conical frustum 的各个参数计算，如何使用torch实现？（利用好张量计算的并行性），应该比较简单
(2) 公式（9）（14）的实现
"""

from pyrsistent import b
import torch
from torch.nn import functional as F

# 根据入射光线计算高斯近似的参数
# 假设最后需要每条光线上有n个cones，那么就需要采样n+1个点，这里只需要zvals
# 输入应该是(num_ray, num_cones + 1)
def coneParameters(zvals:torch.Tensor, r:float):
    mid_vals = (zvals[:, 1:] + zvals[:, :-1]) / 2               # t_mu
    diff_vals = ((zvals[:, 1:] - zvals[:, :-1]) / 2) ** 2             # t_sigma^2
    tmp_1 = 3 * mid_vals ** 2 + diff_vals

    mu_t:torch.Tensor = mid_vals + 2 * mid_vals * diff_vals / tmp_1
    sigma_t2:torch.Tensor = diff_vals / 3 - 4 * (diff_vals ** 2) * (12 * mid_vals ** 2 - diff_vals) / 15 / (tmp_1 ** 2)
    sigma_r2:torch.Tensor = (r ** 2) * (0.25 * mid_vals ** 2 + 5 / 12 * diff_vals - 4 * diff_vals ** 2 / (15 * tmp_1)) 
    return mu_t, sigma_t2, sigma_r2

# calculate the covariance matrix and mean vector of Gaussian approx for a cone
# 需要张量计算, cam_rays的shape是 (ray_num, 6), mu_t的shape是(ray_num, num_cones) 所以需要None来增加维度
def coneMeanCov(cam_rays:torch.Tensor, mu_t:torch.Tensor, sigma_t2:torch.Tensor, sigma_r2:torch.Tensor) -> torch.Tensor:
    target_device = mu_t.device()
    mu:torch.Tensor = cam_rays[:, :3] + mu_t[:, :, None] * cam_rays[:, None, 3:]
    dd:torch.Tensor = cam_rays[:, 3:] * cam_rays[:, 3:]        # (ray_num, 3)
    i_m_ddt:torch.Tensor = torch.ones(3, device = target_device)[None, ...] - dd / cam_rays[:, 3:].norm()       # (ray_num, 3, 3)
    diag_sigma:torch.Tensor = sigma_t2[:, :, None] * dd[:, None, ...] + sigma_r2[:, :, None] * i_m_ddt[:, None, ...]        # (ray_num, num_cones, 3, 3)
    return mu, diag_sigma

# 上面这个函数最后用于计算positional encoding的计算量太大，我们只需要计算对角线
def multFreq(freq_lvs:int, mu:torch.Tensor, diag_sigma:torch.Tensor):
    target_device = mu.device()
    # mu shape(ray_num, cone_num, 3), diag_sigma shape: (ray_num, cone_num, 3)
    P:torch.Tensor = torch.cat([torch.eye(3, device = target_device) * (2 ** i) for i in range(freq_lvs)], dim = 0)
    diag_P:torch.Tensor = torch.FloatTensor([4 ** i for i in range(freq_lvs)], device = target_device)
    # diag_sigma_r 的shape应该是 (ray_num, cone_num, 3L)
    ray_num, cone_num, _ = diag_sigma.shape
    diag_sigma_r:torch.Tensor = (diag_P[None, None, :, None] * diag_sigma[..., None, :]).view(ray_num, cone_num, -1)
    mu_r:torch.Tensor = (P @ mu[..., None]).squeeze(-1)
    return mu_r, diag_sigma_r

def ipe_feature(zvals:torch.Tensor, cam_rays:torch.Tensor, freq_lvs:int, r:float):
    mu_t, sigma_t2, sigma_r2 = coneParameters(zvals, r)
    mu, diag_sigma = coneMeanCov(cam_rays, mu_t, sigma_t2, sigma_r2)
    mu_r, diag_sigma_r = multFreq(freq_lvs, mu, diag_sigma)
    ray_num, cone_num, _ = mu_r.shape
    diag_sigma_r = torch.exp(-0.5 * diag_sigma_r)
    sin_part = torch.sin(mu_r) * diag_sigma_r
    cos_part = torch.cos(mu_r) * diag_sigma_r
    sin_part = sin_part.view(ray_num, cone_num, -1, 3)
    cos_part = cos_part.view(ray_num, cone_num, -1, 3)
    return torch.cat((sin_part, cos_part), dim = -1).view(ray_num, cone_num, -1)
