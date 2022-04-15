#-*-coding:utf-8-*-

"""
    Utility function for NeRF
    @author Enigmatisms @date 2022.3.22
"""
import os
import torch
import shutil
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

def generateTestSamples(ray_num:int, coarse_pnum:int, sigma_factor:float = 0.1):
    def gaussian(x:float, mean:float, std:float):
        return 1./(np.sqrt(2 * np.pi) * std) * np.exp(-((x - mean)**2) / (2 * std ** 2))
    result = []
    for _ in range(ray_num):
        gauss = gaussian(np.linspace(2, 6, coarse_pnum), 4, 4 * sigma_factor)
        gauss += np.random.uniform(0, np.mean(gauss) * 0.1, size = gauss.shape)
        gauss /= np.sum(gauss)
        result.append(torch.from_numpy(gauss).view(1, -1))
    return torch.cat(result, dim = 0).float().cuda()

# float32. Shape of ray: (ray_num, 6) --> (origin, direction)
def inverseSample(weights:torch.Tensor, coarse_depth:torch.Tensor, sample_pnum:int) -> torch.Tensor:
    if weights.requires_grad == True:
        weights = weights.detach()
    # cdf = torch.cumsum(weights, dim = -1)
    z_vals_mid = .5 * (coarse_depth[...,1:] + coarse_depth[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], sample_pnum, det=False, pytest=False)
    # invTransformSample(cdf, sample_depth, sample_pnum, near, far)
    return z_samples          # depth is used for rendering

# input (all training images, center_crop ratio)
def randomFromOneImage(img:torch.Tensor, center_crop:float):
    target_device = img.device
    if img.dim() > 3:
        img = img.squeeze(0)
    half_w = img.shape[2] // 2
    half_h = img.shape[1] // 2
    if center_crop < 9.9e-1:
        x_lb, x_ub = int(half_w * (1. - center_crop)), int(half_w + half_w * center_crop)
        y_lb, y_ub = int(half_h * (1. - center_crop)), int(half_h + half_h * center_crop)
    else:
        x_lb = y_lb = 0
        x_ub, y_ub = img.shape[2], img.shape[1]
    row_ids, col_ids = torch.meshgrid(torch.arange(x_lb, x_ub), torch.arange(y_lb, y_ub), indexing = 'ij')
    coords = torch.stack((col_ids - half_w, half_h - row_ids), dim = -1).to(target_device)
    # returned values are flattened
    if center_crop < 9.9e-1:
        return img[:, row_ids, col_ids].view(3, -1).transpose(0, 1).contiguous(), coords.view(-1, 2)
    else:
        return img.view(3, -1).transpose(0, 1).contiguous(), coords.view(-1, 2)

# new valid sampler returns sampled points, sampled length, rgb gt value and camera ray origin and direction
def validSampler(rgbs:torch.Tensor, coords:torch.Tensor, cam_tf:torch.Tensor, ray_num:int, point_num:int, w:int, h:int, focal:float, near:float, far:float):
    target_device = rgbs.device
    max_id = coords.shape[0]
    indices = torch.randint(0, max_id, (ray_num,)).to(target_device)
    output_rgb = rgbs[indices]
    sampled_coords = coords[indices]
    resolution = (far - near) / point_num
    all_lengths = torch.linspace(near, far - resolution, point_num).to(target_device)
    lengths = all_lengths + torch.rand((ray_num, point_num)).to(target_device) * resolution
    # sampled coords is (col_id, col_id)
    ray_raw = torch.sum(torch.cat([sampled_coords / focal, -torch.ones(sampled_coords.shape[0], 1, dtype = torch.float32).to(target_device)], dim = -1).unsqueeze(-2) * cam_tf[:, :-1], dim = -1)
    # return shape (ray_num, point_num, 3), (ray_num, point_num), rgb(ray_num, rgb), cams(ray_num, ray_dir, ray_t)
    pts = cam_tf[:, -1] + ray_raw[:, None, :] * lengths[:, :, None]
    # ray_raw is of shape (ray_num, 3)
    return torch.cat((pts, ray_raw.unsqueeze(-2).repeat(1, point_num, 1)), dim = -1), lengths, output_rgb, torch.cat((cam_tf[:, -1].unsqueeze(0).repeat(ray_raw.shape[0], 1), ray_raw), dim = -1)

def fov2Focal(fov:float, img_width:float) -> float:
    return .5 * img_width / np.tan(.5 * fov)

# From official implementation, since using APEX will change the dtype of inputs, plain CUDA (with no template) won't work
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.cuda().contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1).cuda()
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds).cuda(), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom).cuda(), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def saveModel(model, path:str, opt = None, amp = None):
    checkpoint = {'model': model.state_dict(),}
    if not amp is None:
        checkpoint['amp'] =  amp.state_dict()
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    torch.save(checkpoint, path)
