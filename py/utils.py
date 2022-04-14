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
from random import randint

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
def inverseSample(weights:torch.Tensor, rays:torch.Tensor, coarse_depth:torch.Tensor, sample_pnum:int, near:float=2., far:float=6.) -> torch.Tensor:
    if weights.requires_grad == True:
        weights = weights.detach()
    # cdf = torch.cumsum(weights, dim = -1)
    z_vals_mid = .5 * (coarse_depth[...,1:] + coarse_depth[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], sample_pnum, det=False, pytest=False)
    # invTransformSample(cdf, sample_depth, sample_pnum, near, far)
    sort_depth, _ = torch.sort(z_samples, dim = -1)          # shape (ray_num, sample_pnum)
    # Use sort depth to calculate sampled points
    raw_pts = rays.repeat(repeats = (1, 1, sample_pnum)).view(rays.shape[0], sample_pnum, -1)
    # depth * ray_direction + origin (this should be further tested)
    raw_pts[:, :, :3] += sort_depth[:, :, None] * raw_pts[:, :, 3:]
    return raw_pts, sort_depth          # depth is used for rendering

# Extract samples of which alpha is bigger than a threshold
def getValidSamples(images:torch.Tensor, invalid_threshold:float = -10) -> torch.Tensor:
    image_result = []
    coord_result = []
    index_result = []
    rows, cols = images.shape[2], images.shape[3]
    row_idxs, col_idxs = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing = 'ij')
    coords = torch.stack((row_idxs, col_idxs), dim = -1).cuda()        # shape (rows, cols, 2) --> image coordinates 
    for i, pic in enumerate(images):
        bools = pic[3] > 1e-3
        invalids = (~bools) & (torch.normal(0, 1, bools.shape).cuda() > invalid_threshold)
        bools = (bools | invalids)          # add some of the invalid points (if invalid_threshold == 0, this means half of the invalid points are added)
        valid_samples = pic[:3, bools].transpose(0, 1)
        valid_coords = coords[bools]
        index = torch.ones(valid_samples.shape[0]) * i
        image_result.append(valid_samples)
        index_result.append(index)
        coord_result.append(valid_coords)
    coords = torch.cat(coord_result, dim = 0)
    indices = torch.cat(index_result, dim = 0).cuda().view(-1, 1)
    stacked = torch.cat((coords, indices), dim = -1)
    return torch.cat(image_result, dim = 0), stacked.int()

# input (all training images, center_crop ratio)
def randomFromOneImage(imgs:torch.Tensor, center_crop:float):
    img_idx = randint(0, imgs.shape[0] - 1)
    select_img = imgs[img_idx]
    half_w = select_img.shape[2] // 2
    half_h = select_img.shape[1] // 2
    if center_crop < 9.9e-1:
        x_lb, x_ub = int(half_w * (1. - center_crop)), int(half_w + half_w * center_crop)
        y_lb, y_ub = int(half_h * (1. - center_crop)), int(half_h + half_h * center_crop)
    else:
        x_lb = y_lb = 0
        x_ub, y_ub = select_img.shape[2], select_img.shape[1]
    row_ids, col_ids = torch.meshgrid(torch.arange(x_lb, x_ub), torch.arange(y_lb, y_ub), indexing = 'ij')
    coords = torch.stack((col_ids - half_w, half_h - row_ids), dim = -1).cuda()
    # returned values are flattened
    if center_crop <9.9e-1:
        return select_img[:, row_ids, col_ids].view(3, -1).transpose(0, 1).contiguous(), coords.view(-1, 2), img_idx
    else:
        return select_img.view(3, -1).transpose(0, 1).contiguous(), coords.view(-1, 2), img_idx

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W / 2)/focal, -(j-H / 2)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


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
