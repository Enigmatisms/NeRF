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
from nerf_helper import invTransformSample, validSampling
from time import time

def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

# input weights shape: (ray_num, point_num), prefix sum should be performed in dim1
def weightPrefixSum(weights:torch.Tensor) -> torch.Tensor:
    result = weights.clone()
    # print(result.shape, result[:, 0].shape)
    for i in range(1, weights.shape[1]):
        result[:, i] += result[:, i-1]
    return result

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
def inverseSample(weights:torch.Tensor, rays:torch.Tensor, sample_pnum:int, near:float=2., far:float=6.) -> torch.Tensor:
    if weights.requires_grad == True:
        weights = weights.detach()
    cdf = weightPrefixSum(weights)
    sample_depth = torch.zeros(rays.shape[0], sample_pnum).float().cuda()
    invTransformSample(cdf, sample_depth, sample_pnum, near, far)
    sort_depth, _ = torch.sort(sample_depth, dim = -1)          # shape (ray_num, sample_pnum)
    # Use sort depth to calculate sampled points
    raw_pts = rays.repeat(repeats = (1, 1, sample_pnum)).view(rays.shape[0], sample_pnum, -1)
    # depth * ray_direction + origin (this should be further tested)
    raw_pts[:, :, :3] += sort_depth[:, :, None] * raw_pts[:, :, 3:]
    return raw_pts, sort_depth          # depth is used for rendering

# Extract samples of which alpha is bigger than a threshold
def getValidSamples(images:torch.Tensor) -> torch.Tensor:
    image_result = []
    coord_result = []
    index_result = []
    # 还需要将图像坐标保存下来
    rows, cols = images.shape[2], images.shape[3]
    row_idxs, col_idxs = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing = 'ij')
    coords = torch.stack((row_idxs, col_idxs), dim = -1).cuda()        # shape (rows, cols, 2) --> image coordinates 
    for i, pic in enumerate(images):
        bools = pic[3] > 1e-3
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

def validSampler(rgbs:torch.Tensor, coords:torch.Tensor, tfs:torch.Tensor, ray_num:int, point_num:int, w:int, h:int, focal:float, near:float, far:float):
    output:torch.Tensor = torch.zeros(ray_num, point_num + 1, 9, dtype = torch.float32).cuda()
    lengths:torch.Tensor = torch.zeros(ray_num, point_num, dtype = torch.float32).cuda()
    validSampling(rgbs, coords, tfs, output, lengths, w, h, ray_num, point_num, focal, near, far)
    return output, lengths

def fov2Focal(fov:float, img_width:float) -> float:
    return (img_width / 2) / np.tan(fov * 0.5)
