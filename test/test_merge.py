import sys
sys.path.append("..")

import torch
from py.model import NeRF
from nerf_helper import sampling
from instances import Rs, ts, BLENDER_FOV
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from py.utils import fov2Focal, inverseSample, generateTestSamples

RAY_NUM = 4096
BIN_NUM = 64
NEAR_T = 2.0
FAR_T = 6.0
RESOLUTION = (FAR_T - NEAR_T) / BIN_NUM
CAM_NUM = 1
LINE_TO_VIZ = 2
INITIAL_SKIP = 0

if __name__ == "__main__":
    o1:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM + 1, 9).cuda()
    l1:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM).cuda()

    focal = fov2Focal(BLENDER_FOV, 800.)
    images = torch.normal(0, 1, (CAM_NUM, 3, 800, 800)).cuda()
    Ts = torch.cat([torch.hstack((R, t)).unsqueeze(dim = 0) for R, t in zip(Rs, ts)], dim = 0).cuda()

    sampling(images, Ts[:CAM_NUM], o1, l1, RAY_NUM, BIN_NUM, focal, NEAR_T, FAR_T)
    weights_pdf = generateTestSamples(RAY_NUM, BIN_NUM)
    cam_rays = o1[:, -1, :-3]
    fine_pts, fine_depth = inverseSample(weights_pdf, cam_rays, BIN_NUM * 2)
    pts, _ = NeRF.coarseFineMerge(cam_rays, l1, fine_depth)

    axis = plt.axes(projection='3d')
    for i in range(INITIAL_SKIP, LINE_TO_VIZ + INITIAL_SKIP):
        point_list = pts[i, :, :].cpu().numpy()
        fine_cpu_pts = fine_pts[i, :, :].cpu().numpy()
        coarse_cpu_pts = o1[i, :-1, :].cpu().numpy()
        axis.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], label = "line %i"%(i))
        axis.scatter(fine_cpu_pts[:, 0], fine_cpu_pts[:, 1], fine_cpu_pts[:, 2], c = 'r', s = 6)
        axis.scatter(coarse_cpu_pts[:, 0], coarse_cpu_pts[:, 1], coarse_cpu_pts[:, 2], c = 'b', s = 6)
    axis.legend()
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()