#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.18
    NeRF CUDA sampler test2. Visualizing results from sampler in a specific view
"""

import torch
import numpy as np
from time import time
from nerf_helper import sampling
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from instances import *

RAY_NUM = 4096
BIN_NUM = 128
NEAR_T = 2.0
FAR_T = 6.0
RESOLUTION = (FAR_T - NEAR_T) / BIN_NUM
CAM_NUM = 1
LINE_TO_VIZ = 32
INITIAL_SKIP = 32

if __name__ == "__main__":
    Ts = torch.cat([torch.hstack((R @ K.inverse(), t)).unsqueeze(dim = 0) for R, t in zip(Rs, ts)], dim = 0).cuda()
    print(Ts[:CAM_NUM])
    images = torch.normal(0, 1, (CAM_NUM, 3, 200, 200)).cuda()
    output:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM, 9).cuda()
    lengths:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM).cuda()
    start_time = time()
    sampling(images, Ts[:CAM_NUM], output, lengths, RAY_NUM, BIN_NUM, NEAR_T, FAR_T)
    end_time = time()
    print("Finished sampling within %.7lf s\n", end_time - start_time)
    axis = plt.axes(projection='3d')
    for i in range(INITIAL_SKIP, LINE_TO_VIZ + INITIAL_SKIP):
        point_list = output[i, :-1, :].cpu().numpy()
        axis.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], label = "line %i"%(i))
        axis.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c = 'r', s = 8)
    axis.legend()
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()
