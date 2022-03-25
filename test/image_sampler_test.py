#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.23
    Last testing module for image sampler
"""

import torch
import numpy as np
from time import time
from nerf_helper import imageSampling
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from instances import BLENDER_FOV, Rs, ts
import sys
sys.path.append("..")
from py.utils import fov2Focal
from py.model import NeRF

POINT_TO_SAMPLE = 128
IMAGE_SIZE = 800
NEAR_T, FAR_T = 2., 6.
INITIAL_SKIP = 0
LINE_TO_VIZ = 16

if __name__ == "__main__":
    output = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, 6, dtype = torch.float32).cuda()
    lengths = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, dtype = torch.float32).cuda()
    tf = np.concatenate((Rs[-1], ts[-1]), axis = -1)
    tf = torch.from_numpy(tf).float().cuda()
    focal = fov2Focal(BLENDER_FOV, IMAGE_SIZE)
    nerf = NeRF(10, 4).cuda()
    imageSampling(tf, output, lengths, IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, focal, NEAR_T, FAR_T)
    print("Forward, here we go")
    with torch.no_grad():
        start_time = time()
        nerf = nerf.eval()
        for i in range(16):
            for j in range(16):
                a = nerf.forward(output[50 * i:50 * (i + 1), 50 * j:50 * (j + 1)].reshape(-1, POINT_TO_SAMPLE, 6))
        end_time = time()
        print("Finished within %.6lf"%(end_time - start_time))
    axis = plt.axes(projection='3d')
    for i in range(INITIAL_SKIP, LINE_TO_VIZ + INITIAL_SKIP):
        point_list = output[i, 0].cpu().numpy()
        axis.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], label = "line %i"%(i))
        axis.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c = 'r', s = 2)
    axis.legend()
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()