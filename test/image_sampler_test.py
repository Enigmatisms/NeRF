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
from torchvision import transforms
import sys
sys.path.append("..")
from py.utils import fov2Focal
from py.dataset import CustomDataSet

POINT_TO_SAMPLE = 16
IMAGE_SIZE = 50
NEAR_T, FAR_T = 2., 6.
INITIAL_SKIP = 0

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    # tf = np.concatenate((Rs[-1], ts[-1]), axis = -1)
    # tf = torch.from_numpy(tf).float().cuda()
    dataset = CustomDataSet("../../dataset/nerf_synthetic/%s/"%("lego"), transforms.ToTensor(), True, use_alpha = True)
    cam_fov_train, train_cam_tf, _ = dataset.get_dataset(to_cuda = True)
    focal = fov2Focal(cam_fov_train, IMAGE_SIZE)
    axis = plt.axes(projection='3d')
    colors = ('r', 'g', 'b', 'y', 'p')
    for k in range(4):
        output = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, 6, dtype = torch.float32).cuda()
        lengths = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, dtype = torch.float32).cuda()
        imageSampling(train_cam_tf[k], output, lengths, IMAGE_SIZE, IMAGE_SIZE, POINT_TO_SAMPLE, focal, NEAR_T, FAR_T)
        for i in range(IMAGE_SIZE):
            for j in range(IMAGE_SIZE):
                point_list = output[i, j].cpu().numpy()
                axis.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], c = colors[k], alpha = 0.7)
                # axis.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c = 'r', s = 2)
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()