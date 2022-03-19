#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.18
    NeRF CUDA sampler test. 4096 sampling rays (per batch) and 128 points per ray: 0.00074ms
"""

import torch
import numpy as np
from nerf_helper import sampling
from scipy.spatial.transform import Rotation as R
from time import time
from instances import *

RAY_NUM = 512
BIN_NUM = 128
NEAR_T = 0.01
RESOLUTION = 0.05
CAM_NUM = 3

if __name__ == "__main__":
    for R in Rs:
        print(R.type())
    Ts = torch.cat([torch.hstack((R @ K.inverse(), t)).unsqueeze(dim = 0) for R, t in zip(Rs, ts)], dim = 0).cuda()
    images = torch.normal(0, 1, (CAM_NUM, 3, 200, 200), dtype = torch.float32).cuda()
    output = torch.zeros(RAY_NUM, BIN_NUM + 1, 3, dtype = torch.float32).cuda()
    lengths = torch.zeros(RAY_NUM, BIN_NUM, dtype = torch.float32).cuda()
    start_time = time()
    sampling(images, Ts[:CAM_NUM], output, lengths, RAY_NUM, BIN_NUM, NEAR_T, RESOLUTION)
    end_time = time()

    # Finish within 1ms
    print("Process completed with %.7lf seconds"%(end_time - start_time))
    print(output[0, :, :].cpu())
    print(lengths[0, :].cpu())