#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.18
    NeRF CUDA sampler test. 512 sampling rays (per batch) and 128 points per ray: 0.00024ms
"""

import torch
import numpy as np
from sampler import sampling
from scipy.spatial.transform import Rotation as R
from time import time

K = torch.FloatTensor([
    [100, 0, 100],
    [0, 100, 100],
    [0, 0, 1]
])

Rs = ([
    torch.eye(3, dtype = torch.float32),
    torch.from_numpy(R.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix()).float(),
    torch.from_numpy(R.from_rotvec(np.pi / 3 * 2 * np.array([0, 0, 1])).as_matrix()).float()
])

ts = ([
    torch.FloatTensor([[0, 1.5, 1.0]]).view(-1, 1),
    torch.FloatTensor([[1.0, 0, 1.0]]).view(-1, 1),
    torch.FloatTensor([[2.0, 1.0, 1.0]]).view(-1, 1),
])

RAY_NUM = 512
BIN_NUM = 128
NEAR_T = 0.01
RESOLUTION = 0.05

if __name__ == "__main__":
    for R in Rs:
        print(R.type())
    Ts = torch.cat([torch.hstack((R @ K.inverse(), t)).unsqueeze(dim = 0) for R, t in zip(Rs, ts)], dim = 0).cuda()
    images = torch.normal(0, 1, (3, 3, 200, 200)).cuda()
    output = torch.zeros(RAY_NUM, BIN_NUM + 1, 3).cuda()
    lengths = torch.zeros(RAY_NUM, BIN_NUM).cuda()
    start_time = time()
    sampling(images, Ts, output, lengths, RAY_NUM, BIN_NUM, NEAR_T, RESOLUTION)
    end_time = time()

    # Finish within 1ms
    print("Process completed with %.7lf seconds"%(end_time - start_time))
    print(output[0, :, :].cpu())
    print(lengths[0, :].cpu())