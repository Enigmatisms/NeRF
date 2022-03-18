#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.18
    NeRF CUDA sampler test2. Visualizing results from sampler in a specific view
"""

import torch
import numpy as np
from sampler import sampling
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

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

RAY_NUM = 32
BIN_NUM = 16
NEAR_T = 0.01
RESOLUTION = 0.05

if __name__ == "__main__":
    for R in Rs:
        print(R.type())
    Ts = torch.cat([torch.hstack((R @ K.inverse(), t)).unsqueeze(dim = 0) for R, t in zip(Rs, ts)], dim = 0).cuda()
    images = torch.normal(0, 1, (3, 3, 200, 200)).cuda()
    output:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM + 1, 3).cuda()
    lengths:torch.Tensor = torch.zeros(RAY_NUM, BIN_NUM).cuda()
    sampling(images, Ts, output, lengths, RAY_NUM, BIN_NUM, NEAR_T, RESOLUTION)
    
    point_list1 = output[0, :-1, :].cpu().numpy()
    point_list2 = output[1, :-1, :].cpu().numpy()
    
    axis = plt.axes(projection='3d')
    print(point_list1)
    axis.plot3D(point_list1[:, 0], point_list1[:, 1], point_list1[:, 2])
    axis.scatter(point_list1[:, 0], point_list1[:, 1], point_list1[:, 2], c = 'r', s = 8)
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()
    # TODO: 所有点之后可以进行可视化
    # Finish within 1ms
    # print(output[:3].cpu())
    # print(lengths[:3].cpu())