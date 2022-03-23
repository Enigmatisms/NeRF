#-*-coding:utf-8-*-
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("..")
from py.utils import fov2Focal

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
    torch.FloatTensor([[0, 2.5, 1.0]]).view(-1, 1),
    torch.FloatTensor([[1.0, 0, 1.0]]).view(-1, 1),
    torch.FloatTensor([[2.0, 1.0, 1.0]]).view(-1, 1),
])

sampled_points = torch.FloatTensor([
    [1.0, 2.0, 3.0],
    [-1.0, 2.5, 0.006],
    [7.0, 18.0, -5.6],
    [0.005, -0.009, 0.0],
    [6.0, -5.6, -2.5]
])

simple_points = torch.FloatTensor([
    [0, 0, 0],
    [1, 1, 1],
    [3, 4, 5]
])

BLENDER_FOV = 0.6194058656692505

if __name__ == "__main__":
    a = torch.FloatTensor([
        [[1, 1, 1],
        [2, 2, 2]],
        [[-1, -1, -1],
        [-2, -2, -2]],
        [[0, 0, 0],
        [1, 1, -1]]
    ])
    
    b = torch.Tensor([
        [0.5, 0.4], 
        [0.2, -0.2],
        [0.8, 0.8]
    ])
    
    print(a.shape, b.shape)
    c = b[:, :, None] * a
    print(c)
    print(torch.sum(c, dim = 1))
    