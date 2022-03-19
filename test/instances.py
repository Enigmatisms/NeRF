#-*-coding:utf-8-*-
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

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

# convert from fov angle to focal length, we assume that the image is square
def fov2Focal(fov:float, img_width:float) -> float:
    return (img_width / 2) / np.tan(fov * 0.5)