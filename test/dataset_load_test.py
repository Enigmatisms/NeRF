#-*-coding:utf-8-*-

"""
    Load dataset from json.
    Blender synthetic realistic dataset from NeRF
"""

import json
import numpy as np

DATASET_PREFIX = "../../dataset/nerf_synthetic/"

def readFromJson(path:str):
    with open(path, "r") as file:
        items = json.load(file)
    print(items["camera_angle_x"])
    for frame in items["frames"]:
        print(np.array(frame["transform_matrix"]))

if __name__ == "__main__":
    readFromJson(DATASET_PREFIX + "drums/transforms_train.json")