#-*-coding:utf-8-*-

"""
    Custom pytorch dataset for training and testing
    Blender synthetic realistic dataset from NeRF
"""

import os
import json
import torch
import natsort
import numpy as np
from PIL import Image
from torch.utils import data

DATASET_PREFIX = "../../dataset/nerf_synthetic/"

class CustomDataSet(data.Dataset):
    def __init__(self, root_dir, transform, is_train = True):
        self.is_train = is_train
        self.root_dir = root_dir
        self.main_dir = root_dir + ("train/" if is_train else "test/")
        self.transform = transform
        print(self.main_dir)
        img_names = filter(lambda x: x.endswith("png"), os.listdir(self.main_dir))
        all_imgs = [name for name in img_names]
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

    @staticmethod
    def readFromJson(path:str):
        with open(path, "r") as file:
            items = json.load(file)
        cam_fov = items["camera_angle_x"]
        print("Camera fov: %lf"%(cam_fov))
        tf_np = np.stack([frame["transform_matrix"] for frame in items["frames"]], axis = 0)
        tfs = torch.from_numpy(tf_np)
        return cam_fov, tfs.float()
    
    """
        Return camera fov, transforms for each image, batchified image data in shape (N, 3, H, W)
    """
    def get_dataset(self, to_cuda:bool):
        result = []
        for img_name in self.total_imgs:
            img_loc = os.path.join(self.main_dir, img_name)
            image = Image.open(img_loc).convert("RGB")
            result.append(self.transform(image))
        json_file = "%stransforms_%s.json"%(self.root_dir, "train" if self.is_train else "test")
        cam_fov, tfs = CustomDataSet.readFromJson(json_file)
        if to_cuda:
            tfs = tfs.cuda()
            return cam_fov, tfs, torch.stack(result, dim = 0).float().cuda()
        return cam_fov, tfs, torch.stack(result, dim = 0).float()