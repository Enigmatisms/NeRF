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

from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F 

DATASET_PREFIX = "../../dataset/nerf_synthetic/"

class AdaptiveResize(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
        self.interpolation = transforms.InterpolationMode.BILINEAR
        self.max_size = None
        self.antialias = None

    def forward(self, input:Image.Image):
        size = (int(input.size[1] * self.ratio), int(input.size[0] * self.ratio))
        return F.resize(input, size, self.interpolation, self.max_size, self.antialias)
        
class CustomDataSet(data.Dataset):
    """ Even in the DDP case, we still load images for all the images (won't be a problem)
        we do not shuffle the data but let the sampler do the work

        TODO: test the sampler
    """
    def __init__(self, root_dir, transform, scene_scale = 1.0, is_train = True, use_alpha = False, white_bkg = False, use_div = False):
        self.is_train = is_train
        self.root_dir = root_dir
        self.main_dir = root_dir + ("train/" if is_train else "test/")
        self.transform = transform
        img_names = filter(lambda x: x.endswith("png") and (not "normal" in x) and (not "alpha" in x), os.listdir(self.main_dir))
        all_imgs = [name for name in img_names]
        self.total_imgs = natsort.natsorted(all_imgs)
        self.use_alpha = use_alpha
        self.scene_scale = scene_scale
        self.white_bkg = white_bkg
        self.use_div = use_div
        self.cam_fov, self.tfs, self.divisions = self.__get_camera_param()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc, mode = 'r').convert("RGBA" if self.use_alpha or self.white_bkg else "RGB")
        tensor_image = self.transform(image)
        tf = self.tfs[idx].clone()
        if self.white_bkg:
            tensor_image = tensor_image[:3, ...]*tensor_image[-1:, ...] + (1.-tensor_image[-1:, ...])
        tf[:3, -1] *= self.scene_scale
        return tensor_image, tf

    def r_c(self):
        image, _ = self.__getitem__(0)
        return image.shape[1], image.shape[2]

    def cuda(self, flag = True):
        self.is_cuda = flag

    @staticmethod
    def readFromJson(path:str, use_div = False):
        with open(path, "r") as file:
            items: dict = json.load(file)
        cam_fov = items["camera_angle_x"]
        if "camera_angle_y" in items:
            cam_fov = (cam_fov, items["camera_angle_y"])
        tf_np = np.stack([frame["transform_matrix"] for frame in items["frames"]], axis = 0)
        tfs = torch.from_numpy(tf_np)[:, :3, :]
        division = None
        if use_div:
            division = items.get('division', None)
        return cam_fov, tfs.float(), division
        
    def __get_camera_param(self):
        json_path = f"{self.root_dir}transforms_{'train' if self.is_train else 'test'}"
        if self.use_div:
            json_path = f"{json_path}_div.json"
        else:
            json_path = f"{json_path}.json"
        return CustomDataSet.readFromJson(json_path, self.use_div)

    def getCameraParam(self):
        return self.cam_fov, self.tfs

    """
        Return camera fov, transforms for each image, batchified image data in shape (N, 3, H, W)
    """
    def get_dataset(self, to_cuda:bool):
        result = []
        for img_name in self.total_imgs:
            img_loc = os.path.join(self.main_dir, img_name)
            image = Image.open(img_loc).convert("RGBA" if self.use_alpha else "RGB")
            result.append(self.transform(image))
        cam_fov, tfs = self.getCameraParam()
        if to_cuda:
            tfs = tfs.cuda()
            return cam_fov, tfs, torch.stack(result, dim = 0).float().cuda()
        return cam_fov, tfs, torch.stack(result, dim = 0).float()
