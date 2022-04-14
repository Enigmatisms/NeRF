#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn
from time import time
from nerf_helper import encoding
from torch.nn import functional as F
from torchvision.transforms import transforms

# from pytorch.org https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/2
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def makeMLP(in_chan, out_chan, act = nn.ReLU(), batch_norm = False):
    modules = [nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(nn.BatchNorm1d(out_chan))
    if not act is None:
        modules.append(act)
    return modules

# This module is shared by coarse and fine network, with no need to modify
class NeRF(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(self, position_flevel, direction_flevel) -> None:
        super().__init__()
        self.position_flevel = position_flevel
        self.direction_flevel = direction_flevel

        module_list = makeMLP(60, 256)
        for _ in range(4):
            module_list.extend(makeMLP(256, 256))

        self.lin_block1 = nn.Sequential(*module_list)       # MLP before skip connection
        self.lin_block2 = nn.Sequential(
            *makeMLP(316, 256),
            *makeMLP(256, 256), *makeMLP(256, 256),
        )

        self.bottle_neck = nn.Sequential(*makeMLP(256, 256, None))

        self.opacity_head = nn.Sequential(                  # authors said that ReLU is used here
            *makeMLP(256, 1)
        )
        self.rgb_layer = nn.Sequential(
            *makeMLP(280, 128),
            *makeMLP(128, 3, nn.Sigmoid())
        )
        self.apply(self.init_weight)

    def loadFromFile(self, load_path:str):
        save = torch.load(load_path)   
        save_model = save['model']                  
        model_dict = self.state_dict()
        state_dict = {k:v for k, v in save_model.items()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict) 
        print("NeRF Model loaded from '%s'"%(load_path))

    @staticmethod
    def positional_encoding(x:torch.Tensor, freq_level:int) -> torch.Tensor:
        result = []
        ray_num, point_num = x.shape[0], x.shape[1]
        for fid in range(freq_level):
            freq = 2. ** fid
            for func in (torch.sin, torch.cos):
                result.append(func(freq * x.unsqueeze(-1)))
        encoded = torch.cat(result, dim = -1).view(ray_num, point_num, -1)
        return encoded

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    # TODO: fine-network输入的point_num是192，会产生影响吗？
    def forward(self, pts:torch.Tensor) -> torch.Tensor:
        flat_batch = pts.shape[0] * pts.shape[1]
        position_dim, direction_dim = 6 * self.position_flevel, 6 * self.direction_flevel
        encoded_x:torch.Tensor = torch.zeros(flat_batch, position_dim).cuda()
        encoded_r:torch.Tensor = torch.zeros(flat_batch, direction_dim).cuda()
        encoding(pts[:, :, :3].reshape(-1, 3), encoded_x, self.position_flevel, False)
        rotation = pts[:, :, 3:6].reshape(-1, 3)
        rotation = rotation / rotation.norm(dim = -1, keepdim = True)
        encoding(rotation, encoded_r, self.direction_flevel, False)
        encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)
        encoded_r = encoded_r.view(pts.shape[0], pts.shape[1], direction_dim)
        tmp = self.lin_block1(encoded_x)
        encoded_x = torch.cat((encoded_x, tmp), dim = -1)
        encoded_x = self.lin_block2(encoded_x)
        opacity = self.opacity_head(encoded_x)
        encoded_x = self.bottle_neck(encoded_x)
        rgb = self.rgb_layer(torch.cat((encoded_x, encoded_r), dim = -1))
        return torch.cat((rgb, opacity), dim = -1)      # output (ray_num, point_num, 4)

    # rays is of shape (ray_num, 6)
    @staticmethod
    def coarseFineMerge(rays:torch.Tensor, c_zvals:torch.Tensor, f_zvals:torch.Tensor) -> torch.Tensor:
        zvals = torch.cat((f_zvals, c_zvals), dim = -1)
        zvals, _ = torch.sort(zvals, dim = -1)
        sample_pnum = f_zvals.shape[1] + c_zvals.shape[1]
        # Use sort depth to calculate sampled points
        pts = rays[...,None,:3] + rays[...,None,3:] * zvals[...,:,None] 
        # depth * ray_direction + origin (this should be further tested)
        return torch.cat((pts, rays[:, 3:].unsqueeze(-2).repeat(1, sample_pnum, 1)), dim = -1), zvals          # output is (ray_num, coarse_pts num + fine pts num, 6)

    """
        This function is important for inverse transform sampling, since for every ray
        we will have 64 normalized weights (summing to 1.) for inverse sampling
    """
    @staticmethod
    def getNormedWeight(opacity:torch.Tensor, depth:torch.Tensor) -> torch.Tensor:
        delta:torch.Tensor = torch.cat((depth[:, 1:] - depth[:, :-1], torch.FloatTensor([1e10]).repeat((depth.shape[0], 1)).cuda()), dim = -1)
        # print(opacity.shape, depth[:, 1:].shape, raw_delta.shape, delta.shape)
        mult:torch.Tensor = torch.exp(-F.relu(opacity) * delta)
        alpha:torch.Tensor = 1. - mult
        # fusion requires normalization, rgb output should be passed through sigmoid
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1] + 1e-9
        weight_sum = torch.sum(weights, dim = -1, keepdim = True)
        return weights, weights / weight_sum

    # depth shape: (ray_num, point_num)
    # need the norm of rays, shape: (ray_num, point_num)
    @staticmethod
    def render(rgbo:torch.Tensor, depth:torch.Tensor, ray_norm:torch.Tensor) -> torch.Tensor:
        depth = depth * ray_norm
        rgb:torch.Tensor = rgbo[..., :3] # shape (ray_num, pnum, 3)
        opacity:torch.Tensor = rgbo[..., -1]             # 1e-5 is used for eliminating numerical instability
        weights, weights_normed = NeRF.getNormedWeight(opacity, depth)
        weighted_rgb:torch.Tensor = weights[:, :, None] * rgb
        return torch.sum(weighted_rgb, dim = -2), weights_normed     # output (ray_num, 3) and (ray_num, point_num)

TEST_RAY_NUM = 4096
TEST_PNUM = 64
TEST_NEAR_T = 2.
TEST_FAR_T = 6.

if __name__ == "__main__":
    from dataset import CustomDataSet
    from utils import fov2Focal
    torch.set_default_tensor_type(torch.FloatTensor)
    nerf_model = NeRF(10, 4).cuda()
    dataset = CustomDataSet("../../dataset/nerf_synthetic/drums/", transforms.ToTensor(), True)
    cam_fov, tfs, images = dataset.get_dataset(to_cuda = True)
    output:torch.Tensor = torch.zeros(TEST_RAY_NUM, TEST_PNUM + 1, 9, dtype = torch.float32).cuda()
    lengths:torch.Tensor = torch.zeros(TEST_RAY_NUM, TEST_PNUM, dtype = torch.float32).cuda()
    focal = fov2Focal(cam_fov, images.shape[2])
    sampling(images, tfs, output, lengths, TEST_RAY_NUM, TEST_PNUM, focal, TEST_NEAR_T, TEST_FAR_T)
    start_time = time()
    output, sampled_cams = output[:, :-1, :].contiguous(), output[:, -1, :-3].contiguous()
    rgbo = nerf_model(output)
    rendered, weights_normed = NeRF.render(rgbo, lengths, output[:, :, 3:6].norm(dim = -1))
    print("Shape of rendered:", rendered.shape, weights_normed.shape)
    end_time = time()
    print("Finished forwarding within %.6lf seconds. Shape:"%(end_time - start_time), rgbo.shape)
    print("Test translation:")
    print(sampled_cams)
    print("Test weight:")
    summation = torch.sum(weights_normed, dim = -1)
    print(summation)
    