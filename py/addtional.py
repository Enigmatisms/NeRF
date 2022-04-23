#-*-coding:utf-8-*-

"""
    Implementation related with mip NeRF 360
"""

import torch
from torch import nn
from torch.nn import functional as F
from apex import amp
from py.utils import makeMLP


# according to calculated weights (of proposal net) and indices of inverse sampling, calculate the bounds required for loss computation
# input weights (from proposal net) shape: (ray_num, num of proposal interval), inds shape (ray_num, fine_sample num + 1? TODO, 2)
def getBounds(weights:torch.Tensor, inds:torch.Tensor):
    pass

class NeRF(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(self, position_flevel, direction_flevel, cat_origin = True) -> None:
        super().__init__()
        
        self.apply(self.init_weight)

    def loadFromFile(self, load_path:str, use_amp = False, opt = None):
        save = torch.load(load_path)   
        save_model = save['model']                  
        model_dict = self.state_dict()
        state_dict = {k:v for k, v in save_model.items()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        if not opt is None:
            opt.load_state_dict(save['optimizer'])
        if use_amp:
            amp.load_state_dict(save['amp'])
        print("NeRF Model loaded from '%s'"%(load_path))

    # 主要问题是，如何将公式(13)复现出来
    def forward(self, pts:torch.Tensor, encoded_pt:torch.Tensor = None) -> torch.Tensor:
        pass
        # return torch.cat((rgb, opacity), dim = -1)      # output (ray_num, point_num, 4)

if __name__ == "__main__":
    print("Hello NeRF world!")