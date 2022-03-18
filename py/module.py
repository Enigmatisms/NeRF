#-*-coding:utf-8-*-

"""
    NeRF network details. To be finished ...
"""

import torch
from torch import nn

class NeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # Positional Encoding is implemented through CUDA