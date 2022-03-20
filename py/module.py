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

"""
TODO:
- 数据集已经可以加载了，也即coarse网络是完全可以跑起来的，现在还有一个主要问题就是
    - 网络的构建以及render操作，主要在render操作上，如何通过网络的输出结果得到一副图，如果写完这部分coarse-fine网络可以复用的部分，那么网络搭建部分就写完了
    - fine网络的sample指导，inverse transform sampling，这是怎么实现的？官方源码中实现了吗？
        - inverse transform sampling 貌似就是之前概率论中学的，如何从简单的均匀分布映射到另一种分布
    - 这两部分完全搞懂，写出来以后就是NeRF了
"""