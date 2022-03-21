#-*-coding:utf-8-*-

"""
    Utility function for NeRF
    @author Enigmatisms @date 2022.3.22
"""
import torch

# input weights shape: (ray_num, point_num), prefix sum should be performed in dim1
def weightPrefixSum(weights:torch.Tensor) -> torch.Tensor:
    result = weights.clone()
    print(result.shape, result[:, 0].shape)
    for i in range(1, weights.shape[1]):
        result[:, i] += result[:, i-1]
    return result