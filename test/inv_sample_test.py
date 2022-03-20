#-*-coding:utf-8-*-

"""
    Test inverse transform sampler.
    @author Enigmatisms @date 2021.3.21
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# from nerf_helper import *

# input weights shape: (ray_num, point_num), prefix sum should be performed in dim1
def weightPrefixSum(weights:torch.Tensor):
    result = weights.clone()
    print(result.shape, result[:, 0].shape)
    for i in range(1, weights.shape[1]):
        result[:, i] += result[:, i-1]
    return result

def generateTestSamples(ray_num:int, coarse_pnum:int, sigma_factor:float = 0.25):
    def gaussian(x:float, mean:float, std:float):
        return 1./(np.sqrt(2 * np.pi) * std) * np.exp(-((x - mean)**2) / (2 * std ** 2))
    result = []
    for _ in range(ray_num):
        gauss = gaussian(np.linspace(2, 6, coarse_pnum), 4, 4 * sigma_factor)
        gauss += np.random.uniform(0, np.mean(gauss), size = gauss.shape)
        gauss /= np.sum(gauss)
        result.append(torch.from_numpy(gauss).view(1, -1))
    return torch.cat(result, dim = 0)

if __name__ == "__main__":
    samples = generateTestSamples(16, 64)
    prefix_sum = weightPrefixSum(samples)
    for i in range(3):
        plt.figure(i)
        plt.subplot(1, 2, 1)
        plt.plot(samples[i, :])
        plt.subplot(1, 2, 2)
        plt.plot(prefix_sum[i, :])
    plt.show()

