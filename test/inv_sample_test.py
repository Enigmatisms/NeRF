#-*-coding:utf-8-*-

"""
    Test inverse transform sampler.
    @author Enigmatisms @date 2021.3.21
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nerf_helper import invTransformSample
from time import time

# input weights shape: (ray_num, point_num), prefix sum should be performed in dim1
def weightPrefixSum(weights:torch.Tensor):
    result = weights.clone()
    print(result.shape, result[:, 0].shape)
    for i in range(1, weights.shape[1]):
        result[:, i] += result[:, i-1]
    return result

def generateTestSamples(ray_num:int, coarse_pnum:int, sigma_factor:float = 0.1):
    def gaussian(x:float, mean:float, std:float):
        return 1./(np.sqrt(2 * np.pi) * std) * np.exp(-((x - mean)**2) / (2 * std ** 2))
    result = []
    for _ in range(ray_num):
        gauss = gaussian(np.linspace(2, 6, coarse_pnum), 4, 4 * sigma_factor)
        gauss += np.random.uniform(0, np.mean(gauss) * 0.1, size = gauss.shape)
        gauss /= np.sum(gauss)
        result.append(torch.from_numpy(gauss).view(1, -1))
    return torch.cat(result, dim = 0)

def sampleVisualize(samples:torch.Tensor):
    prefix_sum = weightPrefixSum(samples)
    for i in range(3):
        plt.figure(i)
        plt.subplot(1, 2, 1)
        plt.plot(samples[i, :])
        plt.subplot(1, 2, 2)
        plt.plot(prefix_sum[i, :])
    plt.show()

def invSamplerTest(samples:torch.Tensor):
    PNUM_SAMPLE = 128
    NEAR = 2
    FAR = 6
    prefix_sum = weightPrefixSum(samples).float().cuda()
    rays = torch.FloatTensor([0, 0, 0, 1, 1, 0]).repeat(repeats = (4096, 1)).cuda()
    output = torch.zeros(4096, 128, 6, dtype = torch.float32).cuda()
    start_time = time()
    invTransformSample(prefix_sum, rays, output, PNUM_SAMPLE, NEAR, FAR)
    end_time = time()
    print("Process completed within %.6lf s"%(end_time - start_time))

    axis = plt.axes(projection='3d')
    points = output.cpu()
    axis.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2])
    axis.legend()
    axis.set_zlabel('Z') 
    axis.set_ylabel('Y')
    axis.set_xlabel('X')
    plt.show()

def sortTest():
    a = torch.randint(0, 100000, (1024, 128)).cuda()
    time_sum = 0.0
    for i in range(100):
        start_time = time()
        torch.sort(a, dim = -1)
        end_time = time()
        time_sum += (end_time - start_time)
        print("%d / 100"%(i))
    print("Finished within %.6lf s"%(time_sum))

# torch.sort implement is even faster than moderngpu implementation
if __name__ == "__main__":
    samples = generateTestSamples(4096, 64)
    invSamplerTest(samples)
    # sortTest()
