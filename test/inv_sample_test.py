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

import sys
sys.path.append("..")

from py.utils import weightPrefixSum, generateTestSamples, inverseSample

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
    rays = torch.FloatTensor([0, 0, 0, 1, 1, 0]).repeat(repeats = (4096, 1)).cuda()
    output, _ = inverseSample(samples, rays, PNUM_SAMPLE)
    axis = plt.axes(projection='3d')
    points = output.cpu()
    axis.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2])
    # axis.legend()
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
    torch.set_default_tensor_type(torch.FloatTensor)
    samples = generateTestSamples(4096, 64)
    invSamplerTest(samples)
    # sortTest()
