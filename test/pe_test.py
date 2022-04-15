#-*-coding:utf-8-*-

"""
    @author Enigmatisms @date 2022.3.19
    Test program for CUDA positional encoding
"""

import torch
from nerf_helper import encoding
from time import time
from instances import *

# This Embedder comes from the official implementation
class Embedder:
    def __init__(self, max_freq):
        self.max_freq = max_freq
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = 3
        out_dim = 0

        freq_bands = 2.0**torch.arange(self.max_freq)

        for freq in freq_bands:
            for p_fn in (torch.sin, torch.cos):
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        start_time = time()
        res = torch.concat([fn(inputs) for fn in self.embed_fns], -1)
        end_time = time()
        print("Time consumption: %.6lf s"%(end_time - start_time))
        return res

FLEVEL = 10

def encode_position(x):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(FLEVEL):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn(2.0 ** i * x))
    return torch.concat(positions, dim=-1)

def positional_encoding(x:torch.Tensor, freq_level:int) -> torch.Tensor:
    result = []
    ray_num, point_num = x.shape[0], x.shape[1]
    for fid in range(freq_level):
        freq = 2. ** fid
        for func in (torch.sin, torch.cos):
            result.append(func(freq * x.unsqueeze(-2)))
    encoded = torch.cat(result, dim = -2).view(ray_num, point_num, -1)
    return encoded

if __name__ == "__main__":
    emb = Embedder(FLEVEL)
    zeros = torch.normal(0, 2, (2, 3, 3)).float().cuda()
    res1 = emb.embed(zeros)
    print("Official:")
    print(res1, res1.shape)
    start_time = time()
    res2 = positional_encoding(zeros, FLEVEL)
    end_time = time()
    print("Python:")
    print(res2, res2.shape)
    print("Time consumption: %.6lf s"%(end_time - start_time))
    # output = torch.zeros(zeros.shape[0], FLEVEL * 6).float().cuda()
    # start_time = time()
    # encoding(zeros, output, FLEVEL, False)
    # end_time = time()
    # print("Time consumption: %.6lf s"%(end_time - start_time))
    # print("CUDA:")
    # print(output, output.shape)
    diff_mat = res1 - res2
    print("Difference:", diff_mat.norm())
    # print(output)
