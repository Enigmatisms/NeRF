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

        freq_bands = 2.0**torch.arange(self.max_freq + 1)

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

FLEVEL = 4

if __name__ == "__main__":
    emb = Embedder(FLEVEL)
    zeros = torch.normal(0, 2, (8192, 3)).float().cuda()
    res1 = emb.embed(zeros)
    print("")
    output = torch.zeros(zeros.shape[0], (FLEVEL + 1) * 6).float().cuda()
    start_time = time()
    encoding(zeros, output, FLEVEL + 1, False)
    end_time = time()
    print("Time consumption: %.6lf s"%(end_time - start_time))
    diff_mat = res1 - output
    print("Difference:", diff_mat.norm())
    # print(output)
