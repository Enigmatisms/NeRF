# NeRF

---

### 8.24 Update

It turns out that PSNR calculation is to blame. My loss function is not MSE (PSNR is calculated using MSE), it is SoftL1 (`sqrt(e^2 + ε)`), which is bigger than expected. Therefore "PSNR" is low (around 19.). Currently, PSNR of the model's (trained for only 7 hours) is around 28.5.

### 8.19 Update

CVPR 2022 best student honorable mention: [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://arxiv.org/abs/2112.03907) is implemented in this repo. This repo can turn Ref NeRF part on/off with one flag: `-t`. Ref NeRF is implemented upon (proposal network + NeRF) framework. Currently, the result is not so satisfying as I expected. This may be caused by insufficient time for training (limited training device, 6GB global memory, can only use up to batch size 2^9 (rays), while the paper uses 2^14).

Ref NeRF is built together with the proposal network proposed in mip NeRF 360, making the model harder to train. The reason behind this is (I suppose) normal prediction in Ref NeRF uses a "back-culling strategy" (orientation loss), which prevents foggy artifacts behind semi-transparent surface. This strategy will both concentrate density and have some strange (magical) effect on the gradients of proposal network. I experimented with original NeRF framework, and things seem to work out fine, with no mosaic-like noise. I made some modification to the current proposal-based framework, which makes things working. To test Ref NeRF, run the following code:

```python
python3 ./train.py -s -t -u  --grad_clip -0.01 --dataset_name helmet --render_depth --render_normal --prop_normal
```

   Note that:

- Gradient clip = -0.01 means no grad clipping (negative numbers indicates none)
- `-s`: use amp (default: nvidia APEX O1 level). `-t`: use ref nerf. `-u`: use srgb prediction.

---

### 7.12 update

Better code structure for running and debugging. The original NeRF (in previous versions, the executable of which is named `train.py`) is no longer supported. Instead, NeRF with mip 360 (proposal network) and auto-mixed precision is currently maintained. Run the following code to find out:

```shell
python ./train.py -w -s --opt_level O1 --dataset_name ship
```

`-w` indicates white background. `-s` and `--opt_level` set up amp env.

---

Reproduction results:

|                    Lego trained for 2.5h                     |                   Hotdog trained for 30min                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![ezgif-4-fe2ea0a6a2](https://user-images.githubusercontent.com/46109954/173533863-499b04bf-4242-41a5-98d4-6fc81ee412b3.gif) | ![ezgif-4-402a3412da](https://user-images.githubusercontent.com/46109954/173536624-beb64cb6-e151-4267-94c9-183793951011.gif) |

---

### 4.25 Update

To boost the resulting quality, two more approaches are incorporated in this repo:

The idea from [ICCV 2021: Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/), using conical frustum instead of pure points in sampling, trying to solve the problem of aliasing under multi-resolution settings. Removed the use of coarse network.

The idea from [CVPR 2022: Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://paperswithcode.com/paper/mip-nerf-360-unbounded-anti-aliased-neural/review/), which currently has no open-sourced code:

- Using shallower proposal network distilled from NeRF MLP weight rendering, in order to reduce the evaluation time for coarse samples.
- Using weight regularizer which aims to concentrate computed weight in a smaller region, make it more "delta-function-like". The final output is more accurate. Here are the result and a small comparison:

The second blog about the latest re-implementation is to be posted.

|   Spherical views (400 * 400)    |          Comparison (no regularizer - left)           |   Proposal network distillation   |
| :------------------------------: | :---------------------------------------------------: | :-------------------------------: |
| <img src="assets/sphere.gif"  /> | <img src="assets/comparison.png" style="zoom:80%;" /> | ![](assets/proposal_dist_076.png) |



---

### Original Implementation

Re-implementation of ECCV 2020 NeRF with PyTorch:

- [Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf)
- More information ? Refer to my blog about this reimplementation: [enigmatisms.github.io/NeRF论文复现](https://enigmatisms.github.io/2022/03/27/NeRF%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/)

Quick overview for a 10-hour training results (single view rendering, APEX O2 optimized) in nerf-blender-synthetic dataset (drums):

<img src="/home/stn/dl/NeRF/assets/dynamic.gif" style="zoom:50%;" />

This repo contains:

- CUDA implemented functions, like inverse transform sampling, image sampler, positional encoding module, etc.
- A simpler version (in terms of readability) of NeRF (comparing with offcial NeRF implementation which is written in TensorFlow)
- Simple APEX accelerated version of NeRF 