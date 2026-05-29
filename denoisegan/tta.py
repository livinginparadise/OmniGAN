"""Zero-Shot Noise2Noise (ZS-N2N) self-supervised pieces for test-time adaptation.

Lets a denoising pipeline adapt to a *single* user image whose noise is unknown,
using only that noisy image (no clean reference). From the noisy image we build
two diagonally-downsampled views that share clean content but carry independent
noise; matching one to the other (Noise2Noise) forces the pipeline to learn the
clean signal. A consistency term ties denoise-then-downsample to downsample-then-
denoise. Ref: Mansour & Heckel, "Zero-Shot Noise2Noise", CVPR 2023.
"""
import torch
import torch.nn.functional as F


def make_pair_kernels(device, dtype=torch.float32):
    # two fixed 2x2 diagonal averaging downsamplers (stride 2)
    k1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype)
    k2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype)
    return k1, k2


def pair_downsample(x, k1, k2):
    c = x.shape[1]
    f1 = k1.to(x.dtype).repeat(c, 1, 1, 1)
    f2 = k2.to(x.dtype).repeat(c, 1, 1, 1)
    a = F.conv2d(x, f1, stride=2, groups=c)
    b = F.conv2d(x, f2, stride=2, groups=c)
    return a, b


def zsn2n_loss(pipeline, y, k1, k2):
    """pipeline(x) is the denoiser being adapted, e.g. D(T(x)).
    Returns (total, residual, consistency)."""
    d1, d2 = pair_downsample(y, k1, k2)
    p1 = pipeline(d1)
    p2 = pipeline(d2)
    l_res = 0.5 * (F.mse_loss(p1, d2) + F.mse_loss(p2, d1))

    fy = pipeline(y)
    fd1, fd2 = pair_downsample(fy, k1, k2)
    l_cons = 0.5 * (F.mse_loss(p1, fd1) + F.mse_loss(p2, fd2))

    return l_res + l_cons, l_res, l_cons
