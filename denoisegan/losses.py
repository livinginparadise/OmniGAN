import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, ave_spectrum=False):
        super().__init__()
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum

    def _fft(self, x):
        return torch.fft.fft2(x, norm='ortho')

    def forward(self, pred, target):
        fp = self._fft(pred)
        ft = self._fft(target)
        diff = fp - ft
        mag2 = diff.real ** 2 + diff.imag ** 2
        with torch.no_grad():
            w = mag2.detach() ** self.alpha
            w = w / (w.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            w = w.clamp(0.0, 1.0)
        return (w * mag2).mean()


def rgb_to_lab(rgb01):
    eps = 6.0 / 29.0
    rgb = rgb01.clamp(0.0, 1.0)
    mask = (rgb > 0.04045).float()
    rgb_lin = mask * ((rgb + 0.055) / 1.055).clamp(min=1e-12) ** 2.4 + (1 - mask) * (rgb / 12.92)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=rgb.device, dtype=rgb.dtype)
    rgb_flat = rgb_lin.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = rgb_flat @ M.T
    ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz = xyz / ref
    f = torch.where(xyz > eps ** 3, xyz.clamp(min=1e-12) ** (1.0 / 3.0), xyz / (3.0 * eps * eps) + 4.0 / 29.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    lab = torch.stack([L, a, b], dim=-1).reshape(rgb.shape[0], rgb.shape[2], rgb.shape[3], 3)
    return lab.permute(0, 3, 1, 2)


def lab_chroma_loss(pred01, target01):
    lab_p = rgb_to_lab(pred01)
    lab_t = rgb_to_lab(target01)
    return (lab_p[:, 1:] - lab_t[:, 1:]).abs().mean()


class GradVarianceLoss(nn.Module):
    def __init__(self, patch_size=8):
        super().__init__()
        self.patch = patch_size
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 4.0
        ky = kx.t().contiguous()
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def _grad_var(self, x):
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        var = F.avg_pool2d(mag * mag, self.patch) - F.avg_pool2d(mag, self.patch) ** 2
        return var

    def forward(self, pred, target):
        return (self._grad_var(pred) - self._grad_var(target)).abs().mean()


class VGGPerceptual(nn.Module):
    def __init__(self, layer_indices=(2, 7, 16, 25, 34), weights=(0.1, 0.1, 1.0, 1.0, 1.0)):
        super().__init__()
        try:
            vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = tvm.vgg19(pretrained=True).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.idxs = layer_indices
        self.weights = weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x01):
        return (x01 - self.mean) / self.std

    def _features(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.idxs:
                feats.append(x)
            if i >= max(self.idxs):
                break
        return feats

    def forward(self, pred01, target01):
        p = self._norm(pred01.clamp(0, 1))
        t = self._norm(target01.clamp(0, 1))
        fp = self._features(p)
        with torch.no_grad():
            ft = self._features(t)
        loss = 0.0
        for w, a, b in zip(self.weights, fp, ft):
            loss = loss + w * F.l1_loss(a, b)
        return loss


def softplus_loss(x):
    return F.softplus(x).mean()


def rpgan_d_loss(d_real, d_fake):
    return F.softplus(d_fake - d_real).mean()


def rpgan_g_loss(d_real, d_fake):
    return F.softplus(d_real - d_fake).mean()


def r1_penalty(d_real_outputs, real_inputs):
    grads = torch.autograd.grad(
        outputs=[o.sum() for o in d_real_outputs],
        inputs=real_inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grads.pow(2).flatten(1).sum(1).mean()


def r2_penalty(d_fake_outputs, fake_inputs):
    grads = torch.autograd.grad(
        outputs=[o.sum() for o in d_fake_outputs],
        inputs=fake_inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grads.pow(2).flatten(1).sum(1).mean()


def aggregate_unet_dino(unet_logit, dino_logits):
    parts = [unet_logit.mean(dim=(1, 2, 3))]
    for d in dino_logits:
        parts.append(d.mean(dim=(1, 2, 3)))
    return torch.stack(parts, dim=0).mean(dim=0)
