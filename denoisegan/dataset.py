import os
import math
import random
import concurrent.futures
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_all_filenames(data_path, keep_list=None):
    noisy_dir, clean_dir = None, None
    for noisy_name in ['bad', 'input', 'low', 'degraded']:
        path = os.path.join(data_path, noisy_name)
        if os.path.exists(path):
            noisy_dir = path
            break
    for clean_name in ['good', 'target', 'high', 'gt', 'ground_truth']:
        path = os.path.join(data_path, clean_name)
        if os.path.exists(path):
            clean_dir = path
            break
    if noisy_dir is None or clean_dir is None:
        raise ValueError(f"Could not find noisy/clean folders in {data_path}.")
    noisy_files = set(os.listdir(noisy_dir))
    clean_files = set(os.listdir(clean_dir))
    names = noisy_files & clean_files
    if keep_list is not None:
        with open(keep_list) as fh:
            keep = {ln.strip() for ln in fh if ln.strip()}
        before = len(names)
        names = names & keep
        print(f"[dataset] keep-list {keep_list}: {len(names)}/{before} pairs retained "
              f"({before - len(names)} contaminated targets excluded)")
    return sorted(names), noisy_dir, clean_dir


def _circular_lowpass_kernel(cutoff, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing='ij',
    )
    r = torch.sqrt(x * x + y * y)
    arg = cutoff * r
    kernel = torch.where(
        r > 0,
        cutoff * _bessel_j1(arg) / (2.0 * math.pi * r + 1e-8),
        torch.tensor(cutoff * cutoff / (4.0 * math.pi)),
    )
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def _bessel_j1(x):
    out = torch.zeros_like(x)
    small = x.abs() < 8.0
    xs = x[small]
    y = (xs / 8.0) ** 2
    num = xs * (
        72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))))
    )
    den = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))))
    out[small] = num / den

    big = ~small
    if big.any():
        xb = x[big].abs()
        z = 8.0 / xb
        y2 = z * z
        p = 1.0 + y2 * (0.00183105 + y2 * (-0.00003516396 + y2 * (0.0000002457520 + y2 * (-0.000000024062))))
        q = 0.04687499995 + y2 * (-0.00200269 + y2 * (0.00008449199 + y2 * (-0.00000882529)))
        xx = xb - 2.356194491
        amp = torch.sqrt(0.636619772 / xb)
        out[big] = amp * (torch.cos(xx) * p - z * torch.sin(xx) * q) * torch.sign(x[big])
    return out


def random_iso_kernel(kernel_size):
    sigma = random.uniform(0.2, 3.0)
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = g[:, None] * g[None, :]
    k = k / k.sum()
    return k


def random_aniso_kernel(kernel_size):
    sx = random.uniform(0.2, 3.0)
    sy = random.uniform(0.2, 3.0)
    theta = random.uniform(0, math.pi)
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xr = cos_t * xx + sin_t * yy
    yr = -sin_t * xx + cos_t * yy
    g = torch.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    k = g / g.sum()
    return k


def random_blur_kernel():
    ks = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    kind = random.random()
    if kind < 0.4:
        return random_iso_kernel(ks)
    elif kind < 0.85:
        return random_aniso_kernel(ks)
    else:
        cutoff = random.uniform(math.pi / 3.0, math.pi)
        return _circular_lowpass_kernel(cutoff, ks)


def apply_kernel(tensor, kernel):
    ks = kernel.shape[-1]
    pad = ks // 2
    c = tensor.shape[0]
    kernel = kernel.to(tensor.device).to(tensor.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(c, 1, ks, ks)
    x = tensor.unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x, kernel, groups=c).squeeze(0)


class PairedRobustDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, filenames, image_size=256, augment=True, in_memory=False):
        self.image_size = image_size
        self.augment = augment
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.filenames = filenames
        self.in_memory = in_memory
        self.noisy_data = []
        self.clean_data = []

        if self.in_memory:
            self.noisy_data = [None] * len(self.filenames)
            self.clean_data = [None] * len(self.filenames)

            def load_pair(idx):
                fname = self.filenames[idx]
                n_path = os.path.join(self.noisy_dir, fname)
                c_path = os.path.join(self.clean_dir, fname)
                n_img = cv2.imread(n_path, cv2.IMREAD_COLOR)
                c_img = cv2.imread(c_path, cv2.IMREAD_COLOR)
                n_img = torch.from_numpy(cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                c_img = torch.from_numpy(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                return idx, n_img, c_img

            max_workers = min(32, (os.cpu_count() or 1) * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(load_pair, i) for i in range(len(self.filenames))]
                pbar = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(self.filenames),
                    desc=f"Loading RAM (Threads: {max_workers})",
                    leave=False,
                    dynamic_ncols=True,
                )
                for future in pbar:
                    idx, n_img, c_img = future.result()
                    self.noisy_data[idx] = n_img
                    self.clean_data[idx] = c_img

    def __len__(self):
        return len(self.filenames)

    def _add_gaussian_noise(self, tensor, sigma_range=(1.0, 30.0)):
        sigma = random.uniform(*sigma_range) / 255.0
        if random.random() < 0.4:
            noise = torch.randn(1, tensor.shape[1], tensor.shape[2]) * sigma
            noise = noise.expand_as(tensor)
        else:
            noise = torch.randn_like(tensor) * sigma
        return tensor + noise

    def _add_poisson_noise(self, tensor, scale_range=(0.05, 3.0)):
        scale = random.uniform(*scale_range)
        gray = random.random() < 0.4
        vals = tensor.clamp(0.0, 1.0)
        if gray:
            mono = (0.299 * vals[0] + 0.587 * vals[1] + 0.114 * vals[2]).unsqueeze(0)
            noisy = torch.poisson(mono * 255.0 * scale) / (255.0 * scale)
            return tensor + (noisy.expand_as(tensor) - mono.expand_as(tensor))
        else:
            noisy = torch.poisson(vals * 255.0 * scale) / (255.0 * scale)
            return tensor + (noisy - vals)

    def _add_speckle_noise(self, tensor, sigma_range=(0.01, 0.06)):
        sigma = random.uniform(*sigma_range)
        return tensor + tensor * torch.randn_like(tensor) * sigma

    def _add_salt_pepper(self, tensor, density_range=(0.0005, 0.02)):
        d = random.uniform(*density_range)
        mask = torch.rand_like(tensor[0:1])
        salt = (mask > 1.0 - d / 2).float()
        pepper = (mask < d / 2).float()
        return tensor * (1.0 - salt - pepper) + salt

    def _add_film_grain(self, tensor, sigma_range=(0.005, 0.05)):
        sigma = random.uniform(*sigma_range)
        lum = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        mask = torch.exp(-((lum - 0.5) ** 2) / 0.08)
        grain = torch.randn(1, tensor.shape[1], tensor.shape[2]) * sigma
        return tensor + grain * mask.unsqueeze(0)

    def _add_banding(self, tensor):
        c, h, w = tensor.shape
        if random.random() < 0.5:
            freq = random.uniform(0.5, 3.0)
            coords = torch.arange(h, dtype=torch.float32).view(-1, 1) / h
        else:
            freq = random.uniform(0.5, 3.0)
            coords = torch.arange(w, dtype=torch.float32).view(1, -1) / w
        band = torch.sin(coords * freq * 2.0 * math.pi)
        strength = random.uniform(0.003, 0.05)
        return tensor + band.unsqueeze(0) * strength

    def _add_chromatic_aberration(self, tensor):
        shift = random.randint(1, 3)
        dy, dx = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        r = torch.roll(tensor[0], shifts=(dy * shift, dx * shift), dims=(0, 1))
        b = torch.roll(tensor[2], shifts=(-dy * shift, -dx * shift), dims=(0, 1))
        return torch.stack([r, tensor[1], b], dim=0)

    def _color_quantize(self, tensor):
        levels = random.choice([4, 6, 8, 12, 16])
        dither = torch.randn_like(tensor) * (1.0 / (levels * 2))
        return torch.round((tensor + dither) * (levels - 1)) / (levels - 1)

    def _color_jitter(self, tensor):
        if random.random() < 0.7:
            tensor = TF.adjust_hue(tensor.clamp(0, 1), random.uniform(-0.05, 0.05))
        if random.random() < 0.7:
            tensor = TF.adjust_saturation(tensor.clamp(0, 1), random.uniform(0.8, 1.2))
        return tensor

    def _brightness_contrast(self, tensor):
        b = random.uniform(-0.05, 0.05)
        c = random.uniform(0.85, 1.15)
        m = tensor.mean()
        return (tensor - m) * c + m + b

    def _apply_sharpen(self, tensor):
        radius = random.uniform(0.8, 2.5)
        amount = random.uniform(0.5, 2.0)
        ks = max(3, int(radius * 3) | 1)
        blurred = TF.gaussian_blur(tensor.clamp(0, 1), [ks, ks], [radius, radius])
        return tensor + (tensor - blurred) * amount

    def _channel_shuffle(self, tensor):
        order = [0, 1, 2]
        random.shuffle(order)
        return tensor[order]

    def _add_chroma_subsample(self, tensor):
        c, h, w = tensor.shape
        if h < 4 or w < 4:
            return tensor
        r, g, b = tensor[0], tensor[1], tensor[2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        U = -0.14713 * r - 0.28886 * g + 0.436 * b
        V = 0.615 * r - 0.51499 * g - 0.10001 * b
        factor = random.choice([2, 4])
        UV = torch.stack([U, V], dim=0).unsqueeze(0)
        UV_small = F.interpolate(UV, scale_factor=1.0 / factor,
                                 mode='bilinear', align_corners=False, antialias=True)
        mode = random.choice(['nearest', 'bilinear'])
        if mode == 'nearest':
            UV_big = F.interpolate(UV_small, size=(h, w), mode='nearest')
        else:
            UV_big = F.interpolate(UV_small, size=(h, w), mode='bilinear', align_corners=False)
        U2, V2 = UV_big[0, 0], UV_big[0, 1]
        r2 = Y + 1.13983 * V2
        g2 = Y - 0.39465 * U2 - 0.58060 * V2
        b2 = Y + 2.03211 * U2
        return torch.stack([r2, g2, b2], dim=0)

    def _add_macroblock_edges(self, tensor):
        c, h, w = tensor.shape
        block = random.choice([8, 16])
        strength = random.uniform(0.005, 0.035)
        mask = torch.zeros(h, w, dtype=tensor.dtype)
        for off in (0,):
            mask[off::block, :] = 1.0
            mask[:, off::block] = 1.0
        mask = mask * torch.rand(h, w, dtype=tensor.dtype)
        bias = (torch.randn(c, 1, 1, dtype=tensor.dtype) * strength)
        return tensor + bias * mask.unsqueeze(0)

    def _add_mosquito_noise(self, tensor):
        c, h, w = tensor.shape
        if h < 3 or w < 3:
            return tensor
        gray = (0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]).unsqueeze(0).unsqueeze(0)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 4.0
        ky = kx.transpose(-1, -2).contiguous()
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
        edge = edge / (edge.amax() + 1e-8)
        edge = F.max_pool2d(edge, 5, stride=1, padding=2)
        edge = TF.gaussian_blur(edge.squeeze(0), [5, 5], [1.2, 1.2]).squeeze(0)
        sigma = random.uniform(0.01, 0.06)
        noise = torch.randn_like(tensor) * sigma * edge.unsqueeze(0)
        return tensor + noise

    def _add_halftone(self, tensor):
        c, h, w = tensor.shape
        angle = random.uniform(0.0, math.pi)
        freq = random.uniform(0.18, 0.55)
        y = torch.arange(h, dtype=torch.float32).view(-1, 1)
        x = torch.arange(w, dtype=torch.float32).view(1, -1)
        pat = torch.sin((x * math.cos(angle) + y * math.sin(angle)) * freq * 2.0 * math.pi)
        if random.random() < 0.5:
            pat = (pat > 0).float() * 2.0 - 1.0
        strength = random.uniform(0.006, 0.03)
        return tensor + pat.unsqueeze(0) * strength

    def _add_dither(self, tensor):
        levels = random.choice([3, 4, 6, 8])
        noise = (torch.rand_like(tensor) - 0.5) * (1.0 / levels)
        out = torch.round((tensor + noise) * (levels - 1)) / (levels - 1)
        if random.random() < 0.4:
            out = out + (torch.rand_like(out) - 0.5) * (0.5 / levels)
        return out

    def _apply_resize(self, tensor, target_h, target_w):
        modes = [TF.InterpolationMode.BILINEAR, TF.InterpolationMode.BICUBIC, TF.InterpolationMode.NEAREST]
        mode = random.choice(modes)
        antialias = mode != TF.InterpolationMode.NEAREST
        return TF.resize(tensor, [target_h, target_w], interpolation=mode, antialias=antialias)

    def _apply_jpeg(self, tensor, q_range):
        q = random.randint(*q_range)
        u8 = (tensor.clamp(0, 1) * 255.0).to(torch.uint8)
        try:
            bts = tv_io.encode_jpeg(u8, quality=q)
            dec = tv_io.decode_jpeg(bts)
            return dec.to(torch.float32) / 255.0
        except Exception:
            return tensor

    def _first_order(self, x):
        if random.random() < 1.0:
            x = apply_kernel(x, random_blur_kernel())
        h, w = x.shape[1], x.shape[2]
        scale = random.uniform(0.5, 1.2)
        if random.random() < 0.5:
            scale = max(0.4, min(scale, 0.95))
        new_h = max(8, int(h * scale))
        new_w = max(8, int(w * scale))
        x = self._apply_resize(x, new_h, new_w)
        x = x.clamp(0.0, 1.0)
        which = random.random()
        if which < 0.5:
            x = self._add_gaussian_noise(x, (1.0, 60.0))
        elif which < 0.8:
            x = self._add_poisson_noise(x, (0.05, 4.0))
        else:
            x = self._add_speckle_noise(x, (0.01, 0.10))
        x = x.clamp(0.0, 1.0)
        if random.random() < 0.85:
            x = self._apply_jpeg(x, (30, 95))
        return x.clamp(0.0, 1.0)

    def _second_order(self, x, target_h, target_w):
        if random.random() < 0.8:
            x = apply_kernel(x, random_blur_kernel())
        scale = random.uniform(0.4, 1.1)
        new_h = max(8, int(target_h * scale))
        new_w = max(8, int(target_w * scale))
        x = self._apply_resize(x, new_h, new_w)
        x = x.clamp(0.0, 1.0)
        which = random.random()
        if which < 0.45:
            x = self._add_gaussian_noise(x, (1.0, 50.0))
        elif which < 0.7:
            x = self._add_poisson_noise(x, (0.05, 3.5))
        else:
            x = self._add_speckle_noise(x, (0.005, 0.09))
        x = x.clamp(0.0, 1.0)
        if random.random() < 0.5:
            x = apply_kernel(x, _circular_lowpass_kernel(
                random.uniform(math.pi / 3.0, math.pi),
                random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
            ))
            x = x.clamp(0.0, 1.0)
            x = self._apply_resize(x, target_h, target_w)
            if random.random() < 0.85:
                x = self._apply_jpeg(x.clamp(0, 1), (30, 95))
        else:
            x = self._apply_resize(x, target_h, target_w)
            if random.random() < 0.85:
                x = self._apply_jpeg(x.clamp(0, 1), (30, 95))
            x = apply_kernel(x.clamp(0, 1), _circular_lowpass_kernel(
                random.uniform(math.pi / 3.0, math.pi),
                random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
            ))
        return x.clamp(0.0, 1.0)

    def _stylistic_extras(self, x):
        if random.random() < 0.30:
            x = self._add_salt_pepper(x).clamp(0, 1)
        if random.random() < 0.30:
            x = self._add_film_grain(x).clamp(0, 1)
        if random.random() < 0.25:
            x = self._add_banding(x).clamp(0, 1)
        if random.random() < 0.25:
            x = self._add_chromatic_aberration(x).clamp(0, 1)
        if random.random() < 0.25:
            x = self._color_quantize(x).clamp(0, 1)
        if random.random() < 0.25:
            x = self._color_jitter(x).clamp(0, 1)
        if random.random() < 0.25:
            x = self._brightness_contrast(x).clamp(0, 1)
        if random.random() < 0.20:
            x = self._apply_sharpen(x).clamp(0, 1)
        if random.random() < 0.10:
            x = self._channel_shuffle(x).clamp(0, 1)
        if random.random() < 0.45:
            x = self._add_chroma_subsample(x).clamp(0, 1)
        if random.random() < 0.30:
            x = self._add_macroblock_edges(x).clamp(0, 1)
        if random.random() < 0.30:
            x = self._add_mosquito_noise(x).clamp(0, 1)
        if random.random() < 0.10:
            x = self._add_halftone(x).clamp(0, 1)
        if random.random() < 0.15:
            x = self._add_dither(x).clamp(0, 1)
        return x

    def _final_block_jpeg(self, x, q_range=(8, 40)):
        # Harsh JPEG as the genuinely LAST op, at target resolution, so the 8x8
        # block grid stays crisp & aligned — matches how a deployment .jpg looks.
        # Optional double-compression reproduces compounded blocking from re-saves.
        x = self._apply_jpeg(x.clamp(0.0, 1.0), q_range)
        if random.random() < 0.5:
            x = self._apply_jpeg(x.clamp(0.0, 1.0), q_range)
        return x

    def _degrade(self, clean01):
        h, w = clean01.shape[1], clean01.shape[2]
        x = clean01.clone()
        x = self._first_order(x)
        x = self._second_order(x, h, w)
        x = self._stylistic_extras(x)
        # Sometimes finish with crisp, grid-aligned block artifacts (no resize/blur
        # after) so the model learns to remove the harsh JPEG it actually meets.
        if random.random() < 0.35:
            x = self._final_block_jpeg(x).clamp(0.0, 1.0)
        if random.random() < 0.05:
            return clean01.clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _augment_residual(self, residual):
        """Stretch the real-noise residual across a wide distribution while keeping
        its character (color correlation, spatial structure). The model then sees
        the same noise family at many intensities/scales/shapes -> robust to the
        'similar but different' noise it meets at deployment."""
        r = residual
        c, h, w = r.shape

        # intensity: dimmer / brighter noise (always applied)
        r = r * random.uniform(0.5, 1.8)

        # per-channel rebalance: luma vs chroma noise mix
        if random.random() < 0.4:
            cs = torch.empty(c, 1, 1).uniform_(0.6, 1.4)
            r = r * cs

        # spatial stretch / resize: coarser or finer grain (anisotropic allowed)
        if random.random() < 0.4:
            sh = random.uniform(0.6, 1.5)
            sw = random.uniform(0.6, 1.5)
            nh, nw = max(8, int(h * sh)), max(8, int(w * sw))
            r = F.interpolate(r.unsqueeze(0), size=(nh, nw),
                              mode='bilinear', align_corners=False)
            r = F.interpolate(r, size=(h, w),
                              mode='bilinear', align_corners=False).squeeze(0)

        # soften (coarse grain) or sharpen (harsh grain) the noise
        if random.random() < 0.35:
            if random.random() < 0.5:
                s = random.uniform(0.4, 1.2)
                r = TF.gaussian_blur(r, [3, 3], [s, s])
            else:
                blur = TF.gaussian_blur(r, [3, 3], [1.0, 1.0])
                r = r + (r - blur) * random.uniform(0.5, 1.5)

        # contrast/gamma reshape of the noise amplitude distribution
        if random.random() < 0.3:
            g = random.uniform(0.7, 1.4)
            r = torch.sign(r) * (r.abs().clamp_min(1e-6) ** g)

        # geometric decorrelation: break noise<->content alignment
        if random.random() < 0.3:
            if random.random() < 0.5:
                r = r.flip(-1)
            if random.random() < 0.5:
                r = r.flip(-2)
            k = random.choice([0, 1, 2, 3])
            if k:
                r = torch.rot90(r, k=k, dims=[-2, -1])

        return r

    def _cutblur(self, noisy, clean, p=0.5, alpha_range=(0.2, 0.7)):
        if random.random() > p:
            return noisy, clean
        c, h, w = clean.shape
        ratio = random.uniform(*alpha_range)
        cut_h = max(1, int(h * ratio))
        cut_w = max(1, int(w * ratio))
        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)
        out = noisy.clone()
        if random.random() < 0.5:
            out[:, cy:cy + cut_h, cx:cx + cut_w] = clean[:, cy:cy + cut_h, cx:cx + cut_w]
        else:
            tmp = clean.clone()
            tmp[:, cy:cy + cut_h, cx:cx + cut_w] = noisy[:, cy:cy + cut_h, cx:cx + cut_w]
            out = tmp
        return out, clean

    def _sync_crop_aug(self, noisy, clean):
        c, h, w = clean.shape
        if w < self.image_size or h < self.image_size:
            pad_w = max(0, self.image_size - w)
            pad_h = max(0, self.image_size - h)
            noisy = F.pad(noisy.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            clean = F.pad(clean.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            h, w = clean.shape[1:]
        i = random.randint(0, h - self.image_size)
        j = random.randint(0, w - self.image_size)
        noisy = noisy[:, i:i + self.image_size, j:j + self.image_size]
        clean = clean[:, i:i + self.image_size, j:j + self.image_size]
        if self.augment:
            if random.random() > 0.5:
                noisy = noisy.flip(-1)
                clean = clean.flip(-1)
            if random.random() > 0.5:
                noisy = noisy.flip(-2)
                clean = clean.flip(-2)
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                noisy = torch.rot90(noisy, k=k, dims=[-2, -1])
                clean = torch.rot90(clean, k=k, dims=[-2, -1])
        return noisy, clean

    def __getitem__(self, idx):
        if self.in_memory:
            noisy_raw = self.noisy_data[idx]
            clean_raw = self.clean_data[idx]
        else:
            fname = self.filenames[idx]
            n_raw = cv2.imread(os.path.join(self.noisy_dir, fname), cv2.IMREAD_COLOR)
            c_raw = cv2.imread(os.path.join(self.clean_dir, fname), cv2.IMREAD_COLOR)
            noisy_raw = torch.from_numpy(cv2.cvtColor(n_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            clean_raw = torch.from_numpy(cv2.cvtColor(c_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        noisy_raw, clean_raw = self._sync_crop_aug(noisy_raw, clean_raw)
        noisy01 = noisy_raw.to(torch.float32) / 255.0
        clean01 = clean_raw.to(torch.float32) / 255.0

        if self.augment:
            roll = random.random()
            if roll < 0.15:
                noisy01 = self._degrade(clean01)
            elif roll < 0.80:
                residual = self._augment_residual(noisy01 - clean01)
                noisy01 = (clean01 + residual).clamp(0.0, 1.0)
            # roll >= 0.80: keep the pure real pair as an anchor
            noisy01, clean01 = self._cutblur(noisy01, clean01, p=0.5)

        noisy = noisy01.clamp(0, 1) * 2.0 - 1.0
        clean = clean01.clamp(0, 1) * 2.0 - 1.0
        return noisy, clean, idx


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)


def build_dataloader(data_path, batch_size, image_size=256, augment=True,
                     num_workers=4, in_memory=False, shuffle=True, keep_list=None):
    files, noisy_dir, clean_dir = get_all_filenames(data_path, keep_list=keep_list)
    dataset = PairedRobustDataset(
        noisy_dir, clean_dir, files,
        image_size=image_size, augment=augment, in_memory=in_memory,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
    return loader


class CleanOnlyDataset(Dataset):
    def __init__(self, clean_dir, filenames, image_size=256, augment=True):
        self.clean_dir = clean_dir
        self.filenames = filenames
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        c_raw = cv2.imread(os.path.join(self.clean_dir, fname), cv2.IMREAD_COLOR)
        clean = torch.from_numpy(cv2.cvtColor(c_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        c, h, w = clean.shape
        if w < self.image_size or h < self.image_size:
            pad_w = max(0, self.image_size - w)
            pad_h = max(0, self.image_size - h)
            clean = F.pad(clean.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            h, w = clean.shape[1:]
        i = random.randint(0, h - self.image_size)
        j = random.randint(0, w - self.image_size)
        clean = clean[:, i:i + self.image_size, j:j + self.image_size]
        if self.augment:
            if random.random() > 0.5:
                clean = clean.flip(-1)
            if random.random() > 0.5:
                clean = clean.flip(-2)
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                clean = torch.rot90(clean, k=k, dims=[-2, -1])
        clean = clean.to(torch.float32) / 255.0
        clean = clean * 2.0 - 1.0
        return clean, idx


def build_clean_loader(data_path, batch_size, image_size=256, augment=True,
                       num_workers=4, shuffle=True, keep_list=None):
    files, _, clean_dir = get_all_filenames(data_path, keep_list=keep_list)
    dataset = CleanOnlyDataset(clean_dir, files, image_size=image_size, augment=augment)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
    return loader


if __name__ == '__main__':
    loader = build_dataloader("/home/algis/Desktop/data/train", batch_size=2, num_workers=0)
    noisy, clean, _ = next(iter(loader))
    print(f"noisy: {tuple(noisy.shape)}  range [{noisy.min().item():.3f},{noisy.max().item():.3f}]")
    print(f"clean: {tuple(clean.shape)}  range [{clean.min().item():.3f},{clean.max().item():.3f}]")
