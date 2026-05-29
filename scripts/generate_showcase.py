import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import math
import random
import argparse
import contextlib

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from denoisegan.models import DenoiseGenerator, NoiseTranslator

IMG_EXT = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp', '*.tif', '*.tiff')


# ----------------------------- io / tensors -----------------------------

def all_images(path):
    if path is None:
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(path, e))
    return sorted(files)


def load_pm1(path, device, max_side=1024):
    img = Image.open(path).convert('RGB')
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0


def pm1_to_u8(t):
    x = ((t.clamp(-1, 1) + 1) * 0.5 * 255.0).round().clamp(0, 255)
    return x[0].permute(1, 2, 0).byte().cpu().numpy()


def amp_ctx(device, amp):
    if amp == 'bf16':
        return autocast(device.type, dtype=torch.bfloat16)
    if amp == 'fp16':
        return autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


# ----------------------------- difficulty selection -----------------------------

def _gray256(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    return cv2.resize(g, (256, 256)).astype(np.float32)


def noise_score(path):
    """Immerkaer noise proxy: higher = noisier."""
    g = _gray256(path)
    if g is None:
        return -1.0
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return float(np.median(np.abs(cv2.filter2D(g, -1, M))))


def detail_score(path):
    """Variance of Laplacian: higher = more texture/detail (harder to preserve)."""
    g = _gray256(path)
    if g is None:
        return -1.0
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def pick_hardest(paths, metric, n):
    scored = [(metric(p), p) for p in paths]
    scored = [(s, p) for s, p in scored if s >= 0]
    scored.sort(reverse=True)
    return [p for _, p in scored[:n]]


def pick_random(paths, n):
    if not paths:
        return []
    return random.sample(paths, k=min(n, len(paths)))


# ----------------------------- noise -----------------------------

def add_noise(clean01, kind):
    x = clean01.clone()
    if kind == 'gaussian':
        x = x + torch.randn_like(x) * (25.0 / 255.0)
    elif kind == 'poisson':
        x = torch.poisson((x.clamp(0, 1) * 255.0)) / 255.0
    elif kind == 'speckle':
        x = x + x * torch.randn_like(x) * 0.12
    elif kind == 'salt_pepper':
        m = torch.rand_like(x[:, :1])
        x = torch.where(m < 0.02, torch.zeros_like(x), x)
        x = torch.where(m > 0.98, torch.ones_like(x), x)
    return x.clamp(0, 1)


# ----------------------------- inference -----------------------------

def hann2d(n, device):
    w = np.maximum(np.hanning(n).astype(np.float32), 1e-3)
    return torch.from_numpy(w[:, None] * w[None, :]).to(device)[None, None]


def tiled_denoise(model, img_pm1, tile, overlap, device, amp, translator=None):
    model.eval()
    if translator is not None:
        translator.eval()
    _, C, H, W = img_pm1.shape
    if H <= tile and W <= tile:
        with torch.no_grad(), amp_ctx(device, amp):
            t = translator(img_pm1) if translator is not None else img_pm1
            return model(t).float().clamp(-1, 1)
    overlap = max(0, min(overlap, tile - 1))
    stride = tile - overlap
    pad_h = (stride - (H - tile) % stride) % stride if H > tile else max(0, tile - H)
    pad_w = (stride - (W - tile) % stride) % stride if W > tile else max(0, tile - W)
    x = F.pad(img_pm1, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x.shape
    ys = list(range(0, Hp - tile + 1, stride)) or [0]
    xs = list(range(0, Wp - tile + 1, stride)) or [0]
    if ys[-1] + tile < Hp:
        ys.append(Hp - tile)
    if xs[-1] + tile < Wp:
        xs.append(Wp - tile)
    win = hann2d(tile, device)
    acc = torch.zeros(1, C, Hp, Wp, device=device)
    wsum = torch.zeros(1, 1, Hp, Wp, device=device)
    with torch.no_grad(), amp_ctx(device, amp):
        for y in ys:
            for xx in xs:
                t = x[:, :, y:y + tile, xx:xx + tile]
                if translator is not None:
                    t = translator(t)
                o = model(t).float()
                acc[:, :, y:y + tile, xx:xx + tile] += o * win
                wsum[:, :, y:y + tile, xx:xx + tile] += win
    return (acc / (wsum + 1e-8)).clamp(-1, 1)[:, :, :H, :W]


def psnr01(a01, b01):
    mse = F.mse_loss(a01.clamp(0, 1), b01.clamp(0, 1))
    return float(10.0 * math.log10(1.0 / max(mse.item(), 1e-12)))


def find_detail_crop(img_pm1, size):
    """Locate the most-textured size x size window (max high-frequency energy)."""
    _, _, H, W = img_pm1.shape
    size = min(size, H, W)
    g = ((img_pm1 + 1) / 2).mean(1, keepdim=True)
    hf = (g - F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode='reflect'), 3, 1)).abs()
    stride = max(1, size // 4)
    pooled = F.avg_pool2d(hf, kernel_size=size, stride=stride)
    idx = int(torch.argmax(pooled.flatten()))
    nw = pooled.shape[-1]
    y = min((idx // nw) * stride, H - size)
    x = min((idx % nw) * stride, W - size)
    return y, x, size


# ----------------------------- visuals -----------------------------

def _font(sz=16):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def captioned(u8, caption, disp_h=320, nearest=False):
    im = Image.fromarray(u8)
    w = max(1, int(im.width * disp_h / im.height))
    im = im.resize((w, disp_h), Image.NEAREST if nearest else Image.LANCZOS)
    bar = 28
    canvas = Image.new('RGB', (im.width, im.height + bar), (22, 22, 26))
    canvas.paste(im, (0, bar))
    ImageDraw.Draw(canvas).text((7, 6), caption, fill=(235, 235, 245), font=_font())
    return canvas


def row(cells, bg=(14, 14, 18), pad=6):
    h = max(c.height for c in cells)
    W = sum(c.width for c in cells) + pad * (len(cells) + 1)
    strip = Image.new('RGB', (W, h + 2 * pad), bg)
    x = pad
    for c in cells:
        strip.paste(c, (x, pad))
        x += c.width + pad
    return strip


def stack(rows, bg=(14, 14, 18)):
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows)
    panel = Image.new('RGB', (W, H), bg)
    y = 0
    for r in rows:
        panel.paste(r, ((W - r.width) // 2, y))
        y += r.height
    return panel


def diagonal_split(noisy_u8, denoised_u8):
    h, w, _ = noisy_u8.shape
    yy, xx = np.mgrid[0:h, 0:w]
    right = (xx / w) > (yy / h)
    out = np.where(right[..., None], denoised_u8, noisy_u8).astype(np.uint8)
    line = np.abs(xx / w - yy / h) < 0.0035
    out[line] = (255, 220, 60)
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    f = _font(22)
    d.text((10, h - 34), "NOISY", fill=(255, 150, 150), font=f)
    tw = d.textlength("DENOISED", font=f)
    d.text((w - tw - 12, 8), "DENOISED", fill=(150, 255, 190), font=f)
    return im


def make_gif(noisy_u8, denoised_u8, path, disp=480):
    def frame(u8, label, color):
        im = Image.fromarray(u8)
        w = max(1, int(im.width * disp / im.height))
        im = im.resize((w, disp), Image.LANCZOS).convert('RGB')
        d = ImageDraw.Draw(im)
        d.rectangle([0, im.height - 32, 168, im.height], fill=(0, 0, 0))
        d.text((10, im.height - 28), label, fill=color, font=_font(20))
        return im
    f1 = frame(noisy_u8, "NOISY", (255, 140, 140))
    f2 = frame(denoised_u8, "DENOISED", (140, 255, 180))
    f2 = f2.resize(f1.size)
    f1.save(path, save_all=True, append_images=[f2], duration=850, loop=0, optimize=True)


def fft_panel(t_pm1, caption):
    x = ((t_pm1 + 1) / 2).mean(1, keepdim=True)
    f = torch.fft.fftshift(torch.fft.fft2(x.float()))
    m = (f.abs() + 1e-6).log()
    m = ((m - m.amin()) / (m.amax() - m.amin() + 1e-8)).clamp(0, 1)
    u8 = (m[0, 0] * 255).byte().cpu().numpy()
    u8 = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)[:, :, ::-1]
    return captioned(np.ascontiguousarray(u8), caption, disp_h=256)


# ----------------------------- architecture diagram -----------------------------

def draw_architecture(path):
    NAF, ATTN, BOT, PROJ = "#cfe3f5", "#d6f0d6", "#fde2c0", "#e4e4ea"
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.8, 10.6)
    ax.axis('off')

    def box(cx, cy, w, h, lines, fc):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.2, edgecolor="#333", facecolor=fc))
        ax.text(cx, cy, "\n".join(lines), ha='center', va='center', fontsize=8.5)

    def arrow(p1, p2, dashed=False, color="#444"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=12,
                     linewidth=1.2, color=color,
                     linestyle='--' if dashed else '-',
                     connectionstyle="arc3,rad=0"))

    # ---- top: inference pipeline ----
    ax.text(6.5, 10.35, "Inference pipeline", ha='center', fontsize=11, weight='bold')
    box(1.6, 9.7, 2.4, 0.7, ["Noisy image"], "#f2f2f2")
    box(5.0, 9.7, 2.8, 0.7, ["Noise Translator", "(optional, ~0.37M, bias-free)"], NAF)
    box(8.6, 9.7, 2.6, 0.7, ["Denoiser  D", "(U-Net, ~21M)"], ATTN)
    box(11.8, 9.7, 2.0, 0.7, ["Clean image"], "#f2f2f2")
    arrow((2.8, 9.7), (3.6, 9.7))
    arrow((6.4, 9.7), (7.3, 9.7))
    arrow((9.9, 9.7), (10.8, 9.7))

    # ---- bottom: U-Net detail ----
    ax.text(6.5, 9.0, "Denoiser  (DenoiseGenerator)", ha='center', fontsize=11, weight='bold')
    ys = [7.4, 6.05, 4.7, 3.35, 2.0]
    ex, dx = 2.3, 10.7
    enc = [("enc0", 48, "NAF", NAF), ("enc1", 96, "NAF", NAF), ("enc2", 192, "NAF", NAF),
           ("enc3", 320, "MDTA", ATTN), ("enc4", 448, "MDTA", ATTN)]
    dec = [("dec0", 48, "NAF", NAF), ("dec1", 96, "NAF", NAF), ("dec2", 192, "NAF", NAF),
           ("dec3", 320, "MDTA", ATTN)]

    box(ex, 8.3, 2.1, 0.55, ["in_proj  3->48"], PROJ)
    box(dx, 8.3, 2.1, 0.55, ["out:  x + residual -> 3"], PROJ)
    for i, (nm, ch, blk, c) in enumerate(enc):
        box(ex, ys[i], 2.1, 0.7, [f"{nm}  ({ch})", blk], c)
    for i, (nm, ch, blk, c) in enumerate(dec):
        box(dx, ys[i], 2.1, 0.7, [f"{nm}  ({ch})", blk], c)
    box(6.5, 0.9, 3.0, 0.8, ["bottleneck", "4x MDTA + BlockAttn (448)"], BOT)

    arrow((ex, 8.025), (ex, ys[0] + 0.35))           # in_proj -> enc0
    for i in range(4):                               # encoder downsamples
        arrow((ex, ys[i] - 0.35), (ex, ys[i + 1] + 0.35))
        ax.text(ex - 1.35, (ys[i] + ys[i + 1]) / 2, "down /2", fontsize=7, color="#666")
    arrow((ex + 1.05, ys[4]), (5.0, 0.9))            # enc4 -> bottleneck
    arrow((8.0, 0.9), (dx - 1.05, ys[3]))            # bottleneck -> dec3
    for i in range(3, 0, -1):                         # decoder upsamples
        arrow((dx, ys[i] + 0.35), (dx, ys[i - 1] - 0.35))
        ax.text(dx + 1.15, (ys[i] + ys[i - 1]) / 2, "up x2", fontsize=7, color="#666")
    arrow((dx, ys[0] + 0.35), (dx, 8.025))           # dec0 -> out

    for i in range(4):                               # attention-gated skips
        arrow((ex + 1.05, ys[i]), (dx - 1.05, ys[i]), dashed=True, color="#b06")
    ax.text(6.5, 7.85, "attention-gated skip connections", ha='center',
            fontsize=7.5, color="#b06")

    arrow((ex, 8.62), (dx, 8.62), dashed=True, color="#2a7")   # global residual
    ax.text(6.5, 8.74, "global residual  (output = input + predicted residual)",
            ha='center', fontsize=7.5, color="#2a7")

    for j, (lab, c) in enumerate([("NAFNet block", NAF), ("MDTA/attention block", ATTN),
                                  ("bottleneck", BOT), ("projection", PROJ)]):
        box(1.7 + j * 2.85, -0.4, 2.0, 0.42, [lab], c)

    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ----------------------------- main -----------------------------

def load_denoiser(path, channels, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['ema'] if isinstance(ck, dict) and ck.get('ema') else (ck.get('G', ck) if isinstance(ck, dict) else ck)
    D = DenoiseGenerator(channels=channels, drop_path_rate=0.0, use_checkpoint=False)
    D.load_state_dict(sd, strict=False)
    D = D.eval().to(device)
    for q in D.parameters():
        q.requires_grad_(False)
    return D


def load_translator(path, device):
    tck = torch.load(path, map_location='cpu', weights_only=False)
    tsd = tck['ema'] if isinstance(tck, dict) and tck.get('ema') else (tck.get('T', tck) if isinstance(tck, dict) else tck)
    cfg = tck.get('args', {}) if isinstance(tck, dict) else {}
    T = NoiseTranslator(nc=3, dim=int(cfg.get('dim', 64)), num_blocks=int(cfg.get('blocks', 10)))
    T.load_state_dict(tsd, strict=False)
    T = T.eval().to(device)
    for q in T.parameters():
        q.requires_grad_(False)
    return T


def main():
    p = argparse.ArgumentParser(description="Generate DenoiseGAN demo images.")
    p.add_argument('--denoiser-ckpt', type=str, default=None)
    p.add_argument('--arch-only', action='store_true',
                   help='only draw the architecture diagram (no checkpoint needed)')
    p.add_argument('--translator-ckpt', type=str, default=None)
    p.add_argument('--clean', type=str, default=None, help='CLEAN images (synthetic noise + GT)')
    p.add_argument('--input', type=str, default=None, help='REAL noisy images (no GT)')
    p.add_argument('--out-dir', type=str, default='./assets')
    p.add_argument('--max-images', type=int, default=3)
    p.add_argument('--pick', type=str, default='random',
                   choices=['hard', 'first', 'random'],
                   help='select examples: random (default), first, or hard')
    p.add_argument('--crop', type=int, default=160, help='detail-crop window size (px)')
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--overlap', type=int, default=32)
    p.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--channels', type=int, nargs=5, default=[48, 96, 192, 320, 448])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    od = args.out_dir

    draw_architecture(os.path.join(od, 'architecture.png'))
    print("[saved] architecture.png")
    if args.arch_only or not args.denoiser_ckpt:
        if not args.denoiser_ckpt:
            print("No --denoiser-ckpt given; drew architecture only.")
        return

    D = load_denoiser(args.denoiser_ckpt, tuple(args.channels), device)
    print(f"[denoiser] {args.denoiser_ckpt}")
    T = None
    if args.translator_ckpt and os.path.exists(args.translator_ckpt):
        T = load_translator(args.translator_ckpt, device)
        print(f"[translator] {args.translator_ckpt}")

    def Den(img, t=None):
        return tiled_denoise(D, img, args.tile, args.overlap, device, args.amp, translator=t)

    real = all_images(args.input)
    clean = all_images(args.clean)
    if args.pick == 'hard':
        real = pick_hardest(real, noise_score, args.max_images)
        clean = pick_hardest(clean, detail_score, args.max_images)
        print(f"[select] hardest {len(real)} noisy + {len(clean)} clean")
    elif args.pick == 'random':
        real = pick_random(real, args.max_images)
        clean = pick_random(clean, args.max_images)
        print(f"[select] random {len(real)} noisy + {len(clean)} clean")
    else:
        real, clean = real[:args.max_images], clean[:args.max_images]
        print(f"[select] first {len(real)} noisy + {len(clean)} clean")

    hero_pair = None  # (noisy_u8, denoised_u8) for gif/split/crops

    # ---- real noisy: denoise_real panel ----
    if real:
        rows = []
        for fp in real:
            n = load_pm1(fp, device)
            d = Den(n)
            cells = [captioned(pm1_to_u8(n), "noisy input"),
                     captioned(pm1_to_u8(d), "denoised")]
            if T is not None:
                cells.append(captioned(pm1_to_u8(Den(n, T)), "denoised + translator"))
            rows.append(row(cells))
        stack(rows).save(os.path.join(od, 'denoise_real.png'))
        print("[saved] denoise_real.png")
        n0 = load_pm1(real[0], device)
        hero_pair = (pm1_to_u8(n0), pm1_to_u8(Den(n0)))
        hero_src = n0

    # ---- clean: gaussian on/off, noise types, fft ----
    if clean:
        rows = []
        for fp in clean:
            c = load_pm1(fp, device)
            c01 = (c + 1) / 2
            n = add_noise(c01, 'gaussian') * 2 - 1
            d = Den(n)
            cells = [captioned(pm1_to_u8(n), f"gaussian  {psnr01((n + 1) / 2, c01):.1f}dB"),
                     captioned(pm1_to_u8(d), f"D(noisy)  {psnr01((d + 1) / 2, c01):.1f}dB")]
            if T is not None:
                dt = Den(n, T)
                cells.append(captioned(pm1_to_u8(dt), f"D(T(noisy))  {psnr01((dt + 1) / 2, c01):.1f}dB"))
            cells.append(captioned(pm1_to_u8(c), "clean"))
            rows.append(row(cells))
        stack(rows).save(os.path.join(od, 'translator_gaussian.png'))
        print("[saved] translator_gaussian.png")

        c = load_pm1(clean[0], device)
        c01 = (c + 1) / 2
        rows = []
        for kind in ['gaussian', 'poisson', 'speckle', 'salt_pepper']:
            n = add_noise(c01, kind) * 2 - 1
            cells = [captioned(pm1_to_u8(n), kind), captioned(pm1_to_u8(Den(n)), "D")]
            if T is not None:
                cells.append(captioned(pm1_to_u8(Den(n, T)), "D + translator"))
            cells.append(captioned(pm1_to_u8(c), "clean"))
            rows.append(row(cells))
        stack(rows).save(os.path.join(od, 'noise_types.png'))
        print("[saved] noise_types.png")

        # fft compare on gaussian
        n = add_noise(c01, 'gaussian') * 2 - 1
        d = Den(n, T) if T is not None else Den(n)
        row([fft_panel(n, "noisy spectrum"), fft_panel(d, "denoised spectrum"),
             fft_panel(c, "clean spectrum")]).save(os.path.join(od, 'fft_compare.png'))
        print("[saved] fft_compare.png")

        if hero_pair is None:
            n = add_noise(c01, 'gaussian') * 2 - 1
            d = Den(n, T) if T is not None else Den(n)
            hero_pair = (pm1_to_u8(n), pm1_to_u8(d))
            hero_src = n

    # ---- hero gif + split + detail crops ----
    if hero_pair is not None:
        noisy_u8, den_u8 = hero_pair
        make_gif(noisy_u8, den_u8, os.path.join(od, 'before_after.gif'))
        print("[saved] before_after.gif")
        diagonal_split(noisy_u8, den_u8).save(os.path.join(od, 'split.png'))
        print("[saved] split.png")

        y, x, s = find_detail_crop(hero_src, args.crop)
        crop_cells = [
            captioned(noisy_u8[y:y + s, x:x + s], "noisy (100%)", disp_h=300, nearest=True),
            captioned(den_u8[y:y + s, x:x + s], "denoised (100%)", disp_h=300, nearest=True),
        ]
        row(crop_cells).save(os.path.join(od, 'detail_crops.png'))
        print("[saved] detail_crops.png")

    if not real and not clean:
        print("Nothing to do: pass --clean and/or --input.")
        return
    print(f"\nDone. Images in {od}/")


if __name__ == '__main__':
    main()
