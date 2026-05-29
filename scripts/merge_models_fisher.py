import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path

from torch.amp import autocast
from tqdm import tqdm

from denoisegan.models import DenoiseGenerator
from denoisegan.dataset import build_dataloader
from denoisegan.losses import charbonnier


def pick_state_dict(ck):
    if isinstance(ck, dict) and 'ema' in ck and ck['ema']:
        return ck['ema'], 'ema'
    if isinstance(ck, dict) and 'G' in ck:
        return ck['G'], 'G'
    return ck, 'raw'


def load_generator(path, channels, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd, tag = pick_state_dict(ck)
    G = DenoiseGenerator(channels=channels, drop_path_rate=0.0, use_checkpoint=False)
    missing, unexpected = G.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [{Path(path).name}] missing {len(missing)} keys (first: {missing[:3]})")
    if unexpected:
        print(f"  [{Path(path).name}] unexpected {len(unexpected)} keys (first: {unexpected[:3]})")
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(True)
    print(f"  [{Path(path).name}] loaded weights from '{tag}'")
    return G


def estimate_fisher(G, loader, device, n_batches, amp, lpips_fn, label):
    fisher = {n: torch.zeros_like(p) for n, p in G.named_parameters()}
    data_iter = iter(loader)
    seen = 0
    amp_dtype = torch.bfloat16 if amp == 'bf16' else (torch.float16 if amp == 'fp16' else None)

    pbar = tqdm(range(n_batches), desc=f"Fisher[{label}]", dynamic_ncols=True)
    for _ in pbar:
        try:
            noisy, clean, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            noisy, clean, _ = next(data_iter)
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        G.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            ctx = autocast(device.type, dtype=amp_dtype)
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            pred = G(noisy)
            loss = charbonnier(pred, clean, eps=1e-3)
            if lpips_fn is not None:
                loss = loss + lpips_fn(pred.float(), clean.float()).mean()
        loss.backward()

        with torch.no_grad():
            for n, p in G.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float() ** 2
        seen += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    for n in fisher:
        fisher[n] /= max(1, seen)
    total = sum(f.sum().item() for f in fisher.values())
    print(f"  [{label}] total Fisher mass: {total:.4e} over {seen} batches")
    return fisher


def merge(sd_a, sd_b, F_a, F_b, eps):
    merged = {}
    fisher_share_a = 0.0
    fisher_share_b = 0.0
    for k in sd_a:
        a = sd_a[k].float()
        b = sd_b[k].float()
        if k in F_a and k in F_b:
            fa = F_a[k]
            fb = F_b[k]
            num = fa * a + fb * b + eps * 0.5 * (a + b)
            den = fa + fb + eps
            merged[k] = (num / den).to(sd_a[k].dtype)
            fisher_share_a += fa.sum().item()
            fisher_share_b += fb.sum().item()
        else:
            if sd_a[k].dtype.is_floating_point:
                merged[k] = (0.5 * (a + b)).to(sd_a[k].dtype)
            else:
                merged[k] = sd_a[k].clone()
    tot = fisher_share_a + fisher_share_b + 1e-12
    print(f"  Fisher-weighted pull:  A={100*fisher_share_a/tot:.1f}%  "
          f"B={100*fisher_share_b/tot:.1f}%")
    return merged


def main():
    p = argparse.ArgumentParser(description="Fisher-weighted merge of two generators.")
    p.add_argument('--ckpt-a', type=str, required=True, help='model A (e.g. PSNR final)')
    p.add_argument('--ckpt-b', type=str, required=True, help='model B (e.g. GAN checkpoint)')
    p.add_argument('--out', type=str, required=True, help='output merged checkpoint')
    p.add_argument('--data', type=str, default='/home/algis/Desktop/data/train')
    p.add_argument('--keep-list', type=str, default=None,
                   help='clean-target keep-list (recommended for faithful Fisher)')
    p.add_argument('--n-batches', type=int, default=150,
                   help='batches per model for Fisher estimation')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--eps', type=float, default=1e-8,
                   help='uniform-average prior strength')
    p.add_argument('--use-lpips', action='store_true',
                   help='include LPIPS in the Fisher loss (slower, more faithful)')
    p.add_argument('--channels', type=int, nargs=5, default=[48, 96, 192, 320, 448])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    channels = tuple(args.channels)

    print("building dataloader...")
    loader = build_dataloader(args.data, args.batch_size,
                              image_size=args.image_size, augment=True,
                              num_workers=args.workers, keep_list=args.keep_list)

    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').to(device)
            for q in lpips_fn.parameters():
                q.requires_grad_(False)
            lpips_fn.eval()
            print("LPIPS term enabled in Fisher loss")
        except Exception as e:
            print(f"LPIPS unavailable ({e}); using Charbonnier only")

    print(f"\nloading model A: {args.ckpt_a}")
    G_a = load_generator(args.ckpt_a, channels, device)
    print(f"loading model B: {args.ckpt_b}")
    G_b = load_generator(args.ckpt_b, channels, device)

    print("\nestimating Fisher for model A...")
    F_a = estimate_fisher(G_a, loader, device, args.n_batches, args.amp, lpips_fn, 'A')
    print("estimating Fisher for model B...")
    F_b = estimate_fisher(G_b, loader, device, args.n_batches, args.amp, lpips_fn, 'B')

    print("\nmerging...")
    sd_a = {n: p.detach() for n, p in G_a.state_dict().items()}
    sd_b = {n: p.detach() for n, p in G_b.state_dict().items()}
    merged = merge(sd_a, sd_b, F_a, F_b, args.eps)

    payload = {
        'G': merged,
        'ema': merged,
        'step': 0,
        'stage': 'psnr',
        'extra': {
            'merge': 'fisher',
            'ckpt_a': args.ckpt_a,
            'ckpt_b': args.ckpt_b,
            'n_batches': args.n_batches,
            'eps': args.eps,
            'use_lpips': bool(lpips_fn is not None),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"\nsaved merged checkpoint: {args.out}")


if __name__ == '__main__':
    main()
