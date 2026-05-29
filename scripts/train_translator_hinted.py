import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import lpips

from denoisegan.models import DenoiseGenerator, GuidedNoiseTranslator
from denoisegan.dataset import build_clean_loader
from denoisegan.losses import charbonnier
from train_noise_translator import make_arbitrary_noise, EMA, amp_ctx, to01


def info_nce(z_a, z_b, tau=0.2):
    """NT-Xent: two views of the same image (same noise) are positives; all other
    images in the batch (different noise) are negatives. Makes the noise embedding
    content/position-invariant and noise-discriminative (DASR-style)."""
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = (z @ z.t()) / tau
    sim.masked_fill_(torch.eye(2 * n, device=z.device, dtype=torch.bool), float('-inf'))
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def load_frozen_denoiser(path, channels, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'ema' in ck and ck['ema']:
        sd, tag = ck['ema'], 'ema'
    elif isinstance(ck, dict) and 'G' in ck:
        sd, tag = ck['G'], 'G'
    else:
        sd, tag = ck, 'raw'
    D = DenoiseGenerator(channels=channels, drop_path_rate=0.0, use_checkpoint=True)
    D.load_state_dict(sd, strict=False)
    D.to(device)
    for p in D.parameters():
        p.requires_grad_(False)
    D.train()  # enables checkpointing; drop_path=0 => deterministic
    print(f"[denoiser] loaded '{tag}', frozen + checkpointed")
    return D


def warm_start_translator(T, path):
    """Best-effort load of an unconditioned NoiseTranslator (Phase 1) into the
    guided backbone: head/tail match by name, body.{2i} -> blocks.{i}."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('ema') if isinstance(ck, dict) and ck.get('ema') else \
        (ck.get('T') if isinstance(ck, dict) and 'T' in ck else ck)
    remap = {}
    if 'head.weight' in sd:
        remap['head.weight'] = sd['head.weight']
    if 'tail.weight' in sd:
        remap['tail.weight'] = sd['tail.weight']
    body = sorted(((int(k.split('.')[1]), k) for k in sd
                   if k.startswith('body.') and k.endswith('.weight')))
    for bi, (_, k) in enumerate(body):
        remap[f'blocks.{bi}.weight'] = sd[k]
    missing, unexpected = T.load_state_dict(remap, strict=False)
    print(f"[warm-start] mapped {len(remap)} backbone tensors from {Path(path).name} "
          f"(conditioning layers fresh)")


def main():
    p = argparse.ArgumentParser(description="Phase-2 hint-conditioned translator.")
    p.add_argument('--denoiser-ckpt', type=str, required=True)
    p.add_argument('--init-translator', type=str, default=None,
                   help='Phase-1 (unconditioned) translator to warm-start the backbone')
    p.add_argument('--data', type=str, default='/home/algis/Desktop/data/train')
    p.add_argument('--keep-list', type=str, default=None)
    p.add_argument('--out-dir', type=str, default='./checkpoints')
    p.add_argument('--sample-dir', type=str, default='./samples')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--iters', type=int, default=60000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--patch', type=int, default=256)
    p.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--blocks', type=int, default=10)
    p.add_argument('--emb', type=int, default=128)
    p.add_argument('--lambda-lpips', type=float, default=1.0)
    p.add_argument('--lambda-patch', type=float, default=1.0)
    p.add_argument('--lambda-contrast', type=float, default=0.1,
                   help='weight of the contrastive degradation-embedding loss (0 disables)')
    p.add_argument('--contrast-tau', type=float, default=0.2)
    p.add_argument('--channels', type=int, nargs=5, default=[48, 96, 192, 320, 448])
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--sample-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)

    D = load_frozen_denoiser(args.denoiser_ckpt, tuple(args.channels), device)

    T = GuidedNoiseTranslator(nc=3, dim=args.dim, num_blocks=args.blocks, emb=args.emb).to(device)
    if args.init_translator and os.path.exists(args.init_translator):
        warm_start_translator(T, args.init_translator)
    n_params = sum(p.numel() for p in T.parameters())
    print(f"[translator] GuidedNoiseTranslator: {n_params/1e6:.3f}M params")

    opt = torch.optim.AdamW(T.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-7)
    ema = EMA(T, decay=args.ema_decay)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        T.load_state_dict(ck['T'])
        if 'ema' in ck:
            ema.shadow = {k: v.to(device) for k, v in ck['ema'].items()}
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])
        start_step = int(ck.get('step', 0))
        for _ in range(start_step):
            sched.step()
        print(f"[resume] from step {start_step}")

    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for q in lpips_fn.parameters():
        q.requires_grad_(False)
    lpips_fn.eval()

    loader = build_clean_loader(args.data, args.batch_size, image_size=args.image_size,
                                augment=True, num_workers=args.workers,
                                keep_list=args.keep_list)
    data_iter = iter(loader)

    S, P = args.image_size, args.patch
    T.train()
    pbar = tqdm(range(start_step, args.iters), initial=start_step, total=args.iters,
                desc="hinted-translator", dynamic_ncols=True)
    for step in pbar:
        try:
            clean_pm1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            clean_pm1, _ = next(data_iter)
        clean_pm1 = clean_pm1.to(device, non_blocking=True)
        B = clean_pm1.shape[0]

        clean01 = to01(clean_pm1)
        with torch.no_grad():
            noisy01 = make_arbitrary_noise(clean01)          # fresh random noise per sample
        noisy_pm1 = noisy01 * 2.0 - 1.0

        def sample_patch():
            ys = [random.randint(0, S - P) for _ in range(B)]
            xs = [random.randint(0, S - P) for _ in range(B)]
            hc = torch.stack([clean_pm1[b, :, ys[b]:ys[b] + P, xs[b]:xs[b] + P] for b in range(B)])
            hn = torch.stack([noisy_pm1[b, :, ys[b]:ys[b] + P, xs[b]:xs[b] + P] for b in range(B)])
            pos = torch.tensor([[(xs[b] + P / 2) / S, (ys[b] + P / 2) / S] for b in range(B)],
                               device=device, dtype=torch.float32)
            return ys, xs, hc, hn, pos

        ys, xs, hint_clean, hint_noisy, pos = sample_patch()          # view A: hint + reconstruction
        _, _, hint_clean_b, hint_noisy_b, _ = sample_patch()          # view B: contrastive positive

        with amp_ctx(device, args.amp):
            trans = T(noisy_pm1, hint_noisy, hint_clean, pos)
            out = D(trans)
            l_main = charbonnier(out, clean_pm1, eps=1e-3)
            l_lpips = lpips_fn(out, clean_pm1).mean()
            out_patch = torch.stack([out[b, :, ys[b]:ys[b] + P, xs[b]:xs[b] + P] for b in range(B)])
            l_patch = charbonnier(out_patch, hint_clean, eps=1e-3)

            if args.lambda_contrast > 0 and B > 1:
                z_a = T.project(T.noise_feature(hint_noisy, hint_clean))
                z_b = T.project(T.noise_feature(hint_noisy_b, hint_clean_b))
                l_con = info_nce(z_a, z_b, tau=args.contrast_tau)
            else:
                l_con = torch.zeros((), device=device)

            loss = (l_main + args.lambda_lpips * l_lpips + args.lambda_patch * l_patch
                    + args.lambda_contrast * l_con)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(T)

        if step % 20 == 0:
            pbar.set_postfix(main=f"{l_main.item():.4f}", lpips=f"{l_lpips.item():.4f}",
                             patch=f"{l_patch.item():.4f}", con=f"{float(l_con):.3f}",
                             lr=f"{sched.get_last_lr()[0]:.1e}")

        if step % args.sample_every == 0:
            with torch.no_grad():
                out_no_hint = D(noisy_pm1)  # denoiser alone, no translation
                grid = torch.cat([
                    to01(noisy_pm1[:4]), to01(trans[:4].detach()),
                    to01(out[:4].detach()), to01(out_no_hint[:4]),
                    to01(clean_pm1[:4]),
                ], dim=0)
            save_image(grid, os.path.join(args.sample_dir, f"hinted_{step:07d}.png"), nrow=4)

        if step > 0 and step % args.save_every == 0:
            torch.save({'T': T.state_dict(), 'ema': ema.shadow, 'opt': opt.state_dict(),
                        'step': step, 'args': vars(args)},
                       os.path.join(args.out_dir, f"hinted_{step:07d}.pt"))

    torch.save({'T': T.state_dict(), 'ema': ema.shadow, 'opt': opt.state_dict(),
                'step': args.iters, 'args': vars(args)},
               os.path.join(args.out_dir, "hinted_final.pt"))
    print(f"\nsaved: {os.path.join(args.out_dir, 'hinted_final.pt')}")


if __name__ == '__main__':
    main()
