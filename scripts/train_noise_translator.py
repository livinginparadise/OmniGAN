import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import contextlib
import time
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
from tqdm import tqdm

import lpips

from denoisegan.models import DenoiseGenerator, NoiseTranslator
from denoisegan.dataset import build_dataloader
from denoisegan.losses import charbonnier


def amp_ctx(device, dtype):
    if dtype == 'bf16':
        return autocast(device.type, dtype=torch.bfloat16)
    if dtype == 'fp16':
        return autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def to01(x):
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def psnr01(p, t):
    mse = F.mse_loss(p.clamp(0, 1), t.clamp(0, 1))
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def ssim01(p, t):
    return float(tm_ssim(p.clamp(0, 1), t.clamp(0, 1), data_range=1.0))


def heat(x_pm1):
    """abs error map (mean over channels) of a [-1,1] residual, normalized 0..1 per-batch."""
    g = x_pm1.abs().mean(dim=1, keepdim=True)
    g = g / (g.amax() + 1e-8)
    return g.clamp(0, 1)


def fft_logmag(x01):
    """log-magnitude FFT spectrum (mean over channels) of a [0,1] image, normalized."""
    f = torch.fft.fftshift(torch.fft.fft2(x01.float().mean(dim=1, keepdim=True)))
    m = (f.abs() + 1e-6).log()
    m = (m - m.amin()) / (m.amax() - m.amin() + 1e-8)
    return m.clamp(0, 1)


class TBLogger:
    def __init__(self, log_dir):
        self.w = SummaryWriter(log_dir=log_dir)

    def scalars(self, d, step):
        for k, v in d.items():
            self.w.add_scalar(k, v, step)

    def grid(self, tag, t01, step, nrow):
        self.w.add_image(tag, make_grid(t01.clamp(0, 1).detach().cpu(), nrow=nrow), step)

    def params(self, model, step):
        for n, p in model.named_parameters():
            t = p.detach().float().cpu()
            if torch.isfinite(t).all():
                self.w.add_histogram(f"weights/{n}", t, step)
            if p.grad is not None:
                g = p.grad.detach().float().cpu()
                if torch.isfinite(g).all():
                    self.w.add_histogram(f"grads/{n}", g, step)

    def hist(self, tag, t, step):
        t = t.detach().float().cpu()
        if torch.isfinite(t).all():
            self.w.add_histogram(tag, t, step)

    def per_layer_grad_norms(self, model, step):
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.w.add_scalar(f"grad_norm/{n}", p.grad.detach().norm(2).item(), step)

    def ema_dist(self, model, ema):
        tot = 0.0
        msd = model.state_dict()
        for k, v in ema.shadow.items():
            if v.dtype.is_floating_point:
                tot += (msd[k].detach().float().cpu() - v.float().cpu()).norm(2).item() ** 2
        return tot ** 0.5


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v.detach())


def make_arbitrary_noise(clean01):
    """Apply a wide, diverse mix of noise types to teach T to handle 'anything'.
    Operates in [0,1]. Per-image random type selection for maximum diversity."""
    B = clean01.shape[0]
    outs = []
    for b in range(B):
        x = clean01[b:b + 1]
        for _ in range(random.randint(1, 2)):
            t = random.random()
            if t < 0.28:                                   # gaussian, very wide sigma
                sig = random.uniform(5.0, 90.0) / 255.0
                noise = torch.randn_like(x) * sig
                if random.random() < 0.3:                  # channel-correlated (gray) noise
                    noise = noise[:, :1].expand_as(x).contiguous()
                x = x + noise
            elif t < 0.42:                                 # poisson / shot noise
                scale = random.uniform(0.02, 3.0)
                x = torch.poisson((x.clamp(0, 1) * 255.0 * scale)) / (255.0 * scale)
            elif t < 0.56:                                 # speckle (multiplicative)
                x = x + x * torch.randn_like(x) * random.uniform(0.02, 0.25)
            elif t < 0.68:                                 # uniform
                a = random.uniform(0.02, 0.30)
                x = x + (torch.rand_like(x) - 0.5) * 2.0 * a
            elif t < 0.80:                                 # salt & pepper
                d = random.uniform(0.005, 0.08)
                m = torch.rand_like(x[:, :1])
                x = torch.where(m < d / 2, torch.zeros_like(x), x)
                x = torch.where(m > 1.0 - d / 2, torch.ones_like(x), x)
            elif t < 0.92:                                 # spatially-correlated gaussian
                sig = random.uniform(10.0, 60.0) / 255.0
                noise = torch.randn_like(x) * sig
                k = random.choice([3, 5, 7])
                noise = TF.gaussian_blur(noise, [k, k])
                x = x + noise
            else:                                          # heavy quantization / banding
                levels = random.choice([3, 4, 6, 8])
                x = torch.round(x.clamp(0, 1) * (levels - 1)) / (levels - 1)
            x = x.clamp(0.0, 1.0)
        outs.append(x)
    return torch.cat(outs, dim=0)


def load_frozen_denoiser(path, channels, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'ema' in ck and ck['ema']:
        sd, tag = ck['ema'], 'ema'
    elif isinstance(ck, dict) and 'G' in ck:
        sd, tag = ck['G'], 'G'
    else:
        sd, tag = ck, 'raw'
    # train()+checkpoint to allow grad-through with low VRAM; drop_path 0 => deterministic
    D = DenoiseGenerator(channels=channels, drop_path_rate=0.0, use_checkpoint=True)
    missing, unexpected = D.load_state_dict(sd, strict=False)
    if missing:
        print(f"[denoiser] missing {len(missing)} keys (first: {missing[:3]})")
    if unexpected:
        print(f"[denoiser] unexpected {len(unexpected)} keys (first: {unexpected[:3]})")
    D.to(device)
    for p in D.parameters():
        p.requires_grad_(False)
    D.train()  # enables internal gradient checkpointing (no BN in this net; drop_path=0)
    print(f"[denoiser] loaded '{tag}' weights, frozen, checkpointed")
    return D


def main():
    p = argparse.ArgumentParser(description="Train a noise-translation front-end.")
    p.add_argument('--denoiser-ckpt', type=str, required=True)
    p.add_argument('--data', type=str, default='/home/algis/Desktop/data/train')
    p.add_argument('--keep-list', type=str, default=None)
    p.add_argument('--out-dir', type=str, default='./checkpoints')
    p.add_argument('--sample-dir', type=str, default='./samples')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--iters', type=int, default=60000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--blocks', type=int, default=10)
    p.add_argument('--lambda-id', type=float, default=0.5)
    p.add_argument('--lambda-clean', type=float, default=0.25)
    p.add_argument('--lambda-lpips', type=float, default=1.0)
    p.add_argument('--channels', type=int, nargs=5, default=[48, 96, 192, 320, 448])
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--sample-every', type=int, default=1000)
    p.add_argument('--log-dir', type=str, default='./logs')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--tb-every', type=int, default=50, help='scalar logging interval')
    p.add_argument('--img-every', type=int, default=1000, help='image logging interval')
    p.add_argument('--hist-every', type=int, default=2000, help='weight/grad histogram interval')
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

    T = NoiseTranslator(nc=3, dim=args.dim, num_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in T.parameters())
    print(f"[translator] NoiseTranslator: {n_params/1e6:.3f}M params "
          f"(dim={args.dim}, blocks={args.blocks})")

    opt = torch.optim.AdamW(T.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-7)
    scaler = GradScaler(device.type, enabled=args.amp == 'fp16')
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
        print(f"[resume] translator from step {start_step}")

    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for q in lpips_fn.parameters():
        q.requires_grad_(False)
    lpips_fn.eval()

    loader = build_dataloader(args.data, args.batch_size, image_size=args.image_size,
                              augment=True, num_workers=args.workers,
                              keep_list=args.keep_list)
    data_iter = iter(loader)

    run_name = args.run_name or f"translator_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    tb = TBLogger(os.path.join(args.log_dir, run_name))
    tb.w.add_text("config/args", str(vars(args)), 0)
    tb.w.add_text("config/model", f"NoiseTranslator {n_params/1e6:.3f}M params; "
                  f"frozen denoiser {args.denoiser_ckpt}", 0)
    print(f"[tb] logging to {os.path.join(args.log_dir, run_name)}")

    T.train()
    t0 = time.time()
    t_last = time.time()
    pbar = tqdm(range(start_step, args.iters), initial=start_step, total=args.iters,
                desc="translator", dynamic_ncols=True)
    for step in pbar:
        try:
            canon_pm1, clean_pm1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            canon_pm1, clean_pm1, _ = next(data_iter)
        canon_pm1 = canon_pm1.to(device, non_blocking=True)
        clean_pm1 = clean_pm1.to(device, non_blocking=True)

        clean01 = to01(clean_pm1)
        with torch.no_grad():
            src01 = make_arbitrary_noise(clean01)
        src_pm1 = (src01 * 2.0 - 1.0)

        with amp_ctx(device, args.amp):
            trans_src = T(src_pm1)
            out_src = D(trans_src)
            l_main = charbonnier(out_src, clean_pm1, eps=1e-3)
            l_lpips = lpips_fn(out_src, clean_pm1).mean()

            trans_canon = T(canon_pm1)
            l_id = charbonnier(trans_canon, canon_pm1, eps=1e-3)

            trans_clean = T(clean_pm1)
            l_clean = charbonnier(trans_clean, clean_pm1, eps=1e-3)

            loss = (l_main + args.lambda_lpips * l_lpips
                    + args.lambda_id * l_id + args.lambda_clean * l_clean)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0))

        # histograms of weights/grads + residual distributions (grads still valid here)
        if step % args.hist_every == 0:
            tb.params(T, step)
            tb.per_layer_grad_norms(T, step)
            tb.hist("dist/translator_change_Tsrc_minus_src", trans_src.detach() - src_pm1, step)
            tb.hist("dist/input_noise_src_minus_clean", src_pm1 - clean_pm1, step)

        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(T)

        if step % 20 == 0:
            pbar.set_postfix(main=f"{l_main.item():.4f}", lpips=f"{l_lpips.item():.4f}",
                             idn=f"{l_id.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        # ---------- detailed scalar logging ----------
        if step % args.tb_every == 0:
            with torch.no_grad():
                out_base = D(src_pm1)                       # denoiser alone, no translator
                pl, base, cl = to01(out_src.detach()), to01(out_base), to01(clean_pm1)
                psnr_pipe, psnr_base = psnr01(pl, cl), psnr01(base, cl)
                ssim_pipe, ssim_base = ssim01(pl, cl), ssim01(base, cl)
                lp_base = lpips_fn(out_base, clean_pm1).mean().item()
                act = (trans_src.detach() - src_pm1).abs().mean().item()
                id_c = (trans_canon.detach() - canon_pm1).abs().mean().item()
                id_cl = (trans_clean.detach() - clean_pm1).abs().mean().item()
                res_std = (trans_src.detach() - src_pm1).std().item()
            now = time.time()
            ips = args.tb_every / max(now - t_last, 1e-6)
            t_last = now
            logs = {
                "loss/total": loss.item(),
                "loss/main_charbonnier": l_main.item(),
                "loss/lpips": l_lpips.item(),
                "loss/identity_canon": l_id.item(),
                "loss/identity_clean": l_clean.item(),
                "loss_weighted/main": l_main.item(),
                "loss_weighted/lpips": args.lambda_lpips * l_lpips.item(),
                "loss_weighted/identity_canon": args.lambda_id * l_id.item(),
                "loss_weighted/identity_clean": args.lambda_clean * l_clean.item(),
                "opt/lr": sched.get_last_lr()[0],
                "opt/grad_norm_preclip": gnorm,
                "opt/grad_clipped": 1.0 if gnorm > 1.0 else 0.0,
                "opt/ema_dist": tb.ema_dist(T, ema),
                "quality/psnr_pipeline": psnr_pipe,
                "quality/psnr_baseline_noT": psnr_base,
                "quality/psnr_GAIN_from_T": psnr_pipe - psnr_base,
                "quality/ssim_pipeline": ssim_pipe,
                "quality/ssim_baseline_noT": ssim_base,
                "quality/ssim_GAIN_from_T": ssim_pipe - ssim_base,
                "quality/lpips_pipeline": l_lpips.item(),
                "quality/lpips_baseline_noT": lp_base,
                "translator/activity_on_src": act,
                "translator/identity_resid_canon": id_c,
                "translator/identity_resid_clean": id_cl,
                "translator/src_residual_std": res_std,
                "perf/it_per_s": ips,
            }
            if torch.cuda.is_available():
                logs["perf/vram_gb"] = torch.cuda.memory_allocated() / 1e9
            tb.scalars(logs, step)

        # ---------- image panels ----------
        if step % args.img_every == 0:
            with torch.no_grad():
                out_base = D(src_pm1)
                n = min(4, src_pm1.shape[0])
                comp = torch.cat([
                    to01(src_pm1[:n]), to01(trans_src.detach()[:n]),
                    to01(out_src.detach()[:n]), to01(out_base[:n]), to01(clean_pm1[:n]),
                ], dim=0)
                tb.grid("img/rows_src_Tsrc_DTsrc_Dsrc_clean", comp, step, nrow=n)
                tb.grid("map/translator_change_absTsrc-src", heat((trans_src.detach() - src_pm1)[:n]), step, nrow=n)
                tb.grid("map/error_pipeline_absDTsrc-clean", heat((out_src.detach() - clean_pm1)[:n]), step, nrow=n)
                tb.grid("map/error_baseline_absDsrc-clean", heat((out_base - clean_pm1)[:n]), step, nrow=n)
                tb.grid("fft/input_noise", fft_logmag((src_pm1 - clean_pm1)[:n]), step, nrow=n)
                tb.grid("fft/remaining_error", fft_logmag((out_src.detach() - clean_pm1)[:n]), step, nrow=n)
            save_image(comp, os.path.join(args.sample_dir, f"translator_{step:07d}.png"), nrow=n)

        if step > 0 and step % args.save_every == 0:
            torch.save({'T': T.state_dict(), 'ema': ema.shadow, 'opt': opt.state_dict(),
                        'step': step, 'args': vars(args)},
                       os.path.join(args.out_dir, f"translator_{step:07d}.pt"))

    torch.save({'T': T.state_dict(), 'ema': ema.shadow, 'opt': opt.state_dict(),
                'step': args.iters, 'args': vars(args)},
               os.path.join(args.out_dir, "translator_final.pt"))
    tb.w.flush()
    tb.w.close()
    print(f"\nsaved: {os.path.join(args.out_dir, 'translator_final.pt')}  "
          f"({(time.time()-t0)/3600:.2f}h)")


if __name__ == '__main__':
    main()
