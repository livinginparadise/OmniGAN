import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import contextlib
import warnings
import time
import datetime
from pathlib import Path
from collections import deque

import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim

import lpips

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console, Group
from rich.columns import Columns

from denoisegan.models import DenoiseGenerator, UNetDiscriminatorSN, DinoProjectedDiscriminator
from denoisegan.dataset import build_dataloader, build_clean_loader
from denoisegan.losses import (
    charbonnier, FocalFrequencyLoss, lab_chroma_loss, GradVarianceLoss,
    rpgan_d_loss, rpgan_g_loss, r1_penalty, r2_penalty,
    aggregate_unet_dino,
)
from denoisegan.soap import SOAP

SPARK = "▁▂▃▄▅▆▇█"


def _raw(m):
    return getattr(m, '_orig_mod', m)


def sparkline(vals, w=20):
    if not vals:
        return ""
    import math
    v = [x for x in list(vals)[-w:] if math.isfinite(x)]
    if not v:
        return "?"
    lo, hi = min(v), max(v)
    r = hi - lo if hi - lo > 1e-12 else 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((x - lo) / r * (len(SPARK) - 1)))] for x in v)


_gpu_stats_cache = {
    "vram_used": 0.0,
    "vram_total": 0.0,
    "vram_pct": 0.0,
    "util": -1,
    "temp": -1,
    "last_query": 0.0
}


def gpu_stats():
    d = {}
    if torch.cuda.is_available():
        now = time.time()
        # Query at most once every 2 seconds to avoid subprocess / API overhead
        if now - _gpu_stats_cache["last_query"] > 2.0:
            _gpu_stats_cache["last_query"] = now
            util, temp = -1, -1
            vram_used, vram_total = -1.0, -1.0

            # 1. Try pynvml (fastest, no subprocess overhead)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = mem.used / 1e9
                vram_total = mem.total / 1e9
                pynvml.nvmlShutdown()
            except Exception:
                # 2. Try nvidia-smi (reliable fallback, minimal impact due to cache)
                try:
                    import subprocess
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        encoding="utf-8"
                    ).strip()
                    parts = [x.strip() for x in out.split(",")]
                    if len(parts) == 4:
                        util = int(parts[0])
                        temp = int(parts[1])
                        vram_used = float(parts[2]) / 1024.0
                        vram_total = float(parts[3]) / 1024.0
                except Exception:
                    pass

            # Fallback to PyTorch metrics if NVML/nvidia-smi failed to get VRAM
            if vram_used < 0:
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            _gpu_stats_cache["util"] = util
            _gpu_stats_cache["temp"] = temp
            _gpu_stats_cache["vram_used"] = vram_used
            _gpu_stats_cache["vram_total"] = vram_total
            _gpu_stats_cache["vram_pct"] = (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0

        d["vram_used"] = _gpu_stats_cache["vram_used"]
        d["vram_total"] = _gpu_stats_cache["vram_total"]
        d["vram_pct"] = _gpu_stats_cache["vram_pct"]
        d["util"] = _gpu_stats_cache["util"]
        d["temp"] = _gpu_stats_cache["temp"]
    return d


def ssim(p, t):
    return tm_ssim(p.clamp(0, 1), t.clamp(0, 1), data_range=1.0).item()


def grad_norm(params):
    ns = [p.grad.detach().flatten() for p in params if p.grad is not None]
    if not ns:
        return 0.0
    return torch.cat(ns).norm(2).item()


class MT:
    def __init__(self, maxlen=200):
        self.d = {}
        self.maxlen = maxlen

    def up(self, k, v):
        if k not in self.d:
            self.d[k] = deque(maxlen=self.maxlen)
        self.d[k].append(v)

    def avg(self, k):
        if k not in self.d or not self.d[k]:
            return 0.0
        return sum(self.d[k]) / len(self.d[k])

    def last(self, k):
        if k not in self.d or not self.d[k]:
            return 0.0
        return self.d[k][-1]

    def hist(self, k):
        return list(self.d.get(k, []))

    def spark(self, k, w=20):
        return sparkline(self.hist(k), w)


class TBL:
    def __init__(self, writer):
        self.w = writer

    def scalars(self, d, step):
        for k, v in d.items():
            self.w.add_scalar(k, v, step)

    def imgs(self, tag, tensors, step, nrow=4):
        grid = make_grid(tensors.clamp(0, 1).detach().cpu(), nrow=nrow, normalize=False)
        self.w.add_image(tag, grid, step)

    def hists(self, model, step, prefix):
        for n, p in _raw(model).named_parameters():
            if p.numel() > 0:
                t = p.detach().cpu().float()
                if torch.isfinite(t).any():
                    try:
                        self.w.add_histogram(f"{prefix}/W/{n}", t, step)
                    except ValueError:
                        pass
                if p.grad is not None:
                    g = p.grad.detach().cpu().float()
                    if torch.isfinite(g).any():
                        try:
                            self.w.add_histogram(f"{prefix}/G/{n}", g, step)
                        except ValueError:
                            pass

    def gnorms(self, model, step, prefix):
        total = 0.0
        for n, p in _raw(model).named_parameters():
            if p.grad is not None:
                gn = p.grad.detach().norm(2).item()
                self.w.add_scalar(f"{prefix}/gn/{n}", gn, step)
                total += gn ** 2
        self.w.add_scalar(f"{prefix}/gn_total", total ** 0.5, step)

    def lr(self, opt, step, tag):
        for i, pg in enumerate(opt.param_groups):
            self.w.add_scalar(f"{tag}/lr_{i}", pg['lr'], step)

    def ema_diff(self, model, ema, step, tag):
        total = 0.0
        cnt = 0
        msd = _raw(model).state_dict()
        for k, v in ema.shadow.items():
            if v.dtype.is_floating_point:
                diff = (msd[k].detach().cpu().float() - v.float()).norm(2).item()
                total += diff ** 2
                cnt += 1
        self.w.add_scalar(f"{tag}/ema_dist", total ** 0.5, step)
        return total ** 0.5

    def fft_img(self, pred01, tgt01, step, tag):
        with torch.no_grad():
            diff = (torch.fft.fft2(pred01[:1].cpu().float()) - torch.fft.fft2(tgt01[:1].cpu().float()))
            mag = (diff.real ** 2 + diff.imag ** 2).sqrt().mean(dim=1, keepdim=True)
            mag = mag / (mag.max() + 1e-8)
            self.w.add_image(f"{tag}/fft_err", mag[0], step)

    def residual(self, pred01, tgt01, step, tag):
        with torch.no_grad():
            r = (pred01[:1] - tgt01[:1]).abs().mean(dim=1, keepdim=True).cpu()
            r = r / (r.max() + 1e-8)
            self.w.add_image(f"{tag}/residual", r[0], step)


def vram_bar(pct, w=20):
    filled = int(pct / 100 * w)
    return "█" * filled + "░" * (w - filled)


def mk_loss_table(title, keys, m):
    t = Table(title=title, expand=True, show_header=True, header_style="bold cyan", border_style="dim")
    t.add_column("Metric", style="white", ratio=2)
    t.add_column("Value", style="green", ratio=1)
    t.add_column("Avg", style="yellow", ratio=1)
    t.add_column("Trend", style="magenta", ratio=3)
    for k, label in keys:
        v = m.last(k)
        a = m.avg(k)
        s = m.spark(k, 20)
        fmt = f"{v:.2e}" if abs(v) < 0.01 or abs(v) > 999 else f"{v:.4f}"
        afmt = f"{a:.2e}" if abs(a) < 0.01 or abs(a) > 999 else f"{a:.4f}"
        t.add_row(label, fmt, afmt, s)
    return t


def mk_dash(stage, step, total, t0, m, progress, extra_panels=None):
    layout = Layout()
    elapsed = time.time() - t0
    its = step / elapsed if elapsed > 0 else 0
    eta_s = (total - step) / its if its > 0 else 0
    eta = str(datetime.timedelta(seconds=int(eta_s)))
    el = str(datetime.timedelta(seconds=int(elapsed)))

    header = Panel(
        Text.from_markup(
            f"[bold cyan]DenoiseGAN[/] · [bold yellow]{stage}[/]  │  "
            f"Step [bold]{step:,}[/]/{total:,}  │  "
            f"⏱ {el}  │  ETA {eta}  │  "
            f"[bold green]{its:.1f}[/] it/s"
        ),
        style="bold", border_style="bright_blue"
    )

    g = gpu_stats()
    if g:
        vp = g.get("vram_pct", 0)
        gpu_txt = (
            f"VRAM: {g.get('vram_used',0):.1f}/{g.get('vram_total',0):.1f} GB  "
            f"[{vram_bar(vp)}] {vp:.0f}%\n"
            f"Util: {g.get('util',-1)}%  Temp: {g.get('temp',-1)}°C"
        )
    else:
        gpu_txt = "No GPU"
    gpu_panel = Panel(gpu_txt, title="🖥 GPU", border_style="dim green")

    info_parts = []
    for k, label in [("lr", "LR"), ("lr_g", "LR(G)"), ("lr_d", "LR(D)")]:
        if k in m.d:
            info_parts.append(f"{label}: {m.last(k):.2e}")
    for k, label in [("gn", "∇G"), ("gn_g", "∇G"), ("gn_d", "∇D")]:
        if k in m.d:
            info_parts.append(f"{label}: {m.last(k):.3f}")
    if "ema_d" in m.d:
        info_parts.append(f"EMA Δ: {m.last('ema_d'):.5f}")
    info_panel = Panel("\n".join(info_parts) if info_parts else "—", title="⚙ Training", border_style="dim yellow")

    panels = extra_panels or []

    quality_keys = []
    if "psnr" in m.d:
        quality_keys.append(("psnr", "PSNR (dB)"))
    if "ssim" in m.d:
        quality_keys.append(("ssim", "SSIM"))
    if quality_keys:
        panels.append(mk_loss_table("📊 Quality", quality_keys, m))

    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(name="body"),
        Layout(name="bottom", size=6),
        Layout(progress, name="footer", size=3),
    )

    if panels:
        if len(panels) == 1:
            layout["body"].update(panels[0])
        elif len(panels) == 2:
            layout["body"].split_row(
                Layout(panels[0]),
                Layout(panels[1]),
            )
        else:
            layout["body"].split_row(*[Layout(p) for p in panels])

    layout["bottom"].split_row(
        Layout(info_panel),
        Layout(gpu_panel),
    )

    return layout


def to01(x):
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def psnr(pred01, target01):
    mse = F.mse_loss(pred01.clamp(0, 1), target01.clamp(0, 1))
    if mse.item() == 0:
        return torch.tensor(99.0, device=pred01.device)
    return 10.0 * torch.log10(1.0 / mse)


class EMA:
    def __init__(self, model, decay=0.999, device='cpu'):
        self.decay = decay
        self.device = device
        self.shadow = {
            k: v.detach().clone().to(device) for k, v in _raw(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        msd = _raw(model).state_dict()
        for k, v in self.shadow.items():
            sv = msd[k].detach().to(self.device)
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(sv, alpha=1.0 - self.decay)
            else:
                v.copy_(sv)

    @torch.no_grad()
    def copy_to(self, model):
        msd = _raw(model).state_dict()
        for k, v in self.shadow.items():
            msd[k].data.copy_(v.to(msd[k].device))


def random_mask(B, H, W, mask_ratio=0.6, patch=16, device='cuda'):
    mh = H // patch
    mw = W // patch
    nm = mh * mw
    keep = int(nm * (1.0 - mask_ratio))
    mask = torch.zeros(B, nm, device=device)
    for i in range(B):
        idx = torch.randperm(nm, device=device)[keep:]
        mask[i, idx] = 1.0
    mask = mask.view(B, 1, mh, mw)
    mask = F.interpolate(mask, scale_factor=patch, mode='nearest')
    return mask


def make_optimizer_groups_g(model):
    no_decay, with_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith('.bias') or 'gamma' in n or 'temperature' in n:
            no_decay.append(p)
        else:
            with_decay.append(p)
    return [
        {'params': with_decay, 'weight_decay': 1e-4},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def save_ckpt(path, G, D, ema, opt_g, opt_d, step, stage, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'G': _raw(G).state_dict(),
        'ema': ema.shadow,
        'step': step,
        'stage': stage,
    }
    if D is not None:
        payload['D'] = _raw(D).state_dict()
    if opt_g is not None:
        payload['opt_g'] = opt_g.state_dict()
    if opt_d is not None:
        payload['opt_d'] = opt_d.state_dict()
    if extra is not None:
        payload['extra'] = extra
    torch.save(payload, path)


def load_resume(path, G, ema):
    print(f"[resume] loading {path}")
    ck = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = _raw(G).load_state_dict(ck['G'], strict=False)
    if missing:
        print(f"[resume] missing G keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[resume] unexpected G keys: {len(unexpected)} (first: {unexpected[:3]})")
    if 'ema' in ck and ema is not None:
        ckpt_shadow = ck['ema']
        restored = 0
        for k in list(ema.shadow.keys()):
            if k in ckpt_shadow:
                ema.shadow[k] = ckpt_shadow[k].to(ema.device)
                restored += 1
        new_keys = [k for k in ema.shadow if k not in ckpt_shadow]
        if new_keys:
            print(f"[resume] EMA shadow: {restored} restored, "
                  f"{len(new_keys)} new keys kept from init "
                  f"(first: {new_keys[:3]})")
        else:
            print(f"[resume] restored EMA shadow ({restored} tensors)")
    return ck


def restore_stage_optim(ck, stage, opt_g=None, opt_d=None, D=None):
    if ck is None or ck.get('stage') != stage:
        return 0, None
    step = int(ck.get('step', 0))
    if D is not None and 'D' in ck:
        try:
            _raw(D).load_state_dict(ck['D'], strict=False)
            print(f"[resume] restored D state for stage '{stage}'")
        except Exception as e:
            print(f"[resume] D load failed: {e}")
    if opt_g is not None and 'opt_g' in ck:
        try:
            opt_g.load_state_dict(ck['opt_g'])
            print(f"[resume] restored opt_g for stage '{stage}'")
        except Exception as e:
            print(f"[resume] opt_g load failed: {e}")
    if opt_d is not None and 'opt_d' in ck:
        try:
            opt_d.load_state_dict(ck['opt_d'])
            print(f"[resume] restored opt_d for stage '{stage}'")
        except Exception as e:
            print(f"[resume] opt_d load failed: {e}")
    return step, ck.get('extra', None)


def amp_ctx(device, dtype):
    if dtype == 'bf16':
        return autocast(device.type, dtype=torch.bfloat16)
    if dtype == 'fp16':
        return autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def stage_mim(args, device, G, ema, log_dir, sample_dir, tb, resume_state=None):
    loader = build_clean_loader(args.data, args.batch_size, keep_list=args.keep_list,
                                image_size=args.image_size,
                                augment=True, num_workers=args.workers)

    g_params = make_optimizer_groups_g(G)
    opt = torch.optim.AdamW(g_params, lr=args.lr_g, betas=(0.9, 0.99))
    scaler = GradScaler(device.type, enabled=args.amp == 'fp16')

    iters_per_epoch = len(loader)
    total_iters = max(iters_per_epoch * args.mim_epochs, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters, eta_min=1e-7)

    start_step, _ = restore_stage_optim(resume_state, 'mim', opt_g=opt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_step):
            sched.step()
    if start_step:
        print(f"[resume] MIM resuming from step {start_step}/{total_iters}")
    if start_step >= total_iters:
        print(f"[resume] MIM already completed (step {start_step} >= {total_iters}); skipping stage.")
        return

    m = MT()
    progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                        BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%",
                        TimeRemainingColumn(), expand=True)
    task_id = progress.add_task("MIM", total=total_iters)
    progress.update(task_id, completed=start_step)

    G.train()
    t0 = time.time()
    data_iter = iter(loader)

    with Live(mk_dash("MIM", start_step, total_iters, t0, m, progress),
              refresh_per_second=4, console=Console(stderr=True)) as live:
        for step in range(start_step, total_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            clean, _ = batch
            clean = clean.to(device, non_blocking=True)
            mask = random_mask(clean.shape[0], clean.shape[2], clean.shape[3],
                               mask_ratio=args.mim_ratio, patch=16, device=device)
            inp = clean * (1.0 - mask)
            with amp_ctx(device, args.amp):
                pred = G(inp)
                loss_pix = charbonnier(pred * mask, clean * mask, eps=1e-3)
                loss = loss_pix

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(G)

            m.up("loss", loss.item())
            m.up("lr", sched.get_last_lr()[0])

            if step % args.tb_every == 0:
                gn = grad_norm(G.parameters())
                m.up("gn", gn)
                with torch.no_grad():
                    p01 = to01(pred).detach()
                    c01 = to01(clean).detach()
                    pv = psnr(p01, c01).item()
                    sv = ssim(p01, c01)
                m.up("psnr", pv)
                m.up("ssim", sv)
                tb.scalars({"MIM/loss": loss.item(), "MIM/psnr": pv, "MIM/ssim": sv,
                            "MIM/grad_norm": gn, "MIM/lr": sched.get_last_lr()[0]}, step)

            if step > 0 and step % args.sample_every == 0:
                with torch.no_grad():
                    grid = torch.cat([to01(inp[:4]), to01(pred[:4]), to01(clean[:4])], dim=0)
                save_image(grid, os.path.join(sample_dir, f"mim_{step:07d}.png"), nrow=4)
                tb.imgs("MIM/samples", grid, step, nrow=4)

            if step > 0 and step % args.hist_every == 0:
                tb.hists(G, step, "MIM")
                tb.gnorms(G, step, "MIM")
                ed = tb.ema_diff(G, ema, step, "MIM")
                m.up("ema_d", ed)

            if step > 0 and step % 2000 == 0:
                with torch.no_grad():
                    p01 = to01(pred).detach()
                    c01 = to01(clean).detach()
                tb.fft_img(p01, c01, step, "MIM")

            if step > 0 and step % args.save_every == 0:
                save_ckpt(os.path.join(args.ckpt_dir, f"mim_{step:07d}.pt"),
                          G, None, ema, opt, None, step, 'mim')

            progress.update(task_id, completed=step)
            if step % 10 == 0:
                loss_keys = [("loss", "Pixel Loss")]
                lp = mk_loss_table("MIM Losses", loss_keys, m)
                live.update(mk_dash("MIM", step, total_iters, t0, m, progress, [lp]))

    save_ckpt(os.path.join(args.ckpt_dir, "mim_final.pt"),
              G, None, ema, opt, None, step, 'mim')


def stage_psnr(args, device, G, ema, log_dir, sample_dir, tb, resume_state=None):
    loader = build_dataloader(args.data, args.batch_size,
                              image_size=args.image_size,
                              augment=True, num_workers=args.workers,
                              keep_list=args.keep_list)

    g_params = make_optimizer_groups_g(G)

    start_step_peek = 0
    if resume_state is not None and resume_state.get('stage') == 'psnr':
        start_step_peek = int(resume_state.get('step', 0))

    total_iters = args.psnr_iters

    using_soap = args.psnr_use_soap and (
        args.adamw_warmup == 0 or start_step_peek >= args.adamw_warmup)
    if using_soap:
        opt = SOAP(g_params, lr=args.lr_g_soap, betas=(0.95, 0.95),
                   weight_decay=0.0, precondition_frequency=10,
                   max_precond_dim=2048, precondition_1d=False)
    else:
        opt = torch.optim.AdamW(g_params, lr=args.lr_g, betas=(0.9, 0.99))

    scaler = GradScaler(device.type, enabled=args.amp == 'fp16')
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters, eta_min=1e-7)

    start_step, _ = restore_stage_optim(resume_state, 'psnr', opt_g=opt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_step):
            sched.step()
    if start_step:
        print(f"[resume] PSNR resuming from step {start_step}/{total_iters}")
    if start_step >= total_iters:
        print(f"[resume] PSNR already completed (step {start_step} >= {total_iters}); skipping stage.")
        return

    ffl = FocalFrequencyLoss(alpha=1.0).to(device)
    gv_loss = GradVarianceLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    lpips_fn.eval()

    m = MT()
    progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                        BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%",
                        TimeRemainingColumn(), expand=True)
    task_id = progress.add_task("PSNR", total=total_iters)
    progress.update(task_id, completed=start_step)

    G.train()
    data_iter = iter(loader)
    t0 = time.time()

    with Live(mk_dash("PSNR", start_step, total_iters, t0, m, progress),
              refresh_per_second=4, console=Console(stderr=True)) as live:
        for step in range(start_step, total_iters):
            if args.psnr_use_soap and (not using_soap) and step == args.adamw_warmup:
                del opt, sched
                torch.cuda.empty_cache()
                opt = SOAP(g_params, lr=args.lr_g_soap, betas=(0.95, 0.95),
                           weight_decay=0.0, precondition_frequency=10,
                           max_precond_dim=2048, precondition_1d=False)
                remaining = max(1, total_iters - step)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=1e-7)
                using_soap = True
                print(f"[step {step}] PSNR: Switched optimizer AdamW → SOAP")

            try:
                noisy, clean, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                noisy, clean, _ = next(data_iter)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with amp_ctx(device, args.amp):
                pred = G(noisy)
                pred01 = to01(pred)
                clean01 = to01(clean)
                l_charb = charbonnier(pred, clean, eps=1e-3)
                l_ffl = ffl(pred01, clean01)
                l_lab = lab_chroma_loss(pred01, clean01)
                l_gv = gv_loss(pred01, clean01)
                l_lpips = lpips_fn(pred, clean).mean()
                loss = (1.0 * l_charb + 0.05 * l_ffl + 0.05 * l_lab +
                        0.1 * l_gv + 5.0 * l_lpips)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(G)

            m.up("total", loss.item())
            m.up("charb", l_charb.item())
            m.up("ffl", l_ffl.item())
            m.up("lab", l_lab.item())
            m.up("gv", l_gv.item())
            m.up("lpips", l_lpips.item())
            m.up("lr", sched.get_last_lr()[0])

            if step % args.tb_every == 0:
                gn = grad_norm(G.parameters())
                m.up("gn", gn)
                with torch.no_grad():
                    pv = psnr(pred01.detach(), clean01.detach()).item()
                    sv = ssim(pred01.detach(), clean01.detach())
                m.up("psnr", pv)
                m.up("ssim", sv)
                tb.scalars({
                    "PSNR/loss_total": loss.item(), "PSNR/loss_charb": l_charb.item(),
                    "PSNR/loss_ffl": l_ffl.item(), "PSNR/loss_lab": l_lab.item(),
                    "PSNR/loss_gv": l_gv.item(), "PSNR/loss_lpips": l_lpips.item(),
                    "PSNR/psnr": pv, "PSNR/ssim": sv,
                    "PSNR/grad_norm": gn, "PSNR/lr": sched.get_last_lr()[0],
                }, step)

            if step % args.sample_every == 0:
                with torch.no_grad():
                    grid = torch.cat([to01(noisy[:4]), pred01[:4].detach(), clean01[:4].detach()], dim=0)
                save_image(grid, os.path.join(sample_dir, f"psnr_{step:07d}.png"), nrow=4)
                tb.imgs("PSNR/samples", grid, step, nrow=4)

            if step % args.hist_every == 0 and step > 0:
                tb.hists(G, step, "PSNR")
                tb.gnorms(G, step, "PSNR")
                ed = tb.ema_diff(G, ema, step, "PSNR")
                m.up("ema_d", ed)

            if step % 2000 == 0 and step > 0:
                with torch.no_grad():
                    tb.fft_img(pred01.detach(), clean01.detach(), step, "PSNR")
                    tb.residual(pred01.detach(), clean01.detach(), step, "PSNR")

            if step > 0 and step % args.save_every == 0:
                save_ckpt(os.path.join(args.ckpt_dir, f"psnr_{step:07d}.pt"),
                          G, None, ema, opt, None, step, 'psnr')

            progress.update(task_id, completed=step)
            if step % 10 == 0:
                loss_keys = [("total", "Total"), ("charb", "Charbonnier"), ("ffl", "FFL"),
                             ("lab", "Lab Chroma"), ("gv", "GradVar"), ("lpips", "LPIPS")]
                lp = mk_loss_table("📉 PSNR Losses", loss_keys, m)
                live.update(mk_dash("PSNR", step, total_iters, t0, m, progress, [lp]))

    save_ckpt(os.path.join(args.ckpt_dir, "psnr_final.pt"),
              G, None, ema, opt, None, step, 'psnr')


def stage_gan(args, device, G, ema, log_dir, sample_dir, tb, resume_state=None):
    loader = build_dataloader(args.data, args.batch_size,
                              image_size=args.image_size,
                              augment=True, num_workers=args.workers,
                              keep_list=args.keep_list)

    D_unet = UNetDiscriminatorSN(in_channels=3, num_feat=args.d_feat).to(device)
    D_dino = DinoProjectedDiscriminator(layers=(2, 5, 8, 11)).to(device)

    start_step_peek = 0
    if resume_state is not None and resume_state.get('stage') == 'gan':
        start_step_peek = int(resume_state.get('step', 0))

    if hasattr(G, '_orig_mod'):
        print("[gan] Unwrapping compiled Generator to avoid compilation instabilities in the GAN stage.")
        G = G._orig_mod
    elif args.compile:
        print("[gan] D_unet and D_dino are intentionally NOT compiled.")

    if args.channels_last:
        G = G.to(memory_format=torch.channels_last)
        D_unet = D_unet.to(memory_format=torch.channels_last)
        print("[gan] channels_last enabled for G + UNet-D.")

    d_params = list(D_unet.parameters())
    for h in D_dino.heads:
        d_params.extend(list(h.parameters()))

    g_groups = make_optimizer_groups_g(G)

    using_soap = (args.adamw_warmup == 0) or (start_step_peek >= args.adamw_warmup)
    if using_soap:
        opt_g = SOAP(g_groups, lr=args.lr_g_soap, betas=(0.95, 0.95),
                     weight_decay=0.0, precondition_frequency=10,
                     max_precond_dim=2048, precondition_1d=False)
    else:
        opt_g = torch.optim.AdamW(g_groups, lr=args.lr_g, betas=(0.9, 0.99))

    opt_d = torch.optim.AdamW(d_params, lr=args.lr_d, betas=(0.5, 0.9), weight_decay=0.0)

    sched_g = torch.optim.lr_scheduler.MultiStepLR(
        opt_g, milestones=[int(args.gan_iters * 0.5), int(args.gan_iters * 0.75)], gamma=0.5)
    sched_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[int(args.gan_iters * 0.5), int(args.gan_iters * 0.75)], gamma=0.5)

    start_step, _ = restore_stage_optim(resume_state, 'gan', opt_g=opt_g, opt_d=opt_d, D=D_unet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_step):
            sched_g.step()
            sched_d.step()
    if start_step:
        print(f"[resume] GAN resuming from step {start_step}/{args.gan_iters} "
              f"(opt={'SOAP' if using_soap else 'AdamW'})")
    if start_step >= args.gan_iters:
        print(f"[resume] GAN already completed (step {start_step} >= {args.gan_iters}); skipping stage.")
        return

    ffl = FocalFrequencyLoss(alpha=1.0).to(device)
    gv_loss = GradVarianceLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    lpips_fn.eval()

    m = MT()
    progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                        BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%",
                        TimeRemainingColumn(), expand=True)
    task_id = progress.add_task("GAN", total=args.gan_iters)
    progress.update(task_id, completed=start_step)

    data_iter = iter(loader)
    t0 = time.time()
    nan_consecutive = 0
    MAX_NAN = 50

    # Snapshot D for NaN recovery (EMA only covers G) — stored on CPU to save VRAM
    def _snapshot_d():
        return {
            'unet': {k: v.detach().cpu().clone() for k, v in _raw(D_unet).state_dict().items()},
            'dino_heads': [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in D_dino.heads],
        }
    def _restore_d(snap):
        _raw(D_unet).load_state_dict({k: v.to(device) for k, v in snap['unet'].items()})
        for h, sd in zip(D_dino.heads, snap['dino_heads']):
            h.load_state_dict({k: v.to(device) for k, v in sd.items()})
    d_snapshot = _snapshot_d()

    with Live(mk_dash("GAN", start_step, args.gan_iters, t0, m, progress),
              refresh_per_second=4, console=Console(stderr=True)) as live:
        for step in range(start_step, args.gan_iters):
            if (not using_soap) and step == args.adamw_warmup:
                # Free AdamW state before allocating SOAP
                del opt_g, sched_g
                torch.cuda.empty_cache()
                opt_g = SOAP(g_groups, lr=args.lr_g_soap, betas=(0.95, 0.95),
                             weight_decay=0.0, precondition_frequency=10,
                             max_precond_dim=2048, precondition_1d=False)
                sched_g = torch.optim.lr_scheduler.MultiStepLR(
                    opt_g,
                    milestones=[max(1, int(args.gan_iters * 0.5) - step),
                                max(1, int(args.gan_iters * 0.75) - step)],
                    gamma=0.5)
                using_soap = True
                print(f"[step {step}] Switched G optimizer: AdamW → SOAP")

            try:
                noisy, clean, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                noisy, clean, _ = next(data_iter)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if args.channels_last:
                noisy = noisy.to(memory_format=torch.channels_last)
                clean = clean.to(memory_format=torch.channels_last)

            for p in D_unet.parameters():
                p.requires_grad = True
            for h in D_dino.heads:
                for p in h.parameters():
                    p.requires_grad = True

            use_dino = (args.dino_every <= 1) or (step % args.dino_every == 0)

            # --- D main loss (in autocast) ---
            with amp_ctx(device, args.amp):
                with torch.no_grad():
                    fake = G(noisy)
                fake_d = fake.detach()
                unet_real = D_unet(clean)
                unet_fake = D_unet(fake_d)
                dino_real = D_dino(clean) if use_dino else []
                dino_fake = D_dino(fake_d) if use_dino else []
                d_real = aggregate_unet_dino(unet_real, dino_real)
                d_fake = aggregate_unet_dino(unet_fake, dino_fake)
                loss_d_main = rpgan_d_loss(d_real, d_fake)

            step_nan = False
            do_r1 = (args.lazy_r1 <= 0) or (step % args.lazy_r1 == 0)
            r1 = torch.zeros(1, device=device)
            r2 = torch.zeros(1, device=device)

            # Early NaN gate: check D outputs before any backward
            d_out_ok = (torch.isfinite(d_real).all() and torch.isfinite(d_fake).all()
                        and torch.isfinite(loss_d_main))
            if not d_out_ok:
                opt_d.zero_grad(set_to_none=True)
                step_nan = True
            else:
                # Phase 1: backward main D loss → frees bf16 graph
                opt_d.zero_grad(set_to_none=True)
                loss_d_main.backward()

                # Check D grads for NaN before stepping
                d_grad_ok = all(
                    torch.isfinite(p.grad).all()
                    for p in d_params if p.grad is not None
                )
                if not d_grad_ok:
                    opt_d.zero_grad(set_to_none=True)
                    step_nan = True
                else:
                    if do_r1:
                        lazy_k = max(float(args.lazy_r1), 1.0)

                        # Phase 2: R1 on real (fp32 forward → backward → freed)
                        clean_r1 = clean.detach().float().requires_grad_(True)
                        with torch.amp.autocast(device.type, enabled=False):
                            ur1 = D_unet(clean_r1)
                            dr1 = D_dino(clean_r1) if use_dino else []
                            d_real_r1 = aggregate_unet_dino(ur1, dr1)
                        r1 = r1_penalty([d_real_r1], clean_r1).clamp(max=1e4)
                        r1_scaled = (args.r1_gamma * lazy_k / 2.0) * r1
                        if torch.isfinite(r1_scaled):
                            r1_scaled.backward()
                        else:
                            step_nan = True
                        del clean_r1, ur1, dr1, d_real_r1, r1_scaled

                        # Phase 3: R2 on fake (fp32 forward → backward → freed)
                        if not step_nan:
                            fake_r2 = fake_d.detach().float().requires_grad_(True)
                            with torch.amp.autocast(device.type, enabled=False):
                                uf2 = D_unet(fake_r2)
                                df2 = D_dino(fake_r2) if use_dino else []
                                d_fake_r2 = aggregate_unet_dino(uf2, df2)
                            r2 = r2_penalty([d_fake_r2], fake_r2).clamp(max=1e4)
                            r2_scaled = (args.r2_gamma * lazy_k / 2.0) * r2
                            if torch.isfinite(r2_scaled):
                                r2_scaled.backward()
                            else:
                                step_nan = True
                            del fake_r2, uf2, df2, d_fake_r2, r2_scaled

                    if not step_nan:
                        torch.nn.utils.clip_grad_norm_(d_params, 5.0)
                        opt_d.step()
                        sched_d.step()
                        # Snapshot D periodically (cheap, only on successful steps)
                        if step % 500 == 0:
                            d_snapshot = _snapshot_d()

            loss_d = loss_d_main.float().detach() + r1.detach() + r2.detach()

            if not step_nan:
                for p in D_unet.parameters():
                    p.requires_grad = False
                for h in D_dino.heads:
                    for p in h.parameters():
                        p.requires_grad = False

                with amp_ctx(device, args.amp):
                    fake = G(noisy)
                    fake01 = to01(fake)
                    clean01 = to01(clean)
                    with torch.no_grad():
                        unet_real_ng = D_unet(clean)
                        dino_real_ng = D_dino(clean) if use_dino else []
                        d_real = aggregate_unet_dino(unet_real_ng, dino_real_ng)
                    unet_fake = D_unet(fake)
                    dino_fake = D_dino(fake) if use_dino else []
                    d_fake = aggregate_unet_dino(unet_fake, dino_fake)
                    l_charb = charbonnier(fake, clean, eps=1e-3)
                    l_ffl = ffl(fake01, clean01)
                    l_lab = lab_chroma_loss(fake01, clean01)
                    l_gv = gv_loss(fake01, clean01)
                    l_lpips = lpips_fn(fake, clean).mean()
                    l_gan = rpgan_g_loss(d_real, d_fake)
                    loss_g = (1.0 * l_charb + 0.1 * l_ffl + 0.2 * l_lab +
                              0.3 * l_gv + 1.0 * l_lpips + args.gan_weight * l_gan)

                if not torch.isfinite(loss_g):
                    opt_g.zero_grad(set_to_none=True)
                    step_nan = True
                else:
                    opt_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
                    opt_g.step()
                    sched_g.step()
                    ema.update(G)

            # --- NaN tracking & recovery ---
            if step_nan:
                nan_consecutive += 1
                m.up("nan_skip", nan_consecutive)
                if nan_consecutive >= MAX_NAN:
                    print(f"\n[!] {MAX_NAN} consecutive NaN steps at step {step}. "
                          f"Rolling back G from EMA + D from snapshot...")
                    ema.copy_to(_raw(G))
                    _restore_d(d_snapshot)
                    # Reset D optimizer to clear corrupted momentum
                    opt_d.zero_grad(set_to_none=True)
                    for state in opt_d.state.values():
                        state.clear()
                    nan_consecutive = 0
            else:
                nan_consecutive = 0
                m.up("g_total", loss_g.item())
                m.up("g_charb", l_charb.item())
                m.up("g_ffl", l_ffl.item())
                m.up("g_lab", l_lab.item())
                m.up("g_gv", l_gv.item())
                m.up("g_lpips", l_lpips.item())
                m.up("g_adv", l_gan.item())
                m.up("d_total", loss_d.item())
                m.up("d_main", loss_d_main.item())
                m.up("d_r1", r1.item())
                m.up("d_r2", r2.item())
                m.up("d_real_m", d_real.detach().mean().item())
                m.up("d_fake_m", d_fake.detach().mean().item())
                m.up("lr_g", opt_g.param_groups[0]['lr'])
                m.up("lr_d", opt_d.param_groups[0]['lr'])

            if not step_nan and step % args.tb_every == 0:
                gn_g = grad_norm(G.parameters())
                gn_d = grad_norm(d_params)
                m.up("gn_g", gn_g)
                m.up("gn_d", gn_d)
                with torch.no_grad():
                    pv = psnr(fake01.detach(), clean01.detach()).item()
                    sv = ssim(fake01.detach(), clean01.detach())
                m.up("psnr", pv)
                m.up("ssim", sv)
                tb.scalars({
                    "GAN/loss_g_total": loss_g.item(), "GAN/loss_g_charb": l_charb.item(),
                    "GAN/loss_g_ffl": l_ffl.item(), "GAN/loss_g_lab": l_lab.item(),
                    "GAN/loss_g_gv": l_gv.item(), "GAN/loss_g_lpips": l_lpips.item(),
                    "GAN/loss_g_adv": l_gan.item(),
                    "GAN/loss_d_total": loss_d.item(), "GAN/loss_d_main": loss_d_main.item(),
                    "GAN/r1": r1.item(), "GAN/r2": r2.item(),
                    "GAN/d_real_mean": d_real.detach().mean().item(),
                    "GAN/d_fake_mean": d_fake.detach().mean().item(),
                    "GAN/psnr": pv, "GAN/ssim": sv,
                    "GAN/grad_norm_G": gn_g, "GAN/grad_norm_D": gn_d,
                    "GAN/lr_g": opt_g.param_groups[0]['lr'],
                    "GAN/lr_d": opt_d.param_groups[0]['lr'],
                }, step)

            if not step_nan and step % args.sample_every == 0:
                with torch.no_grad():
                    grid = torch.cat([to01(noisy[:4]), fake01[:4].detach(), clean01[:4].detach()], dim=0)
                save_image(grid, os.path.join(sample_dir, f"gan_{step:07d}.png"), nrow=4)
                tb.imgs("GAN/samples", grid, step, nrow=4)

            if not step_nan and step % args.hist_every == 0 and step > 0:
                tb.hists(G, step, "GAN_G")
                tb.hists(D_unet, step, "GAN_D")
                tb.gnorms(G, step, "GAN_G")
                tb.gnorms(D_unet, step, "GAN_D")
                ed = tb.ema_diff(G, ema, step, "GAN")
                m.up("ema_d", ed)

            if not step_nan and step % 2000 == 0 and step > 0:
                with torch.no_grad():
                    tb.fft_img(fake01.detach(), clean01.detach(), step, "GAN")
                    tb.residual(fake01.detach(), clean01.detach(), step, "GAN")
                    d_heatmap = torch.sigmoid(unet_real.detach()[:1]).cpu()
                    d_heatmap = d_heatmap / (d_heatmap.max() + 1e-8)
                    tb.w.add_image("GAN/d_heatmap", d_heatmap[0], step)

            if not step_nan and args.viz_every > 0 and step % args.viz_every == 0:
                try:
                    import viz
                    extra = []
                    for _ in range(max(0, args.viz_umap_batches - 1)):
                        try:
                            vn, vc, _ = next(data_iter)
                        except StopIteration:
                            data_iter = iter(loader)
                            vn, vc, _ = next(data_iter)
                        extra.append((vn, vc))
                    viz.log_gan_visuals(
                        tb, step, "GAN", G, D_dino,
                        cur_batch=(noisy, fake.detach(), clean),
                        extra_batches=extra,
                        dscores=(d_real.detach(), d_fake.detach()),
                        device=device, amp=args.amp,
                        n_umap_batches=args.viz_umap_batches,
                    )
                except Exception as ex:
                    print(f"[viz] step {step} failed (non-fatal): "
                          f"{type(ex).__name__}: {ex}")

            if step > 0 and step % args.save_every == 0:
                save_ckpt(os.path.join(args.ckpt_dir, f"gan_{step:07d}.pt"),
                          G, D_unet, ema, opt_g, opt_d, step, 'gan')

            progress.update(task_id, completed=step)
            if step % 10 == 0:
                g_keys = [("g_total", "Total"), ("g_charb", "Charb"), ("g_ffl", "FFL"),
                          ("g_lab", "Lab"), ("g_gv", "GradVar"), ("g_lpips", "LPIPS"), ("g_adv", "Adv")]
                d_keys = [("d_total", "Total"), ("d_main", "Main"), ("d_r1", "R1"), ("d_r2", "R2"),
                          ("d_real_m", "D(real)"), ("d_fake_m", "D(fake)")]
                gp = mk_loss_table("📉 Generator", g_keys, m)
                dp = mk_loss_table("📉 Discriminator", d_keys, m)
                extra_info = ""
                if nan_consecutive > 0:
                    extra_info = f" ⚠ NaN streak: {nan_consecutive}/{MAX_NAN}"
                live.update(mk_dash(f"GAN{extra_info}", step, args.gan_iters, t0, m, progress, [gp, dp]))

    save_ckpt(os.path.join(args.ckpt_dir, "gan_final.pt"),
              G, D_unet, ema, opt_g, opt_d, step, 'gan')


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='/home/algis/Desktop/data/train')
    p.add_argument('--keep-list', type=str, default=None,
                   help='Path to keep_files.txt from scan_clean_noise.py; restricts training '
                        'to verified-clean targets (used as both recon target and D real set).')
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    p.add_argument('--sample-dir', type=str, default='./samples')
    p.add_argument('--log-dir', type=str, default='./logs')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--reset-step', action='store_true',
                   help='Resume weights (G + EMA) only; ignore saved step / opt / D / scheduler. '
                        'Use this to fine-tune from an existing checkpoint with a fresh budget.')
    p.add_argument('--stage', type=str, default='all', choices=['all', 'mim', 'psnr', 'gan'])

    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])

    p.add_argument('--mim-epochs', type=int, default=5)
    p.add_argument('--mim-ratio', type=float, default=0.6)

    p.add_argument('--psnr-iters', type=int, default=200_000)
    p.add_argument('--gan-iters', type=int, default=400_000)

    p.add_argument('--lr-g', type=float, default=2e-4)
    p.add_argument('--lr-g-soap', type=float, default=2e-4)
    p.add_argument('--lr-d', type=float, default=2e-4)
    p.add_argument('--adamw-warmup', type=int, default=10_000)
    p.add_argument('--psnr-use-soap', action='store_true',
                   help='Use SOAP optimizer in PSNR stage (after adamw-warmup steps)')

    p.add_argument('--gan-weight', type=float, default=0.1)
    p.add_argument('--r1-gamma', type=float, default=0.1)
    p.add_argument('--r2-gamma', type=float, default=0.1)
    p.add_argument('--lazy-r1', type=int, default=16,
                   help='Compute R1/R2 gradient penalty every k D steps (StyleGAN2-style lazy regularization). '
                        'Higher = faster training, less NaN risk. 0 = every step.')

    p.add_argument('--d-feat', type=int, default=64)
    p.add_argument('--dino-every', type=int, default=1,
                   help='Run the DINO discriminator only every k GAN steps (UNet-SN disc runs every '
                        'step). 2 ~halves the DINO ViT cost; 1 = every step.')
    p.add_argument('--channels-last', action=argparse.BooleanOptionalAction, default=True,
                   help='Use channels_last memory format for the conv nets (G + UNet-D) in the GAN stage.')
    p.add_argument('--viz-every', type=int, default=5000,
                   help='Log rich diagnostics (UMAP, spectra, edges, histograms) every k GAN steps (incl. 0). 0 disables.')
    p.add_argument('--viz-umap-batches', type=int, default=24,
                   help='Number of batches gathered for the DINO-feature UMAP projection.')

    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=10_000)
    p.add_argument('--tb-every', type=int, default=50)
    p.add_argument('--hist-every', type=int, default=500)

    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=1234)

    p.add_argument('--use-checkpoint', dest='use_checkpoint',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Activation checkpointing in G. Off = faster, more VRAM. '
                        'On = ~30%% slower, ~2-3x less peak VRAM. Use --no-use-checkpoint to disable explicitly.')
    p.add_argument('--g-drop-path', type=float, default=0.1,
                   help='Stochastic depth max rate for G blocks.')

    # Advanced compile options
    p.add_argument('--compile', action='store_true', help='Use torch.compile to speed up training')
    p.add_argument('--compile-mode', type=str, default='default',
               choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'])

    p.add_argument('--compile-dynamic', action='store_true', help='Use dynamic shape tracing in torch.compile')
    p.add_argument('--compile-fullgraph', action='store_true', help='Require full graph compilation (no graph breaks)')
    p.add_argument('--compile-backend', type=str, default='inductor', help='Compiler backend to use')

    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    tb = TBL(writer)

    writer.add_text("config/args", str(vars(args)), 0)

    G = DenoiseGenerator(channels=(48, 96, 192, 320, 448),
                         drop_path_rate=args.g_drop_path,
                         use_checkpoint=args.use_checkpoint).to(device)
    n_g = sum(p.numel() for p in G.parameters())

    writer.add_text("config/model", f"G params: {n_g/1e6:.2f} M", 0)

    console = Console(stderr=True)
    console.print(
        f"[bold cyan]DenoiseGAN[/] | G params: [bold]{n_g/1e6:.2f}M[/] | "
        f"device: [bold]{device}[/] | "
        f"ckpt={'ON' if args.use_checkpoint else 'OFF'} | "
        f"compile={'ON' if args.compile else 'OFF'}"
    )

    ema = EMA(G, decay=args.ema_decay, device='cpu')

    resume_state = None
    if args.resume is not None and os.path.exists(args.resume):
        resume_state = load_resume(args.resume, G, ema)
        console.print(
            f"[bold green]Resumed[/] from {args.resume}  "
            f"(stage={resume_state.get('stage')}, step={resume_state.get('step')})"
        )
        if args.reset_step:
            for k in ('step', 'opt_g', 'opt_d', 'D'):
                resume_state.pop(k, None)
            resume_state['stage'] = '__weights_only__'
            console.print(
                "[bold yellow]--reset-step set:[/] using weights only "
                "(step / optimizer / D state discarded)."
            )
    elif args.resume is not None:
        console.print(f"[bold red]--resume given but file not found:[/] {args.resume}")

    if args.compile:
        console.print(
            f"[bold yellow]Compiling G[/] (mode={args.compile_mode}, "
            f"dynamic={args.compile_dynamic}, fullgraph={args.compile_fullgraph}, "
            f"backend={args.compile_backend})"
        )
        G = torch.compile(
            G,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
            fullgraph=args.compile_fullgraph,
            backend=args.compile_backend,
        )

    stages = ['mim', 'psnr', 'gan'] if args.stage == 'all' else [args.stage]
    if 'mim' in stages:
        stage_mim(args, device, G, ema, args.log_dir, args.sample_dir, tb, resume_state)
        resume_state = None
    if 'psnr' in stages:
        stage_psnr(args, device, G, ema, args.log_dir, args.sample_dir, tb, resume_state)
        resume_state = None
    if 'gan' in stages:
        stage_gan(args, device, G, ema, args.log_dir, args.sample_dir, tb, resume_state)

    writer.close()
    console.print("[bold green]Training complete.[/]")


if __name__ == '__main__':
    main()
