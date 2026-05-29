import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def _to01(x):
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def _safe(name, fn):
    try:
        fn()
    except Exception as ex:
        print(f"[viz] {name} failed (non-fatal): {type(ex).__name__}: {ex}")


@torch.no_grad()
def _dino_cls(D_dino, imgs):
    """Per-image DINOv2 CLS embedding. imgs in [-1, 1]."""
    x = D_dino._preprocess(imgs)
    feats = D_dino.backbone.get_intermediate_layers(
        x, n=1, return_class_token=True, norm=True)
    cls = feats[0][1]
    return cls.float().cpu().numpy()


def _project(feats):
    feats = np.asarray(feats, dtype=np.float32)
    n = feats.shape[0]
    if n < 6:
        return None, None
    if _HAS_UMAP:
        try:
            nn = max(2, min(15, n - 1))
            reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2,
                                metric='cosine', random_state=42)
            return reducer.fit_transform(feats), 'UMAP'
        except Exception as ex:
            print(f"[viz] UMAP failed, falling back to PCA: {ex}")
    f = feats - feats.mean(0, keepdims=True)
    U, S, _Vt = np.linalg.svd(f, full_matrices=False)
    return U[:, :2] * S[:2], 'PCA'


# ---------------------------------------------------------------- UMAP ------
def _log_umap(tb, step, tag, G, D_dino, cur, extra, device, amp, n_batches):
    noisy, fake, clean = cur
    real_e = [_dino_cls(D_dino, clean)]
    fake_e = [_dino_cls(D_dino, fake)]
    nois_e = [_dino_cls(D_dino, noisy)]

    was_training = G.training
    G.eval()
    ac = torch.autocast(device.type, dtype=torch.bfloat16, enabled=(amp == 'bf16'))
    try:
        for vn, vc in extra[:max(0, n_batches - 1)]:
            vn = vn.to(device, non_blocking=True)
            vc = vc.to(device, non_blocking=True)
            with torch.no_grad(), ac:
                vf = G(vn)
            real_e.append(_dino_cls(D_dino, vc))
            fake_e.append(_dino_cls(D_dino, vf))
            nois_e.append(_dino_cls(D_dino, vn))
    finally:
        if was_training:
            G.train()

    real = np.concatenate(real_e, 0)
    fake = np.concatenate(fake_e, 0)
    nois = np.concatenate(nois_e, 0)
    proj, method = _project(np.concatenate([real, fake, nois], 0))
    if proj is None:
        return
    nr, nf = len(real), len(fake)
    pr, pf, pn = proj[:nr], proj[nr:nr + nf], proj[nr + nf:]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
    ax.scatter(pn[:, 0], pn[:, 1], s=10, alpha=0.35, c='#888', label=f'noisy in ({len(pn)})')
    ax.scatter(pr[:, 0], pr[:, 1], s=14, alpha=0.75, c='#33aa66', label=f'real clean ({nr})')
    ax.scatter(pf[:, 0], pf[:, 1], s=14, alpha=0.75, c='#dd4444', label=f'fake out ({nf})')
    ax.set_title(f'{tag} DINO-CLS {method}  (step {step})\nfake should drift onto real')
    ax.legend(loc='best', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    tb.w.add_figure(f'{tag}/umap_dino', fig, step)
    plt.close(fig)


# --------------------------------------------------- frequency spectrum -----
def _radial_psd(img01):
    g = img01.mean(1)
    Fc = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
    psd = (Fc.abs() ** 2).mean(0).cpu().numpy()
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - cx, yy - cy).astype(int)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _log_radial_psd(tb, step, tag, noisy, fake, clean):
    n01, f01, c01 = _to01(noisy), _to01(fake), _to01(clean)
    rn, rf, rc = _radial_psd(n01), _radial_psd(f01), _radial_psd(c01)
    k = min(len(rn), len(rf), len(rc))
    freq = np.arange(k)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax.plot(freq, np.log10(rn[:k] + 1e-8), c='#888', label='noisy in')
    ax.plot(freq, np.log10(rc[:k] + 1e-8), c='#33aa66', label='real clean')
    ax.plot(freq, np.log10(rf[:k] + 1e-8), c='#dd4444', label='fake out')
    ax.set_xlabel('radial spatial frequency'); ax.set_ylabel('log10 power')
    ax.set_title(f'{tag} radial power spectrum (step {step})\n'
                 'fake under clean at high-freq = over-smoothing')
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    tb.w.add_figure(f'{tag}/radial_psd', fig, step)
    plt.close(fig)


# ------------------------------------------------ removed / residual maps ---
def _norm_map(t):
    t = t - t.amin()
    return t / (t.amax() + 1e-8)


def _log_removed_residual(tb, step, tag, noisy, fake, clean, k=4):
    n01, f01, c01 = _to01(noisy[:k]), _to01(fake[:k]), _to01(clean[:k])
    removed = _norm_map((n01 - f01).abs().mean(1, keepdim=True)).cpu()
    resid = _norm_map((f01 - c01).abs().mean(1, keepdim=True)).cpu()
    tb.w.add_image(f'{tag}/removed_noisy_minus_fake',
                   make_grid(removed, nrow=k), step)
    tb.w.add_image(f'{tag}/residual_fake_minus_clean',
                   make_grid(resid, nrow=k), step)


# -------------------------------------------------------- sobel edges -------
def _sobel(img01):
    g = img01.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=g.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return (gx * gx + gy * gy).sqrt()


def _log_edges(tb, step, tag, noisy, fake, clean, k=3):
    rows = []
    for t in (noisy, fake, clean):
        e = _sobel(_to01(t[:k]))
        rows.append(_norm_map(e).cpu())
    grid = make_grid(torch.cat(rows, 0), nrow=k)
    tb.w.add_image(f'{tag}/edges_noisy_fake_clean', grid, step)


# ----------------------------------------------------- color histograms -----
def _log_color_hist(tb, step, tag, fake, clean):
    f01 = _to01(fake).cpu().numpy()
    c01 = _to01(clean).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=110)
    for ci, (name, col) in enumerate([('R', '#d33'), ('G', '#3a3'), ('B', '#33d')]):
        axes[ci].hist(c01[:, ci].ravel(), bins=64, range=(0, 1), color=col,
                      alpha=0.45, density=True, label='real')
        axes[ci].hist(f01[:, ci].ravel(), bins=64, range=(0, 1), histtype='step',
                      color='k', density=True, label='fake')
        axes[ci].set_title(f'{name}'); axes[ci].set_yticks([])
        if ci == 0:
            axes[ci].legend(fontsize=7)
    fig.suptitle(f'{tag} channel histograms — fake vs real (step {step})')
    fig.tight_layout()
    tb.w.add_figure(f'{tag}/color_hist', fig, step)
    plt.close(fig)


# --------------------------------------------- discriminator score hist -----
def _log_dscore_hist(tb, step, tag, d_real, d_fake):
    dr = d_real.detach().float().cpu().numpy().ravel()
    df = d_fake.detach().float().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=110)
    lo = float(min(dr.min(), df.min()))
    hi = float(max(dr.max(), df.max()))
    rng = (lo, hi) if hi > lo else (lo - 1, hi + 1)
    ax.hist(dr, bins=24, range=rng, color='#33aa66', alpha=0.55, label='D(real)')
    ax.hist(df, bins=24, range=rng, color='#dd4444', alpha=0.55, label='D(fake)')
    ax.set_title(f'{tag} discriminator scores (step {step})\noverlap = G fooling D')
    ax.legend(fontsize=8)
    fig.tight_layout()
    tb.w.add_figure(f'{tag}/d_scores', fig, step)
    plt.close(fig)


def log_gan_visuals(tb, step, tag, G, D_dino, cur_batch, extra_batches,
                    dscores, device, amp='bf16', n_umap_batches=24):
    """Entry point. cur_batch=(noisy,fake,clean) on device in [-1,1];
    extra_batches=list[(noisy_cpu,clean_cpu)]; dscores=(d_real,d_fake) or None."""
    noisy, fake, clean = cur_batch
    _safe('removed/residual', lambda: _log_removed_residual(tb, step, tag, noisy, fake, clean))
    _safe('edges', lambda: _log_edges(tb, step, tag, noisy, fake, clean))
    _safe('radial_psd', lambda: _log_radial_psd(tb, step, tag, noisy, fake, clean))
    _safe('color_hist', lambda: _log_color_hist(tb, step, tag, fake, clean))
    if dscores is not None:
        _safe('d_scores', lambda: _log_dscore_hist(tb, step, tag, dscores[0], dscores[1]))
    _safe('umap', lambda: _log_umap(tb, step, tag, G, D_dino, cur_batch,
                                    extra_batches, device, amp, n_umap_batches))
