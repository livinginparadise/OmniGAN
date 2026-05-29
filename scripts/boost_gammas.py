import argparse
import torch
from pathlib import Path


def collect_gammas(sd):
    return [(k, v) for k, v in sd.items()
            if k.endswith('.gamma1') or k.endswith('.gamma2')]


def report(sd, label):
    gs = collect_gammas(sd)
    if not gs:
        print(f"[{label}] no gamma tensors found")
        return None
    abs_max = max(v.abs().max().item() for _, v in gs)
    abs_min = min(v.abs().min().item() for _, v in gs)
    means = torch.stack([v.float().mean() for _, v in gs])
    print(f"[{label}] {len(gs)} gamma tensors  |  "
          f"|γ| range: [{abs_min:.3e}, {abs_max:.3e}]  |  "
          f"mean across tensors: {means.mean().item():.3e}")
    return abs_max


def scale(sd, factor, label):
    gs = collect_gammas(sd)
    for k, v in gs:
        sd[k] = v * factor
    new_max = max(v.abs().max().item() for _, v in collect_gammas(sd))
    print(f"[{label}] applied factor {factor:.4f}×  |  new max |γ|: {new_max:.3e}")


def wake_dead_tensors(sd, target, dead_threshold, label):
    gs = collect_gammas(sd)
    woken, skipped = [], []
    for k, v in gs:
        m = v.abs().max().item()
        if m < dead_threshold:
            if m == 0:
                sd[k] = torch.full_like(v, float(target))
                woken.append((k, m, target))
            else:
                factor = target / m
                sd[k] = v * factor
                woken.append((k, m, target))
        else:
            skipped.append((k, m))
    print(f"[{label}] woken: {len(woken)} dead tensor(s)  |  "
          f"skipped: {len(skipped)} healthy tensor(s)")
    for k, old, new in woken:
        print(f"  + {k}: max|γ| {old:.2e} → {new:.2e}")
    for k, m in skipped:
        print(f"  · {k}: max|γ| {m:.2e} (kept)")


def main():
    p = argparse.ArgumentParser(
        description="Boost residual layer-scale gammas in a checkpoint.")
    p.add_argument('--src', type=str, required=True,
                   help='source checkpoint (.pt)')
    p.add_argument('--dst', type=str, required=True,
                   help='output checkpoint path')
    p.add_argument('--target', type=float, default=0.1,
                   help='desired max |gamma| after boosting (default 0.1)')
    p.add_argument('--max-factor', type=float, default=1000.0,
                   help='hard cap on scale factor for safety (default 1000)')
    p.add_argument('--inspect', action='store_true',
                   help='print gamma stats only; do not modify or save')
    p.add_argument('--mode', type=str, default='uniform',
                   choices=['uniform', 'wake'],
                   help='uniform: scale all gammas by one factor. '
                        'wake: only revive tensors whose max|γ| is below dead-threshold.')
    p.add_argument('--dead-threshold', type=float, default=1e-2,
                   help='in wake mode, a tensor is considered dead if max|γ| is below this')
    args = p.parse_args()

    print(f"loading: {args.src}")
    ck = torch.load(args.src, map_location='cpu', weights_only=False)

    g_max = report(ck['G'], 'G')
    ema_max = None
    if 'ema' in ck:
        ema_max = report(ck['ema'], 'ema')

    if args.inspect:
        print("[inspect] exiting without saving")
        return

    ref_max = g_max if g_max is not None else ema_max
    if ref_max is None or ref_max == 0:
        print("no usable gammas; nothing to do")
        return

    if args.mode == 'wake':
        print(f"\n[mode=wake]  target={args.target}  dead_threshold={args.dead_threshold}")
        wake_dead_tensors(ck['G'], args.target, args.dead_threshold, 'G')
        if 'ema' in ck:
            wake_dead_tensors(ck['ema'], args.target, args.dead_threshold, 'ema')
    else:
        factor = min(args.max_factor, args.target / max(ref_max, 1e-12))
        factor = max(1.0, factor)
        if factor <= 1.0:
            print(f"gammas already at or above target "
                  f"(current max |γ|={ref_max:.3e} >= target {args.target}); "
                  f"no boost applied")
        else:
            scale(ck['G'], factor, 'G')
            if 'ema' in ck:
                scale(ck['ema'], factor, 'ema')

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.dst)
    print(f"saved: {args.dst}")


if __name__ == '__main__':
    main()
