import os
import csv
import math
import argparse
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

_M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_SCALE = math.sqrt(0.5 * math.pi) / 6.0

_GOOD_DIR = None
_CROP = 512
_BLOCK = 16
_PCTL = 10.0


def _flat_sigma(gray):
    """Low-percentile block sigma -> noise floor, robust to texture/edges."""
    conv = np.abs(cv2.filter2D(gray, cv2.CV_32F, _M, borderType=cv2.BORDER_REFLECT))
    h, w = conv.shape
    bh, bw = max(1, h // _BLOCK), max(1, w // _BLOCK)
    blocks = cv2.resize(conv, (bw, bh), interpolation=cv2.INTER_AREA)
    return float(np.percentile(blocks, _PCTL) * _SCALE)


def _scan_one(fname):
    try:
        img = cv2.imread(os.path.join(_GOOD_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return fname, -1.0
        h, w = img.shape
        if h > _CROP:
            y = (h - _CROP) // 2
            img = img[y:y + _CROP]
        if w > _CROP:
            x = (w - _CROP) // 2
            img = img[:, x:x + _CROP]
        return fname, _flat_sigma(img.astype(np.float32))
    except Exception:
        return fname, -1.0


def _init(good_dir, crop, block, pctl):
    global _GOOD_DIR, _CROP, _BLOCK, _PCTL
    _GOOD_DIR, _CROP, _BLOCK, _PCTL = good_dir, crop, block, pctl


def scan(good_dir, workers, crop, block, pctl, sample):
    files = sorted(os.listdir(good_dir))
    if sample > 0:
        import random
        random.seed(0)
        files = random.sample(files, min(sample, len(files)))
    results = []
    with mp.Pool(workers, initializer=_init,
                 initargs=(good_dir, crop, block, pctl)) as pool:
        for fname, sigma in tqdm(pool.imap_unordered(_scan_one, files, chunksize=64),
                                 total=len(files), desc="scanning", dynamic_ncols=True):
            results.append((fname, sigma))
    return results


def report_and_write(results, threshold, out_dir):
    sig = np.array([s for _, s in results if s >= 0])
    print(f"\nscanned {len(sig)} images ({sum(1 for _, s in results if s < 0)} unreadable)")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:>2}: sigma={np.percentile(sig, q):6.2f} (/255)")
    for thr in [1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"  sigma > {thr:>3}: {(sig > thr).mean() * 100:5.1f}% rejected")

    keep = [f for f, s in results if 0 <= s <= threshold]
    reject = [f for f, s in results if s > threshold or s < 0]
    keep_p = os.path.join(out_dir, "keep_files.txt")
    rej_p = os.path.join(out_dir, "reject_files.txt")
    with open(keep_p, "w") as fh:
        fh.write("\n".join(sorted(keep)) + "\n")
    with open(rej_p, "w") as fh:
        fh.write("\n".join(sorted(reject)) + "\n")
    print(f"\nthreshold sigma <= {threshold}/255")
    print(f"  keep   : {len(keep):>7} -> {keep_p}")
    print(f"  reject : {len(reject):>7} -> {rej_p}  (unreadable rejected too)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--good', type=str, default='/home/algis/Desktop/data/train/good')
    p.add_argument('--out-dir', type=str, default='.')
    p.add_argument('--csv', type=str, default='noise_sigmas.csv')
    p.add_argument('--from-csv', type=str, default=None,
                   help='Re-threshold an existing CSV without re-scanning.')
    p.add_argument('--threshold', type=float, default=1.5,
                   help='Reject good images with flat-region sigma above this (0-255 units).')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--block', type=int, default=16)
    p.add_argument('--pctl', type=float, default=10.0)
    p.add_argument('--sample', type=int, default=0, help='0 = all; else dry-run on N random files.')
    args = p.parse_args()

    if args.from_csv:
        with open(args.from_csv) as fh:
            results = [(r[0], float(r[1])) for r in csv.reader(fh) if r]
    else:
        results = scan(args.good, args.workers, args.crop, args.block, args.pctl, args.sample)
        with open(os.path.join(args.out_dir, args.csv), "w", newline="") as fh:
            csv.writer(fh).writerows(results)
        print(f"wrote {os.path.join(args.out_dir, args.csv)}")

    report_and_write(results, args.threshold, args.out_dir)


if __name__ == '__main__':
    main()
