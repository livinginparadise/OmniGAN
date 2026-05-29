import os
import csv
import argparse
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

FULL_MAX = 720
CROP_PANEL = 460


class Viewer:
    def __init__(self, root, rows, good_dir, threshold, sort_mode='desc'):
        self.root = root
        self.good_dir = good_dir
        self.threshold = threshold
        self.rows = rows
        self.sort_mode = sort_mode
        self.zoom = 2
        self.idx = 0
        self._full_ref = None
        self._crop_ref = None

        self._build_ui()
        label = {'desc': 'sigma desc', 'asc': 'sigma asc', 'file': 'filename'}[self.sort_mode]
        self.sort_btn.config(text=f"Sort: {label} (s)")
        self._apply_sort()

    def _build_ui(self):
        self.root.title("noise_sigmas checker")
        self.root.configure(bg='#1e1e1e')

        top = tk.Frame(self.root, bg='#1e1e1e')
        top.pack(fill='x', padx=8, pady=6)

        tk.Label(top, text="Threshold", fg='#ccc', bg='#1e1e1e').pack(side='left')
        self.thr_var = tk.StringVar(value=str(self.threshold))
        e = tk.Entry(top, textvariable=self.thr_var, width=6)
        e.pack(side='left', padx=(4, 2))
        e.bind('<Return>', lambda _ev: self._set_threshold())
        tk.Button(top, text="Apply", command=self._set_threshold).pack(side='left', padx=2)

        tk.Button(top, text="Jump to threshold (t)",
                  command=self._jump_threshold).pack(side='left', padx=10)

        tk.Label(top, text="Go to #", fg='#ccc', bg='#1e1e1e').pack(side='left')
        self.jump_var = tk.StringVar()
        je = tk.Entry(top, textvariable=self.jump_var, width=8)
        je.pack(side='left', padx=(4, 2))
        je.bind('<Return>', lambda _ev: self._jump_to())
        tk.Button(top, text="Go", command=self._jump_to).pack(side='left', padx=2)

        self.sort_btn = tk.Button(top, text="Sort: sigma desc (s)", command=self._cycle_sort)
        self.sort_btn.pack(side='left', padx=10)

        self.zoom_lbl = tk.Label(top, text="zoom 2x", fg='#8cf', bg='#1e1e1e')
        self.zoom_lbl.pack(side='right')

        self.info = tk.Label(self.root, text="", fg='#eee', bg='#1e1e1e',
                             font=('monospace', 11), anchor='w', justify='left')
        self.info.pack(fill='x', padx=10)

        imgs = tk.Frame(self.root, bg='#1e1e1e')
        imgs.pack(fill='both', expand=True, padx=8, pady=8)
        self.full_lbl = tk.Label(imgs, bg='#111')
        self.full_lbl.pack(side='left', padx=(0, 8))
        right = tk.Frame(imgs, bg='#1e1e1e')
        right.pack(side='left', fill='y')
        tk.Label(right, text="center crop (native px, NEAREST)",
                 fg='#888', bg='#1e1e1e').pack()
        self.crop_lbl = tk.Label(right, bg='#111')
        self.crop_lbl.pack()

        tk.Label(self.root,
                 text="<- ->  prev/next   Home/End   PgUp/PgDn jump50   "
                      "1 2 4 zoom   s sort   t threshold",
                 fg='#777', bg='#1e1e1e').pack(pady=(0, 6))

        b = self.root.bind
        b('<Left>', lambda _e: self._step(-1))
        b('<Right>', lambda _e: self._step(1))
        b('<Prior>', lambda _e: self._step(-50))
        b('<Next>', lambda _e: self._step(50))
        b('<Home>', lambda _e: self._goto(0))
        b('<End>', lambda _e: self._goto(len(self.rows) - 1))
        b('<Key-1>', lambda _e: self._set_zoom(1))
        b('<Key-2>', lambda _e: self._set_zoom(2))
        b('<Key-4>', lambda _e: self._set_zoom(4))
        b('<Key-s>', lambda _e: self._cycle_sort())
        b('<Key-t>', lambda _e: self._jump_threshold())

    def _apply_sort(self):
        if self.sort_mode == 'desc':
            self.rows.sort(key=lambda r: r[1], reverse=True)
        elif self.sort_mode == 'asc':
            self.rows.sort(key=lambda r: r[1])
        else:
            self.rows.sort(key=lambda r: r[0])
        self.idx = 0
        self._show()

    def _cycle_sort(self):
        order = {'desc': 'asc', 'asc': 'file', 'file': 'desc'}
        self.sort_mode = order[self.sort_mode]
        label = {'desc': 'sigma desc', 'asc': 'sigma asc', 'file': 'filename'}[self.sort_mode]
        self.sort_btn.config(text=f"Sort: {label} (s)")
        self._apply_sort()

    def _set_threshold(self):
        try:
            self.threshold = float(self.thr_var.get())
        except ValueError:
            return
        self._show()

    def _set_zoom(self, z):
        self.zoom = z
        self.zoom_lbl.config(text=f"zoom {z}x")
        self._show()

    def _step(self, d):
        self._goto(self.idx + d)

    def _goto(self, i):
        self.idx = max(0, min(len(self.rows) - 1, i))
        self._show()

    def _jump_to(self):
        try:
            self._goto(int(self.jump_var.get()))
        except ValueError:
            pass

    def _jump_threshold(self):
        """When sorted by sigma, jump to the threshold boundary."""
        for i, (_f, s) in enumerate(self.rows):
            if (self.sort_mode == 'desc' and s <= self.threshold) or \
               (self.sort_mode != 'desc' and s > self.threshold):
                self._goto(i)
                return
        self._goto(len(self.rows) - 1)

    # ---------- render ----------
    def _show(self):
        if not self.rows:
            return
        fname, sigma = self.rows[self.idx]
        path = os.path.join(self.good_dir, fname)
        status = "REJECT" if sigma > self.threshold else "keep"
        color = '#f66' if sigma > self.threshold else '#6f6'
        self.info.config(
            text=f"[{self.idx + 1}/{len(self.rows)}]  sigma={sigma:7.3f}  "
                 f"thr={self.threshold:.2f}  {status}\n{fname}",
            fg=color)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as ex:
            self.full_lbl.config(image='', text=f"cannot open:\n{ex}", fg='#f66')
            self.crop_lbl.config(image='', text='')
            return

        full = img.copy()
        full.thumbnail((FULL_MAX, FULL_MAX), Image.Resampling.LANCZOS)
        self._full_ref = ImageTk.PhotoImage(full)
        self.full_lbl.config(image=self._full_ref, text='')

        w, h = img.size
        cs = max(8, CROP_PANEL // self.zoom)
        x0 = max(0, (w - cs) // 2)
        y0 = max(0, (h - cs) // 2)
        crop = img.crop((x0, y0, min(w, x0 + cs), min(h, y0 + cs)))
        crop = crop.resize((crop.width * self.zoom, crop.height * self.zoom),
                           Image.Resampling.NEAREST)
        self._crop_ref = ImageTk.PhotoImage(crop)
        self.crop_lbl.config(image=self._crop_ref, text='')


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline='') as fh:
        for r in csv.reader(fh):
            if len(r) >= 2:
                try:
                    rows.append((r[0], float(r[1])))
                except ValueError:
                    pass
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='noise_sigmas.csv')
    p.add_argument('--good', default='/home/algis/Desktop/data/train/good')
    p.add_argument('--threshold', type=float, default=1.5)
    p.add_argument('--sort', choices=['desc', 'asc', 'file'], default='desc')
    args = p.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows loaded from {args.csv}")

    root = tk.Tk()
    Viewer(root, rows, args.good, args.threshold, sort_mode=args.sort)
    root.mainloop()


if __name__ == '__main__':
    main()
