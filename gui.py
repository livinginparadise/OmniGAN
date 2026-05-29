import os
import sys
import time
import traceback
from pathlib import Path


def _ensure_flatpak_cuda():
    # VSCode-as-Flatpak passes /dev/nvidia* through but not the host's
    # libcuda.so.1, so torch silently falls back to CPU. Prepend the host
    # driver dir to LD_LIBRARY_PATH and re-exec so the linker picks it up
    # before torch dlopens the driver.
    if os.environ.get("_DENOISEGAN_CUDA_LDPATH") or not os.path.exists("/.flatpak-info"):
        return
    host_lib = "/run/host/usr/lib"
    if not os.path.exists(os.path.join(host_lib, "libcuda.so.1")):
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if host_lib not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (host_lib, cur) if p)
    os.environ["_DENOISEGAN_CUDA_LDPATH"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_flatpak_cuda()

import numpy as np
import torch
from torch.amp import autocast

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QRectF
from PyQt6.QtGui import (
    QAction, QImage, QPixmap, QPainter, QFont, QPalette, QColor, QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QMessageBox, QStatusBar, QToolBar, QSplitter, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QProgressBar, QVBoxLayout, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QStyleFactory, QSizePolicy, QGridLayout,
)

from denoisegan.models import DenoiseGenerator, NoiseTranslator


WIN98_QSS = """
QWidget {
    background-color: #c0c0c0;
    color: black;
    font-family: "MS Sans Serif", "Sans Serif", "DejaVu Sans";
    font-size: 11px;
}
QMainWindow, QDialog {
    background-color: #c0c0c0;
}
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #808080;
}
QPushButton, QToolButton {
    background-color: #c0c0c0;
    color: black;
    border-top: 1px solid #ffffff;
    border-left: 1px solid #ffffff;
    border-right: 1px solid #404040;
    border-bottom: 1px solid #404040;
    padding: 3px 12px;
    min-width: 60px;
    min-height: 18px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #d4d0c8;
}
QPushButton:pressed, QToolButton:pressed, QPushButton:checked, QToolButton:checked {
    border-top: 1px solid #404040;
    border-left: 1px solid #404040;
    border-right: 1px solid #ffffff;
    border-bottom: 1px solid #ffffff;
    background-color: #c0c0c0;
}
QPushButton:disabled, QToolButton:disabled {
    color: #808080;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: white;
    color: black;
    border-top: 1px solid #404040;
    border-left: 1px solid #404040;
    border-right: 1px solid #ffffff;
    border-bottom: 1px solid #ffffff;
    padding: 2px 4px;
    selection-background-color: #000080;
    selection-color: white;
}
QComboBox::drop-down {
    width: 16px;
    background-color: #c0c0c0;
    border-left: 1px solid #ffffff;
}
QComboBox QAbstractItemView {
    background-color: white;
    color: black;
    border: 1px solid #404040;
    selection-background-color: #000080;
    selection-color: white;
}
QGroupBox {
    border: 1px solid #808080;
    margin-top: 12px;
    padding-top: 6px;
    background-color: #c0c0c0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    background-color: #c0c0c0;
}
QStatusBar {
    background-color: #c0c0c0;
    border-top: 1px solid #ffffff;
}
QStatusBar::item { border: none; }
QStatusBar QLabel {
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #ffffff;
    border-bottom: 1px solid #ffffff;
    padding: 2px 6px;
}
QToolBar {
    background-color: #c0c0c0;
    border: none;
    spacing: 2px;
    padding: 2px;
    border-bottom: 1px solid #808080;
}
QToolBar::separator {
    background-color: #808080;
    width: 1px;
    margin: 2px 4px;
}
QMenuBar {
    background-color: #c0c0c0;
    border-bottom: 1px solid #808080;
}
QMenuBar::item {
    padding: 3px 8px;
    background: transparent;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background-color: #000080;
    color: white;
}
QMenu {
    background-color: #c0c0c0;
    border-top: 1px solid #ffffff;
    border-left: 1px solid #ffffff;
    border-right: 1px solid #404040;
    border-bottom: 1px solid #404040;
}
QMenu::item {
    padding: 4px 24px 4px 24px;
}
QMenu::item:selected {
    background-color: #000080;
    color: white;
}
QMenu::separator {
    height: 1px;
    background: #808080;
    margin: 4px 2px;
}
QProgressBar {
    background-color: white;
    color: black;
    border-top: 1px solid #404040;
    border-left: 1px solid #404040;
    border-right: 1px solid #ffffff;
    border-bottom: 1px solid #ffffff;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #000080;
}
QGraphicsView {
    background-color: #808080;
    border-top: 1px solid #404040;
    border-left: 1px solid #404040;
    border-right: 1px solid #ffffff;
    border-bottom: 1px solid #ffffff;
}
QCheckBox {
    spacing: 6px;
    padding: 2px;
}
QCheckBox:disabled {
    color: #808080;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #c0c0c0;
    border: 1px solid #808080;
}
QScrollBar::handle {
    background-color: #c0c0c0;
    border-top: 1px solid #ffffff;
    border-left: 1px solid #ffffff;
    border-right: 1px solid #404040;
    border-bottom: 1px solid #404040;
    min-height: 16px;
    min-width: 16px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background-color: #c0c0c0;
    border: 1px outset #ffffff;
}
"""


def numpy_to_qpixmap(arr_uint8):
    h, w, c = arr_uint8.shape
    if c == 3:
        contig = np.ascontiguousarray(arr_uint8)
        qimg = QImage(contig.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    else:
        contig = np.ascontiguousarray(arr_uint8)
        qimg = QImage(contig.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def load_image_rgb(path):
    qimg = QImage(path)
    if qimg.isNull():
        raise IOError(f"Could not read image: {path}")
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(qimg.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())[:, : w * 3]
    arr = arr.reshape(h, w, 3).copy()
    return arr


def save_image_rgb(arr_uint8, path):
    h, w, _ = arr_uint8.shape
    contig = np.ascontiguousarray(arr_uint8)
    qimg = QImage(contig.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return qimg.save(path)


def hann_window_2d(size):
    w = np.hanning(size).astype(np.float32)
    w = np.maximum(w, 1e-3)
    return (w[:, None] * w[None, :])


class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._zoom = 1.0

    def set_image(self, arr_uint8):
        if arr_uint8 is None:
            self._pix_item.setPixmap(QPixmap())
            return
        pix = numpy_to_qpixmap(arr_uint8)
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.reset_view()

    def reset_view(self):
        self._zoom = 1.0
        self.resetTransform()
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pix_item.pixmap().isNull():
            return
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self._zoom *= factor
        self._zoom = max(0.05, min(40.0, self._zoom))
        self.resetTransform()
        self.scale(self._zoom, self._zoom)


class TileWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(np.ndarray, float)
    failed = pyqtSignal(str)

    def __init__(self, model, image_rgb_u8, tile, overlap, batch, device, amp,
                 normalize=True, translator=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.translator = translator
        self.image = image_rgb_u8
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.batch = int(batch)
        self.device = device
        self.amp = amp
        self.normalize = normalize
        self._abort = False

    def abort(self):
        self._abort = True

    def _amp_ctx(self):
        if self.amp == 'bf16':
            return autocast(self.device.type, dtype=torch.bfloat16)
        if self.amp == 'fp16':
            return autocast(self.device.type, dtype=torch.float16)
        import contextlib
        return contextlib.nullcontext()

    def run(self):
        try:
            t0 = time.time()
            tile = self.tile
            overlap = max(0, min(self.overlap, tile - 1))
            stride = tile - overlap

            img = self.image.astype(np.float32) / 255.0
            h, w, _ = img.shape

            pad_h = 0 if h <= tile else (stride - (h - tile) % stride) % stride
            pad_w = 0 if w <= tile else (stride - (w - tile) % stride) % stride
            if h < tile:
                pad_h = tile - h
            if w < tile:
                pad_w = tile - w

            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            H, W, _ = padded.shape

            y_starts = list(range(0, H - tile + 1, stride))
            x_starts = list(range(0, W - tile + 1, stride))
            if y_starts[-1] + tile < H:
                y_starts.append(H - tile)
            if x_starts[-1] + tile < W:
                x_starts.append(W - tile)
            positions = [(y, x) for y in y_starts for x in x_starts]
            total = len(positions)
            self.log.emit(f"Image {w}x{h}  padded {W}x{H}  tile={tile}  overlap={overlap}  tiles={total}")

            win = hann_window_2d(tile)
            win_t = torch.from_numpy(win).to(self.device)

            out_acc = torch.zeros((3, H, W), dtype=torch.float32, device=self.device)
            wsum = torch.zeros((H, W), dtype=torch.float32, device=self.device)

            self.model.eval()
            if self.translator is not None:
                self.translator.eval()
            done = 0
            for i in range(0, total, self.batch):
                if self._abort:
                    self.failed.emit("Cancelled.")
                    return
                batch_pos = positions[i:i + self.batch]
                tiles = []
                for (y, x) in batch_pos:
                    p = padded[y:y + tile, x:x + tile]
                    t = torch.from_numpy(p).permute(2, 0, 1)
                    if self.normalize:
                        t = t * 2.0 - 1.0
                    tiles.append(t)
                inp = torch.stack(tiles, dim=0).to(self.device, non_blocking=True)

                with torch.no_grad(), self._amp_ctx():
                    if self.translator is not None:
                        inp = self.translator(inp)
                    out = self.model(inp)
                out = out.float()
                if self.normalize:
                    out = (out.clamp(-1.0, 1.0) + 1.0) * 0.5
                else:
                    out = out.clamp(0.0, 1.0)

                for j, (y, x) in enumerate(batch_pos):
                    out_acc[:, y:y + tile, x:x + tile] += out[j] * win_t[None, :, :]
                    wsum[y:y + tile, x:x + tile] += win_t

                done += len(batch_pos)
                elapsed = time.time() - t0
                self.progress.emit(done, total, elapsed)

            out_acc = out_acc / (wsum[None, :, :] + 1e-8)
            out_acc = out_acc.clamp(0.0, 1.0)
            out_cpu = (out_acc[:, :h, :w] * 255.0).round().clamp(0, 255).to(torch.uint8)
            out_np = out_cpu.permute(1, 2, 0).cpu().numpy()

            elapsed = time.time() - t0
            self.finished_ok.emit(out_np, elapsed)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


class AdaptWorker(QThread):
    """Test-time adaptation: fine-tune the translator on the user's single noisy
    image via ZS-N2N self-supervision (no clean reference). Denoiser stays frozen;
    only the translator is updated, on a deep copy so the base T is preserved."""
    progress = pyqtSignal(int, int, float)
    finished_ok = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, denoiser, translator, image_rgb_u8, steps, lr, crop,
                 device, amp, parent=None):
        super().__init__(parent)
        self.denoiser = denoiser
        self.translator = translator
        self.image = image_rgb_u8
        self.steps = int(steps)
        self.lr = float(lr)
        self.crop = int(crop)
        self.device = device
        self.amp = amp
        self._abort = False

    def abort(self):
        self._abort = True

    def _amp_ctx(self):
        # bf16 is safe without a grad scaler; fp16/fp32 -> fp32 for stable TTA
        if self.amp == 'bf16':
            return autocast(self.device.type, dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def run(self):
        try:
            import copy
            import random as _random
            from denoisegan.tta import make_pair_kernels, zsn2n_loss

            dev = self.device
            D = self.denoiser
            D.eval()
            for p in D.parameters():
                p.requires_grad_(False)

            T = copy.deepcopy(self.translator).to(dev)
            T.train()
            for p in T.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(T.parameters(), lr=self.lr, betas=(0.9, 0.99))

            img = torch.from_numpy(self.image.astype(np.float32) / 255.0)
            img = img.permute(2, 0, 1).unsqueeze(0).to(dev) * 2.0 - 1.0
            _, _, H, W = img.shape
            crop = max(64, (self.crop // 32) * 32)  # divisible by 32 for D's /16 grid
            ph = max(0, crop - H)
            pw = max(0, crop - W)
            if ph or pw:
                img = F.pad(img, (0, pw, 0, ph), mode='reflect')
                _, _, H, W = img.shape

            k1, k2 = make_pair_kernels(dev)

            def pipeline(x):
                return D(T(x))

            last = 0.0
            for step in range(self.steps):
                if self._abort:
                    self.failed.emit("Adaptation cancelled.")
                    return
                iy = _random.randint(0, H - crop)
                ix = _random.randint(0, W - crop)
                y = img[:, :, iy:iy + crop, ix:ix + crop]
                opt.zero_grad(set_to_none=True)
                with self._amp_ctx():
                    loss, _, _ = zsn2n_loss(pipeline, y, k1, k2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0)
                opt.step()
                last = float(loss.item())
                self.progress.emit(step + 1, self.steps, last)

            sd = {k: v.detach().cpu().clone() for k, v in T.state_dict().items()}
            self.finished_ok.emit(sd, last)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class SettingsDialog(QDialog):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.state = state

        lay = QGridLayout()
        row = 0
        lay.addWidget(QLabel("Tile size:"), row, 0)
        self.tile = QSpinBox()
        self.tile.setRange(64, 1024)
        self.tile.setSingleStep(64)
        self.tile.setValue(state['tile'])
        lay.addWidget(self.tile, row, 1)
        row += 1

        lay.addWidget(QLabel("Tile overlap:"), row, 0)
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 512)
        self.overlap.setSingleStep(8)
        self.overlap.setValue(state['overlap'])
        lay.addWidget(self.overlap, row, 1)
        row += 1

        lay.addWidget(QLabel("Batch size:"), row, 0)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(state['batch'])
        lay.addWidget(self.batch, row, 1)
        row += 1

        lay.addWidget(QLabel("Device:"), row, 0)
        self.device = QComboBox()
        opts = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
        self.device.addItems(opts)
        idx = self.device.findText(state['device'])
        if idx >= 0:
            self.device.setCurrentIndex(idx)
        lay.addWidget(self.device, row, 1)
        row += 1

        lay.addWidget(QLabel("Precision:"), row, 0)
        self.amp = QComboBox()
        self.amp.addItems(['bf16', 'fp16', 'fp32'])
        idx = self.amp.findText(state['amp'])
        if idx >= 0:
            self.amp.setCurrentIndex(idx)
        lay.addWidget(self.amp, row, 1)
        row += 1

        self.use_ema = QCheckBox("Load EMA weights when present in checkpoint")
        self.use_ema.setChecked(state['use_ema'])
        lay.addWidget(self.use_ema, row, 0, 1, 2)
        row += 1

        self.use_translator = QCheckBox("Apply translator before denoiser (D(T(x)))")
        self.use_translator.setChecked(state['use_translator'])
        lay.addWidget(self.use_translator, row, 0, 1, 2)
        row += 1

        lay.addWidget(QLabel("TTA steps:"), row, 0)
        self.tta_steps = QSpinBox()
        self.tta_steps.setRange(10, 5000)
        self.tta_steps.setSingleStep(50)
        self.tta_steps.setValue(state['tta_steps'])
        lay.addWidget(self.tta_steps, row, 1)
        row += 1

        lay.addWidget(QLabel("TTA crop:"), row, 0)
        self.tta_crop = QSpinBox()
        self.tta_crop.setRange(64, 512)
        self.tta_crop.setSingleStep(32)
        self.tta_crop.setValue(state['tta_crop'])
        lay.addWidget(self.tta_crop, row, 1)
        row += 1

        lay.addWidget(QLabel("TTA learning rate:"), row, 0)
        self.tta_lr = QDoubleSpinBox()
        self.tta_lr.setDecimals(6)
        self.tta_lr.setRange(1e-6, 1e-2)
        self.tta_lr.setSingleStep(1e-5)
        self.tta_lr.setValue(state['tta_lr'])
        lay.addWidget(self.tta_lr, row, 1)
        row += 1

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns, row, 0, 1, 2)

        self.setLayout(lay)
        self.setFixedWidth(320)

    def values(self):
        return {
            'tile': self.tile.value(),
            'overlap': self.overlap.value(),
            'batch': self.batch.value(),
            'device': self.device.currentText(),
            'amp': self.amp.currentText(),
            'use_ema': self.use_ema.isChecked(),
            'use_translator': self.use_translator.isChecked(),
            'tta_steps': self.tta_steps.value(),
            'tta_crop': self.tta_crop.value(),
            'tta_lr': self.tta_lr.value(),
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DenoiseGAN  -  Image Restore")
        self.resize(1200, 760)

        self.model = None
        self.model_path = None
        self.translator = None
        self.translator_path = None
        self.translator_cfg = (64, 10)
        self.input_image = None
        self.input_path = None
        self.output_image = None
        self.worker = None
        self.adapt_worker = None
        self.state = {
            'tile': 256,
            'overlap': 32,
            'batch': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'amp': 'bf16',
            'use_ema': True,
            'use_translator': True,
            'tta_steps': 400,
            'tta_crop': 128,
            'tta_lr': 1e-4,
        }

        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._update_actions()

    def _build_menus(self):
        mb = self.menuBar()

        file_m = mb.addMenu("&File")
        self.act_open = QAction("&Open Image...", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self.open_image)
        file_m.addAction(self.act_open)

        self.act_save = QAction("&Save Output...", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self.save_output)
        file_m.addAction(self.act_save)

        file_m.addSeparator()
        self.act_exit = QAction("E&xit", self)
        self.act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self.act_exit.triggered.connect(self.close)
        file_m.addAction(self.act_exit)

        model_m = mb.addMenu("&Model")
        self.act_load_model = QAction("&Load Checkpoint...", self)
        self.act_load_model.setShortcut(QKeySequence("Ctrl+L"))
        self.act_load_model.triggered.connect(self.load_checkpoint)
        model_m.addAction(self.act_load_model)

        self.act_unload_model = QAction("&Unload", self)
        self.act_unload_model.triggered.connect(self.unload_model)
        model_m.addAction(self.act_unload_model)

        model_m.addSeparator()
        self.act_load_translator = QAction("Load &Translator...", self)
        self.act_load_translator.setShortcut(QKeySequence("Ctrl+T"))
        self.act_load_translator.triggered.connect(self.load_translator)
        model_m.addAction(self.act_load_translator)

        self.act_unload_translator = QAction("Unload Tra&nslator", self)
        self.act_unload_translator.triggered.connect(self.unload_translator)
        model_m.addAction(self.act_unload_translator)

        proc_m = mb.addMenu("&Process")
        self.act_process = QAction("&Run", self)
        self.act_process.setShortcut(QKeySequence("F5"))
        self.act_process.triggered.connect(self.run_process)
        proc_m.addAction(self.act_process)

        self.act_adapt = QAction("&Adapt Translator to Image (TTA)", self)
        self.act_adapt.setShortcut(QKeySequence("F6"))
        self.act_adapt.triggered.connect(self.adapt_translator)
        proc_m.addAction(self.act_adapt)

        self.act_cancel = QAction("&Cancel", self)
        self.act_cancel.setShortcut(QKeySequence("Esc"))
        self.act_cancel.triggered.connect(self.cancel_process)
        proc_m.addAction(self.act_cancel)

        proc_m.addSeparator()
        self.act_settings = QAction("&Settings...", self)
        self.act_settings.triggered.connect(self.open_settings)
        proc_m.addAction(self.act_settings)

        view_m = mb.addMenu("&View")
        self.act_fit_in = QAction("Fit &Input to Window", self)
        self.act_fit_in.triggered.connect(lambda: self.view_in.reset_view())
        view_m.addAction(self.act_fit_in)
        self.act_fit_out = QAction("Fit &Output to Window", self)
        self.act_fit_out.triggered.connect(lambda: self.view_out.reset_view())
        view_m.addAction(self.act_fit_out)

        help_m = mb.addMenu("&Help")
        self.act_about = QAction("&About", self)
        self.act_about.triggered.connect(self.show_about)
        help_m.addAction(self.act_about)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setIconSize(QSize(16, 16))
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        for act in [self.act_open, self.act_save]:
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_load_model)
        tb.addAction(self.act_load_translator)
        tb.addSeparator()
        tb.addAction(self.act_process)
        tb.addAction(self.act_adapt)
        tb.addAction(self.act_cancel)
        tb.addSeparator()
        tb.addAction(self.act_settings)

    def _build_central(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        self.lbl_in = QLabel("Input:  (no image)")
        self.lbl_out = QLabel("Output: (not processed)")
        info_row.addWidget(self.lbl_in, 1)
        info_row.addWidget(self.lbl_out, 1)
        outer.addLayout(info_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.view_in = ImageView()
        self.view_out = ImageView()
        split.addWidget(self.view_in)
        split.addWidget(self.view_out)
        split.setSizes([600, 600])
        outer.addWidget(split, 1)

        self.setCentralWidget(central)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.lbl_model = QLabel("Model: (none)")
        self.lbl_device = QLabel(f"Device: {self.state['device']}")
        self.lbl_amp = QLabel(f"AMP: {self.state['amp']}")
        self.lbl_status = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        self.progress.setFixedWidth(220)

        self.lbl_translator = QLabel("Translator: (none)")
        sb.addPermanentWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.progress, 0)
        sb.addPermanentWidget(self.lbl_model, 0)
        sb.addPermanentWidget(self.lbl_translator, 0)
        sb.addPermanentWidget(self.lbl_device, 0)
        sb.addPermanentWidget(self.lbl_amp, 0)
        self.setStatusBar(sb)

    def _busy(self):
        return ((self.worker is not None and self.worker.isRunning()) or
                (self.adapt_worker is not None and self.adapt_worker.isRunning()))

    def _update_actions(self):
        has_img = self.input_image is not None
        has_model = self.model is not None
        has_trans = self.translator is not None
        running = self._busy()
        self.act_process.setEnabled(has_img and has_model and not running)
        self.act_adapt.setEnabled(has_img and has_model and not running)
        self.act_cancel.setEnabled(running)
        self.act_save.setEnabled(self.output_image is not None and not running)
        self.act_open.setEnabled(not running)
        self.act_load_model.setEnabled(not running)
        self.act_unload_model.setEnabled(has_model and not running)
        self.act_load_translator.setEnabled(not running)
        self.act_unload_translator.setEnabled(has_trans and not running)
        self.act_settings.setEnabled(not running)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"
        )
        if not path:
            return
        try:
            arr = load_image_rgb(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.input_image = arr
        self.input_path = path
        self.output_image = None
        h, w, _ = arr.shape
        self.lbl_in.setText(f"Input: {os.path.basename(path)}   ({w} x {h})")
        self.lbl_out.setText("Output: (not processed)")
        self.view_in.set_image(arr)
        self.view_out.set_image(None)
        self.lbl_status.setText("Image loaded.")
        self._update_actions()

    def save_output(self):
        if self.output_image is None:
            return
        suggest = "output.png"
        if self.input_path:
            stem = Path(self.input_path).stem
            suggest = f"{stem}_restored.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", suggest,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;WebP (*.webp);;All files (*)"
        )
        if not path:
            return
        if not save_image_rgb(self.output_image, path):
            QMessageBox.critical(self, "Save failed", f"Could not save image to {path}")
            return
        self.lbl_status.setText(f"Saved: {path}")

    def load_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", "checkpoints",
            "PyTorch checkpoint (*.pt *.pth);;All files (*)"
        )
        if not path:
            return
        try:
            self.lbl_status.setText("Loading model...")
            QApplication.processEvents()

            device = torch.device(self.state['device'])
            G = DenoiseGenerator(channels=(48, 96, 192, 320, 448),
                                 drop_path_rate=0.0,
                                 use_checkpoint=False).to(device)
            ck = torch.load(path, map_location='cpu', weights_only=False)

            used_ema = False
            if self.state['use_ema'] and isinstance(ck, dict) and 'ema' in ck and ck['ema']:
                missing, unexpected = G.load_state_dict(ck['ema'], strict=False)
                used_ema = True
            elif isinstance(ck, dict) and 'G' in ck:
                missing, unexpected = G.load_state_dict(ck['G'], strict=False)
            else:
                missing, unexpected = G.load_state_dict(ck, strict=False)
            G.eval()
            for p in G.parameters():
                p.requires_grad = False

            self.model = G
            self.model_path = path
            tag = "EMA" if used_ema else "G"
            stage = ck.get('stage', '?') if isinstance(ck, dict) else '?'
            step = ck.get('step', '?') if isinstance(ck, dict) else '?'
            self.lbl_model.setText(f"Model: {os.path.basename(path)} [{tag}, {stage}@{step}]")
            self.lbl_status.setText(f"Model loaded ({tag} weights).")
            if missing or unexpected:
                msg = (
                    f"Model loaded ({tag}) with mismatches:\n"
                    f"  missing keys (in arch but not in ckpt): {len(missing)}\n"
                    f"  unexpected keys (in ckpt but not in arch): {len(unexpected)}\n\n"
                )
                if missing:
                    msg += f"first missing: {missing[:5]}\n"
                if unexpected:
                    msg += f"first unexpected: {unexpected[:5]}\n"
                msg += ("\nIf the architecture (channels / depth / bias) was changed "
                        "since training, output may be wrong.")
                if len(missing) > 5 or len(unexpected) > 5:
                    QMessageBox.warning(self, "Checkpoint mismatch", msg)
                self.lbl_status.setText(
                    f"Model loaded ({tag}). missing={len(missing)} unexpected={len(unexpected)}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load checkpoint:\n{e}\n\n{traceback.format_exc()}")
        finally:
            self._update_actions()

    def unload_model(self):
        self.model = None
        self.model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.lbl_model.setText("Model: (none)")
        self.lbl_status.setText("Model unloaded.")
        self._update_actions()

    def load_translator(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Translator", "checkpoints",
            "PyTorch checkpoint (*.pt *.pth);;All files (*)"
        )
        if not path:
            return
        try:
            device = torch.device(self.state['device'])
            ck = torch.load(path, map_location='cpu', weights_only=False)
            cfg = ck.get('args', {}) if isinstance(ck, dict) else {}
            dim = int(cfg.get('dim', 64))
            blocks = int(cfg.get('blocks', 10))
            T = NoiseTranslator(nc=3, dim=dim, num_blocks=blocks).to(device)

            if (self.state['use_ema'] and isinstance(ck, dict)
                    and 'ema' in ck and ck['ema']):
                sd, tag = ck['ema'], 'EMA'
            elif isinstance(ck, dict) and 'T' in ck:
                sd, tag = ck['T'], 'T'
            else:
                sd, tag = ck, 'raw'
            missing, unexpected = T.load_state_dict(sd, strict=False)
            T.eval()
            for p in T.parameters():
                p.requires_grad_(False)

            self.translator = T
            self.translator_path = path
            self.translator_cfg = (dim, blocks)
            self.lbl_translator.setText(f"Translator: {os.path.basename(path)} [{tag}]")
            self.lbl_status.setText(f"Translator loaded ({tag}).")
            if (len(missing) > 2 or len(unexpected) > 2):
                QMessageBox.warning(
                    self, "Translator mismatch",
                    f"Loaded with mismatches: missing={len(missing)} "
                    f"unexpected={len(unexpected)}.\nfirst missing: {missing[:4]}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load translator:\n{e}\n\n{traceback.format_exc()}")
        finally:
            self._update_actions()

    def unload_translator(self):
        self.translator = None
        self.translator_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.lbl_translator.setText("Translator: (none)")
        self.lbl_status.setText("Translator unloaded.")
        self._update_actions()

    def open_settings(self):
        dlg = SettingsDialog(self, self.state)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_state = dlg.values()
            device_changed = new_state['device'] != self.state['device']
            self.state = new_state
            self.lbl_device.setText(f"Device: {self.state['device']}")
            self.lbl_amp.setText(f"AMP: {self.state['amp']}")
            if device_changed:
                try:
                    if self.model is not None:
                        self.model.to(torch.device(self.state['device']))
                    if self.translator is not None:
                        self.translator.to(torch.device(self.state['device']))
                    self.lbl_status.setText(f"Models moved to {self.state['device']}.")
                except Exception as e:
                    QMessageBox.warning(self, "Device", f"Could not move model: {e}")
            self._update_actions()

    def run_process(self):
        if self.input_image is None or self.model is None:
            return
        device = torch.device(self.state['device'])
        if next(self.model.parameters()).device != device:
            try:
                self.model.to(device)
            except Exception as e:
                QMessageBox.critical(self, "Device", f"Could not move model: {e}")
                return

        translator = None
        if self.state['use_translator'] and self.translator is not None:
            self.translator.to(device)
            translator = self.translator

        self.progress.setValue(0)
        self.lbl_status.setText("Processing..." + (" [+translator]" if translator else ""))
        self.worker = TileWorker(
            self.model, self.input_image,
            tile=self.state['tile'], overlap=self.state['overlap'],
            batch=self.state['batch'], device=device, amp=self.state['amp'],
            translator=translator,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()
        self._update_actions()

    def cancel_process(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            self.lbl_status.setText("Cancelling...")
        if self.adapt_worker is not None and self.adapt_worker.isRunning():
            self.adapt_worker.abort()
            self.lbl_status.setText("Cancelling adaptation...")

    def _on_progress(self, done, total, elapsed):
        pct = int(done * 100 / max(1, total))
        self.progress.setValue(pct)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        self.lbl_status.setText(
            f"Processing tile {done}/{total}  ({rate:.1f} tiles/s, ETA {eta:.1f}s)"
        )

    def _on_log(self, msg):
        self.lbl_status.setText(msg)

    def _on_done(self, arr, elapsed):
        self.output_image = arr
        self.view_out.set_image(arr)
        h, w, _ = arr.shape
        self.lbl_out.setText(f"Output: ({w} x {h})   {elapsed:.2f}s")
        self.progress.setValue(100)
        self.lbl_status.setText(f"Done in {elapsed:.2f}s.")
        self.worker = None
        self._update_actions()

    def _on_fail(self, msg):
        self.progress.setValue(0)
        self.lbl_status.setText("Failed.")
        self.worker = None
        QMessageBox.critical(self, "Processing failed", msg)
        self._update_actions()

    def adapt_translator(self):
        if self.input_image is None or self.model is None:
            return
        device = torch.device(self.state['device'])
        try:
            self.model.to(device)
        except Exception as e:
            QMessageBox.critical(self, "Device", f"Could not move model: {e}")
            return

        if self.translator is None:
            dim, blocks = self.translator_cfg
            self.translator = NoiseTranslator(nc=3, dim=dim, num_blocks=blocks).to(device)
            self.translator.eval()
            self.translator_path = None
            self.lbl_translator.setText("Translator: (fresh, identity)")
        else:
            self.translator.to(device)

        self.progress.setValue(0)
        self.lbl_status.setText("Adapting translator (TTA)...")
        self.adapt_worker = AdaptWorker(
            self.model, self.translator, self.input_image,
            steps=self.state['tta_steps'], lr=self.state['tta_lr'],
            crop=self.state['tta_crop'], device=device, amp=self.state['amp'],
        )
        self.adapt_worker.progress.connect(self._on_adapt_progress)
        self.adapt_worker.finished_ok.connect(self._on_adapt_done)
        self.adapt_worker.failed.connect(self._on_adapt_fail)
        self.adapt_worker.start()
        self._update_actions()

    def _on_adapt_progress(self, step, total, loss):
        pct = int(step * 100 / max(1, total))
        self.progress.setValue(pct)
        self.lbl_status.setText(f"TTA step {step}/{total}   loss={loss:.5f}")

    def _on_adapt_done(self, state_dict, final_loss):
        try:
            self.translator.load_state_dict(state_dict)
            self.translator.eval()
            for p in self.translator.parameters():
                p.requires_grad_(False)
        except Exception as e:
            QMessageBox.warning(self, "TTA", f"Could not apply adapted weights: {e}")
        self.progress.setValue(100)
        self.lbl_status.setText(
            f"Translator adapted (final loss {final_loss:.5f}). Run to denoise.")
        base = os.path.basename(self.translator_path) if self.translator_path else "fresh"
        self.lbl_translator.setText(f"Translator: {base} [adapted]")
        self.adapt_worker = None
        self._update_actions()

    def _on_adapt_fail(self, msg):
        self.progress.setValue(0)
        self.lbl_status.setText("TTA failed.")
        self.adapt_worker = None
        QMessageBox.critical(self, "Adaptation failed", msg)
        self._update_actions()

    def show_about(self):
        QMessageBox.information(
            self, "About",
            "DenoiseGAN Image Restore\n\n"
            "1-step GAN restoration model with tiled Hann-blended inference.\n"
            "Optional noise translator front-end: D(T(x)) handles foreign noise.\n"
            "Process - Adapt Translator (TTA) self-tunes the translator to the\n"
            "current image's noise via ZS-N2N (no clean reference needed).\n"
        )

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(2000)
        if self.adapt_worker is not None and self.adapt_worker.isRunning():
            self.adapt_worker.abort()
            self.adapt_worker.wait(2000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    if 'Fusion' in QStyleFactory.keys():
        app.setStyle('Fusion')
    pal = QPalette()
    gray = QColor(192, 192, 192)
    pal.setColor(QPalette.ColorRole.Window, gray)
    pal.setColor(QPalette.ColorRole.Button, gray)
    pal.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.AlternateBase, gray)
    pal.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(0, 0, 128))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    app.setStyleSheet(WIN98_QSS)
    app.setFont(QFont("MS Sans Serif", 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
