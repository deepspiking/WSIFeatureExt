import argparse
import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


FEATURES_DIR = Path("./features")
TILES_DIR = Path("./datas/tiles")
PREVIEW_SIZE = 300
MAP_W = 900
MAP_H = 760


def list_feature_files(slide: str = "", model: str = ""):
    pts = sorted(FEATURES_DIR.glob("*.pt"))
    if slide:
        pts = [p for p in pts if slide in p.stem]
    if model:
        pts = [p for p in pts if f".{model}" in p.stem or p.stem.endswith(model)]
    return pts


def parse_tile_coord(path: Path):
    m = re.search(r"tile_r(\d+)_c(\d+)", path.name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def slide_name_from_pt(pt_path: Path) -> str:
    stem = pt_path.stem
    if stem.endswith(".ctranspath"):
        return stem[:-11]
    if stem.endswith(".retccl"):
        return stem[:-7]
    return stem


class Viewer(tk.Tk):
    def __init__(self, feature_files: list[Path], initial_index: int = 0):
        super().__init__()
        self.configure(bg="#1e1e2e")
        self.title("WSI Feature Viewer")

        self.feature_files = feature_files
        self.current_feature_path = None
        self.features = None
        self.tile_paths = []
        self.tile_coords = []
        self.coord_to_idx = {}
        self.model_name = "unknown"
        self.slide_name = ""

        self.map_bg_image = None
        self.selection_rect = None
        self.map_scale_x = 1.0
        self.map_scale_y = 1.0
        self.read_size = 1.0
        self.read_stride = 1.0
        self.selected = None

        self._build_layout()
        self._set_feature_list_values()

        if self.feature_files:
            initial_index = max(0, min(initial_index, len(self.feature_files) - 1))
            self.feature_combo.current(initial_index)
            self._load_selected_feature()

    def _build_layout(self):
        left = tk.Frame(self, bg="#1e1e2e")
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)

        tk.Label(
            left,
            text="Tile Map",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        self.grid_canvas = tk.Canvas(
            left,
            bg="#181825",
            highlightthickness=0,
            bd=0,
            width=MAP_W,
            height=MAP_H,
        )
        self.grid_canvas.pack(fill="both", expand=False)
        self.grid_canvas.bind("<Button-1>", self._on_map_click)

        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        control = tk.Frame(right, bg="#1e1e2e")
        control.pack(fill="x", pady=(0, 8))
        tk.Label(
            control,
            text="Result",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 10, "bold"),
        ).pack(side="left", padx=(0, 8))
        self.feature_combo = ttk.Combobox(control, state="readonly", width=70)
        self.feature_combo.pack(side="left", fill="x", expand=True)
        self.feature_combo.bind("<<ComboboxSelected>>", lambda e: self._load_selected_feature())

        self.summary_label = tk.Label(
            right,
            text="",
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Courier", 10),
            justify="left",
            anchor="w",
        )
        self.summary_label.pack(fill="x", pady=(0, 8))

        tk.Label(
            right,
            text="Selected Tile",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w")

        self.preview_label = tk.Label(right, bg="#181825", relief="flat", bd=0)
        self.preview_label.pack(pady=(4, 8))

        self.tile_info_label = tk.Label(
            right,
            text="",
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Helvetica", 10),
        )
        self.tile_info_label.pack(anchor="w")

        self.stats_label = tk.Label(
            right,
            text="",
            bg="#1e1e2e",
            fg="#89dceb",
            font=("Courier", 10),
            justify="left",
        )
        self.stats_label.pack(anchor="w", pady=(4, 8))

        self.feature_title = tk.Label(
            right,
            text="Feature Vector",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 12, "bold"),
        )
        self.feature_title.pack(anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(6, 2.4))
        self.fig.patch.set_facecolor("#181825")
        self.ax.set_facecolor("#181825")
        self.fig.tight_layout(pad=1.2)

        self.chart = FigureCanvasTkAgg(self.fig, master=right)
        self.chart.get_tk_widget().pack(fill="x", expand=False)

    def _set_feature_list_values(self):
        self.feature_combo["values"] = [p.name for p in self.feature_files]

    def _load_selected_feature(self):
        idx = self.feature_combo.current()
        if idx < 0 or idx >= len(self.feature_files):
            return
        pt_path = self.feature_files[idx]
        self.current_feature_path = pt_path
        data = torch.load(pt_path, map_location="cpu")
        self.features = data["features"].numpy()
        self.tile_paths = [Path(p) for p in data["tile_paths"]]
        self.tile_coords = [parse_tile_coord(p) for p in self.tile_paths]
        self.coord_to_idx = {coord: i for i, coord in enumerate(self.tile_coords)}
        self.model_name = data.get("model", "unknown")
        self.slide_name = slide_name_from_pt(pt_path)

        dim = self.features.shape[1] if self.features is not None else 0
        missing = sum(1 for p in self.tile_paths if not p.exists())
        self.summary_label.configure(
            text=(
                f"slide={self.slide_name}\n"
                f"model={self.model_name}  dim={dim}  tiles={len(self.tile_paths)}  missing_files={missing}"
            )
        )
        self.feature_title.configure(text=f"Feature Vector ({dim} dims, {self.model_name})")

        self._render_tile_map()
        first_existing = next((i for i, p in enumerate(self.tile_paths) if p.exists()), None)
        if first_existing is not None:
            self._select(first_existing)

    def _load_overview_and_geometry(self):
        overview_path = TILES_DIR / self.slide_name / "overview.png"
        spec_path = TILES_DIR / self.slide_name / "specification.json"

        level0_w = 1.0
        level0_h = 1.0
        read_size = 1.0
        read_stride = 1.0
        if spec_path.exists():
            spec = json.loads(spec_path.read_text())
            ext = spec.get("extraction", {})
            level0_w = float(ext.get("level0_width", 1.0))
            level0_h = float(ext.get("level0_height", 1.0))
            read_size = float(ext.get("read_size", 1.0))
            read_stride = float(ext.get("read_stride", 1.0))

        self.read_size = max(read_size, 1.0)
        self.read_stride = max(read_stride, 1.0)

        if not overview_path.exists():
            self.map_scale_x = MAP_W / max(level0_w, 1.0)
            self.map_scale_y = MAP_H / max(level0_h, 1.0)
            self.map_bg_image = None
            return

        image = Image.open(overview_path).convert("RGB")
        w0, h0 = image.size
        scale = min(MAP_W / max(w0, 1), MAP_H / max(h0, 1))
        tw = max(1, int(w0 * scale))
        th = max(1, int(h0 * scale))
        thumb = image.resize((tw, th), Image.BILINEAR)
        self.map_bg_image = ImageTk.PhotoImage(thumb)
        self.map_scale_x = tw / max(level0_w, 1.0)
        self.map_scale_y = th / max(level0_h, 1.0)
        self.grid_canvas.create_image(0, 0, anchor="nw", image=self.map_bg_image)

    def _render_tile_map(self):
        self.grid_canvas.delete("all")
        self._load_overview_and_geometry()

        for row, col in self.tile_coords:
            x0 = (col * self.read_stride) * self.map_scale_x
            y0 = (row * self.read_stride) * self.map_scale_y
            x1 = x0 + (self.read_size * self.map_scale_x)
            y1 = y0 + (self.read_size * self.map_scale_y)
            self.grid_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill="",
                outline="#89b4fa",
                width=1,
            )

    def _on_map_click(self, event):
        if self.features is None:
            return
        lx = event.x / max(self.map_scale_x, 1e-6)
        ly = event.y / max(self.map_scale_y, 1e-6)
        col = int(lx // self.read_stride)
        row = int(ly // self.read_stride)
        idx = self.coord_to_idx.get((row, col))
        if idx is not None:
            self._select(idx)

    def _select(self, idx: int):
        self.selected = idx
        row, col = self.tile_coords[idx]
        x0 = (col * self.read_stride) * self.map_scale_x
        y0 = (row * self.read_stride) * self.map_scale_y
        x1 = x0 + (self.read_size * self.map_scale_x)
        y1 = y0 + (self.read_size * self.map_scale_y)

        if self.selection_rect is not None:
            self.grid_canvas.delete(self.selection_rect)
        self.selection_rect = self.grid_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#f38ba8",
            width=2,
        )

        self._update_preview(idx)
        self._update_chart(idx)

    def _update_preview(self, idx: int):
        path = self.tile_paths[idx]
        if not path.exists():
            self.preview_label.configure(image="")
            row_s, col_s = self.tile_coords[idx]
            self.tile_info_label.configure(text=f"  missing: {path.name} (row={row_s}, col={col_s})")
            self.stats_label.configure(text="  file not found")
            return

        img = Image.open(path).convert("RGB")
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        ph = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=ph)
        self.preview_label.image = ph

        row_s, col_s = self.tile_coords[idx]
        self.tile_info_label.configure(text=f"  {path.name}   (grid row={row_s}, col={col_s})")

        feat = self.features[idx]
        l2 = float(np.linalg.norm(feat))
        self.stats_label.configure(
            text=(
                f"  mean={feat.mean():.4f}   std={feat.std():.4f}\n"
                f"  min={feat.min():.4f}   max={feat.max():.4f}   l2={l2:.4f}"
            )
        )

    def _update_chart(self, idx: int):
        feat = self.features[idx]
        x = np.arange(len(feat))

        self.ax.clear()
        self.ax.set_facecolor("#181825")
        colors = ["#89b4fa" if v >= 0 else "#f38ba8" for v in feat]
        self.ax.bar(x, feat, color=colors, width=1.0, linewidth=0)
        self.ax.axhline(0, color="#585b70", linewidth=0.6)

        self.ax.set_xlim(0, len(feat))
        self.ax.set_xlabel("dimension", color="#6c7086", fontsize=8)
        self.ax.set_ylabel("value", color="#6c7086", fontsize=8)
        self.ax.tick_params(colors="#6c7086", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#313244")

        self.fig.tight_layout(pad=1.2)
        self.chart.draw()


def resolve_initial_index(feature_files: list[Path], feature_file: str):
    if not feature_file:
        return 0
    target = Path(feature_file).name
    for i, p in enumerate(feature_files):
        if p.name == target:
            return i
    return 0


def main():
    parser = argparse.ArgumentParser(description="Visual QA for extracted WSI features")
    parser.add_argument("--slide", default="", help="filter feature files by slide name")
    parser.add_argument("--model", choices=["ctranspath", "retccl"], default="", help="filter by model")
    parser.add_argument("--feature-file", default="", help="exact feature .pt filename to open first")
    args = parser.parse_args()

    feature_files = list_feature_files(slide=args.slide, model=args.model)
    if not feature_files:
        sys.exit("No feature files found. Run run_inference.py first.")

    initial_index = resolve_initial_index(feature_files, args.feature_file)
    app = Viewer(feature_files, initial_index=initial_index)
    app.mainloop()


if __name__ == "__main__":
    main()
