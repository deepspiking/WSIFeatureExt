"""
Tile-Feature Viewer
  - 왼쪽: tile 썸네일 그리드 (스크롤 가능)
  - 오른쪽 상단: 선택한 tile 원본 이미지
  - 오른쪽 하단: 해당 feature 벡터 시각화 (bar chart + 수치)

실행: python viewer.py [slide_name]
      slide_name 생략 시 features/ 에서 첫 번째 슬라이드 자동 선택
"""

import sys
import tkinter as tk
from tkinter import ttk, font
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_DIR = Path("./features")
THUMB_SIZE   = 80    # 그리드 썸네일 크기 (px)
GRID_COLS    = 8     # 한 행에 표시할 tile 수
PREVIEW_SIZE = 300   # 우측 상단 미리보기 크기 (px)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_slide(pt_path: Path):
    data = torch.load(pt_path, map_location="cpu")
    features   = data["features"].numpy()   # (N, 768)
    tile_paths = [Path(p) for p in data["tile_paths"]]
    return features, tile_paths


def pick_slide(name_hint: str | None) -> Path:
    pts = sorted(FEATURES_DIR.glob("*.pt"))
    if not pts:
        sys.exit("features/ 에 .pt 파일이 없습니다. run_inference.py 를 먼저 실행하세요.")
    if name_hint:
        for p in pts:
            if name_hint in p.stem:
                return p
        sys.exit(f"'{name_hint}' 와 일치하는 슬라이드를 찾을 수 없습니다.")
    return pts[0]

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class Viewer(tk.Tk):
    def __init__(self, pt_path: Path):
        super().__init__()
        self.title(f"Tile-Feature Viewer  —  {pt_path.stem}")
        self.configure(bg="#1e1e2e")

        self.features, self.tile_paths = load_slide(pt_path)
        self.n_tiles = len(self.tile_paths)
        self.selected = None

        self._build_layout()
        self._load_thumbnails()
        self._populate_grid()

        # 첫 번째 tile 자동 선택
        self._select(0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        # ── 왼쪽 패널: tile 그리드 ──────────────────────────────────
        self._grid_canvas_w = (THUMB_SIZE + 4) * GRID_COLS
        left = tk.Frame(self, bg="#1e1e2e")
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)

        header = tk.Label(left, text=f"Tiles  ({self.n_tiles})",
                          bg="#1e1e2e", fg="#cdd6f4",
                          font=("Helvetica", 12, "bold"))
        header.pack(anchor="w", pady=(0, 6))

        # 스크롤 가능한 캔버스
        canvas_frame = tk.Frame(left, bg="#1e1e2e")
        canvas_frame.pack(fill="both", expand=True)

        self.grid_canvas = tk.Canvas(canvas_frame, bg="#181825",
                                     highlightthickness=0, bd=0,
                                     width=self._grid_canvas_w)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                   command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.grid_canvas.pack(side="left", fill="both", expand=True)

        self.grid_frame = tk.Frame(self.grid_canvas, bg="#181825")
        self.grid_window = self.grid_canvas.create_window(
            (0, 0), window=self.grid_frame, anchor="nw"
        )
        self.grid_frame.bind("<Configure>", self._on_grid_resize)
        self.grid_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.grid_canvas.bind("<Button-4>",   self._on_mousewheel)
        self.grid_canvas.bind("<Button-5>",   self._on_mousewheel)

        # ── 오른쪽 패널: 미리보기 + feature ────────────────────────
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        # 미리보기 이미지
        preview_header = tk.Label(right, text="Selected Tile",
                                   bg="#1e1e2e", fg="#cdd6f4",
                                   font=("Helvetica", 12, "bold"))
        preview_header.pack(anchor="w")

        self.preview_label = tk.Label(right, bg="#181825",
                                       relief="flat", bd=0)
        self.preview_label.pack(pady=(4, 8))

        self.tile_info_label = tk.Label(right, text="",
                                         bg="#1e1e2e", fg="#a6adc8",
                                         font=("Helvetica", 10))
        self.tile_info_label.pack(anchor="w")

        # feature 통계
        self.stats_label = tk.Label(right, text="",
                                     bg="#1e1e2e", fg="#89dceb",
                                     font=("Courier", 10), justify="left")
        self.stats_label.pack(anchor="w", pady=(4, 8))

        # feature bar chart
        feat_header = tk.Label(right, text="Feature Vector  (768 dims)",
                                bg="#1e1e2e", fg="#cdd6f4",
                                font=("Helvetica", 12, "bold"))
        feat_header.pack(anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(6, 2.4))
        self.fig.patch.set_facecolor("#181825")
        self.ax.set_facecolor("#181825")
        self.fig.tight_layout(pad=1.2)

        self.chart = FigureCanvasTkAgg(self.fig, master=right)
        self.chart.get_tk_widget().pack(fill="x", expand=False)

    # ------------------------------------------------------------------
    # Thumbnail grid
    # ------------------------------------------------------------------

    def _load_thumbnails(self):
        self.thumbs = []
        self.thumb_imgs = []   # PhotoImage 참조 유지용
        for p in self.tile_paths:
            img = Image.open(p).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            self.thumbs.append(img)

    def _populate_grid(self):
        self.thumb_btns = []
        for idx, img in enumerate(self.thumbs):
            ph = ImageTk.PhotoImage(img)
            self.thumb_imgs.append(ph)

            row, col = divmod(idx, GRID_COLS)
            btn = tk.Label(self.grid_frame, image=ph,
                           bg="#181825", cursor="hand2",
                           relief="flat", bd=2)
            btn.grid(row=row, column=col, padx=2, pady=2)
            btn.bind("<Button-1>", lambda e, i=idx: self._select(i))
            self.thumb_btns.append(btn)

    def _on_grid_resize(self, event):
        self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.grid_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.grid_canvas.yview_scroll(1, "units")
        else:
            self.grid_canvas.yview_scroll(int(-event.delta / 60), "units")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, idx: int):
        # 이전 선택 해제
        if self.selected is not None:
            self.thumb_btns[self.selected].configure(bg="#181825", relief="flat")

        self.selected = idx
        self.thumb_btns[idx].configure(bg="#89b4fa", relief="solid")

        self._update_preview(idx)
        self._update_chart(idx)

    def _update_preview(self, idx: int):
        path = self.tile_paths[idx]
        img  = Image.open(path).convert("RGB")
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        ph = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=ph)
        self.preview_label.image = ph   # 참조 유지

        # row / col 파싱
        import re
        m = re.search(r"tile_r(\d+)_c(\d+)", path.name)
        row_s = m.group(1) if m else "?"
        col_s = m.group(2) if m else "?"
        self.tile_info_label.configure(
            text=f"  {path.name}   (grid row={row_s}, col={col_s})"
        )

        feat = self.features[idx]
        self.stats_label.configure(
            text=(f"  mean={feat.mean():.4f}   std={feat.std():.4f}"
                  f"   min={feat.min():.4f}   max={feat.max():.4f}")
        )

    def _update_chart(self, idx: int):
        feat = self.features[idx]          # (768,)
        x    = np.arange(len(feat))

        self.ax.clear()
        self.ax.set_facecolor("#181825")

        # 양수/음수 색 구분
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hint = sys.argv[1] if len(sys.argv) > 1 else None
    pt_path = pick_slide(hint)
    print(f"Loading: {pt_path}")
    app = Viewer(pt_path)
    app.mainloop()
