"""
CTransPath feature extraction for WSI tiles.

Pipeline:
  datas/svs/*.svs  →  [wsi_read.py]  →  datas/tiles/{slide}/tiles/*.png
                   →  [this script]  →  features/{slide}.pt  + features/{slide}.csv
                                        CSV columns: tile_path, row, col, feat_0 … feat_767
"""

import sys
import re
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ctran.py lives in TransPath/
sys.path.insert(0, str(Path(__file__).parent / "TransPath"))
from ctran import ctranspath

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TILES_ROOT   = Path("./datas/tiles")   # wsi_read.py 출력 루트
FEATURES_OUT = Path("./features")      # feature 저장 위치
MODEL_PATH   = Path("./model/ctranspath.pth")

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# CTransPath 공식 normalization
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TileDataset(Dataset):
    def __init__(self, tile_paths: list[Path]):
        self.paths = tile_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return transform(img)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: str) -> nn.Module:
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(model_path, map_location=device)
    model.load_state_dict(td["model"], strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Per-slide inference
# ---------------------------------------------------------------------------

def extract_features(model, tile_paths: list[Path], device: str) -> torch.Tensor:
    dataset = TileDataset(tile_paths)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model(batch)          # (B, 768)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)  # (N_tiles, 768)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FEATURES_OUT.mkdir(exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)
    print(f"Device: {DEVICE}")

    slide_dirs = sorted([d for d in TILES_ROOT.iterdir() if d.is_dir()])
    if not slide_dirs:
        print(f"No slide directories found under {TILES_ROOT}")
        return

    for slide_dir in slide_dirs:
        slide_name = slide_dir.name
        tiles_dir  = slide_dir / "tiles"

        tile_paths = sorted(tiles_dir.glob("*.png"))
        if not tile_paths:
            print(f"[{slide_name}] No tiles found, skipping.")
            continue

        print(f"\n[{slide_name}] {len(tile_paths)} tiles → extracting features...")
        features = extract_features(model, tile_paths, DEVICE)
        print(f"[{slide_name}] feature shape: {features.shape}")  # (N, 768)

        # .pt 저장 (학습/분석용 원본)
        pt_path = FEATURES_OUT / f"{slide_name}.pt"
        torch.save({
            "features": features,
            "tile_paths": [str(p) for p in tile_paths],
        }, pt_path)
        print(f"[{slide_name}] Saved .pt → {pt_path}")

        # CSV 저장 (눈으로 확인용)
        # tile_path, row, col, feat_0 ... feat_767
        feat_cols = [f"feat_{i}" for i in range(features.shape[1])]
        rows, cols = [], []
        for p in tile_paths:
            m = re.search(r"tile_r(\d+)_c(\d+)", p.name)
            rows.append(int(m.group(1)) if m else -1)
            cols.append(int(m.group(2)) if m else -1)

        df = pd.DataFrame(features.numpy(), columns=feat_cols)
        df.insert(0, "tile_path", [str(p) for p in tile_paths])
        df.insert(1, "row", rows)
        df.insert(2, "col", cols)

        csv_path = FEATURES_OUT / f"{slide_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{slide_name}] Saved .csv → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
