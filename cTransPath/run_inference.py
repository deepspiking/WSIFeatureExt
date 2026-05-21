"""
CTransPath feature extraction for WSI tiles.

Pipeline:
  datas/svs/*.svs  →  [wsi_read.py]  →  datas/tiles/{slide}/tiles/*.png
                   →  [this script]  →  features/{slide}.pt  + features/{slide}.csv
                                        CSV columns: tile_path, row, col, feat_0 … feat_767
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TILES_ROOT = Path("./datas/tiles")
FEATURES_OUT = Path("./features")
CTRANSPATH_MODEL_PATH = Path("./model/ctranspath.pth")
RETCCL_MODEL_PATH = Path("./model/best_ckpt.pth")

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

def load_ctranspath_model(model_path: Path, device: str) -> nn.Module:
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "TransPath"))
    from ctran import ctranspath

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(model_path, map_location=device)
    model.load_state_dict(td["model"], strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded cTransPath model from {model_path}")
    return model


def load_retccl_model(model_path: Path, device: str) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded RetCCL model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Per-slide inference
# ---------------------------------------------------------------------------

def extract_features(model, tile_paths: list[Path], device: str, batch_size: int) -> torch.Tensor:
    dataset = TileDataset(tile_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    parser = argparse.ArgumentParser(description="Unified feature extraction from tiled PNGs")
    parser.add_argument("--model", choices=["ctranspath", "retccl"], default="ctranspath")
    parser.add_argument("--tiles-root", type=Path, default=TILES_ROOT)
    parser.add_argument("--features-out", type=Path, default=FEATURES_OUT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=DEVICE)
    args = parser.parse_args()

    features_out = args.features_out
    tiles_root = args.tiles_root

    default_model_path = CTRANSPATH_MODEL_PATH if args.model == "ctranspath" else RETCCL_MODEL_PATH
    model_path = args.model_path if args.model_path is not None else default_model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    features_out.mkdir(exist_ok=True)

    if args.model == "ctranspath":
        model = load_ctranspath_model(model_path, args.device)
    else:
        model = load_retccl_model(model_path, args.device)
    print(f"Device: {args.device}")

    slide_dirs = sorted([d for d in tiles_root.iterdir() if d.is_dir()])
    if not slide_dirs:
        print(f"No slide directories found under {tiles_root}")
        return

    for slide_dir in slide_dirs:
        slide_name = slide_dir.name
        tiles_dir  = slide_dir / "tiles"

        tile_paths = sorted(tiles_dir.glob("*.png"))
        if not tile_paths:
            print(f"[{slide_name}] No tiles found, skipping.")
            continue

        print(f"\n[{slide_name}] {len(tile_paths)} tiles → extracting features...")
        features = extract_features(model, tile_paths, args.device, args.batch_size)
        print(f"[{slide_name}] feature shape: {features.shape}")  # (N, 768)

        # .pt 저장 (학습/분석용 원본)
        pt_path = features_out / f"{slide_name}.{args.model}.pt"
        torch.save({
            "features": features,
            "tile_paths": [str(p) for p in tile_paths],
            "model": args.model,
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

        csv_path = features_out / f"{slide_name}.{args.model}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{slide_name}] Saved .csv → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
