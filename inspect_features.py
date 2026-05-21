import torch
import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("./features")

for pt_path in sorted(FEATURES_DIR.glob("*.pt")):
    data = torch.load(pt_path, map_location="cpu")
    features   = data["features"]       # (N, 768)
    tile_paths = data["tile_paths"]
    model_name = data.get("model", "unknown")

    print(f"{'='*60}")
    print(f"Slide   : {pt_path.stem}")
    print(f"Model   : {model_name}")
    print(f"{'='*60}")
    print(f"Tiles   : {features.shape[0]}")
    print(f"Feat dim: {features.shape[1]}")
    print()

    feat_np = features.numpy()
    print(f"[Feature 통계]")
    print(f"  mean : {feat_np.mean():.4f}")
    print(f"  std  : {feat_np.std():.4f}")
    print(f"  min  : {feat_np.min():.4f}")
    print(f"  max  : {feat_np.max():.4f}")
    print()

    print(f"[첫 3개 tile]")
    for i in range(min(3, len(tile_paths))):
        vec = feat_np[i]
        print(f"  [{i}] {Path(tile_paths[i]).name}")
        print(f"       feat[:5] = {vec[:5].round(4)}")
    print()
