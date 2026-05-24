import argparse
import json
from pathlib import Path

import pandas as pd
import torch


DEFAULT_TILES_ROOT = Path("./data_ex/tiles")
DEFAULT_MODELS_ROOT = Path(".")


def compute_tile_grid(spec_path: Path, target_mpp: float, patch_size: int, stride: int) -> tuple[int, int, int]:
    with open(spec_path) as f:
        spec = json.load(f)

    mpp_x = float(spec.get("mpp_x") or 0)
    if mpp_x <= 0:
        raise ValueError(f"Invalid mpp_x in {spec_path}")

    width, height = spec["level_dimensions"][0]
    read_size = max(1, int(round(patch_size * target_mpp / mpp_x)))
    read_stride = max(1, int(round(stride * target_mpp / mpp_x)))

    cols = max((width - read_size) // read_stride + 1, 0)
    rows = max((height - read_size) // read_stride + 1, 0)
    return rows, cols, rows * cols


def summarize_slide(slide_dir: Path, model: str, model_dir: Path, target_mpp: float, patch_size: int, stride: int) -> dict:
    slide_id = slide_dir.name
    spec_path = slide_dir / "specification.json"
    tiles_dir = slide_dir / "tiles"
    tile_paths = sorted(tiles_dir.glob("*.png"))

    row = {
        "slide_id": slide_id,
        "tile_grid": "",
        "tile_grid_rows": None,
        "tile_grid_cols": None,
        "tile_grid_count": None,
        "saved_tile_count": len(tile_paths),
        "feature_count": None,
        "feature_dim": None,
        "count_delta": None,
        "status": "ok",
    }

    try:
        grid_rows, grid_cols, grid_count = compute_tile_grid(spec_path, target_mpp, patch_size, stride)
        row["tile_grid_rows"] = grid_rows
        row["tile_grid_cols"] = grid_cols
        row["tile_grid_count"] = grid_count
        row["tile_grid"] = f"{grid_rows} x {grid_cols} = {grid_count}"
    except Exception as exc:
        row["status"] = f"spec_error:{exc}"

    feature_path = model_dir / f"{slide_id}.{model}.pt"
    if not feature_path.exists():
        row["status"] = "missing_features" if row["status"] == "ok" else row["status"]
        return row

    try:
        data = torch.load(feature_path, map_location="cpu")
        features = data["features"]
        feature_count = int(features.shape[0])
        feature_dim = int(features.shape[1]) if features.ndim > 1 else 1

        row["feature_count"] = feature_count
        row["feature_dim"] = feature_dim
        row["count_delta"] = feature_count - len(tile_paths)

        if feature_count != len(tile_paths):
            row["status"] = "count_mismatch"
    except Exception as exc:
        row["status"] = f"feature_error:{exc}"

    return row


def summarize_model(
    tiles_root: Path,
    models_root: Path,
    output_dir: Path,
    model: str,
    target_mpp: float,
    patch_size: int,
    stride: int,
) -> Path:
    slide_dirs = sorted(d for d in tiles_root.iterdir() if d.is_dir())
    model_dir = models_root / model if (models_root / model).is_dir() else models_root
    rows = [
        summarize_slide(
            slide_dir=slide_dir,
            model=model,
            model_dir=model_dir,
            target_mpp=target_mpp,
            patch_size=patch_size,
            stride=stride,
        )
        for slide_dir in slide_dirs
    ]

    df = pd.DataFrame(rows)
    int_columns = [
        "tile_grid_rows",
        "tile_grid_cols",
        "tile_grid_count",
        "saved_tile_count",
        "feature_count",
        "feature_dim",
        "count_delta",
    ]
    for col in int_columns:
        df[col] = pd.array(df[col], dtype="Int64")
        df[col] = df[col].map(lambda value: "" if pd.isna(value) else str(int(value)))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model}_feat.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Summarize per-slide tile and feature counts for each model")
    parser.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["ctranspath", "retccl"])
    parser.add_argument("--target-mpp", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.models_root

    for model in args.models:
        out_path = summarize_model(
            tiles_root=args.tiles_root,
            models_root=args.models_root,
            output_dir=output_dir,
            model=model,
            target_mpp=args.target_mpp,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        print(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
