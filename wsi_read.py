import json
import argparse
from pathlib import Path

import openslide
from PIL import Image
import numpy as np


INPUT_DIR = "./datas/svs"
OUTPUT_DIR = "./datas/tiles"
TARGET_MPP = 1.0
PATCH_SIZE = 224
STRIDE = 224
WHITE_THRESHOLD = 240
MAX_WHITE_RATIO = 0.75
BLACK_THRESHOLD = 15
MAX_BLACK_RATIO = 0.10

def is_foreground(
    tile: Image.Image,
    white_threshold: int,
    max_white_ratio: float,
    black_threshold: int,
    max_black_ratio: float,
) -> bool:
    arr = np.asarray(tile)
    white = np.all(arr >= white_threshold, axis=2)
    black = np.all(arr <= black_threshold, axis=2)
    white_ratio = white.mean()
    black_ratio = black.mean()
    return white_ratio <= max_white_ratio and black_ratio <= max_black_ratio


def extract_slide_spec(slide: openslide.OpenSlide):
    spec = {
        "vendor": slide.properties.get(openslide.PROPERTY_NAME_VENDOR),

        "level_count": slide.level_count,

        "level_dimensions": slide.level_dimensions,

        "level_downsamples": slide.level_downsamples,

        "mpp_x": slide.properties.get(
            openslide.PROPERTY_NAME_MPP_X
        ),

        "mpp_y": slide.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        ),

        "objective_power": slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER
        ),

        "dimensions": slide.dimensions,

        "properties": dict(slide.properties)
    }

    return spec

def process_svs_file(
    svs_path: Path,
    output_dir: Path,
    target_mpp: float,
    patch_size: int,
    stride: int,
    white_threshold: int,
    max_white_ratio: float,
    black_threshold: int,
    max_black_ratio: float,
    limit: int,
):

    print(f"\nProcessing: {svs_path.name}")

    slide = openslide.OpenSlide(str(svs_path))

    slide_name = svs_path.stem
    slide_output_dir = output_dir / slide_name

    tiles_dir = slide_output_dir / "tiles"

    slide_output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    spec = extract_slide_spec(slide)

    spec_path = slide_output_dir / "specification.json"
    overview_path = slide_output_dir / "overview.png"

    thumb = slide.get_thumbnail((1200, 1200)).convert("RGB")
    thumb.save(overview_path)

    print(f"Saved specification -> {spec_path}")
    print(f"Saved overview -> {overview_path}")

    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
    if mpp_x <= 0:
        raise ValueError("MPP 정보가 없습니다. TARGET_MPP 기반 추출 불가.")

    read_size = round(patch_size * target_mpp / mpp_x)
    read_stride = round(stride * target_mpp / mpp_x)
    effective_mpp = read_size * mpp_x / patch_size

    read_size = max(1, int(read_size))
    read_stride = max(1, int(read_stride))

    width, height = slide.level_dimensions[0]

    cols = max((width - read_size) // read_stride + 1, 0)
    rows = max((height - read_size) // read_stride + 1, 0)

    spec["extraction"] = {
        "target_mpp": target_mpp,
        "patch_size": patch_size,
        "stride": stride,
        "read_size": read_size,
        "read_stride": read_stride,
        "level0_width": width,
        "level0_height": height,
        "rows": rows,
        "cols": cols,
        "outer_border_dropped": True,
    }

    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"Slide MPP: {mpp_x:.4f}  →  read_size: {read_size}px  →  effective MPP: {effective_mpp:.4f}")
    print(f"Slide size: {width} x {height}")
    print(f"Tiles: {cols} x {rows}")

    tile_count = 0

    for row in range(rows):
        for col in range(cols):
            if row == 0 or col == 0 or row == rows - 1 or col == cols - 1:
                continue

            x = col * read_stride
            y = row * read_stride

            tile = slide.read_region((x, y), 0, (read_size, read_size))

            tile = tile.convert("RGB")
            tile = tile.resize((patch_size, patch_size), Image.BILINEAR)

            if not is_foreground(
                tile,
                white_threshold,
                max_white_ratio,
                black_threshold,
                max_black_ratio,
            ):
                continue

            tile_filename = f"tile_r{row}_c{col}.png"

            tile_path = tiles_dir / tile_filename

            tile.save(tile_path)

            tile_count += 1
            if limit > 0 and tile_count >= limit:
                print(f"Saved {tile_count} tiles (limited)")
                slide.close()
                return

    print(f"Saved {tile_count} tiles (each {patch_size}×{patch_size}px, sampled from level 0)")

    slide.close()


def main():
    parser = argparse.ArgumentParser(description="Unified WSI to PNG tiling (cTransPath level-0 sampling + strict filters)")
    parser.add_argument("--input-dir", type=Path, default=Path(INPUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--target-mpp", type=float, default=TARGET_MPP)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--white-threshold", type=int, default=WHITE_THRESHOLD)
    parser.add_argument("--max-white-ratio", type=float, default=MAX_WHITE_RATIO)
    parser.add_argument("--black-threshold", type=int, default=BLACK_THRESHOLD)
    parser.add_argument("--max-black-ratio", type=float, default=MAX_BLACK_RATIO)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = args.input_dir

    svs_files = list(input_dir.glob("*.svs"))

    print(f"Found {len(svs_files)} SVS files")

    for svs_file in svs_files:
        try:
            process_svs_file(
                svs_file,
                output_dir=args.output_dir,
                target_mpp=args.target_mpp,
                patch_size=args.patch_size,
                stride=args.stride,
                white_threshold=args.white_threshold,
                max_white_ratio=args.max_white_ratio,
                black_threshold=args.black_threshold,
                max_black_ratio=args.max_black_ratio,
                limit=args.limit,
            )
        except Exception as e:
            print(f"ERROR processing {svs_file.name}")
            print(e)


if __name__ == "__main__":
    main()
