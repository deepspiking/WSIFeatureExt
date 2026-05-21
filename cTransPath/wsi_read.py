import os
import math
import json
from pathlib import Path

import openslide
from PIL import Image


# =========================
# Configuration
# =========================

INPUT_DIR = "./datas/svs"
OUTPUT_DIR = "./datas/tiles"

# CTransPath는 1.0 MPP 해상도에서 학습됨.
# read_size = round(PATCH_SIZE * TARGET_MPP / slide_mpp) 픽셀을 level 0에서 읽고
# PATCH_SIZE로 resize → 정확히 TARGET_MPP 해상도가 됨.
TARGET_MPP = 1.0
PATCH_SIZE = 224   # 모델 입력 크기 (저장 크기 = read_size, resize는 inference에서)

# background filtering
WHITE_THRESHOLD = 240
MIN_TISSUE_RATIO = 0.05


# =========================
# Utility Functions
# =========================

def is_background(tile: Image.Image,
                  white_threshold=WHITE_THRESHOLD,
                  min_tissue_ratio=MIN_TISSUE_RATIO):
    """
    Skip mostly white/background tiles.
    """

    gray = tile.convert("L")
    pixels = gray.load()

    w, h = gray.size
    total = w * h

    tissue = 0

    for y in range(h):
        for x in range(w):
            if pixels[x, y] < white_threshold:
                tissue += 1

    tissue_ratio = tissue / total

    return tissue_ratio < min_tissue_ratio


def extract_slide_spec(slide: openslide.OpenSlide):
    """
    Extract slide metadata/specification.
    """

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


# =========================
# Main Processing
# =========================

def process_svs_file(svs_path: Path):

    print(f"\nProcessing: {svs_path.name}")

    slide = openslide.OpenSlide(str(svs_path))

    # -------------------------
    # Create output folder
    # -------------------------

    slide_name = svs_path.stem
    slide_output_dir = Path(OUTPUT_DIR) / slide_name

    tiles_dir = slide_output_dir / "tiles"

    slide_output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Save specification
    # -------------------------

    spec = extract_slide_spec(slide)

    spec_path = slide_output_dir / "specification.json"

    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"Saved specification -> {spec_path}")

    # -------------------------
    # Tile extraction
    # -------------------------

    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
    if mpp_x <= 0:
        raise ValueError("MPP 정보가 없습니다. TARGET_MPP 기반 추출 불가.")

    # level 0에서 읽어야 할 픽셀 크기 (= TARGET_MPP에 해당하는 물리적 영역)
    read_size = round(PATCH_SIZE * TARGET_MPP / mpp_x)
    effective_mpp = read_size * mpp_x / PATCH_SIZE

    width, height = slide.level_dimensions[0]

    cols = math.ceil(width / read_size)
    rows = math.ceil(height / read_size)

    print(f"Slide MPP: {mpp_x:.4f}  →  read_size: {read_size}px  →  effective MPP: {effective_mpp:.4f}")
    print(f"Slide size: {width} x {height}")
    print(f"Tiles: {cols} x {rows}")

    tile_count = 0

    for row in range(rows):
        for col in range(cols):

            x = col * read_size
            y = row * read_size

            tile = slide.read_region(
                (x, y),
                0,                       # 항상 level 0에서 읽음
                (read_size, read_size)
            )

            tile = tile.convert("RGB")

            # skip background
            if is_background(tile):
                continue

            # read_size 그대로 저장 (inference 시 224로 resize됨)
            tile_filename = f"tile_r{row}_c{col}.png"

            tile_path = tiles_dir / tile_filename

            tile.save(tile_path)

            tile_count += 1

    print(f"Saved {tile_count} tiles (each {read_size}×{read_size}px at level 0)")

    slide.close()


def main():

    input_dir = Path(INPUT_DIR)

    svs_files = list(input_dir.glob("*.svs"))

    print(f"Found {len(svs_files)} SVS files")

    for svs_file in svs_files:
        try:
            process_svs_file(svs_file)

        except Exception as e:
            print(f"ERROR processing {svs_file.name}")
            print(e)


if __name__ == "__main__":
    main()
