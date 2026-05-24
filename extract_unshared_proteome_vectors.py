import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "./data_ex/proteome/CPTAC2_Breast_Prospective_Collection_BI_Proteome.tmt10.tsv"
)
DEFAULT_OUTPUT = Path(
    "./data_ex/proteome/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors.pkl"
)

STATS_ROWS = {"Mean", "Median", "StdDev"}
RETROIR_EXCLUDE = {"RetroIR.1", "RetroIR.2"}
UNSHARED_SUFFIX = " Unshared Log Ratio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract unshared proteome vectors by aliquot")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def split_versioned_aliquot(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*)\.(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, 0


def select_unshared_columns(columns: list[str]) -> tuple[dict[str, str], list[dict[str, object]]]:
    candidates: dict[str, list[tuple[int, str]]] = {}
    for column in columns:
        if not column.endswith(UNSHARED_SUFFIX):
            continue

        aliquot_name = column[: -len(UNSHARED_SUFFIX)]
        if aliquot_name in RETROIR_EXCLUDE:
            continue

        base_aliquot, version = split_versioned_aliquot(aliquot_name)
        candidates.setdefault(base_aliquot, []).append((version, column))

    selected: dict[str, str] = {}
    selection_rows: list[dict[str, object]] = []
    for base_aliquot, versions in sorted(candidates.items()):
        chosen_version, chosen_column = max(versions, key=lambda item: item[0])
        selected[base_aliquot] = chosen_column
        selection_rows.append(
            {
                "aliquot_id": base_aliquot,
                "selected_column": chosen_column,
                "selected_version": chosen_version,
                "candidate_count": len(versions),
            }
        )

    return selected, selection_rows


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t")
    df = df[~df["Gene"].isin(STATS_ROWS)].copy()

    selected_columns, selection_rows = select_unshared_columns(df.columns.tolist())

    aliquot_to_vector = {
        aliquot_id: df[column].to_numpy(dtype=np.float32, copy=True)
        for aliquot_id, column in selected_columns.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(aliquot_to_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

    selection_path = args.output.with_suffix(".selection.csv")
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)

    genes_path = args.output.with_suffix(".genes.txt")
    df["Gene"].to_csv(genes_path, index=False, header=False)

    print(f"Saved vectors -> {args.output}")
    print(f"Saved selection -> {selection_path}")
    print(f"Saved gene order -> {genes_path}")
    print(f"genes={len(df)}")
    print(f"aliquots={len(aliquot_to_vector)}")


if __name__ == "__main__":
    main()
