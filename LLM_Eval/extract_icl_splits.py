#!/usr/bin/env python3
"""
Create in-context learning (ICL) splits by extracting the last 10% of rows
per language (and per contains_slang bucket for detection) from the original
datasets, writing them to `*_icl.csv`, and shrinking the source CSVs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

CSV_ENCODING = "utf-8-sig"
BASE_DIR = Path(__file__).resolve().parent


DATASET_CONFIG = {
    "detection": {
        "dir": BASE_DIR / "Detection",
        "filename_fmt": "{lang}_detection.csv",
        "group_col": "contains_slang",
        "icl_fmt": "{lang}_detection_icl.csv",
    },
    "generation": {
        "dir": BASE_DIR / "Generation",
        "filename_fmt": "{lang}_generation.csv",
        "group_col": None,
        "icl_fmt": "{lang}_generation_icl.csv",
    },
    "interpretation": {
        "dir": BASE_DIR / "Interpretation",
        "filename_fmt": "{lang}_interpretation.csv",
        "group_col": None,
        "icl_fmt": "{lang}_interpretation_icl.csv",
    },
}

LANGUAGES = ("en", "zh", "ru")
TAIL_FRACTION = 0.10


def determine_tail_size(size: int, fraction: float = TAIL_FRACTION) -> int:
    if size <= 0:
        return 0
    count = math.ceil(size * fraction)
    return min(size, max(1, count))


def extract_tail_indices(index_list: List[int], count: int) -> List[int]:
    if count <= 0:
        return []
    return index_list[-count:]


def process_language_file(
    dataset_type: str,
    lang: str,
    cfg: Dict[str, Path],
) -> pd.DataFrame:
    path = cfg["dir"] / cfg["filename_fmt"].format(lang=lang)
    if not path.exists():
        raise FileNotFoundError(f"{dataset_type} dataset missing: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    group_col = cfg["group_col"]
    icl_frames: List[pd.DataFrame] = []
    indices_to_drop: List[int] = []

    if group_col and group_col in df.columns:
        for value in df[group_col].unique():
            group_idx = df.index[df[group_col] == value].tolist()
            tail_size = determine_tail_size(len(group_idx))
            selected_idx = extract_tail_indices(group_idx, tail_size)
            if not selected_idx:
                continue
            icl_frames.append(df.loc[selected_idx].copy())
            indices_to_drop.extend(selected_idx)
    else:
        all_idx = df.index.tolist()
        tail_size = determine_tail_size(len(all_idx))
        selected_idx = extract_tail_indices(all_idx, tail_size)
        if selected_idx:
            icl_frames.append(df.loc[selected_idx].copy())
            indices_to_drop.extend(selected_idx)

    if indices_to_drop:
        remaining = df.drop(indices_to_drop).reset_index(drop=True)
        remaining.to_csv(path, index=False, encoding=CSV_ENCODING)
    else:
        remaining = df

    if icl_frames:
        icl_df = pd.concat(icl_frames, ignore_index=True).reset_index(drop=True)
        output_path = cfg["dir"] / cfg["icl_fmt"].format(lang=lang)
        icl_df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
        print(
            f"[INFO] {dataset_type}:{lang} wrote {len(icl_df)} rows to {output_path} "
            f"and retained {len(remaining)} rows in source."
        )
    else:
        print(f"[WARN] {dataset_type}:{lang} had no rows to extract.")

    return icl_df if icl_frames else pd.DataFrame()


def main() -> None:
    for dataset_type, cfg in DATASET_CONFIG.items():
        for lang in LANGUAGES:
            process_language_file(dataset_type, lang, cfg)


if __name__ == "__main__":
    main()


