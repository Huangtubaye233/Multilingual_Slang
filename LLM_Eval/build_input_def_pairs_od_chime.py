#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build input_def_pairs_OD_chime.csv by replacing the English rows in "
            "input_def_pairs_chime.csv with merged en_interpretation_OD.csv + "
            "en_interpretation_OD_icl.csv."
        )
    )
    parser.add_argument(
        "--base-pairs",
        type=Path,
        default=PROJECT_ROOT / "slanggen/Demo/input_def_pairs_chime.csv",
        help="Base input_def_pairs_chime CSV path.",
    )
    parser.add_argument(
        "--en-od",
        type=Path,
        default=BASE_DIR / "Interpretation/en_interpretation_OD.csv",
        help="English OD interpretation CSV path.",
    )
    parser.add_argument(
        "--en-od-icl",
        type=Path,
        default=BASE_DIR / "Interpretation/en_interpretation_OD_icl.csv",
        help="English OD ICL CSV path (merged with --en-od).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "slanggen/Demo/input_def_pairs_OD_chime.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_en_block(en_df: pd.DataFrame, target_cols: List[str], src_name: str) -> pd.DataFrame:
    require_columns(en_df, ["slang", "conv_def", "slang_definition"], src_name)
    en_block = pd.DataFrame(
        {
            "lang": "en",
            "slang": en_df["slang"],
            "conv_def_en": en_df["conv_def"],
            "conv_def": en_df["conv_def"],
            "slang_definition": en_df["slang_definition"],
            "slang_definition_en": en_df["slang_definition"],
        }
    )
    # Keep the exact same column order as base pairs file.
    return en_block[target_cols]


def main() -> None:
    args = parse_args()

    base_df = pd.read_csv(args.base_pairs)
    en_od_df = pd.read_csv(args.en_od)
    en_od_icl_df = pd.read_csv(args.en_od_icl)

    target_cols = [
        "lang",
        "slang",
        "conv_def_en",
        "conv_def",
        "slang_definition",
        "slang_definition_en",
    ]
    require_columns(base_df, target_cols, "input_def_pairs_chime.csv")

    en_block_main = build_en_block(en_od_df, target_cols, "en_interpretation_OD.csv")
    en_block_icl = build_en_block(en_od_icl_df, target_cols, "en_interpretation_OD_icl.csv")
    en_block = pd.concat([en_block_main, en_block_icl], ignore_index=True)
    en_block = en_block.drop_duplicates(
        subset=["lang", "slang", "conv_def", "slang_definition"]
    ).reset_index(drop=True)

    # Keep non-English rows from base file untouched.
    base_non_en = base_df[base_df["lang"] != "en"].copy()

    # Preserve grouped language order from the base file while swapping English rows.
    lang_order = list(dict.fromkeys(base_df["lang"].astype(str).tolist()))
    chunks: List[pd.DataFrame] = []
    for lang in lang_order:
        if lang == "en":
            chunks.append(en_block)
        else:
            chunks.append(base_non_en[base_non_en["lang"] == lang])

    # If base has no language order for some reason, still output valid merged data.
    if not chunks:
        chunks = [en_block, base_non_en]

    out_df = pd.concat(chunks, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out_df)} rows to {args.output}")
    print(f"English rows from OD main: {len(en_block_main)}")
    print(f"English rows from OD ICL: {len(en_block_icl)}")
    print(f"English rows after merge+dedup: {len(en_block)}")
    print(f"Non-English rows kept from base: {len(base_non_en)}")


if __name__ == "__main__":
    main()
