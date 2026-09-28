#!/usr/bin/env python3
"""
Resample closed-ended distractors with semantic cosine similarity.

For each language pair (main + ICL CSV):
1) Merge both files into one distractor pool.
2) Encode `slang_definition` with a multilingual MPNet model.
3) For each row, sample up to 3 distractors from rows with cosine <= threshold.
4) If fewer than 3 exist, fill from least-similar rows until 3 distractors are set.
5) Split back and save files with `cossim_sampled_` prefix.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from sentence_transformers import SentenceTransformer, util


CSV_ENCODING = "utf-8-sig"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PREFIX = "cossim_sampled_"


@dataclass(frozen=True)
class DatasetPair:
    language: str
    base_file: str
    icl_file: str


DATASET_PAIRS: Sequence[DatasetPair] = (
    DatasetPair("en", "en_generation_OD.csv", "en_generation_OD_icl.csv"),
    DatasetPair("zh", "zh_generation_chime.csv", "zh_generation_chime_icl.csv"),
    DatasetPair("ru", "ru_generation.csv", "ru_generation_icl.csv"),
)

REQUIRED_COLUMNS = (
    "slang",
    "slang_definition",
    "distractor_1",
    "distractor_1_slang_def",
    "distractor_2",
    "distractor_2_slang_def",
    "distractor_3",
    "distractor_3_slang_def",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample distractors using cosine similarity on slang_definition."
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Generation",
        help="Directory that contains generation CSV files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model for multilingual embeddings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for primary distractor candidates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _deterministic_rng(seed: int, language: str, row_index: int) -> random.Random:
    rng = random.Random()
    rng.seed(f"{seed}-{language}-{row_index}")
    return rng


def _sample_distractor_indices(
    *,
    cosine_row: Sequence[float],
    self_index: int,
    threshold: float,
    rng: random.Random,
) -> tuple[List[int], bool]:
    all_candidates = [i for i in range(len(cosine_row)) if i != self_index]
    if len(all_candidates) < 3:
        raise ValueError(
            f"Need at least 4 total rows to create 3 distractors, got {len(cosine_row)}."
        )

    eligible = [i for i in all_candidates if cosine_row[i] <= threshold]
    rng.shuffle(eligible)
    chosen = eligible[:3]
    used_fallback = len(chosen) < 3

    if len(chosen) < 3:
        fallback_sorted = sorted(all_candidates, key=lambda i: cosine_row[i])
        for cand in fallback_sorted:
            if cand in chosen:
                continue
            chosen.append(cand)
            if len(chosen) == 3:
                break

    if len(chosen) < 3:
        raise RuntimeError(f"Failed to collect 3 distractors for row index {self_index}.")
    return chosen, used_fallback


def process_pair(
    pair: DatasetPair,
    model: SentenceTransformer,
    generation_dir: Path,
    threshold: float,
    seed: int,
    batch_size: int,
) -> None:
    base_path = generation_dir / pair.base_file
    icl_path = generation_dir / pair.icl_file

    base_df = pd.read_csv(base_path, encoding=CSV_ENCODING)
    icl_df = pd.read_csv(icl_path, encoding=CSV_ENCODING)
    _require_columns(base_df, base_path)
    _require_columns(icl_df, icl_path)

    base_df_tagged = base_df.copy()
    icl_df_tagged = icl_df.copy()
    base_df_tagged["_source"] = "base"
    icl_df_tagged["_source"] = "icl"
    combined = pd.concat([base_df_tagged, icl_df_tagged], ignore_index=True)

    definitions = combined["slang_definition"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        definitions,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    cosine = util.cos_sim(embeddings, embeddings).cpu().numpy()

    fallback_rows = 0
    for i in range(len(combined)):
        rng = _deterministic_rng(seed=seed, language=pair.language, row_index=i)
        sampled_indices, used_fallback = _sample_distractor_indices(
            cosine_row=cosine[i],
            self_index=i,
            threshold=threshold,
            rng=rng,
        )
        if used_fallback:
            fallback_rows += 1

        for slot, sampled_idx in enumerate(sampled_indices, start=1):
            combined.at[i, f"distractor_{slot}"] = combined.at[sampled_idx, "slang"]
            combined.at[i, f"distractor_{slot}_slang_def"] = combined.at[
                sampled_idx, "slang_definition"
            ]

    base_len = len(base_df)
    base_out = combined.iloc[:base_len].drop(columns=["_source"])
    icl_out = combined.iloc[base_len:].drop(columns=["_source"])

    out_base_path = generation_dir / f"{PREFIX}{pair.base_file}"
    out_icl_path = generation_dir / f"{PREFIX}{pair.icl_file}"
    base_out.to_csv(out_base_path, index=False, encoding=CSV_ENCODING)
    icl_out.to_csv(out_icl_path, index=False, encoding=CSV_ENCODING)

    print(
        f"[{pair.language}] done: {pair.base_file}, {pair.icl_file} "
        f"-> fallback used on {fallback_rows}/{len(combined)} rows"
    )


def main() -> None:
    args = parse_args()
    generation_dir = args.generation_dir.resolve()
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation directory not found: {generation_dir}")

    model = SentenceTransformer(args.model_name)
    for pair in DATASET_PAIRS:
        process_pair(
            pair=pair,
            model=model,
            generation_dir=generation_dir,
            threshold=args.threshold,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    print("All files generated successfully.")


if __name__ == "__main__":
    main()
