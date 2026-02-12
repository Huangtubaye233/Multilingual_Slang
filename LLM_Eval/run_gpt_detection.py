#!/usr/bin/env python3
"""
Run GPT detection inference for slang detection datasets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Detection"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_detection.csv",
    "ru": DATA_DIR / "ru_detection.csv",
    "zh": DATA_DIR / "zh_detection.csv",
}

SENTENCE_SYSTEM_PROMPT = (
    "You are a sociolinguist who determines whether sentences contain slang. "
    "Only respond with compact JSON following this schema: "
    '{"contains_slang": "yes" or "no", "rationale":"string"}. '
    "Use lowercase yes/no and provide a short rationale. Do not add extra text."
)

WORD_SYSTEM_PROMPT = (
    "You are a sociolinguist who extracts slang terms from sentences. "
    "Only respond with compact JSON following this schema: "
    '{"slang_terms": "slang_term" or "N/A", "rationale":"string"}. '
    "Return N/A when no slang is present. Do not add extra text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-based slang detection.")
    parser.add_argument(
        "--task",
        choices=["sentence", "word", "both"],
        default="both",
        help="Which detection task(s) to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to query.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to store CSV outputs.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=BASE_DIR / "openai_api_key.txt",
        help="Path to a text file that stores the OpenAI API key.",
    )
    return parser.parse_args()


def load_api_key_from_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file {path} is empty.")
    return api_key


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_model_json(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                return candidate
        return content
    return content


def parse_json_response(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    cleaned = clean_model_json(raw)
    try:
        parsed = json.loads(cleaned)
        return parsed, json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return None, cleaned


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            sleep_for = 2 ** attempt
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenAI API call failed after 3 attempts") from last_error


def sentence_prompt(sentence: str, language: str) -> str:
    return (
        "Determine whether the following sentence contains any slang expressions. "
        "Answer only with the requested JSON schema.\n"
        f"Language: {language}\n"
        f"Sentence: {sentence}"
    )


def word_prompt(sentence: str, language: str) -> str:
    return (
        "Extract the slang expression(s) that appears in the sentence. "
        "Return N/A if none occur.\n"
        f"Language: {language}\n"
        f"Sentence: {sentence}"
    )


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def run_sentence_task(
    df: pd.DataFrame,
    client: OpenAI,
    args: argparse.Namespace,
    language: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Sentence-{language}",
        leave=False,
    )
    for idx, record in iterator:
        sentence_value = record.get("sentence", "")
        sentence = str(sentence_value).strip()
        if not sentence:
            continue
        try:
            raw = call_openai_with_retry(
                client=client,
                model=args.model,
                system_prompt=SENTENCE_SYSTEM_PROMPT,
                user_prompt=sentence_prompt(sentence, language),
            )
            parsed, structured = parse_json_response(raw)
            rows.append(
                {
                    "row_id": idx,
                    "word_to_focus_on": record.get("word_to_focus_on", ""),
                    "sentence": sentence,
                    "language": language,
                    "model_response_json": structured,
                    "contains_slang": parsed.get("contains_slang") if parsed else None,
                    "raw_response": raw,
                    "json_valid": parsed is not None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "row_id": idx,
                    "word_to_focus_on": record.get("word_to_focus_on", ""),
                    "sentence": sentence,
                    "language": language,
                    "model_response_json": "",
                    "contains_slang": None,
                    "raw_response": "",
                    "json_valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def run_word_task(
    df: pd.DataFrame,
    client: OpenAI,
    args: argparse.Namespace,
    language: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Word-{language}",
        leave=False,
    )
    for idx, record in iterator:
        sentence_value = record.get("sentence", "")
        sentence = str(sentence_value).strip()
        if not sentence:
            continue
        try:
            raw = call_openai_with_retry(
                client=client,
                model=args.model,
                system_prompt=WORD_SYSTEM_PROMPT,
                user_prompt=word_prompt(sentence, language),
            )
            parsed, structured = parse_json_response(raw)
            slang_terms = parsed.get("slang_terms") if parsed else None
            rows.append(
                {
                    "row_id": idx,
                    "word_to_focus_on": record.get("word_to_focus_on", ""),
                    "sentence": sentence,
                    "language": language,
                    "model_response_json": structured,
                    "slang_terms": slang_terms if isinstance(slang_terms, str) else "",
                    "raw_response": raw,
                    "json_valid": parsed is not None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "row_id": idx,
                    "word_to_focus_on": record.get("word_to_focus_on", ""),
                    "sentence": sentence,
                    "language": language,
                    "model_response_json": "",
                    "slang_terms": "",
                    "raw_response": "",
                    "json_valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, output_dir: Path, model_name: str, task_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    output_path = output_dir / f"{safe_model}_detection_{task_name}.csv"
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def main() -> None:
    args = parse_args()
    api_key = load_api_key_from_file(args.api_key_file)
    client = OpenAI(api_key=api_key)

    tasks_to_run = ["sentence", "word"] if args.task == "both" else [args.task]
    task_frames: Dict[str, List[pd.DataFrame]] = {task: [] for task in tasks_to_run}

    for language, dataset_path in LANG_TO_DATASET.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path)
        if "sentence" in tasks_to_run:
            task_frames.setdefault("sentence", []).append(
                run_sentence_task(df, client, args, language)
            )
        if "word" in tasks_to_run:
            task_frames.setdefault("word", []).append(
                run_word_task(df, client, args, language)
            )

    for task_name in tasks_to_run:
        frames = task_frames.get(task_name, [])
        if not frames:
            continue
        result_df = pd.concat(frames, ignore_index=True)
        output_path = save_results(result_df, args.output_dir, args.model, task_name)
        print(f"Saved {task_name} results to {output_path}")


if __name__ == "__main__":
    main()

