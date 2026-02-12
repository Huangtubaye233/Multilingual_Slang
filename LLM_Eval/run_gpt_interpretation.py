#!/usr/bin/env python3
"""
Run GPT interpretation inference to elicit slang definitions.
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
DATA_DIR = BASE_DIR / "Interpretation"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_interpretation.csv",
    "ru": DATA_DIR / "ru_interpretation.csv",
    "zh": DATA_DIR / "zh_interpretation.csv",
}

INTERPRET_SYSTEM_PROMPT = (
    "You are a sociolinguist who explains slang. "
    "Only respond with compact JSON following this schema: "
    '{"definition":"string","rationale":"string"}. '
    "Keep the definition short, cite context in the rationale, "
    "and do not add extra text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-based slang interpretation.")
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
    raise RuntimeError("OpenAI API call failed after 3 attempts") from last_error


def interpretation_prompt(slang: str, example_sentence: str, language: str) -> str:
    return (
        "Provide the definition of the given slang term, using the example sentence for context. "
        "Return only the requested JSON schema.\n"
        f"Language: {language}\n"
        f"Slang: {slang}\n"
        f"Sentence: {example_sentence}"
    )


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def run_interpretation_task(
    df: pd.DataFrame,
    client: OpenAI,
    args: argparse.Namespace,
    language: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Interpret-{language}",
        leave=False,
    )
    for idx, record in iterator:
        slang_value = record.get("slang", "")
        sentence_value = record.get("example_sentence", "")
        slang = str(slang_value).strip()
        sentence = str(sentence_value).strip()
        if not slang or not sentence:
            continue
        try:
            raw = call_openai_with_retry(
                client=client,
                model=args.model,
                system_prompt=INTERPRET_SYSTEM_PROMPT,
                user_prompt=interpretation_prompt(slang, sentence, language),
            )
            parsed, structured = parse_json_response(raw)
            rows.append(
                {
                    "row_id": idx,
                    "slang": slang,
                    "example_sentence": sentence,
                    "language": language,
                    "gold_definition": record.get("slang_definition", ""),
                    "model_response_json": structured,
                    "predicted_definition": parsed.get("definition") if parsed else None,
                    "confidence": parsed.get("confidence") if parsed else None,
                    "rationale": parsed.get("rationale") if parsed else None,
                    "raw_response": raw,
                    "json_valid": parsed is not None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "row_id": idx,
                    "slang": slang,
                    "example_sentence": sentence,
                    "language": language,
                    "gold_definition": record.get("slang_definition", ""),
                    "model_response_json": "",
                    "predicted_definition": None,
                    "confidence": None,
                    "rationale": None,
                    "raw_response": "",
                    "json_valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    output_path = output_dir / f"{safe_model}_interpretation.csv"
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def main() -> None:
    args = parse_args()
    api_key = load_api_key_from_file(args.api_key_file)
    client = OpenAI(api_key=api_key)

    frames: List[pd.DataFrame] = []
    for language, dataset_path in LANG_TO_DATASET.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path)
        frames.append(run_interpretation_task(df, client, args, language))

    if not frames:
        raise RuntimeError("No interpretation results were produced.")

    result_df = pd.concat(frames, ignore_index=True)
    output_path = save_results(result_df, args.output_dir, args.model)
    print(f"Saved interpretation results to {output_path}")


if __name__ == "__main__":
    main()

