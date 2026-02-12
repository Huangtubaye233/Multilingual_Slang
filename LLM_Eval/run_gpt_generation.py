#!/usr/bin/env python3
"""
Run GPT generation-style multiple-choice evaluation for slang definitions.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Generation"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_generation.csv",
    "ru": DATA_DIR / "ru_generation.csv",
    "zh": DATA_DIR / "zh_generation.csv",
}

TASK_CHOICES = ["def_only", "context", "both"]

DEF_SYSTEM_PROMPT = (
    "You are a sociolinguist who answers multiple-choice questions about slang definitions. "
    "Only respond with compact JSON following this schema: "
    '{"choice":"A|B|C|D","rationale":"string"}. '
    "Choose the option that matches the definition and keep the rationale short."
)

CONTEXT_SYSTEM_PROMPT = (
    "You are a sociolinguist who answers multiple-choice questions about slang definitions "
    "using both a definition and an example sentence. Only respond with compact JSON following "
    "this schema: "
    '{"choice":"A|B|C|D","rationale":"string"}. '
    "Choose the option that best fits the provided context."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-based slang generation evaluation.")
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="both",
        help="Which generation task(s) to run.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for shuffling answer options.",
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


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def deterministic_rng(seed: int, language: str, row_idx: int) -> random.Random:
    key = f"{seed}-{language}-{row_idx}"
    rng = random.Random()
    rng.seed(key)
    return rng


def prepare_options(record: pd.Series, rng: random.Random) -> Tuple[Dict[str, str], str]:
    candidates = [
        {"text": str(record.get("slang", "")).strip(), "is_correct": True},
        {"text": str(record.get("distractor_1", "")).strip(), "is_correct": False},
        {"text": str(record.get("distractor_2", "")).strip(), "is_correct": False},
        {"text": str(record.get("distractor_3", "")).strip(), "is_correct": False},
    ]
    if any(not c["text"] for c in candidates):
        raise ValueError("Missing option text in record; cannot build options.")
    rng.shuffle(candidates)
    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    correct_label: Optional[str] = None
    for label, candidate in zip(labels, candidates):
        options[label] = candidate["text"]
        if candidate["is_correct"]:
            correct_label = label
    if not correct_label:
        raise RuntimeError("Failed to locate correct option after shuffling.")
    return options, correct_label


def format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def def_only_prompt(definition: str, options: Dict[str, str], language: str) -> str:
    return (
        "Select the slang term that matches the definition. "
        "Respond only with the JSON schema provided earlier.\n"
        f"Language: {language}\n"
        f"Definition: {definition}\n"
        "Options:\n"
        f"{format_options(options)}"
    )


def context_prompt(
    definition: str,
    sentence: str,
    options: Dict[str, str],
    language: str,
) -> str:
    return (
        "Select the slang term that fits the definition and completes the example sentence. "
        "Respond only with the JSON schema provided earlier.\n"
        f"Language: {language}\n"
        f"Definition: {definition}\n"
        f"Sentence: {sentence}\n"
        "Options:\n"
        f"{format_options(options)}"
    )


def build_result_row(
    idx: int,
    language: str,
    record: pd.Series,
    options: Dict[str, str],
    correct_label: str,
    parsed: Optional[Dict[str, Any]],
    structured_json: str,
    raw_response: str,
) -> Dict[str, Any]:
    model_choice = None
    rationale = None
    if parsed:
        choice_value = parsed.get("choice")
        if isinstance(choice_value, str):
            model_choice = choice_value.strip().upper()
        rationale = parsed.get("rationale")
    return {
        "row_id": idx,
        "slang": record.get("slang", ""),
        "slang_definition": record.get("slang_definition", ""),
        "sentence_with_blank": record.get("sentence_with_blank", ""),
        "language": language,
        "option_a": options.get("A"),
        "option_b": options.get("B"),
        "option_c": options.get("C"),
        "option_d": options.get("D"),
        "correct_option": correct_label,
        "model_choice": model_choice,
        "rationale": rationale,
        "is_correct": (model_choice == correct_label) if model_choice else None,
        "model_response_json": structured_json,
        "raw_response": raw_response,
        "json_valid": parsed is not None,
    }


def run_def_only_task(
    df: pd.DataFrame,
    client: OpenAI,
    args: argparse.Namespace,
    language: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"DefOnly-{language}",
        leave=False,
    )
    for idx, record in iterator:
        definition = str(record.get("slang_definition", "")).strip()
        if not definition:
            continue
        rng = deterministic_rng(args.seed, language, idx)
        try:
            options, correct_label = prepare_options(record, rng)
            raw = call_openai_with_retry(
                client=client,
                model=args.model,
                system_prompt=DEF_SYSTEM_PROMPT,
                user_prompt=def_only_prompt(definition, options, language),
            )
            parsed, structured = parse_json_response(raw)
            rows.append(
                build_result_row(
                    idx,
                    language,
                    record,
                    options,
                    correct_label,
                    parsed,
                    structured,
                    raw,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "row_id": idx,
                    "slang": record.get("slang", ""),
                    "slang_definition": definition,
                    "sentence_with_blank": record.get("sentence_with_blank", ""),
                    "language": language,
                    "option_a": None,
                    "option_b": None,
                    "option_c": None,
                    "option_d": None,
                    "correct_option": None,
                    "model_choice": None,
                    "rationale": None,
                    "is_correct": None,
                    "model_response_json": "",
                    "raw_response": "",
                    "json_valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def run_context_task(
    df: pd.DataFrame,
    client: OpenAI,
    args: argparse.Namespace,
    language: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Context-{language}",
        leave=False,
    )
    for idx, record in iterator:
        definition = str(record.get("slang_definition", "")).strip()
        sentence = str(record.get("sentence_with_blank", "")).strip()
        if not definition or not sentence:
            continue
        rng = deterministic_rng(args.seed, language, idx)
        try:
            options, correct_label = prepare_options(record, rng)
            raw = call_openai_with_retry(
                client=client,
                model=args.model,
                system_prompt=CONTEXT_SYSTEM_PROMPT,
                user_prompt=context_prompt(definition, sentence, options, language),
            )
            parsed, structured = parse_json_response(raw)
            rows.append(
                build_result_row(
                    idx,
                    language,
                    record,
                    options,
                    correct_label,
                    parsed,
                    structured,
                    raw,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "row_id": idx,
                    "slang": record.get("slang", ""),
                    "slang_definition": definition,
                    "sentence_with_blank": sentence,
                    "language": language,
                    "option_a": None,
                    "option_b": None,
                    "option_c": None,
                    "option_d": None,
                    "correct_option": None,
                    "model_choice": None,
                    "rationale": None,
                    "is_correct": None,
                    "model_response_json": "",
                    "raw_response": "",
                    "json_valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, output_dir: Path, model_name: str, task_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    output_path = output_dir / f"{safe_model}_generation_{task_name}.csv"
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def main() -> None:
    args = parse_args()
    api_key = load_api_key_from_file(args.api_key_file)
    client = OpenAI(api_key=api_key)

    tasks_to_run = ["def_only", "context"] if args.task == "both" else [args.task]
    task_frames: Dict[str, List[pd.DataFrame]] = {task: [] for task in tasks_to_run}

    for language, dataset_path in LANG_TO_DATASET.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path)
        if "def_only" in tasks_to_run:
            task_frames.setdefault("def_only", []).append(
                run_def_only_task(df, client, args, language)
            )
        if "context" in tasks_to_run:
            task_frames.setdefault("context", []).append(
                run_context_task(df, client, args, language)
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

