#!/usr/bin/env python3
"""
Run GPT generation-style multiple-choice evaluation for slang definitions.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from run_gpt_detection import (  # type: ignore
    load_api_key_from_file,
    parse_json_response,
    sanitize_model_name,
)


PROMPT_LANG_CHOICES = ("en", "zh")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Generation"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"

OPENAI_API_KEY = ""

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_generation_OD.csv",
    "ru": DATA_DIR / "ru_generation.csv",
    "zh": DATA_DIR / "zh_generation.csv",
}

DEF_SYSTEM_PROMPTS = {
    "en": (
        "You are a sociolinguist who answers multiple-choice questions about slang definitions. "
        "IMPORTANT: ONLY RESPOND WITH COMPACT JSON FOLLOWING THIS SCHEMA: "
        '{"choice":"A|B|C|D","rationale":"string"}. '
        "Choose the option that matches the definition and keep the rationale short."
    ),
    "zh": (
        "你是一名社会语言学家，需要回答关于俚语定义的多选题。"
        "只能使用如下 JSON 模板作答："
        '{"choice":"A|B|C|D","rationale":"字符串"}。'
        "请选择最符合定义的选项，理由保持简洁，不要添加其他文字。"
    ),
}

CONTEXT_SYSTEM_PROMPTS = {
    "en": (
        "You are a sociolinguist who answers multiple-choice questions about slang definitions "
        "using both a definition and an example sentence. Only respond with compact JSON following "
        "this schema: "
        '{"choice":"A|B|C|D","rationale":"string"}. '
        "Choose the option that best fits the provided context."
    ),
    "zh": (
        "你是一名社会语言学家，需要同时依据俚语定义和示例句回答多选题。"
        "只能使用如下 JSON 模板作答："
        '{"choice":"A|B|C|D","rationale":"字符串"}。'
        "请选择最符合给定语境的选项。"
    ),
}


class GPTDetector:
    def __init__(
        self,
        model_id: str,
        api_key_file: Path,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        api_key = OPENAI_API_KEY.strip() or load_api_key_from_file(api_key_file)
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError("OpenAI API call failed after 3 attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT slang generation evaluation.")
    parser.add_argument(
        "--mode",
        choices=["multi", "mono", "multi_legacy"],
        default="mono",
        help=(
            "multi: language-specific conv_def + sentence_with_blank, options are slang_definition; "
            "mono: conv_def_en + sentence_with_blank_en (en uses conv_def/sentence_with_blank), options are slang_definition_en; "
            "multi_legacy: original pipeline using slang_definition + sentence_with_blank with slang/distractor options."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="gpt-4o-mini",
        help="OpenAI model ID to query.",
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
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature; set 0 for deterministic decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for shuffling answer options.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (sample a few rows per language).",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=5,
        help="Rows per language when --debug is set.",
    )
    parser.add_argument(
        "--prompt-language",
        choices=PROMPT_LANG_CHOICES,
        default="en",
        help="Language used for system/user prompts.",
    )
    parser.add_argument(
        "--icl-shots",
        type=int,
        default=0,
        help="Number of ICL shots inserted into the prompt.",
    )
    parser.add_argument(
        "--chime",
        action="store_true",
        help="Only run zh on chime dataset (zh_generation_chime.csv, zh_generation_chime_icl.csv); output suffix _chime.",
    )
    parser.add_argument(
        "--od",
        action="store_true",
        help="Run OD-only English evaluation (lang=en only) using OD English datasets; output suffix _OD_only.",
    )
    return parser.parse_args()


def deterministic_rng(seed: int, language: str, row_idx: int) -> random.Random:
    key = f"{seed}-{language}-{row_idx}"
    rng = random.Random()
    rng.seed(key)
    return rng


def prepare_options(
    record: pd.Series,
    rng: random.Random,
    mode: str,
    language: str,
) -> Tuple[Dict[str, str], str]:
    def pick(field: str) -> str:
        return str(record.get(field, "") or "").strip()

    if mode == "multi_legacy":
        candidates = [
            {"text": pick("slang"), "is_correct": True},
            {"text": pick("distractor_1"), "is_correct": False},
            {"text": pick("distractor_2"), "is_correct": False},
            {"text": pick("distractor_3"), "is_correct": False},
        ]
    elif mode == "mono":
        gold = pick("slang_definition" if language == "en" else "slang_definition_en")
        d1 = pick(
            "distractor_1_slang_def" if language == "en" else "distractor_1_slang_def_en"
        )
        d2 = pick(
            "distractor_2_slang_def" if language == "en" else "distractor_2_slang_def_en"
        )
        d3 = pick(
            "distractor_3_slang_def" if language == "en" else "distractor_3_slang_def_en"
        )
        candidates = [
            {"text": gold, "is_correct": True},
            {"text": d1, "is_correct": False},
            {"text": d2, "is_correct": False},
            {"text": d3, "is_correct": False},
        ]
    else:  # multi
        gold = pick("slang_definition")
        d1 = pick("distractor_1_slang_def")
        d2 = pick("distractor_2_slang_def")
        d3 = pick("distractor_3_slang_def")
        candidates = [
            {"text": gold, "is_correct": True},
            {"text": d1, "is_correct": False},
            {"text": d2, "is_correct": False},
            {"text": d3, "is_correct": False},
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


def context_prompt(
    definition: str,
    sentence: str,
    options: Dict[str, str],
    language: str,
    prompt_language: str,
    mode: str,
    icl_text: str,
) -> str:
    choosing_definition = mode in {"mono", "multi"}

    if prompt_language == "zh":
        return (
            f"{'请根据常规定义和示例句来推断该俚语的含义。' if choosing_definition else '请选择既符合定义又能填入示例句的俚语'} "
            "只能按照前述 JSON 模板回答。\n"
            f"{icl_text}"
            f"定义: {definition}\n"
            f"句子: {sentence}\n"
            "选项:\n"
            f"{format_options(options)}"
        )
    return (
        f"{'Given the conventional definition and how the slang is used (sentence), infer the slang meaning.' if choosing_definition else 'Select the slang term that fits the definition and completes the example sentence.'} "
        "Respond only with the JSON schema provided earlier.\n"
        f"{icl_text}"
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
    prompt_text: Optional[str] = None,
    *,
    conv_def_input: str,
    sentence_input: str,
    gold_definition: str,
    icl_shots: int,
    mode: str,
) -> Dict[str, Any]:
    model_choice = None
    rationale = None
    if parsed:
        choice_value = parsed.get("choice")
        if isinstance(choice_value, str):
            model_choice = choice_value.strip().upper()
        rationale = parsed.get("rationale")

    row = {
        "row_id": idx,
        "slang": record.get("slang", ""),
        "slang_definition": gold_definition,
        "conv_def_input": conv_def_input,
        "sentence_with_blank_input": sentence_input,
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
        "icl_shots": icl_shots,
        "mode": mode,
    }
    if prompt_text is not None:
        row["prompt_text"] = prompt_text
    return row


def has_valid_blank(record: pd.Series, mode: str, language: str) -> bool:
    if mode == "mono":
        sentence = str(
            record.get("sentence_with_blank", "")
            if language == "en"
            else record.get("sentence_with_blank_en", "")
        )
    else:
        sentence = str(record.get("sentence_with_blank", ""))
    return "<BLANK>" in sentence if sentence else False


def get_conv_and_sentence(record: pd.Series, mode: str, language: str) -> Tuple[str, str]:
    if mode == "mono":
        conv_def = str(
            record.get("conv_def", "") if language == "en" else record.get("conv_def_en", "")
        ).strip()
        sentence = str(
            record.get("sentence_with_blank", "")
            if language == "en"
            else record.get("sentence_with_blank_en", "")
        ).strip()
    elif mode == "multi":
        conv_def = str(record.get("conv_def", "")).strip()
        sentence = str(record.get("sentence_with_blank", "")).strip()
    else:  # multi_legacy
        conv_def = str(record.get("slang_definition", "")).strip()
        sentence = str(record.get("sentence_with_blank", "")).strip()
    return conv_def, sentence


def load_icl_examples(
    language: str,
    mode: str,
    prompt_language: str,
    shots: int,
    seed: int,
    chime: bool = False,
) -> List[str]:
    if shots <= 0:
        return []

    if chime and language == "zh":
        icl_path = DATA_DIR / "zh_generation_chime_icl.csv"
    elif language == "en":
        icl_path = DATA_DIR / "en_generation_OD_icl.csv"
    else:
        icl_path = DATA_DIR / f"{language}_generation_icl.csv"

    if not icl_path.exists():
        return []

    df = pd.read_csv(icl_path)
    df = df[df.apply(lambda r: has_valid_blank(r, mode, language), axis=1)]
    if df.empty:
        return []

    sample_n = min(shots, len(df))
    df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    examples: List[str] = []
    for idx, record in df.iterrows():
        rng = deterministic_rng(seed, language, idx)
        try:
            options, correct_label = prepare_options(record, rng, mode, language)
            conv_def, sentence = get_conv_and_sentence(record, mode, language)
            definition = conv_def if mode in {"mono", "multi"} else str(record.get("slang_definition", "")).strip()
            block = (
                f"Example {idx + 1}:\n"
                f"Definition: {definition}\n"
                f"Sentence: {sentence}\n"
                "Options:\n"
                f"{format_options(options)}\n"
                f"Answer: {correct_label}\n"
            )
            examples.append(block)
        except Exception:
            continue

    if not examples:
        return []

    joiner = "\n" if prompt_language == "en" else "\n"
    return [joiner.join(examples) + "\n\n"]


def load_dataset(path: Path, debug: bool, debug_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if debug:
        sample_n = min(debug_samples, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


def run_context_task(
    df: pd.DataFrame,
    detector: GPTDetector,
    args: argparse.Namespace,
    language: str,
    store_prompt: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    icl_blocks = load_icl_examples(
        language=language,
        mode=args.mode,
        prompt_language=args.prompt_language,
        shots=args.icl_shots,
        seed=args.seed,
        chime=getattr(args, "chime", False),
    )
    icl_text = "".join(icl_blocks)

    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Context-{language}", leave=False)
    for idx, record in iterator:
        prompt_payload = None

        if args.mode == "mono":
            conv_def = str(
                record.get("conv_def", "") if language == "en" else record.get("conv_def_en", "")
            ).strip()
            sentence = str(
                record.get("sentence_with_blank", "")
                if language == "en"
                else record.get("sentence_with_blank_en", "")
            ).strip()
            gold_definition = str(
                record.get("slang_definition", "")
                if language == "en"
                else record.get("slang_definition_en", "")
            ).strip()
            definition = conv_def
        elif args.mode == "multi":
            definition = str(record.get("conv_def", "")).strip()
            sentence = str(record.get("sentence_with_blank", "")).strip()
            gold_definition = str(record.get("slang_definition", "")).strip()
        else:  # multi_legacy
            definition = str(record.get("slang_definition", "")).strip()
            sentence = str(record.get("sentence_with_blank", "")).strip()
            gold_definition = definition

        if not definition or not sentence:
            continue

        rng = deterministic_rng(args.seed, language, idx)
        try:
            options, correct_label = prepare_options(record, rng, args.mode, language)
            system_prompt = CONTEXT_SYSTEM_PROMPTS[args.prompt_language]
            user_prompt = context_prompt(
                definition,
                sentence,
                options,
                language,
                args.prompt_language,
                args.mode,
                icl_text,
            )
            if store_prompt:
                prompt_payload = f"[SYSTEM]\n{system_prompt}\n[USER]\n{user_prompt}"

            raw = detector.generate(system_prompt=system_prompt, user_prompt=user_prompt)
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
                    prompt_payload,
                    conv_def_input=definition,
                    sentence_input=sentence,
                    gold_definition=gold_definition,
                    icl_shots=args.icl_shots,
                    mode=args.mode,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            rows.append(
                {
                    "row_id": idx,
                    "slang": record.get("slang", ""),
                    "slang_definition": gold_definition if "gold_definition" in locals() else "",
                    "conv_def_input": definition,
                    "sentence_with_blank_input": sentence,
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
                    "icl_shots": args.icl_shots,
                    "mode": args.mode,
                    **({"prompt_text": prompt_payload} if store_prompt else {}),
                }
            )

    return pd.DataFrame(rows)


def save_results(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    mode: str,
    icl_shots: int,
    chime: bool = False,
    od_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    suffix = "_debug" if df.attrs.get("debug_mode") else ""
    chime_suffix = "_chime" if chime else ""
    od_only_suffix = "_OD_only" if od_only else ""
    shots_suffix = f"_{icl_shots}shots" if icl_shots else "_0shots"
    output_path = output_dir / f"{safe_model}_gpt_generation_{mode}{shots_suffix}{chime_suffix}{od_only_suffix}{suffix}.csv"
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def main() -> None:
    args = parse_args()
    detector = GPTDetector(
        model_id=args.model_id,
        api_key_file=args.api_key_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    lang_to_dataset = dict(LANG_TO_DATASET)
    if args.od:
        lang_to_dataset = {"en": LANG_TO_DATASET["en"]}
    elif args.chime:
        lang_to_dataset["zh"] = DATA_DIR / "zh_generation_chime.csv"

    frames: List[pd.DataFrame] = []
    for language, dataset_path in lang_to_dataset.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path, debug=args.debug, debug_samples=args.debug_samples)
        if args.debug:
            df.attrs["debug_mode"] = True
        frames.append(run_context_task(df, detector, args, language, args.debug))

    if not frames:
        raise RuntimeError("No generation results were produced.")

    result_df = pd.concat(frames, ignore_index=True)
    if args.debug:
        result_df.attrs["debug_mode"] = True

    output_path = save_results(
        result_df,
        args.output_dir,
        args.model_id,
        args.mode,
        args.icl_shots,
        chime=args.chime,
        od_only=args.od,
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
