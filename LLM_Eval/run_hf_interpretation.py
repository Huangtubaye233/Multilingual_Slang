#!/usr/bin/env python3
"""
Run HF/vLLM interpretation inference to elicit slang definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from run_hf_detection import (  # type: ignore
    HFDetector,
    VLLMDetector,
    ensure_cache_dir,
    parse_json_response,
    resolve_dtype,
    resolve_vllm_dtype,
    sanitize_model_name,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Interpretation"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"
DEFAULT_CACHE_DIR = Path("/projects/b1170/users/kyx8046/hf-cache")

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_interpretation.csv",
    "ru": DATA_DIR / "ru_interpretation.csv",
    "zh": DATA_DIR / "zh_interpretation.csv",
}

PROMPT_LANG_CHOICES = ("en", "zh")

INTERPRET_SYSTEM_PROMPTS = {
    "en": (
        "You are a sociolinguist who explains slang. "
        "Only respond with compact JSON following this schema: "
        '{"definition":"string","rationale":"string"}. '
        "Keep the definition short, cite context in the rationale, and do not add extra text."
    ),
    "zh": (
        "你是一名社会语言学家，负责解释俚语。"
        "只能使用如下 JSON 模板作答："
        '{"definition":"字符串","rationale":"字符串"}。'
        "请给出简短定义，并在理由中引用语境，不要添加额外文字。"
    ),
}

# Placeholder for future ICL examples. When populated, prepend to the user prompt.
ICL_PLACEHOLDER = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF/vLLM slang interpretation.")
    parser.add_argument(
        "--model-id",
        default="allenai/Olmo-3-7B-Think",
        help="Hugging Face model ID to load.",
    )
    parser.add_argument(
        "--engine",
        choices=["hf", "vllm"],
        default="hf",
        help="Generation backend to use.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for Hugging Face model cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to store CSV outputs.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map for HF model loading (e.g., "auto", "cuda", "cpu").',
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch/vLLM dtype for model weights.",
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
        help="Sampling temperature; set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size when using vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization ratio (0-1).",
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
        "--mode",
        choices=["multi", "mono", "multi_legacy"],
        default="mono",
        help=(
            "multi: use language-specific conv_def + sentence_with_blank with gold slang_definition; "
            "mono: use conv_def_en + sentence_with_blank_en (en uses conv_def/sentence_with_blank) with gold slang_definition_en; "
            "multi_legacy: original slang + example_sentence prompt with gold slang_definition."
        ),
    )
    parser.add_argument(
        "--icl-shots",
        type=int,
        default=0,
        help="Number of ICL shots inserted into the prompt (placeholder only for now).",
    )
    return parser.parse_args()


def load_dataset(path: Path, debug: bool, debug_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if debug:
        sample_n = min(debug_samples, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


def has_valid_blank(record: pd.Series, mode: str, language: str) -> bool:
    if mode == "multi_legacy":
        return True
    if mode == "mono":
        sentence = str(
            record.get("sentence_with_blank", "")
            if language == "en"
            else record.get("sentence_with_blank_en", "")
        )
    else:
        sentence = str(record.get("sentence_with_blank", ""))
    return "<BLANK>" in sentence if sentence else False


def get_inputs(record: pd.Series, mode: str, language: str) -> Dict[str, str]:
    if mode == "multi_legacy":
        slang = str(record.get("slang", "")).strip()
        example_sentence = str(record.get("example_sentence", "")).strip()
        gold_definition = str(record.get("slang_definition", "")).strip()
        return {
            "slang": slang,
            "example_sentence": example_sentence,
            "conv_def": "",
            "sentence_with_blank": "",
            "gold_definition": gold_definition,
        }
    if mode == "mono":
        conv_def = str(
            record.get("conv_def", "") if language == "en" else record.get("conv_def_en", "")
        ).strip()
        sentence_with_blank = str(
            record.get("sentence_with_blank", "")
            if language == "en"
            else record.get("sentence_with_blank_en", "")
        ).strip()
        gold_definition = str(
            record.get("slang_definition", "")
            if language == "en"
            else record.get("slang_definition_en", "")
        ).strip()
        return {
            "slang": str(record.get("slang", "")).strip(),
            "example_sentence": "",
            "conv_def": conv_def,
            "sentence_with_blank": sentence_with_blank,
            "gold_definition": gold_definition,
        }
    conv_def = str(record.get("conv_def", "")).strip()
    sentence_with_blank = str(record.get("sentence_with_blank", "")).strip()
    gold_definition = str(record.get("slang_definition", "")).strip()
    return {
        "slang": str(record.get("slang", "")).strip(),
        "example_sentence": "",
        "conv_def": conv_def,
        "sentence_with_blank": sentence_with_blank,
        "gold_definition": gold_definition,
    }


def load_icl_examples(
    language: str,
    mode: str,
    prompt_language: str,
    shots: int,
    seed: int,
) -> str:
    if shots <= 0:
        return ""
    icl_path = DATA_DIR / f"{language}_interpretation_icl.csv"
    if not icl_path.exists():
        return ""
    df = pd.read_csv(icl_path)
    df = df[df.apply(lambda r: has_valid_blank(r, mode, language), axis=1)]
    if df.empty:
        return ""
    sample_n = min(shots, len(df))
    df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    blocks: List[str] = []
    for idx, record in df.iterrows():
        try:
            inputs = get_inputs(record, mode, language)
            if mode == "multi_legacy":
                block = (
                    f"Example {idx + 1}:\n"
                    f"Slang: {inputs['slang']}\n"
                    f"Sentence: {inputs['example_sentence']}\n"
                    f"Answer: {inputs['gold_definition']}\n"
                )
            else:
                block = (
                    f"Example {idx + 1}:\n"
                    f"Conventional definition: {inputs['conv_def']}\n"
                    f"Sentence with blank: {inputs['sentence_with_blank']}\n"
                    f"Answer: {inputs['gold_definition']}\n"
                )
            blocks.append(block)
        except Exception:
            continue
    return ("\n".join(blocks) + "\n\n") if blocks else ""


def interpretation_prompt(
    mode: str,
    language: str,
    prompt_language: str,
    *,
    slang: str,
    example_sentence: str,
    conv_def: str,
    sentence_with_blank: str,
    icl_text: str,
) -> str:
    icl_block = ICL_PLACEHOLDER.strip()
    icl_prefix = f"{icl_block}\n\n" if icl_block else ""
    icl_section = f"{icl_prefix}{icl_text}\n" if icl_text else icl_prefix

    if mode == "multi_legacy":
        if prompt_language == "zh":
            return (
                "请根据示例句解释该俚语，只能使用前述 JSON 模板作答。\n"
                f"{icl_section}"
                f"俚语: {slang}\n"
                f"句子: {example_sentence}"
            )
        return (
            "Provide the definition of the given slang term, using the example sentence for context. "
            "Return only the requested JSON schema.\n"
            f"{icl_section}"
            f"Slang: {slang}\n"
            f"Sentence: {example_sentence}"
        )

    # mono / multi share the same structure but differ in which columns feed conv_def/sentence_with_blank
    if prompt_language == "zh":
        return (
            "根据常规定义和含空白的句子，推断对应俚语的定义，常规定义仅作补充信息。只能使用前述 JSON 模板作答。\n"
            f"{icl_section}"
            f"常规定义: {conv_def}\n"
            f"句子(含空白): {sentence_with_blank}"
        )
    return (
        "Given the conventional definition (as supporting info) and the sentence with a blank, provide the slang definition. "
        "Return only the requested JSON schema.\n"
        f"{icl_section}"
        f"Conventional definition: {conv_def}\n"
        f"Sentence with blank: {sentence_with_blank}"
    )


def run_interpretation_task(
    df: pd.DataFrame,
    detector: Any,
    language: str,
    prompt_language: str,
    store_prompt: bool,
    mode: str,
    icl_shots: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    icl_text = load_icl_examples(
        language=language,
        mode=mode,
        prompt_language=prompt_language,
        shots=icl_shots,
        seed=42,
    )
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Interpret-{language}",
        leave=False,
    )
    for idx, record in iterator:
        slang = str(record.get("slang", "") or "").strip()
        example_sentence = str(record.get("example_sentence", "") or "").strip()
        conv_def = ""
        sentence_with_blank = ""
        gold_definition = ""

        if mode == "multi_legacy":
            if not slang or not example_sentence:
                continue
            gold_definition = str(record.get("slang_definition", "") or "")
        elif mode == "mono":
            conv_def = (
                str(
                    record.get("conv_def", "")
                    if language == "en"
                    else record.get("conv_def_en", "")
                )
                or ""
            ).strip()
            sentence_with_blank = (
                str(
                    record.get("sentence_with_blank", "")
                    if language == "en"
                    else record.get("sentence_with_blank_en", "")
                )
                or ""
            ).strip()
            if not conv_def or not sentence_with_blank:
                continue
            gold_definition = str(
                record.get(
                    "slang_definition",
                    record.get("slang_definition_en", ""),
                )
                or ""
            )
        else:  # multi
            conv_def = str(record.get("conv_def", "") or "").strip()
            sentence_with_blank = str(record.get("sentence_with_blank", "") or "").strip()
            if not conv_def or not sentence_with_blank:
                continue
            gold_definition = str(record.get("slang_definition", "") or "")

        system_prompt = INTERPRET_SYSTEM_PROMPTS[prompt_language]
        user_prompt = interpretation_prompt(
            mode=mode,
            language=language,
            prompt_language=prompt_language,
            slang=slang,
            example_sentence=example_sentence,
            conv_def=conv_def,
            sentence_with_blank=sentence_with_blank,
            icl_text=icl_text,
        )
        prompt_payload = (
            f"[SYSTEM]\n{system_prompt}\n[USER]\n{user_prompt}"
            if store_prompt
            else None
        )
        try:
            raw = detector.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            parsed, structured = parse_json_response(raw)
            row = {
                "row_id": idx,
                "slang": slang,
                "example_sentence": example_sentence,
                "conv_def": conv_def,
                "sentence_with_blank": sentence_with_blank,
                "language": language,
                "gold_definition": gold_definition,
                "model_response_json": structured,
                "predicted_definition": parsed.get("definition") if parsed else None,
                "rationale": parsed.get("rationale") if parsed else None,
                "raw_response": raw,
                "json_valid": parsed is not None,
                "icl_shots": icl_shots,
                "mode": mode,
            }
            if store_prompt:
                row["prompt_text"] = prompt_payload
            rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            error_row = {
                "row_id": idx,
                "slang": slang,
                "example_sentence": example_sentence,
                "conv_def": conv_def,
                "sentence_with_blank": sentence_with_blank,
                "language": language,
                "gold_definition": gold_definition,
                "model_response_json": "",
                "predicted_definition": None,
                "rationale": None,
                "raw_response": "",
                "json_valid": False,
                "error": str(exc),
                "icl_shots": icl_shots,
                "mode": mode,
            }
            if store_prompt:
                error_row["prompt_text"] = prompt_payload
            rows.append(error_row)
    return pd.DataFrame(rows)


def save_results(
    df: pd.DataFrame, output_dir: Path, model_name: str, mode: str, icl_shots: int
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    suffix = "_debug" if df.attrs.get("debug_mode") else ""
    shots_suffix = f"_{icl_shots}shots" if icl_shots else "_0shots"
    output_path = output_dir / f"{safe_model}_hf_interpretation_{mode}{shots_suffix}{suffix}.csv"
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def main() -> None:
    args = parse_args()
    ensure_cache_dir(args.hf_cache)

    if args.engine == "hf":
        torch_dtype = resolve_dtype(args.dtype)
        detector = HFDetector(
            model_id=args.model_id,
            cache_dir=args.hf_cache,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        vllm_dtype = resolve_vllm_dtype(args.dtype)
        detector = VLLMDetector(
            model_id=args.model_id,
            dtype=vllm_dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    frames: List[pd.DataFrame] = []
    for language, dataset_path in LANG_TO_DATASET.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path, debug=args.debug, debug_samples=args.debug_samples)
        if args.debug:
            df.attrs["debug_mode"] = True
        frames.append(
            run_interpretation_task(
                df,
                detector,
                language,
                args.prompt_language,
                args.debug,
                args.mode,
                args.icl_shots,
            )
        )

    if not frames:
        raise RuntimeError("No interpretation results were produced.")

    result_df = pd.concat(frames, ignore_index=True)
    if args.debug:
        result_df.attrs["debug_mode"] = True
    output_path = save_results(
        result_df, args.output_dir, args.model_id, args.mode, args.icl_shots
    )
    print(f"Saved interpretation results to {output_path}")


if __name__ == "__main__":
    main()


