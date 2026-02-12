#!/usr/bin/env python3
"""
Run Hugging Face causal LM inference for slang detection datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Detection"
RESULTS_DIR = BASE_DIR / "Results"
CSV_ENCODING = "utf-8-sig"
DEFAULT_CACHE_DIR = Path("/projects/b1170/users/kyx8046/hf-cache")
THINK_START = "<think>"
THINK_END = "</think>"

LANG_TO_DATASET = {
    "en": DATA_DIR / "en_detection.csv",
    "ru": DATA_DIR / "ru_detection.csv",
    "zh": DATA_DIR / "zh_detection.csv",
}

PROMPT_LANG_CHOICES = ("en", "zh")

SENTENCE_SYSTEM_PROMPTS = {
    "en": (
        "You are a sociolinguist who determines whether sentences contain slang. "
        "Only respond with compact JSON following this schema: "
        '{"contains_slang": "yes" or "no", "rationale":"string"}. '
        "Use lowercase yes/no and provide a short rationale. Do not add extra text."
    ),
    "zh": (
        "你是一名研究俚语的社会语言学家，负责判断句子是否包含俚语。"
        "只能使用如下 JSON 模板作答："
        '{"contains_slang": "yes" 或 "no", "rationale":"字符串" }。'
        "请使用小写 yes/no，并给出简短理由，不要添加额外文字。"
    ),
}

WORD_SYSTEM_PROMPTS = {
    "en": (
        "You are a sociolinguist who extracts slang terms from sentences. "
        "Only respond with compact JSON following this schema: "
        '{"slang_terms": "slang_term" or "N/A", "rationale":"string"}. '
        "Return N/A when no slang is present. Do not add extra text."
    ),
    "zh": (
        "你是一名研究俚语的社会语言学家，负责从句子中抽取俚语。"
        "只能使用如下 JSON 模板作答："
        '{"slang_terms": "俚语" 或 "N/A", "rationale":"字符串" }。'
        "如果句子中没有俚语，请返回 N/A，不要添加额外文字。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF-based slang detection.")
    parser.add_argument(
        "--task",
        choices=["sentence", "word", "both"],
        default="both",
        help="Which detection task(s) to run.",
    )
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
        help='Device map for model loading (e.g., "auto", "cuda", "cpu").',
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype for model weights.",
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
        "--prompt-language",
        choices=PROMPT_LANG_CHOICES,
        default="en",
        help="Language used for system/user prompts.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
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
    return parser.parse_args()


def ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(path))


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


def strip_think_blocks(text: str) -> str:
    if THINK_START not in text:
        return text
    stripped = []
    cursor = 0
    while True:
        start = text.find(THINK_START, cursor)
        if start == -1:
            stripped.append(text[cursor:])
            break
        stripped.append(text[cursor:start])
        end = text.find(THINK_END, start)
        if end == -1:
            break
        cursor = end + len(THINK_END)
    return "".join(stripped).strip()


def parse_json_response(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    stripped = strip_think_blocks(raw)
    cleaned = clean_model_json(stripped)
    try:
        parsed = json.loads(cleaned)
        return parsed, json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return None, cleaned


def sentence_prompt(sentence: str, language: str, prompt_language: str) -> str:
    if prompt_language == "zh":
        return (
            "请判断下面的句子是否包含任何俚语表达，并且只用上述 JSON 模板回答。\n"
            f"句子: {sentence}"
        )
    return (
        "Determine whether the following sentence contains any slang expressions. "
        "Answer only with the requested JSON schema.\n"
        f"Sentence: {sentence}"
    )


def word_prompt(sentence: str, language: str, prompt_language: str) -> str:
    if prompt_language == "zh":
        return (
            "请从下面的句子中抽取出现的俚语表达；若没有请返回 N/A。只使用前述 JSON 模板回答。\n"
            f"句子: {sentence}"
        )
    return (
        "Extract the slang expression(s) that appears in the sentence. "
        "Return N/A if none occur.\n"
        f"Sentence: {sentence}"
    )


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def resolve_vllm_dtype(dtype_arg: str) -> str:
    if dtype_arg in {"float32", "float16", "bfloat16"}:
        return dtype_arg
    return "auto"


@dataclass
class HFDetector:
    model_id: str
    cache_dir: Path
    device_map: str
    torch_dtype: torch.dtype
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            use_fast=False,
        )
        if self.tokenizer.pad_token_id is None:
            # OLMo style models often do not define pad tokens.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        # Prefer chat template if available (e.g., Qwen) to support options like disable thinking.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # disable thinking mode when supported
                )
            except TypeError:
                # Some tokenizers don't accept enable_thinking; fall back without it.
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return (
            f"[SYSTEM]\n{system_prompt}\n"
            f"[USER]\n{user_prompt}\n"
            "[ASSISTANT]\n"
        )

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self.build_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_k=self.top_k if do_sample else None,
            top_p=self.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()


@dataclass
class VLLMDetector:
    model_id: str
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self) -> None:
        if LLM is None or SamplingParams is None:  # pragma: no cover - runtime guard
            raise ImportError(
                "vLLM is not installed. Please `pip install vllm` in the current environment."
            )
        # Ensure vLLM always receives numeric temperature; some versions reject None.
        temperature = float(self.temperature) if self.temperature is not None else 0.0
        do_sample = temperature > 0
        # vLLM requires top_p to be in (0, 1]; avoid None when not sampling.
        top_p = float(self.top_p) if self.top_p is not None else 1.0
        top_p = top_p if do_sample else 1.0
        top_k = int(self.top_k) if self.top_k is not None else 0
        top_k = top_k if do_sample else 0
        self.sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_k=top_k,
            top_p=top_p,
            max_tokens=self.max_new_tokens,
        )
        self.llm = LLM(
            model=self.model_id,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"[SYSTEM]\n{system_prompt}\n"
            f"[USER]\n{user_prompt}\n"
            "[ASSISTANT]\n"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self.build_prompt(system_prompt, user_prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        text = outputs[0].outputs[0].text
        return text.strip()


def run_sentence_task(
    df: pd.DataFrame,
    detector: Any,
    language: str,
    prompt_language: str,
    store_prompt: bool,
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
        system_prompt = SENTENCE_SYSTEM_PROMPTS[prompt_language]
        user_prompt = sentence_prompt(sentence, language, prompt_language)
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
                "word_to_focus_on": record.get("word_to_focus_on", ""),
                "sentence": sentence,
                "language": language,
                "reference_contains_slang": record.get("contains_slang", ""),
                "model_response_json": structured,
                "contains_slang": parsed.get("contains_slang") if parsed else None,
                "raw_response": raw,
                "json_valid": parsed is not None,
            }
            if store_prompt:
                row["prompt_text"] = prompt_payload
            rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            error_row = {
                "row_id": idx,
                "word_to_focus_on": record.get("word_to_focus_on", ""),
                "sentence": sentence,
                "language": language,
                "reference_contains_slang": record.get("contains_slang", ""),
                "model_response_json": "",
                "contains_slang": None,
                "raw_response": "",
                "json_valid": False,
                "error": str(exc),
            }
            if store_prompt:
                error_row["prompt_text"] = prompt_payload
            rows.append(error_row)
    return pd.DataFrame(rows)


def run_word_task(
    df: pd.DataFrame,
    detector: Any,
    language: str,
    prompt_language: str,
    store_prompt: bool,
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
        system_prompt = WORD_SYSTEM_PROMPTS[prompt_language]
        user_prompt = word_prompt(sentence, language, prompt_language)
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
            slang_terms = parsed.get("slang_terms") if parsed else None
            row = {
                "row_id": idx,
                "word_to_focus_on": record.get("word_to_focus_on", ""),
                "sentence": sentence,
                "language": language,
                "reference_bio_tags": record.get("bio_tags", ""),
                "model_response_json": structured,
                "slang_terms": slang_terms if isinstance(slang_terms, str) else "",
                "raw_response": raw,
                "json_valid": parsed is not None,
            }
            if store_prompt:
                row["prompt_text"] = prompt_payload
            rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            error_row = {
                "row_id": idx,
                "word_to_focus_on": record.get("word_to_focus_on", ""),
                "sentence": sentence,
                "language": language,
                "reference_bio_tags": record.get("bio_tags", ""),
                "model_response_json": "",
                "slang_terms": "",
                "raw_response": "",
                "json_valid": False,
                "error": str(exc),
            }
            if store_prompt:
                error_row["prompt_text"] = prompt_payload
            rows.append(error_row)
    return pd.DataFrame(rows)


def load_dataset(path: Path, debug: bool, debug_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if debug:
        sample_n = min(debug_samples, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, output_dir: Path, model_name: str, task_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(model_name)
    suffix = "_debug" if df.attrs.get("debug_mode") else ""
    output_path = output_dir / f"{safe_model}_hf_detection_{task_name}{suffix}.csv"
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

    tasks_to_run = ["sentence", "word"] if args.task == "both" else [args.task]
    task_frames: Dict[str, List[pd.DataFrame]] = {task: [] for task in tasks_to_run}

    for language, dataset_path in LANG_TO_DATASET.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = load_dataset(dataset_path, debug=args.debug, debug_samples=args.debug_samples)
        if args.debug:
            df.attrs["debug_mode"] = True
        if "sentence" in tasks_to_run:
            task_frames.setdefault("sentence", []).append(
                run_sentence_task(
                    df, detector, language, args.prompt_language, args.debug
                )
            )
        if "word" in tasks_to_run:
            task_frames.setdefault("word", []).append(
                run_word_task(
                    df, detector, language, args.prompt_language, args.debug
                )
            )

    for task_name in tasks_to_run:
        frames = task_frames.get(task_name, [])
        if not frames:
            continue
        result_df = pd.concat(frames, ignore_index=True)
        if args.debug:
            result_df.attrs["debug_mode"] = True
        output_path = save_results(result_df, args.output_dir, args.model_id, task_name)
        print(f"Saved {task_name} results to {output_path}")


if __name__ == "__main__":
    main()

