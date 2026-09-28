#!/usr/bin/env python3
"""
Grade interpretation outputs with a Hugging Face causal LM judge.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "Results"


SYSTEM_PROMPT = """You evaluate semantic similarity between a predicted slang definition and a gold definition.
Ignore fluency, style, and length. Judge ONLY meaning.
Output valid JSON only."""

USER_TEMPLATE = """Slang term: {term}
Language: {language}
Gold definition: {gold_def}
Predicted definition: {pred_def}

Score the prediction from 1 to 5 using these anchors:
5 = EQUIVALENT: near-synonym; interchangeable in most contexts; no key unsupported constraints
4 = CLOSE: same general meaning but differs in intensity/scope/conditions
3 = PARTIAL: captures only part; too broad/narrow; unreliable as a definition
2 = RELATED_NOT_DEFINITION: related but not an actual definitional paraphrase
1 = WRONG: different meaning or contradiction

Also output ONLY these three diagnostics:
- definition_form: is_definition / usage_only / example_only / meta
- core_meaning_overlap: high / medium / low / none
- polarity_match: match / mismatch / unclear

Return JSON:
{{
  "score": <1-5>,
  "definition_form": "...",
  "core_meaning_overlap": "...",
  "polarity_match": "...",
  "reason": "one short sentence"
}}"""


def normalize_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in {"true", "1", "yes"}


def pick_prediction(row: pd.Series) -> str:
    """Choose text sent to the judge: predicted_definition if json_valid else raw_response."""
    is_valid = normalize_bool(row.get("json_valid"))
    pred = str(row.get("predicted_definition", "")).strip()
    raw = str(row.get("raw_response", "")).strip()
    return pred if is_valid and pred else raw


def build_messages(row: pd.Series) -> Dict[str, str]:
    slang = str(row.get("slang", "")).strip()
    language = str(row.get("language", "")).strip()
    gold_def = str(row.get("gold_definition", "")).strip()
    pred_def = pick_prediction(row)
    user_text = USER_TEMPLATE.format(
        term=slang,
        language=language,
        gold_def=gold_def,
        pred_def=pred_def,
    )
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start_idx:])
            if isinstance(obj, dict):
                return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model output.")


def prepare_patched_tokenizer_dir(model_id: str, cache_dir: Optional[Path]) -> Path:
    """Patch tokenizer config fields that break on some Gemma releases."""
    repo_dir = Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=[
                "tokenizer*",
                "special_tokens_map.json",
                "added_tokens.json",
                "config.json",
            ],
        )
    )
    base_cache = cache_dir if cache_dir else (Path.home() / ".cache" / "huggingface")
    patched_dir = base_cache / "patched_tokenizers" / sanitize_model_name(model_id)
    patched_dir.mkdir(parents=True, exist_ok=True)

    for item in repo_dir.iterdir():
        target = patched_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    config_path = patched_dir / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        modified = False
        extra_special_tokens = config.get("extra_special_tokens")
        if isinstance(extra_special_tokens, list):
            config["extra_special_tokens"] = {}
            modified = True
        vocab_file = config.get("vocab_file")
        if isinstance(vocab_file, list):
            config["vocab_file"] = vocab_file[0] if vocab_file else None
            modified = True
        if modified:
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return patched_dir


class HFJudge:
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path],
        device_map: str,
        dtype_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        torch_dtype = resolve_dtype(dtype_name)
        tokenizer_kwargs: Dict[str, Any] = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = str(cache_dir)
        tokenizer_error: Optional[Exception] = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                **tokenizer_kwargs,
            )
        except AttributeError as exc:
            # Some transformers versions have a Gemma fast-tokenizer bug:
            # AttributeError: 'list' object has no attribute 'keys'
            # Fallback to the slow tokenizer for compatibility.
            if "has no attribute 'keys'" not in str(exc):
                tokenizer_error = exc
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        use_fast=False,
                        trust_remote_code=True,
                        **tokenizer_kwargs,
                    )
                except Exception as slow_exc:  # noqa: BLE001
                    tokenizer_error = slow_exc
        except Exception as exc:  # noqa: BLE001
            tokenizer_error = exc

        if tokenizer_error is not None and not hasattr(self, "tokenizer"):
            try:
                patched_dir = prepare_patched_tokenizer_dir(model_id, cache_dir)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(patched_dir),
                        trust_remote_code=True,
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(patched_dir),
                        use_fast=False,
                        trust_remote_code=True,
                    )
            except Exception as patched_exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to initialize tokenizer for "
                    f"{model_id}: {tokenizer_error}; patched fallback failed: {patched_exc}"
                ) from patched_exc

        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if cache_dir:
            model_kwargs["cache_dir"] = str(cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _build_prompt(self, messages: Dict[str, str]) -> str:
        chat_messages = [
            {"role": "system", "content": messages["system"]},
            {"role": "user", "content": messages["user"]},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_template = getattr(self.tokenizer, "chat_template", None)
            if chat_template:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return (
            f"System: {messages['system']}\n\n"
            f"User: {messages['user']}\n\n"
            "Assistant:"
        )

    def generate(self, messages: Dict[str, str]) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            generate_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def call_model(
    judge: HFJudge,
    messages: Dict[str, str],
    retry: int = 3,
    sleep: float = 2.0,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for _ in range(retry):
        try:
            raw = judge.generate(messages)
            json_text = extract_json_object(raw)
            return json.loads(json_text)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep)
    raise RuntimeError(f"HF model call failed after retries: {last_err}")


def grade_one_file(
    judge: HFJudge,
    input_path: Path,
    output_path: Path,
    *,
    max_rows: Optional[int],
    sleep: float,
    output_column: str,
    num_shards: int,
    shard_index: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = {
        "slang",
        "language",
        "gold_definition",
        "model_response_json",
        "predicted_definition",
        "json_valid",
        "raw_response",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path.name}: {missing}")

    if max_rows is not None:
        df = df.head(max_rows).copy()

    # Keep stable row ids so shard outputs can be merged/reordered deterministically.
    df = df.reset_index().rename(columns={"index": "source_row_id"})
    if num_shards > 1:
        df = df[df["source_row_id"] % num_shards == shard_index].copy()
    if df.empty:
        print(
            f"No rows selected for shard {shard_index}/{num_shards} in {input_path.name}; writing empty output."
        )

    judgments = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Grading {input_path.name}"):
        msgs = build_messages(row)
        try:
            judgment = call_model(judge, msgs)
        except Exception as exc:  # noqa: BLE001
            judgment = {
                "score": None,
                "definition_form": "meta",
                "core_meaning_overlap": "none",
                "polarity_match": "unclear",
                "reason": f"judge_error: {str(exc)[:180]}",
            }
        judgments.append(json.dumps(judgment, ensure_ascii=False))
        if sleep > 0:
            time.sleep(sleep)

    df[output_column] = judgments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote graded file to: {output_path}")


def default_output_path(input_path: Path, output_dir: Optional[Path], output_prefix: str) -> Path:
    target_dir = output_dir if output_dir else input_path.parent
    return target_dir / f"{output_prefix}{input_path.name}"


def with_shard_suffix(path: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return path
    return path.with_name(
        f"{path.stem}_shard{shard_index + 1}of{num_shards}{path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a Hugging Face model to grade interpretation outputs."
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Single input CSV path containing interpretation outputs.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help="Multiple input CSV paths to process in one run.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write graded CSV for single-file mode only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Optional output directory for multi-file mode (or single-file without --output-file).",
    )
    parser.add_argument(
        "--output-prefix",
        default="Gemma_graded_",
        help="Output filename prefix when --output-file is not provided.",
    )
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E4B-it",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="Directory for Hugging Face model cache.",
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
        default=256,
        help="Maximum tokens to generate per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to process (for quick tests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--output-column",
        default="gpt_judge",
        help="Output column name for judge JSON strings.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards. Use >1 to split dataset across multiple jobs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index. Must be in [0, num_shards).",
    )
    args = parser.parse_args()

    input_paths = []
    if args.input_files:
        input_paths.extend(Path(p) for p in args.input_files)
    if args.input_file:
        input_paths.append(Path(args.input_file))
    if not input_paths:
        raise ValueError("Please provide --input-file or --input-files.")

    if args.output_file and len(input_paths) > 1:
        raise ValueError("--output-file is only valid when processing a single input file.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")

    hf_cache = args.hf_cache
    if hf_cache:
        hf_cache.mkdir(parents=True, exist_ok=True)

    judge = HFJudge(
        model_id=args.model_id,
        cache_dir=hf_cache,
        device_map=args.device_map,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(f"Judge model: {sanitize_model_name(args.model_id)}")

    for input_path in input_paths:
        if len(input_paths) == 1 and args.output_file:
            output_path = Path(args.output_file)
        else:
            output_path = default_output_path(
                input_path=input_path,
                output_dir=args.output_dir,
                output_prefix=args.output_prefix,
            )
        output_path = with_shard_suffix(
            output_path,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        grade_one_file(
            judge=judge,
            input_path=input_path,
            output_path=output_path,
            max_rows=args.max_rows,
            sleep=args.sleep,
            output_column=args.output_column,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )


if __name__ == "__main__":
    main()
