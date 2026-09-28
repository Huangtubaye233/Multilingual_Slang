import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Explicit API key placeholder; replace or set OPENAI_API_KEY env var.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


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
    user_text = USER_TEMPLATE.format(term=slang, language=language, gold_def=gold_def, pred_def=pred_def)
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


def call_model(client: OpenAI, model: str, messages: Dict[str, str], retry: int = 3, sleep: float = 2.0) -> Dict[str, Any]:
    last_err = None
    for _ in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": messages["system"]},
                    {"role": "user", "content": messages["user"]},
                ],
            )
            content = resp.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an OpenAI model to grade interpretation outputs.")
    parser.add_argument(
        "--input-file",
        default="/Users/kefanyu/Desktop/KefanYu/NU/TTIC/LLM_side/Results_Raw_3.14/gpt-4.1_gpt_interpretation_mono_0shots_chime.csv",
        help="Input CSV path under Results_Raw_2.8 containing interpretation results.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write graded CSV. Defaults to graded_<input name>.csv next to input.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: env OPENAI_API_KEY).",
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
        help="Sleep seconds between requests to avoid rate limits.",
    )
    args = parser.parse_args()

    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set and --api-key not provided.")

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output_file)
        if args.output_file
        else input_path.with_name(f"graded_{input_path.name}")
    )

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
        raise ValueError(f"Missing required columns: {missing}")

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = OpenAI(api_key=api_key)
    judgments = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grading"):
        msgs = build_messages(row)
        judgment = call_model(client, args.model, msgs)
        judgments.append(json.dumps(judgment, ensure_ascii=False))
        if args.sleep > 0:
            time.sleep(args.sleep)

    df["gpt_judge"] = judgments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote graded file to: {output_path}")


if __name__ == "__main__":
    main()
