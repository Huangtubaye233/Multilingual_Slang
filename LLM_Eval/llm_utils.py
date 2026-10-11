"""Small evaluation helpers shared by the GPT entry points."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_api_key_from_file(path: Path) -> str:
    """Read a plain-text key, falling back to OPENAI_API_KEY if the file is absent."""
    if path.is_file():
        key = path.read_text(encoding="utf-8-sig").strip()
        if key:
            return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(f"Set OPENAI_API_KEY or provide a nonempty API key file: {path}")
    return key


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def parse_json_response(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Accept a JSON object, including fenced JSON or surrounding model prose."""
    if not text:
        return None, ""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, json.dumps(obj, ensure_ascii=False)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start_idx:])
            if isinstance(obj, dict):
                return obj, json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    return None, ""
