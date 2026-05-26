"""One-call sanity check for TUM Ollama + qwen3.5:122b.

Usage::

    $env:OLLAMA_HOST = "https://ollama.sps.ed.tum.de"
    uv run python scripts/check_ollama.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    host = os.environ.get("OLLAMA_HOST", "(unset)")
    print(f"OLLAMA_HOST = {host}")

    from src.representation.llm_client import LLMClient

    client = LLMClient({
        "llm_provider": "ollama",
        "llm_model": "qwen3.5:122b",
        "llm_temperature": 0.0,
    })

    system_prompt = (
        'You output a single JSON object with one field "mode" whose value '
        'is the string "charging". No other text.'
    )
    user_prompt = "Return JSON now."

    try:
        resp = client.generate(system_prompt, user_prompt, json_mode=True)
    except Exception:
        print("\nFAILED — exception during complete():")
        traceback.print_exc()
        return 1

    print("\nRaw response:")
    print(repr(resp))

    # Try to parse as JSON if it's a string
    if isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            print(f"\nParsed JSON: {parsed}")
        except json.JSONDecodeError as e:
            print(f"\nNote: response was a string but not parseable JSON: {e}")

    print("\nClient metrics:")
    for k, v in client.get_metrics().items():
        print(f"  {k} = {v}")
    print(f"  last_provider = {getattr(client, '_last_provider', 'unknown')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
