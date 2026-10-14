"""Gemma smoke test (Google AI Studio).

Was: test.py (root, contained a real API key — now redacted).
Now: tools/gemma-smoke-test.py — minimal manual chat test.

Setup:
    pip install google-genai
    set GEMMA_API_KEY=...   (never commit the real key)
    python tools/gemma-smoke-test.py
"""

from __future__ import annotations

import os
import sys

MODEL = "models/gemma-4-31b-it"


def main() -> int:
    try:
        from google import genai
    except ImportError:
        print("Missing dependency: pip install google-genai")
        return 1

    api_key = os.environ.get("GEMMA_API_KEY", "")
    if not api_key:
        print("ERROR: set GEMMA_API_KEY env var first (never hardcode keys).")
        return 1

    client = genai.Client(api_key=api_key)
    print("=" * 50)
    print("GEMMA SMOKE TEST")
    print("=" * 50)
    try:
        prompt = input("\nEnter prompt: ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0

    try:
        chat = client.chats.create(model=MODEL)
        response = chat.send_message(prompt)
        print("\n--- RESPONSE ---")
        print(response.text)
        print("\n--- INFO ---")
        print("Model :", MODEL)
        print("Status: OK")
        return 0
    except Exception as exc:  # noqa: BLE001 — smoke script, show everything
        print("\n--- ERROR ---")
        print(type(exc).__name__)
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
