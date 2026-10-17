"""
llm_adapter.py — Adapter multi-provider untuk webapp.
====================================================

MASALAH: tidak semua "provider LLM" bicara format OpenAI.
  - openai / zhipu / ollama / custom / google(via .../v1beta/openai):
    format OpenAI — Bearer + POST {base}/chat/completions. (jalur lama)
  - anthropic native (api.anthropic.com):
    header x-api-key + versi, POST {base}/messages, system terpisah,
    usage bernama input_tokens/output_tokens, SSE event message_delta.
    Endpoint OpenAI di host ini TIDAK ADA (404) — tanpa adapter,
    preset anthropic di UI mati total.

Aturan routing (di rasa()):
  provider == "anthropic" DAN "anthropic.com" dalam base_url -> "anthropic".
  Selain itu -> "openai" (termasuk google preset, yang memang URL
  OpenAI-compatible, dan custom/proxy kompatibel).

KELUARAN SELALU BENTUK OPENAI agar engine/streaming/UI tak berubah:
  message = {"role": "assistant", "content": str|None,
             "tool_calls": [{id, type, function:{name, arguments:str}}]|None}
  usage mentah diteruskan apa adanya; normalisasi di
  llm_provider.normalisasi_usage().
"""

from __future__ import annotations

import json


VERSI_ANTHROPIC = "2023-06-01"


def rasa(provider: str | None, base_url: str | None) -> str:
    """Tentukan rasa API dari pilihan provider + URL."""
    p = (provider or "").strip().lower()
    u = (base_url or "").strip().lower()
    if p == "anthropic" and "anthropic.com" in u:
        return "anthropic"
    return "openai"


# ------------------------------------------------ konverter pesan/tools
def _pesan_ke_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pesan OpenAI -> (system, messages) ala Anthropic."""
    sistem: list[str] = []
    keluar: list[dict] = []
    for m in messages or []:
        peran = m.get("role")
        isi = m.get("content")
        if peran == "system":
            if isi:
                sistem.append(str(isi))
            continue
        if peran == "assistant" and m.get("tool_calls"):
            blok: list[dict] = []
            if isi:
                blok.append({"type": "text", "text": str(isi)})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                try:
                    arg = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arg = {}
                blok.append({"type": "tool_use", "id": tc.get("id"),
                             "name": fn.get("name"), "input": arg})
            keluar.append({"role": "assistant", "content": blok})
            continue
        if peran == "tool":
            keluar.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": str(isi or "")}]})
            continue
        keluar.append({"role": "assistant" if peran == "assistant" else "user",
                       "content": str(isi or "")})
    return "\n\n".join(sistem), keluar


def _tools_ke_anthropic(tools: list[dict] | None) -> list[dict]:
    """Skema function OpenAI -> input_schema Anthropic."""
    keluar = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        keluar.append({"name": fn.get("name"),
                       "description": fn.get("description") or "",
                       "input_schema": fn.get("parameters")
                       or {"type": "object"}})
    return keluar


def _pesan_dari_anthropic(content) -> tuple[str | None, list | None]:
    """Blok content Anthropic -> (content, tool_calls bentuk OpenAI)."""
    teks: list[str] = []
    tc = []
    for b in content or []:
        if b.get("type") == "text":
            teks.append(b.get("text") or "")
        elif b.get("type") == "tool_use":
            tc.append({"id": b.get("id"), "type": "function",
                       "function": {"name": b.get("name"),
                                    "arguments": json.dumps(
                                        b.get("input") or {},
                                        ensure_ascii=False)}})
    return ("".join(teks) or None, tc or None)


def _kepala_anthropic(api_key: str) -> dict:
    return {"x-api-key": api_key,
            "anthropic-version": VERSI_ANTHROPIC,
            "content-type": "application/json"}


# ------------------------------------------------------------ non-stream
def complete_anthropic(base_url: str, api_key: str, model: str,
                       messages: list[dict], tools: list[dict] | None,
                       temperature: float | None,
                       max_tokens: int | None) -> tuple[dict, dict]:
    """Satu panggilan non-stream Anthropic -> (message, usage_mentah)."""
    import requests

    system, msgs = _pesan_ke_anthropic(messages)
    payload: dict = {"model": model, "max_tokens": max_tokens or 1024,
                     "messages": msgs}
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    conv = _tools_ke_anthropic(tools)
    if conv:
        payload["tools"] = conv
        payload["tool_choice"] = {"type": "auto"}

    url = base_url.rstrip("/") + "/messages"
    resp = requests.post(url, json=payload,
                         headers=_kepala_anthropic(api_key), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} dari Anthropic: {resp.text[:300]}")
    data = resp.json()
    isi, tc = _pesan_dari_anthropic(data.get("content"))
    return ({"role": "assistant", "content": isi, "tool_calls": tc},
            data.get("usage") or {})


# --------------------------------------------------------------- stream
def stream_anthropic_iter(base_url: str, api_key: str, model: str,
                          messages: list[dict], tools: list[dict] | None,
                          temperature: float | None,
                          max_tokens: int | None):
    """Generator SSE Anthropic -> event {"t": token|usage|message} sama
    seperti chat_stream_iter jalur OpenAI (lihat streaming.py)."""
    import requests

    system, msgs = _pesan_ke_anthropic(messages)
    payload: dict = {"model": model, "max_tokens": max_tokens or 1024,
                     "messages": msgs, "stream": True}
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    conv = _tools_ke_anthropic(tools)
    if conv:
        payload["tools"] = conv
        payload["tool_choice"] = {"type": "auto"}

    url = base_url.rstrip("/") + "/messages"
    try:
        resp = requests.post(url, json=payload,
                             headers={**_kepala_anthropic(api_key),
                                      "accept": "text/event-stream"},
                             stream=True, timeout=(30, 300))
    except Exception as e:
        raise RuntimeError(f"Anthropic tak terjangkau ({base_url}): {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} dari Anthropic: "
            f"{resp.text[:300] if not resp.raw else ''}")

    teks: list[str] = []
    alat: dict[int, dict] = {}
    guna: dict = {}
    jenis_event = ""
    try:
        for baris in resp.iter_lines(decode_unicode=True):
            if not baris:
                continue
            if baris.startswith("event:"):
                jenis_event = baris[6:].strip()
                continue
            if not baris.startswith("data:"):
                continue
            data = baris[5:].strip()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if jenis_event == "message_start":
                u = ((obj.get("message") or {}).get("usage")) or {}
                if u:
                    guna = {**guna, **u}
                    yield {"t": "usage", "usage": dict(guna)}
            elif jenis_event == "content_block_start":
                blok = obj.get("content_block") or {}
                if blok.get("type") == "tool_use":
                    i = int(obj.get("index", 0))
                    alat[i] = {"id": blok.get("id") or "",
                               "name": blok.get("name") or "",
                               "input": ""}
            elif jenis_event == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    teks.append(delta["text"])
                    yield {"t": "token", "text": delta["text"]}
                elif "partial_json" in delta:
                    i = int(obj.get("index", 0))
                    alat.setdefault(i, {"id": "", "name": "",
                                        "input": ""})
                    alat[i]["input"] += delta.get("partial_json") or ""
            elif jenis_event == "message_delta":
                u = obj.get("usage") or {}
                if u:
                    guna = {**guna, **u}
                    yield {"t": "usage", "usage": dict(guna)}
    finally:
        resp.close()

    tc = None
    if alat:
        tc = []
        for i, s in sorted(alat.items()):
            try:
                arg = json.loads(s["input"] or "{}")
            except json.JSONDecodeError:
                arg = {}
            tc.append({"id": s["id"] or f"call_{i}", "type": "function",
                       "function": {"name": s["name"],
                                    "arguments": json.dumps(
                                        arg, ensure_ascii=False)}})
    yield {"t": "message",
           "message": {"role": "assistant",
                       "content": "".join(teks) or None,
                       "tool_calls": tc}}
