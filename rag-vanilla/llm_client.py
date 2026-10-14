"""
llm_client.py — Klien LLM "telanjang" (tanpa SDK framework).
=============================================================

MENGAPA requests SAJA, TANPA SDK openai?
  Agar anatomi protokol terlihat: panggilan chat LLM hanyalah SATU POST
  HTTP ke endpoint OpenAI-compatible dengan payload JSON. Semua provider
  modern memakai format yang sama (OpenAI, OpenRouter, Groq, DeepSeek,
  bahkan Ollama lokal di /v1). Memahami payload = bisa memakai semua.

ANATOMI PESAN (messages)
  [
    {"role":"system",    "content":"...aturan keras..."},
    {"role":"user",      "content":"pertanyaan"},
    {"role":"assistant", "content":null, "tool_calls":[{"id":..,
                          "type":"function","function":{"name":..,
                          "arguments":"{...json string...}"}}]},
    {"role":"tool", "tool_call_id":"...", "content":"hasil tool JSON"}
  ]
  Perhatikan: ARGUMEN tool dari LLM datang sebagai STRING JSON (harus
  json.loads), dan hasil tool dikirim balik dengan role "tool" + id yang
  cocok. Inilah seluruh rahasia tool calling.

MODE TANPA JARINGAN
  - dry_run(): cukup CETAK payload yang akan dikirim (bagus untuk belajar).
  - MockLLM  : LLM PALSU terjadwal — jawaban diambil dari daftar skrip,
    sehingga loop agent di agent.py bisa didemokan tanpa API key.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import requests

import config


# ------------------------------------------------------------- klien nyata
def chat_with_usage(messages: list[dict], tools: list[dict] | None = None,
                    temperature: float | None = None,
                    max_tokens: int | None = None) -> tuple[dict, dict]:
    """
    Seperti chat() tapi mengembalikan (message, usage_mentah).
    usage_mentah = objek "usage" provider apa adanya
    ({prompt_tokens, completion_tokens, total_tokens, ...} atau {} bila
    provider tidak mengirim). Normalisasi bentuknya di llm_provider
    webapp (normalisasi_usage) agar konsisten antar-provider.
    """
    cfg = config.get_llm_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "LLM_API_KEY kosong. Isi file .env (lihat .env.example) atau "
            "jalankan dengan --dry-run untuk mode tanpa jaringan.")

    payload: dict = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"] if temperature is None else temperature,
        "max_tokens": cfg["max_tokens"] if max_tokens is None else max_tokens,
    }
    if tools:                      # <- daftar skema fungsi utk tool calling
        payload["tools"] = tools
        payload["tool_choice"] = "auto"   # LLM yang memutuskan memanggil apa

    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}

    resp = requests.post(url, json=payload, headers=headers,
                         timeout=cfg["timeout"])
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} dari LLM: {resp.text[:300]}\n"
            f"-> cek LLM_BASE_URL ({cfg['base_url']}), LLM_MODEL "
            f"({cfg['model']}), dan validitas LLM_API_KEY.")

    data = resp.json()
    pesan = data["choices"][0]["message"]
    # model reasoning menyertakan penalaran di field ini (chat biasa: tidak ada)
    if pesan.get("reasoning_content") and not pesan.get("reasoning"):
        pesan["reasoning"] = pesan.pop("reasoning_content")
    return pesan, data.get("usage") or {}


def chat(messages: list[dict], tools: list[dict] | None = None,
         temperature: float | None = None,
         max_tokens: int | None = None) -> dict:
    """
    Satu panggilan chat completion. Kembalikan dict "message" LLM:
      {"role":"assistant", "content": str | None, "tool_calls": [...] | None}
    Melempar RuntimeError dengan pesan yang merujuk perbaikan konkret.
    (Butuh usage? pakai chat_with_usage.)
    """
    msg, _ = chat_with_usage(messages, tools=tools, temperature=temperature,
                             max_tokens=max_tokens)
    return msg


# ------------------------------------------------------------- mode cetak
def dry_run(messages: list[dict], tools: list[dict] | None = None) -> str:
    """
    Mode TANPA JARINGAN: tampilkan payload yang SEHARUSNYA dikirim.
    Berguna untuk (1) memeriksa prompt sebelum membayar API,
    (2) mempelajari struktur messages/tools.
    """
    cfg = config.get_llm_config()
    garis = "=" * 72
    out = [f"\n{garis}\nDRY-RUN — payload yang akan dikirim ke LLM\n{garis}",
           f"POST {cfg['base_url'].rstrip('/')}/chat/completions",
           f"model: {cfg['model']}\n"]
    if tools:
        out.append(f"tools: {[t['function']['name'] for t in tools]}")
    for m in messages:
        role = m["role"].upper()
        isi = m.get("content") or f"<meminta tool: {[t['function']['name'] for t in m.get('tool_calls', [])]}>"
        out.append(f"\n----- [{role}] -----\n{isi}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------- mock LLM
class MockLLM:
    """
    LLM PALSU untuk demo loop agent TANPA jaringan/API.
    Dipakai seperti:  jawab = mock.chat(messages) — antarmukanya SAMA
    dengan chat() asli sehingga agent.py tidak perlu tahu bedanya.

    Skrip berisi urutan "putaran": tiap putaran bisa berupa
      {"tool_calls":[{name, arguments:dict}]}   -> LLM minta tool, atau
      {"content":"jawaban final"}               -> LLM yakin, loop selesai.
    """

    def __init__(self, skrip: list[dict]) -> None:
        self.skrip = list(skrip)
        self.panggilan = 0

    def chat(self, messages: list[dict], tools=None, **kw) -> dict:
        if self.panggilan >= len(self.skrip):
            return {"role": "assistant",
                    "content": "(mock: skrip habis — jawab ringkas)"}
        langkah = self.skrip[self.panggilan]
        self.panggilan += 1

        if "tool_calls" in langkah:
            # Bentuk ulang persis seperti respons API asli:
            # arguments berupa STRING JSON, bukan objek.
            tc = [{"id": f"call_{self.panggilan}", "type": "function",
                   "function": {"name": t["name"],
                                "arguments": json.dumps(t["arguments"],
                                                        ensure_ascii=False)}}
                  for t in langkah["tool_calls"]]
            return {"role": "assistant", "content": None, "tool_calls": tc}
        return {"role": "assistant", "content": langkah["content"]}


if __name__ == "__main__":
    # `python llm_client.py` -> demo dry-run + mock, tanpa jaringan
    pesan = [{"role": "system", "content": "Kamu asisten hukum."},
             {"role": "user", "content": "Contoh pertanyaan?"}]
    print(dry_run(pesan))
    mock = MockLLM([{"tool_calls": [{"name": "search_norma",
                                     "arguments": {"question": "kedaulatan"}}]},
                    {"content": "Jawaban final contoh."}])
    print(mock.chat(pesan))
