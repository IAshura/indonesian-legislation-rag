"""
llm_provider.py — Preset provider LLM & adapter ke modul llm_client.py lama.
============================================================================

Spec request /api/ask membawa:
  llm: {provider: "openai"|"anthropic"|"google"|"zhipu"|"ollama"|"custom",
        base_url?, api_key?, model?}

Provider bawaan bisa dikonfigurasi full via base_url + model; preset ini
HANYA mengisi default yang salah/berlebih, bukan keharusan. Provider pertama
yang mengirim header x-api-key = Antropic; google pakai x-goog-api-key,
sisanya Authorization: Bearer.

Anthropic native (api.anthropic.com) bukan format OpenAI — adapter-nya di
llm_adapter.py (dipilih otomatis bila provider anthropic + URL anthropic).
Google preset memakai endpoint OpenAI-compatible (.../v1beta/openai) jadi
jalur OpenAI biasa. Custom/proxy kompatibel ikut jalur OpenAI.

api_key tidak pernah di-log: di engine.py di-strip dari error message.
"""

from __future__ import annotations

# base_url default. provider custom WAJIB isi base_url sendiri.
PRESETS: dict[str, dict] = {
    "openai":    {"base_url": "https://api.openai.com/v1",  "model_saran": "gpt-4o-mini"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "model_saran": "claude-sonnet-4-20250514"},
    "google":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model_saran": "gemini-2.0-flash"},
    "zhipu":     {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model_saran": "glm-4-flash"},
    "ollama":    {"base_url": "http://localhost:11434/v1", "model_saran": "qwen3:4b"},
    "custom":    {"base_url": "", "model_saran": ""},
}


def preset(provider: str) -> dict:
    return PRESETS.get((provider or "").strip().lower(), PRESETS["custom"])


def normalisasi(provider: str, base_url: str | None, model: str | None) -> dict:
    """Campur input pengguna dengan preset -> {provider, base_url, model}."""
    p = preset(provider)
    provider = (provider or "custom").strip().lower()
    return {
        "provider": provider,
        "base_url": (base_url or p["base_url"]).strip(),
        "model": (model or p["model_saran"]).strip(),
    }


def normalisasi_usage(obj: dict | None) -> dict:
    """Samakan bentuk usage antar-provider menjadi:
    {prompt_tokens, completion_tokens, total_tokens, cached_tokens}.
    Dikenali 3 bentuk (transport kode ini OpenAI-compatible, tapi proxy/
    gateway kadang meneruskan bentuk asli — semuanya dinormalkan sama):
      OpenAI   : prompt_tokens / completion_tokens / total_tokens,
                 cache lewat prompt_tokens_details.cached_tokens.
      Anthropic: input_tokens / output_tokens,
                 cache = cache_creation_input_tokens
                       + cache_read_input_tokens.
      Google   : promptTokenCount / candidatesTokenCount / totalTokenCount,
                 cache = cachedContentTokenCount.
    Hilang/tak dikirim provider -> 0 (bukan None) agar bisa dijumlah.
    """
    obj = obj or {}
    detail = obj.get("prompt_tokens_details") or {}
    pt = int(obj.get("prompt_tokens") or obj.get("input_tokens")
             or obj.get("promptTokenCount") or 0)
    ct = int(obj.get("completion_tokens") or obj.get("output_tokens")
             or obj.get("candidatesTokenCount") or 0)
    tot = int(obj.get("total_tokens") or obj.get("totalTokenCount")
              or (pt + ct))
    cached = int(detail.get("cached_tokens")
                 or obj.get("prompt_cache_hit_tokens")
                 or obj.get("cached_tokens")  # sudah ternormalisasi
                 or obj.get("cachedContentTokenCount")
                 or (int(obj.get("cache_creation_input_tokens") or 0)
                     + int(obj.get("cache_read_input_tokens") or 0))
                 or 0)
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": tot, "cached_tokens": cached}


def rapikan_nalar(teks: str | None) -> str | None:
    """Bersihkan teks penalaran agar tampil rapi di UI.

    Latar (terbukti dari history.jsonl): provider/gateway mengirim tiap
    potongan penalaran dipisah whitespace ACAK — kadang newline per kata
    ('H alo'), kadang menempel ('said"Halo"', 'question,but'), kadang sudah
    benar ('a at'). Karena itu SEMUA whitespace dibuang dulu (pecah per
    kata), lalu gabung ulang per kata dengan aturan: tanpa spasi bila
    tanda tutup, kutip tutup berpasangan, awalan [_-]/kontraksi, setelah
    tanda buka, atau lanjutan kata; kalimat diputus tiap akhir (.?!...).
    Spasi dipaksa setelah [,;:] dan [—–] yang menempel kata.
    Data asli tak ada yang hilang — cuma whitespace.
    """
    import re
    t = str(teks or "")
    if not t.strip():
        return None
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    tutup = re.compile(r"^[\)\].,!?;:…%”’]+$")
    kutip = re.compile(r"^['\"]+$")
    tempel = re.compile(r"^[_-]|^['’](ll|s|t|re|ve|m|d)\b", re.IGNORECASE)
    kalimat: list[str] = []
    kini = ""
    hitung_kutip: dict[str, int] = {}

    def _diakhiri_buka(s: str) -> bool:
        if not s:
            return False
        ch = s[-1]
        if ch in "\"'":
            # kutip penutup (genap) bukan pembuka — jangan ditempeli
            return (hitung_kutip.get(ch, 0) % 2 == 1)
        return bool(re.search(r"[(\"'\“‘]$", s))

    def _dorong(p: str) -> None:
        nonlocal kini
        if not kini:
            kini = p
            return
        if tutup.match(p):
            kini += p
            return
        if kutip.match(p):
            ch = p[0]
            hitung_kutip[ch] = hitung_kutip.get(ch, 0) + 1
            kini += p if hitung_kutip[ch] % 2 == 0 else " " + p
            return
        if tempel.match(p) or _diakhiri_buka(kini):
            kini += p
            return
        akhir = kini.rsplit(" ", 1)[-1]
        if (len(p) == 1 and p.isalnum()
                and ("_" in akhir or re.fullmatch(r".\.", akhir))):
            kini += p  # '_norm'+'a', 'e.'+'g' (bukan 'is'+'a', 'with'+'I')
            return
        if (2 <= len(p) <= 3 and p.replace("_", "").isalnum()
                and re.search(r"['\"“‘][A-Za-z]$", akhir)):
            kini += p  # '"H'+'alo' (bukan 'search_norma'+'at')
            return
        kini += " " + p

    # Pecah per KATA dulu (semua whitespace dibuang): spasi asli provider
    # tidak bisa dipercaya (kadang hilang seperti 'nota', kadang benar
    # seperti 'a at'), jadi keputusan gabung/spasi murni dari aturan.
    for kata in t.split():
        # pisahkan deret kutip di ujung kata ("said'" -> 'said' + '"'),
        # tapi kontraksi ("'ll") utuh agar menempel ("I'll")
        for p in re.findall(
                r"['’](?:ll|s|t|re|ve|m|d)\b|[\"'“‘”’]+|[^\"'“‘”’]+",
                kata, re.IGNORECASE):
            _dorong(p)
        if re.search(r"[.!?…]$", kata):
            kalimat.append(kini)
            kini = ""
    if kini:
        kalimat.append(kini)
    bersih = "\n\n".join(kalimat)
    bersih = re.sub(r"([,;:])([^\s\d])", r"\1 \2", bersih)
    bersih = re.sub(r"([—–])([^\s])", r"\1 \2", bersih)
    bersih = re.sub(r"\s+([,;:.!?…%”’\)\]])", r"\1", bersih)
    return bersih.strip() or None


def tambah_usage(akumulasi: dict | None, tambahan: dict | None) -> dict:
    """Jumlahkan dua usage ternormalisasi (untuk loop agent multi-putaran)."""
    a = normalisasi_usage(akumulasi)
    b = normalisasi_usage(tambahan)
    return {k: a.get(k, 0) + b.get(k, 0) for k in
            ("prompt_tokens", "completion_tokens", "total_tokens",
             "cached_tokens")}


def validasi(llm: dict | None) -> tuple[str, str, str]:
    """
    Kembalikan (base_url, api_key, model) yang siap dipakai llm_client.
    Dry-run tidak butuh api_key; live LLM tanpa key melempar RuntimeError
    yang jelas (dipotong sebelum ke klien - lihat engine.py).

    Fallback: bila request browser tidak membawa api_key, pakai LLM_API_KEY
    dari env (dimuat app.py dari webapp/backend/.env). Kunci tidak pernah
    dikirim balik ke browser.
    """
    if not llm:
        return "", "", ""
    n = normalisasi(llm.get("provider", "custom"), llm.get("base_url"),
                    llm.get("model"))
    api_key = (llm.get("api_key") or "").strip()
    if not api_key:
        import os
        api_key = os.environ.get("LLM_API_KEY", "").strip()
    return n["base_url"], api_key, n["model"]
