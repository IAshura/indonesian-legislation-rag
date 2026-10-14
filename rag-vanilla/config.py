"""
config.py — Single door for ALL configuration of `rag-vanilla/`.
================================================================================

MENGAPA FILE INI ADA?
  Setiap file lain (dataset.py, retriever.py, answer.py, agent.py, ...)
  membutuhkan informasi yang sama: "di mana data berada" dan "LLM backend mana
  yang dipakai". Menaruhnya di satu tempat berarti:
    - mengganti backend LLM (OpenAI / OpenRouter / Groq / Ollama lokal) cukup
      diedit di SATU tempat (atau lewat file .env) tanpa menyentuh kode lain;
    - tidak ada path hardcoded tersebar yang rentan salah ketik.

DUA HAL YANG DITANGANI FILE INI
  1. PATH  : lokasi hasil pipeline (chunks.jsonl, embeddings, dst).
  2. LLM   : base_url + api_key + model, dibaca dari environment variable
             atau file `.env` (lihat .env.example).

ATURAN BESI PROYEK YANG TERCATAT DI SINI
  - Model embedding query = model embedding index. Folder ini HANYA memakai
    Qwen3-Embedding-0.6B (dim 1024) — sama seperti yang ada di Milvus.
  - Hanya part == "norma" yang boleh masuk konteks jawaban.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- PATH PROYEK
# This file lives at <repo-root>/rag-vanilla/config.py
# parents[0] = rag-vanilla/  ->  parents[1] = repo root.
# Dengan begitu skrip bisa dijalankan dari folder mana pun.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Hasil tahap build (v4.1): unit norma siap-embed + metadata provenance.
RESULT_DIR: Path = PROJECT_ROOT / "pipeline" / "result"

CHUNKS_JSONL: Path = RESULT_DIR / "chunks.jsonl"          # 74,218 chunks (all parts)
PARENTS_JSONL: Path = RESULT_DIR / "parents.jsonl"        # 11.413 pasal/blok utuh
PARENTS_WINDOWS_JSONL: Path = RESULT_DIR / "parents_windows.jsonl"  # 14.689 window <=1.800 char
DOCUMENTS_JSONL: Path = RESULT_DIR / "documents.jsonl"    # 378 peraturan
QUARANTINE_JSONL: Path = RESULT_DIR / "quarantine.jsonl"  # 2.209 item kuarantin
MANIFEST_JSON: Path = RESULT_DIR / "manifest.json"        # angka resmi build

# ------------------------------------------------------------- EMBEDDINGS
# Folder embeddings aktif ditandai file `ACTIVE` (isinya "qwen3-06b").
# Pola ini memungkinkan beberapa model embedding hidup berdampingan
# (di proyek ini: qwen3-06b dim 1024 dan qwen3-4b dim 2560).
EMB_ROOT: Path = RESULT_DIR / "embeddings"


def active_embedding_dir() -> Path:
    """Kembalikan folder embeddings aktif (dibaca dari file ACTIVE)."""
    name = (EMB_ROOT / "ACTIVE").read_text(encoding="utf-8").strip()
    return EMB_ROOT / name


EMB_DIR: Path = active_embedding_dir()
EMB_NPY: Path = EMB_DIR / "embeddings.npy"   # matriks float32 (count x dim), L2-normal
EMB_IDS: Path = EMB_DIR / "ids.jsonl"        # baris-i matriks <-> metadata chunk-nya
EMB_META: Path = EMB_DIR / "meta.json"       # model, dim, pooling, query_instruction, ...

# ------------------------------------------------------- KONSTANTA RETRIEVAL
TOP_K: int = 8                 # jumlah kandidat yang dicari (recall@8 diukur = 0.75)
CONTEXT_CHAR_BUDGET: int = 12000   # anggaran karakter konteks sebelum dipotong
MIN_SCORE: float = 0.0         # ambang skor cosine minimum (0 = ambil semua)

# ------------------------------------------------------- KONSTANTA RERANKER
# Pencarian 2 fase: fase 1 (murah) ambil RERANK_POOL_SIZE kandidat via cosine,
# fase 2 (akurat) Qwen3-Reranker-0.6B (cross-encoder) menilai ulang tiap
# pasangan (query, pasal) lalu urutan final diambil dari skor rerank.
# Set ON/OFF via env RERANK=1|0 (default: aktif bila model ada di cache).
RERANK: bool = os.environ.get("RERANK", "1") == "1"
RERANK_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
RERANK_POOL_SIZE: int = 12     # kandidat fase-1 yang ikut direrank
RERANK_BATCH_SIZE: int = 4     # pasangan per forward batch (CPU-friendly)
RERANK_MAX_LENGTH: int = 1024  # token maks per pasangan (potong teks panjang)

# ------------------------------------------------------- KONSTANTA HYBRID
# Hybrid search: DUA jalur independen lalu digabung RRF (Reciprocal Rank
# Fusion) — bukan penjumlahan skor, karena skala skor dense (0..1) dan BM25
# (0..~20) tidak sebanding. RRF hanya melihat PERINGKAT:
#     skor_fusi(c) = SUM_jalur 1/(RRF_K + rank_jalur(c))
# (rumus standar Cormack et al. 2009; RRF_K=60 adalah nilai baku).
# Jalur 1 (DENSE)  : cosine embeddings.npy (ExactRetriever).
# Jalur 2 (LEXICAL): BM25 atas token teks pasal + sitasi + judul
#                    (index leksikal dibuat sekali oleh bm25.py --build).
# ON/OFF via env HYBRID=1|0 (default aktif; indeks hilang -> fallback dense).
HYBRID: bool = os.environ.get("HYBRID", "1") == "1"
HYBRID_PER_JALUR: int = 12    # top-N tiap jalur sebelum difusi
RRF_K: int = 60               # konstanta RRF standar
LEXICAL_DIR: Path = RESULT_DIR / "lexical"
LEXICAL_INDEX: Path = LEXICAL_DIR / "norma_tokens.txt"

# ------------------------------------------------------------------ KONFIG LLM
# Baca file .env (jika ada) TANPA dependensi python-dotenv:
# format baris sederhana `KEY=VALUE`, baris kosong / `#` diabaikan.
_ENV_FILE: Path = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())  # env asli lebih diutamakan


def get_llm_config() -> dict:
    """
    Kumpulkan konfigurasi LLM dari environment (.env atau env asli).

    Kunci yang dibaca:
      LLM_BASE_URL  : endpoint OpenAI-compatible.
                      OpenAI     -> https://api.openai.com/v1
                      OpenRouter -> https://openrouter.ai/api/v1
                      Groq       -> https://api.groq.com/openai/v1
                      DeepSeek   -> https://api.deepseek.com/v1
                      Ollama     -> http://localhost:11434/v1   (key bebas)
      LLM_API_KEY   : kunci API (untuk Ollama lokal isi "ollama").
      LLM_MODEL     : nama model, mis. "gpt-4o-mini", "qwen/qwen3-4b-2507",
                      "qwen3:4b" (Ollama).
    """
    return {
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
        "timeout": int(os.environ.get("LLM_TIMEOUT", "120")),
    }
