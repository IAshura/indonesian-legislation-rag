"""
config.py — Satu pintu konfigurasi folder `rag-langchain/`.
======================================================================

Isi dan fungsinya IDENTIK dengan config.py di folder `rag-vanilla/`
(sengaja dibuat mirror agar kedua folder mudah dibandingkan berdampingan):
  1. PATH  : lokasi dataset & embeddings hasil pipeline (v4.1).
  2. LLM   : backend di baca dari environment / file .env.

Perbedaan penting dengan folder tanpa-framework:
  folder ini TIDAK membangun klien HTTP sendiri. Konfigurasi di sini
  hanya dipakai oleh `llm.py` untuk membuat objek model LangChain
  (ChatOpenAI / ChatOllama). Itulah kerja framework: lapisan klien
  dibungkus abstraksi `BaseChatModel`.

ATURAN BESI yang tetap berlaku:
  - model embedding query = model embedding index (Qwen3-Embedding-0.6B);
  - hanya part == "norma" yang masuk konteks jawaban.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULT_DIR: Path = PROJECT_ROOT / "pipeline" / "result"

CHUNKS_JSONL: Path = RESULT_DIR / "chunks.jsonl"
PARENTS_JSONL: Path = RESULT_DIR / "parents.jsonl"
PARENTS_WINDOWS_JSONL: Path = RESULT_DIR / "parents_windows.jsonl"
DOCUMENTS_JSONL: Path = RESULT_DIR / "documents.jsonl"
MANIFEST_JSON: Path = RESULT_DIR / "manifest.json"

EMB_ROOT: Path = RESULT_DIR / "embeddings"


def active_embedding_dir() -> Path:
    """Folder embeddings aktif (dibaca dari file ACTIVE, isi 'qwen3-06b')."""
    return EMB_ROOT / (EMB_ROOT / "ACTIVE").read_text(encoding="utf-8").strip()


EMB_DIR: Path = active_embedding_dir()
EMB_NPY: Path = EMB_DIR / "embeddings.npy"
EMB_IDS: Path = EMB_DIR / "ids.jsonl"
EMB_META: Path = EMB_DIR / "meta.json"

# ------------------------------------------------------- KONSTANTA RETRIEVAL
TOP_K: int = 8
CONTEXT_CHAR_BUDGET: int = 12000

# ------------------------------------------------------- KONSTANTA RERANKER
# Pencarian 2 fase: fase-1 cosine ambil RERANK_POOL_SIZE kandidat, fase-2
# Qwen3-Reranker-0.6B (cross-encoder) menilai ulang pasangan (query, pasal).
# ON/OFF via env RERANK=1|0 (default aktif; fallback aman bila model tak ada).
RERANK: bool = os.environ.get("RERANK", "1") == "1"
RERANK_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
RERANK_POOL_SIZE: int = 12
RERANK_BATCH_SIZE: int = 4     # CPU-friendly
RERANK_MAX_LENGTH: int = 1024

# ------------------------------------------------------- KONSTANTA HYBRID
# Identik dengan folder tanpa-framework: dua jalur (DENSE + BM25) digabung
# Reciprocal Rank Fusion — hanya pakai PERINGKAT, bukan skala skor.
# ON/OFF via env HYBRID=1|0; index leksikal dibuat oleh bm25.py --build.
HYBRID: bool = os.environ.get("HYBRID", "1") == "1"
HYBRID_PER_JALUR: int = 12
RRF_K: int = 60
LEXICAL_DIR: Path = RESULT_DIR / "lexical"
LEXICAL_INDEX: Path = LEXICAL_DIR / "norma_tokens.txt"

# ---------------------------------------------------------------- LLM (.env)
_ENV_FILE: Path = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())


def get_llm_config() -> dict:
    """
    Kunci yang dibaca (sama dengan folder tanpa-framework):
      LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (+ temperature, max_tokens).
      LLM_BACKEND : "api" (default, ChatOpenAI-compatible) | "ollama"
                    ("ollama" memakai ChatOllama native, bukan /v1 endpoint).
    """
    return {
        "backend": os.environ.get("LLM_BACKEND", "api"),
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
    }
