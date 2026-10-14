"""
bridge.py — Jembatan ke mesin RAG yang SUDAH ADA (tanpa menulis-ulang logika).
==============================================================================

Folder `rag-tanpa-framework/` memuat seluruh kecerdasan retrieval:
  embedder.py   : pertanyaan -> vektor (Qwen3-Embedding, instruksi dari meta.json)
  retriever.py  : ExactRetriever (cosine exact), HybridRetriever (dense+BM25->RRF),
                  konteks_kaya (small-to-big), fusi_rrf
  bm25.py       : BM25Scorer Okapi k1=1.2 b=0.75 + index leksikal
  reranker.py   : Qwen3-Reranker-0.6B cross-encoder (fallback aman)
  dataset.py    : chunks/parents/windows/documents (lookup by-ID, cache)
  prompt.py     : aturan sitasi keras [S1]/[S2] + blok konteks
  tools.py      : 4 tool agent (skema + execute_tool)
  llm_client.py : dry_run() payload printer + MockLLM

Prinsip file webapp: HANYA MEMBUNGKUS. Rumus BM25/RRF/rerank/small-to-big/
prompt tetap 100% di modul lama — tidak ada duplikasi di webapp.

MASALAH GANTI EMBEDDING
  config.py lama membaca file `pipeline/result/embeddings/ACTIVE` SAAT IMPORT
  (config.EMB_DIR dibind sekali). Jadi mengganti tag embedding = tulis file
  ACTIVE lalu `importlib.reload(config)` dan reset cache singleton di
  embedder (meta + model). Objek retriever lama harus dibuang (matrix lama).

API_KEY
  api_key pengguna TIDAK PERNAH di-log: hanya lewat memori proses (payload
  HTTP ke provider). File ini tidak menyentuhnya.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../Indonesian legislation .../
RAG_DIR = PROJECT_ROOT / "rag-tanpa-framework"
ENB_ROOT = PROJECT_ROOT / "pipeline" / "result" / "embeddings"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TAGS_VALID = {"qwen3-06b", "qwen3-4b"}


def active_tag() -> str:
    """Tag embedding aktif dari file ACTIVE (sumber kebenaran tunggal)."""
    return (ENB_ROOT / "ACTIVE").read_text(encoding="utf-8").strip()


def antre_reload_embedding(tag: str) -> dict:
    """
    Tulis file ACTIVE + reload modul lama agar import berikutnya memakai
    embeddings baru. Mengembalikan info: {tag, dim, count, warning}.
    Model 4b butuh GPU; 06b jalan di CPU/laptop.
    """
    tag = (tag or "").strip()
    if tag not in TAGS_VALID:
        raise ValueError(f"tag embedding tidak dikenal: {tag} (pilihan: {sorted(TAGS_VALID)})")

    info: dict = {}
    lama = active_tag()
    (ENB_ROOT / "ACTIVE").write_text(tag, encoding="utf-8")

    import config as mod_config
    importlib.reload(mod_config)          # EMB_DIR/EMB_NPY/EMB_META terbaharu

    import embedder as mod_embedder
    mod_embedder._META = None
    mod_embedder._MODEL = None
    mod_embedder._TOKENIZER = None
    try:                                   # kosongkan cache GPU bila ada
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    meta = mod_embedder.load_meta()
    hits = {"dim": meta.get("dim"), "count": meta.get("count"),
            "model": meta.get("model")}

    # Index leksikal (norma_tokens.txt) baris-nya sejajar embeddings.npy —
    # jumlah baris beda = index BM25 tidak cocok, beri peringatan (jangan matikan).
    try:
        lex_lines = sum(1 for line in open(mod_config.LEXICAL_INDEX, encoding="utf-8")
                        if line.strip())
        if lex_lines != hits["count"]:
            hits["warning"] = (f"jumlah baris leksikal ({lex_lines}) != embeddings "
                               f"({hits['count']}) — skor BM25 bisa meleset")
    except FileNotFoundError:
        hits["warning"] = "index leksikal norma_tokens.txt tidak ditemukan — hybrid fallback dense-only"

    if lama != tag:
        hits["switched"] = f"{lama} -> {tag}"
    hits["tag"] = tag
    hits["is_gpu_tag"] = tag == "qwen3-4b"
    return hits


# Garansi: modul `config` lama bisa di-import dari modul webapp mana pun.
import config                                    # noqa: E402  (setelah sys.path)
