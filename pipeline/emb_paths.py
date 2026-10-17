"""Resolusi folder artefak embedding AKTIF untuk seluruh skrip pipeline.

Konvensi folder (per model — memungkinkan benchmark multi-model berdampingan):

    result/embeddings/
    ├── ACTIVE          ← satu baris: tag model aktif, mis. "qwen3-4b"
    ├── qwen3-4b/       ← embeddings.npy + ids.jsonl + meta.json (+ golden_report.json)
    ├── qwen3-06b/      ← eksperimen model ringan (Qwen3-Embedding-0.6B, dim 1024)
    └── <tag>/          ← eksperimen berikutnya (mis. bge-m3)

Skrip default (verify/insert/smoke/eval/embed_query) membaca ACTIVE, bukan
menebak folder — jadi berganti model = tulis satu baris di ACTIVE, bukan
edit banyak skrip. Aturan besi tetap: model index = model query.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
EMBEDDINGS_ROOT = HERE / "result" / "embeddings"


def active_embeddings_dir() -> Path:
    """Folder artefak model aktif — dari file ACTIVE (isi: tag model)."""
    active = EMBEDDINGS_ROOT / "ACTIVE"
    if active.exists():
        tag = active.read_text(encoding="utf-8").strip()
        if tag:
            d = EMBEDDINGS_ROOT / tag
            if d.is_dir():
                return d
    return EMBEDDINGS_ROOT          # fallback: artefak langsung di embeddings/
