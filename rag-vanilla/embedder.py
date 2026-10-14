"""
embedder.py — Mengubah PERTANYAAN pengguna menjadi VEKTOR (1024 dim).
=====================================================================

MENGAPA FILE INI PENTING?
  Pencarian vektor hanya bermakna jika PERTANYAAN dan DOKUMEN diubah oleh
  MODEL YANG SAMA — itulah "aturan besi" proyek ini (model index = model
  query). Index dibangun dengan Qwen3-Embedding-0.6B (dim 1024, pooling
  last_token, L2-normal), maka di sini query DIPERLAKUKAN SAMA PERSIS.
  Salah satu saja (mis. salah format instruksi) -> skor turun drastis dan
  hasil pencarian tidak bisa dipercaya.

TIGA DETAIL TEKNIS YANG MEMBEDAKAN QWEN3-EMBEDDING
  1. FORMAT INSTRUKSI: query DIBUNGKUS
        "Instruct: {tugas}\nQuery: {teks}"
     (dokumen saat indexing TIDAK diberi prefix ini). Instruksi dari
     meta.json — diambil langsung dari sana, tidak di-hardcode.
  2. POOLING last_token: Qwen3-Embedding adalah model DECODER (seperti LLM).
     Representasi kalimat diambil dari vektor TERAKHIR yang diperhatikan
     (token EOS), bukan rata-rata semua token (mean-pooling gaya E5/BGE).
  3. LEFT PADDING: karena representasi diambil di posisi TERAKHIR, baris
     pendek diberi padding di KIRI agar token terakhir semua baris sejajar.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import numpy as np
import torch

import config

# Muat model/tokenizer SEKALI per proses (bukan per query) — loading model
# butuh beberapa detik; penyematan ulang tiap query akan memboroskan waktu.
_MODEL = None
_TOKENIZER = None
_META: dict | None = None


def load_meta() -> dict:
    """Baca meta.json embeddings aktif (sumber kebenaran parameter)."""
    global _META
    if _META is None:
        _META = json.loads(config.EMB_META.read_text(encoding="utf-8"))
    return _META


def _lazy_load():
    """Muat tokenizer + model sekali. local_files_only=True: mesin ini
    sudah punya cache HF (models--Qwen--Qwen3-Embedding-0.6B), jadi bekerja
    TANPA internet. Jika cache belum ada, ubah jadi False sekali saja."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    meta = load_meta()
    name = meta["model"]
    from transformers import AutoModel, AutoTokenizer  # import lokal: cepat saat di-import modul lain
    _TOKENIZER = AutoTokenizer.from_pretrained(name, local_files_only=True)
    _MODEL = AutoModel.from_pretrained(name, local_files_only=True)
    _MODEL.eval()  # mode inferensi: matikan dropout dsb.
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL.to(_DEVICE)
    print(f"[embedder] model {name} siap di {_DEVICE} "
          f"(dim {meta['dim']}, pooling {meta['pooling']})")


def embed_query(text: str) -> np.ndarray:
    """
    Teks pertanyaan -> vektor np.float32 (1024,) L2-normal.

    Langkah:
      1. Bungkus dengan instruksi retrieval (format khusus Qwen3).
      2. Tokenisasi dengan padding KIRI + truncation 1024 token.
      3. Forward pass -> hidden states.
      4. Ambil vektor token TERAKHIR tiap baris (last-token pooling).
      5. Normalisasi L2 -> dot-product = cosine similarity.
    """
    _lazy_load()
    meta = load_meta()

    # (1) format instruksi — persis seperti saat build index
    prompt = f"Instruct: {meta['query_instruction']}\nQuery: {text}"

    # (2) padding_side kiri: token terakhir nempel di posisi -1
    batch = _TOKENIZER([prompt], padding=True, truncation=True,
                       max_length=meta.get("max_len", 1024),
                       return_tensors="pt", padding_side="left")
    batch = {k: v.to(_MODEL.device) for k, v in batch.items()}

    with torch.no_grad():  # inferensi: tidak perlu graf gradien (hemat RAM)
        out = _MODEL(**batch)
        hidden = out.last_hidden_state          # (1, seq_len, 1024)

    # (4) last-token: ambil posisi paling kanan dari panjang tiap baris
    lengths = batch["attention_mask"].sum(dim=1) - 1          # index token terakhir
    idx = torch.arange(hidden.size(0), device=hidden.device)  # baris ke-i
    vec = hidden[idx, lengths]                                # (1, 1024)

    # .float() dulu: model ini mengeluarkan BFloat16 yang tidak didukung
    # numpy (TypeError) — konversi ke float32 sebelum keluar dari torch.
    vec = vec.float().cpu().numpy().astype(np.float32)[0]
    # (5) L2-normalisasi: panjang vektor dijadikan 1
    return vec / np.linalg.norm(vec)


if __name__ == "__main__":
    # `python embedder.py` -> uji cepat: satu query, cek dim & norma
    v = embed_query("Di tangan siapa kedaulatan berada menurut UUD 1945?")
    print("dim  :", v.shape)
    print("norma:", float(np.linalg.norm(v)))          # harus 1.0
    print("8 nilai pertama:", np.round(v[:8], 4))
