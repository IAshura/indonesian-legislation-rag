"""
answer.py — ARSITEKTUR A: fixed-pipeline RAG (rekomendasi tahap awal).
==========================================================================

ALUR DETERMINISTIK (setiap langkah selalu jalan, urutannya tetap):
    pertanyaan -> embed query -> cari top-k -> join parent (small-to-big)
    -> susun prompt -> panggil LLM -> jawaban + sitasi

Ini arsitektur yang direkomendasikan dokumen panduan untuk dimulai:
    + 1 panggilan LLM saja (murah, cepat, mudah diaudit)
    + setiap tahap bisa dites terpisah
    - tidak fleksibel: LLM tidak bisa memutuskan "cari lagi dengan kata lain"
      (itu tugas Arsitektur B di agent.py / folder LangGraph).

CARA PAKAI
    python answer.py --question "Di tangan siapa kedaulatan menurut UUD?"
    python answer.py --question "..." --top-k 5 --doc-type UUD --dry-run
    python answer.py --question "..." --show-context   (lihat konteks penuh)
    python answer.py --question "..." --no-hybrid     (dense-only, DPR Umum)
    python answer.py --question "..." --no-rerank     (lewat fase reranker)
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse

import config
import prompt
from embedder import embed_query
from llm_client import chat, dry_run
from reranker import rerank_kandidat
from retriever import ExactRetriever, HybridRetriever, konteks_kaya


def jawab(question: str, top_k: int = config.TOP_K,
          doc_type: str | None = None, year: int | None = None,
          dry: bool = False, show_context: bool = False,
          rerank: bool = config.RERANK,
          hybrid: bool = config.HYBRID) -> dict:
    """
    Jalankan satu siklus RAG penuh. Kembalikan dict:
      {"jawaban": str, "kandidat": [...], "messages": [...], "mode": ...}
    """
    # --- TAHAP 1-2: pertanyaan menjadi vektor query (model = model index!)
    vektor = embed_query(question)

    # --- TAHAP 3: pencarian, hanya part "norma" (filter sudah terlanjur
    #     diterapkan saat build: embeddings hanya memuat norma).
    #     HYBRID (default): dense cosine + BM25 leksikal digabung RRF
    #     (pasal yang benar tapi cosine-nya rendah bisa naik berkat skor
    #     BM25 kata-kata persisnya). Bila indeks leksikal belum dibangun
    #     (bm25.py --build), keduanya otomatis melempar ke dense-only.
    #     Bila rerank aktif, ambil pool lebih besar lagi dulu: fase rerank
    #     cross-encoder yang akhirnya menilai ulang pasangan (query, pasal).
    if hybrid:
        k_pool = max(top_k, config.RERANK_POOL_SIZE) if rerank else top_k
        hasil = HybridRetriever().search(
            vektor, question, k=k_pool, doc_type=doc_type, year=year)
    else:
        k_pool = max(top_k, config.RERANK_POOL_SIZE) if rerank else top_k
        hasil = ExactRetriever().search(
            vektor, k=k_pool, doc_type=doc_type, year=year)
    kandidat = konteks_kaya(hasil)

    if not kandidat:
        return {"jawaban": "Tidak ditemukan dasar dalam korpus.",
                "kandidat": [], "messages": [], "mode": "kosong"}

    # --- TAHAP 3b (opsional): reranker cross-encoder Qwen3-Reranker-0.6B.
    #     Gagal memuat model -> kandidat kembali apa adanya (fallback aman).
    if rerank:
        kandidat = rerank_kandidat(question, kandidat, top_n=top_k)

    # --- TAHAP 4-5: konteks ber-sitasi -> pesan lengkap
    blok = prompt.build_context_block(kandidat)
    messages = prompt.build_messages(question, blok)

    if show_context or dry:
        print(prompt.daftar_sumber(kandidat))
        print(f"\n(konteks {len(blok):,} char)")

    # --- TAHAP 6: panggil LLM (atau cetak payload di mode dry-run)
    if dry:
        print(dry_run(messages))
        return {"jawaban": "(dry-run: tidak memanggil LLM)",
                "kandidat": kandidat, "messages": messages, "mode": "dry-run"}

    pesan = chat(messages)
    return {"jawaban": pesan.get("content") or "",
            "kandidat": kandidat, "messages": messages, "mode": "live"}


# --------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="RAG fixed-pipeline — Peraturan Indonesia")
    ap.add_argument("--question", "-q", required=True, help="pertanyaan hukum")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--doc-type", default=None,
                    help="filter jenis: UUD, TAP MPR, UU, PP, PERPRES, ...")
    ap.add_argument("--year", type=int, default=None, help="filter tahun")
    ap.add_argument("--no-rerank", action="store_true",
                    help="matikan reranker (urutan cosine asli, lebih cepat)")
    ap.add_argument("--no-hybrid", action="store_true",
                    help="matikan hybrid: dense-only (cosine) saja")
    ap.add_argument("--dry-run", action="store_true",
                    help="cetak prompt saja, tidak panggil LLM (tanpa API)")
    ap.add_argument("--show-context", action="store_true",
                    help="tampilkan sumber & ukuran konteks")
    args = ap.parse_args()

    hasil = jawab(args.question, top_k=args.top_k, doc_type=args.doc_type,
                  year=args.year, dry=args.dry_run,
                  show_context=args.show_context,
                  rerank=not args.no_rerank, hybrid=not args.no_hybrid)

    print("\n" + "=" * 72)
    print("JAWABAN" if hasil["mode"] == "live" else "HASIL")
    print("=" * 72)
    print(hasil["jawaban"])
    if hasil["kandidat"] and hasil["mode"] == "live":
        print("\nSumber yang disuntikkan ke konteks:")
        print(prompt.daftar_sumber(hasil["kandidat"]))


if __name__ == "__main__":
    main()
