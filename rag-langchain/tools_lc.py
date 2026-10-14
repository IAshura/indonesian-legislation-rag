"""
tools_lc.py — Tools gaya LangChain: decorator `@tool`.
=====================================================

PERBEDAAN DENGAN tools.py (folder tanpa-framework)
  Tanpa framework: kita MENULIS skema JSON tool satu per satu (nama,
  description, parameters). Dengan LangChain: tulis fungsi Python biasa
  + docstring, beri dekorator @tool — SKEMA DIKELUARKAN OTOMATIS dari
  tipe anotasi (type hints) dan docstring:

      @tool
      def search_norma(question: str, top_k: int = 5) -> dict:
          "Cari pasal ... Pertanyaan hukum wajib diisi."
      -> search_norma.tool.get_input_schema()  = JSON Schema siap pakai

  Docstring bukan basa-basi: deskripsi itulah yang DIBACA LLM untuk
  memutuskan kapan tool dipakai. Tulis sejelas deskripsi di panduan bab 6.

  Fungsi tetap BERTANGGUNG JAWAB: LLM tidak menjalankan apa pun —
  ToolNode di agent_graph.py yang memanggil fungsi ini dengan
  argumen hasil json.loads dari permintaan LLM. Validasi argumen salah
  tetap ditangani (ToolNode menangkap error dan mengirim balik ke LLM).
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

import json

from langchain_core.tools import tool

from loader import doc_pasal, doc_peraturan, dataset
from retriever import LegislationRetriever
from bm25 import BM25Scorer

# Retriever dipakai bersama antar-pemanggilan tool (model embedding hanya
# dimuat sekali per proses — lihat retriever.py).
_retriever: LegislationRetriever | None = None


def _get_retriever() -> LegislationRetriever:
    global _retriever
    if _retriever is None:
        # Pool fase-1 lebih besar agar reranker punya bahan menyaring.
        _retriever = LegislationRetriever(k=config.RERANK_POOL_SIZE)
    return _retriever


# ------------------------------------------------------------- TOOL 1
@tool
def search_norma(question: str, doc_type: str | None = None,
                 year: int | None = None, top_k: int = 5,
                 rerank: bool = True, hybrid: bool = True) -> dict:
    """Cari pasal/norma peraturan Indonesia paling relevan dengan pertanyaan
    hukum. Pencarian hybrid: semantik (dense) + kata kunci (BM25) digabung
    RRF — set hybrid=false untuk dense-only; kandidat lalu diurutkan ulang
    oleh reranker cross-encoder Qwen3 (set rerank=false untuk urutan asli).
    Mengembalikan daftar pasal (teks utuh) + sitasi resmi + skor.
    Selalu mulai dari tool ini sebelum menjawab."""
    retriever = _get_retriever()
    if not hybrid:
        retriever = LegislationRetriever(k=config.RERANK_POOL_SIZE,
                                         hybrid=False)
    docs = retriever.invoke(question)
    if rerank:                            # fallback aman bila model tak termuat
        from reranker import QwenReranker
        docs = QwenReranker(top_n=max(1, min(top_k, 10))).rerank(question, docs)
    kandidat = docs[:max(1, min(top_k, 10))]
    hasil = []
    for d in kandidat:
        m = d.metadata
        if doc_type and m.get("doc_type") != doc_type:
            continue
        if year and int(m.get("year") or 0) != int(year):
            continue
        hasil.append({"parent_id": m["parent_id"],
                      "skor": m.get("score_rerank", m["score"]),
                      "skor_cosine": m["score"],
                      "skor_fusi": m.get("skor_fusi"),
                      "skor_bm25": m.get("skor_bm25"),
                      "official_title": m["official_title"],
                      "citation_str": m["citation_str"],
                      "halaman": m.get("page"),
                      "teks": d.page_content[:600]})   # ringkas utk hemat token
    return {"jumlah": len(hasil), "hasil": hasil}


# ------------------------------------------------------------- TOOL 2
@tool
def get_article(parent_id: str) -> dict:
    """Ambil teks pasal utuh berdasarkan parent_id dari hasil search_norma
    (termasuk ayat-ayat lain dalam pasal yang sama)."""
    d = doc_pasal(parent_id)                      # KeyError = error jelas
    m = d.metadata
    return {"parent_id": parent_id, "sumber": m["sumber"],
            "official_title": m["official_title"],
            "citation_str": m["citation_str"],
            "char_len": m["char_len"], "teks": d.page_content}


# ------------------------------------------------------------- TOOL 3
@tool
def get_document(document_id: str) -> dict:
    """Ambil metadata satu peraturan: judul resmi, jenis, tahun, topik,
    posisi hirarki hukum (legal_force_axis)."""
    d = doc_peraturan(document_id)
    if d is None:
        return {"error": f"dokumen tidak ditemukan: {document_id}"}
    return d.metadata


# ------------------------------------------------------------- TOOL 4
@tool
def check_validity(document_id: str) -> dict:
    """Cek status berlaku peraturan dari validity.jsonl. JANGAN menebak
    status hukum: bila hasil pending_enrichment, sampaikan bahwa status
    belum terverifikasi."""
    import config
    path = config.RESULT_DIR / "validity.jsonl"
    if not path.exists():
        return {"document_id": document_id,
                "legal_status": "pending_enrichment",
                "catatan": "file validity.jsonl belum ada"}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            if rec.get("document_id") == document_id:
                return {"document_id": document_id,
                        "legal_status": rec.get("legal_status")
                                        or "pending_enrichment"}
    return {"document_id": document_id, "legal_status": "pending_enrichment"}


# Empat objek @tool ini yang di-bind ke LLM / dimasukkan ke ToolNode.
TOOLS = [search_norma, get_article, get_document, check_validity]


if __name__ == "__main__":
    # `python tools_lc.py` -> lihat skema otomatis + uji fungsi asli
    t = search_norma
    print("nama tool :", t.name)
    print("deskripsi :", t.description[:80], "...")
    print("skema args:", json.dumps(t.args_schema.model_json_schema()
                                     ["properties"], ensure_ascii=False)[:300])
    print("\nuji get_document:", get_document.invoke({"document_id": "uud-1945"}))
    print("uji tool salah :", get_article.invoke({"parent_id": "x::y::z"})[:80])
