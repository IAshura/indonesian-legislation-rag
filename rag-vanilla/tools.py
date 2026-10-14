"""
tools.py — "Tangan" yang boleh dipakai LLM: 4 fungsi + skema resminya.
======================================================================

PRINSIP (dari panduan, bab 6)
  Tool dibuat SEMPIT & BERTANGGUNG JAWAB. LLM TIDAK PERNAH menyentuh
  database — ia hanya MENGIRIM PERMINTAAN (nama fungsi + argumen JSON),
  KODE INI yang mengeksekusi pencarian sesungguhnya dan mengirim balik
  hasil JSON. Sitasi dalam hasil berasal dari dataset, bukan dari LLM.

STRUKTUR
  TOOLS        : daftar skema (format OpenAI — dipahami juga Qwen3/Ollama).
                 "description" yang bagus = prompt bagi LLM: tulis jelas
                 kapan tool dipakai & apa yang dikembalikannya.
  execute_tool : dispatcher — validasi argumen LLM (bisa salah/kasar!),
                 panggil fungsi asli, bungkus hasil jadi JSON string.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):   # teks pasal bisa memuat karakter
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-cp1252

import config
from dataset import dataset
from reranker import rerank_kandidat
from retriever import ExactRetriever, HybridRetriever, konteks_kaya
from embedder import embed_query

DOC_TYPES = ["UUD", "TAP MPR", "UU", "PERPPU", "PP", "PERPRES",
             "PERMEN", "PERLEMBAGA", "PERBAWASLU", "KEPPRES"]

# ------------------------------------------------------------- skema tools
TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_norma",
        "description": "Mencari pasal/norma peraturan Indonesia paling relevan "
                       "dengan pertanyaan hukum. Mengembalikan daftar pasal "
                       "(teks pasal utuh) + sitasi resmi + skor relevansi. "
                       "Selalu mulai dari tool ini.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "pertanyaan hukum dalam Bahasa Indonesia"},
            "doc_type": {"type": "string", "enum": DOC_TYPES,
                         "description": "batasi ke jenis peraturan tertentu"},
            "year": {"type": "integer",
                     "description": "batasi ke tahun peraturan"},
            "top_k": {"type": "integer", "default": 5,
                      "description": "jumlah pasal yang dikembalikan (1-10)"},
            "rerank": {"type": "boolean", "default": True,
                       "description": "urutkan ulang kandidat dengan "
                                      "reranker cross-encoder (lebih akurat, "
                                      "sedikit lebih lambat)"},
            "hybrid": {"type": "boolean", "default": True,
                       "description": "gabungkan pencarian semantik + BM25 "
                                      "kata kunci (hybrid RRF) — lebih akurat "
                                      "untuk nomor pasal, tahun, dan istilah "
                                      "teknis"}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "get_article",
        "description": "Mengambil teks pasal utuh berdasarkan parent_id dari "
                       "hasil search_norma, termasuk pasal lain di dalamnya. "
                       "Dipakai bila butuh konteks lebih dari potongan hasil cari.",
        "parameters": {"type": "object", "properties": {
            "parent_id": {"type": "string", "example": "uud-1945::p::1"}},
            "required": ["parent_id"]}}},
    {"type": "function", "function": {
        "name": "get_document",
        "description": "Mengambil metadata satu peraturan: judul resmi, jenis, "
                       "tahun, topik, posisi hirarki (legal_force_axis).",
        "parameters": {"type": "object", "properties": {
            "document_id": {"type": "string", "example": "uu-no-7-tahun-2017"}},
            "required": ["document_id"]}}},
    {"type": "function", "function": {
        "name": "check_validity",
        "description": "Memeriksa status berlaku satu peraturan dari file "
                       "validity.jsonl. JANGAN menebak status hukum: bila "
                       "hasilnya pending_enrichment, sampaikan bahwa status "
                       "belum terverifikasi.",
        "parameters": {"type": "object", "properties": {
            "document_id": {"type": "string"}},
            "required": ["document_id"]}}},
]

# ----------------------------------------------------- implementasi tool


def _search_norma(question: str, doc_type: str | None = None,
                  year: int | None = None, top_k: int = 5,
                  rerank: bool = True, hybrid: bool = True) -> dict:
    """Tool utama: pertanyaan -> vektor -> (hybrid|dense) search -> rerank."""
    top_k = max(1, min(int(top_k or 5), 10))       # CLAMP: LLM bisa minta 999
    vektor = embed_query(question)
    k_pool = max(top_k, config.RERANK_POOL_SIZE) if rerank else top_k
    if hybrid:
        # Hybrid: dense + BM25 -> RRF; fallback aman bila indeks leksikal
        # belum dibangun (sinkron dengan peringatan di HybridRetriever).
        hasil = HybridRetriever().search(
            vektor, question, k=k_pool, doc_type=doc_type, year=year)
    else:
        hasil = ExactRetriever().search(
            vektor, k=k_pool, doc_type=doc_type, year=year)
    kandidat = konteks_kaya(hasil)
    if rerank:
        # Fallback aman: model tak termuat -> urutan cosine dipakai kembali.
        kandidat = rerank_kandidat(question, kandidat, top_n=top_k)
    return {"jumlah": len(kandidat), "hasil": [
        {"parent_id": k["parent_id"],
         "skor": k.get("score_rerank", k["score"]),
         "skor_cosine": k["score"],
         "skor_fusi": k.get("skor_fusi"),
         "skor_bm25": k.get("skor_bm25"),
         "official_title": k["official_title"],
         "citation_str": k["citation_str"], "halaman": k["page"],
         "teks": k["text"][:600]}                    # RINGKAS: hemat token
        for k in kandidat]}


def _get_article(parent_id: str) -> dict:
    """Ambil teks pasal utuh/window + sitasi (join ke documents utk judul)."""
    ctx = dataset().context_for_parent(parent_id)   # KeyError = error jelas
    return {"parent_id": parent_id, "sumber": ctx["sumber"],
            "official_title": ctx["official_title"],
            "citation_str": ctx["citation_str"],
            "char_len": ctx["char_len"], "teks": ctx["text"]}


def _get_document(document_id: str) -> dict:
    d = dataset().get_document(document_id)
    if d is None:
        return {"error": f"dokumen tidak ditemukan: {document_id}"}
    return {"document_id": d["document_id"], "official_title": d["official_title"],
            "doc_type": d["doc_type"], "year": d.get("year"),
            "topic": d.get("topic"), "legal_force_axis": d.get("legal_force_axis")}


def _check_validity(document_id: str) -> dict:
    """
    Status hukum dari validity.jsonl hasil enrich_validity.py.
    File bisa berisi status kosong -> jawab JUJUR "pending_enrichment".
    Menebak status hukum = kesalahan fatal untuk sistem seperti ini.
    """
    import config
    path = config.RESULT_DIR / "validity.jsonl"
    if not path.exists():
        return {"document_id": document_id,
                "legal_status": "pending_enrichment",
                "catatan": "file validity.jsonl belum ada — status tidak diverifikasi"}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("document_id") == document_id:
            return {"document_id": document_id,
                    "legal_status": rec.get("legal_status") or "pending_enrichment"}
    return {"document_id": document_id, "legal_status": "pending_enrichment"}


# ------------------------------------------------------------- dispatcher
_FNS = {"search_norma": _search_norma, "get_article": _get_article,
        "get_document": _get_document, "check_validity": _check_validity}


def execute_tool(name: str, arguments: str | dict) -> str:
    """
    Jalankan SATU permintaan tool dari LLM.
      arguments : string JSON (dari API asli) ATAU dict (dari MockLLM).
    Kembalikan STRING JSON siap dikirim balik sebagai role="tool".
    Selalu tangkap error -> hasil JSON {"error": ...} agar LLM bisa
    menyesuaikan, bukan membuat seluruh sesi crash.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) \
            else dict(arguments or {})
        fn = _FNS.get(name)
        if fn is None:
            return json.dumps({"error": f"tool tidak dikenal: {name}"},
                              ensure_ascii=False)
        hasil = fn(**args)                      # Python akan error jika
        return json.dumps(hasil, ensure_ascii=False)  # argumen tidak cocok
    except TypeError as e:                      # argumen salah/kasar dari LLM
        return json.dumps({"error": f"argumen tidak valid: {e}",
                           "diterima": str(arguments)[:200]},
                          ensure_ascii=False)
    except Exception as e:                      # KeyError, file hilang, dsb.
        return json.dumps({"error": f"{type(e).__name__}: {e}"},
                          ensure_ascii=False)


if __name__ == "__main__":
    # `python tools.py` -> uji dispatcher langsung (tanpa LLM sama sekali)
    print(execute_tool("search_norma",
                       {"question": "kedaulatan di tangan siapa", "top_k": 2,
                        "rerank": False})[:400])   # cepat: tanpa rerank
    print(execute_tool("get_article", {"parent_id": "uud-1945::p::1"})[:300])
    print(execute_tool("get_document", {"document_id": "uud-1945"}))
    print(execute_tool("tool_ngawur", {}))          # -> error JSON rapi
