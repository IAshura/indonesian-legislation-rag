"""
loader.py — Jembatan: dataset proyek  <->  objek `Document` milik LangChain.
============================================================================

APA ITU Document DALAM LANGCHAIN?
  LangChain tidak menentukan "database harus begini". Hampir semua
  komponennya (retriever, vector store, chain) hanya mengenal dua hal:
      Document(page_content=teks, metadata={...})
  yaitu TEKS plus KAMUS metadata bebas. Karena dataset proyek ini sudah
  jauh lebih kaya dari sekadar teks (provenance halaman/baris, sitasi
  resmi, peran norma, axis hukum), yang kita lakukan di sini adalah
  MEMETAKAN record JSONL menjadi Document dengan metadata lengkap.

MENGAPA TIDAK MEMAKAI InMemoryVectorStore / FAISS?
  Embeddings 63.231 vektor SUDAH dihitung sebelumnya (Colab) dan disimpan
  di embeddings.npy — membuat vector store LangChain berarti me-re-embed
  semuanya (jam kerja GPU + memori). Jalan benar: bungkus pencarian yang
  sudah ada menjadi custom retriever (lihat retriever.py). Vector store
  LangChain baru masuk akal bila embedding dihitung ulang/dibangun ulang.
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.documents import Document

import config


def load_jsonl(path) -> list[dict]:
    """Baca JSONL menjadi list[dict] (file kecil-menengah)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@lru_cache(maxsize=1)
def dataset() -> "DatasetRingkas":
    """Singleton dataset (sama pola dengan folder tanpa-framework)."""
    return DatasetRingkas()


class DatasetRingkas:
    """Empat koleksi JSONL diindeks by-ID (lookup O(1), muat lazy)."""

    def __init__(self) -> None:
        self._idx: dict[str, dict[str, dict] | None] = {
            "chunks": None, "parents": None, "windows": None, "documents": None}

    def _load(self, name: str, path, key: str) -> dict[str, dict]:
        if self._idx[name] is None:
            self._idx[name] = {r[key]: r for r in load_jsonl(path)}
        return self._idx[name]

    @property
    def chunks(self):
        return self._load("chunks", config.CHUNKS_JSONL, "chunk_id")

    @property
    def parents(self):
        return self._load("parents", config.PARENTS_JSONL, "parent_id")

    @property
    def windows(self):
        return self._load("windows", config.PARENTS_WINDOWS_JSONL, "window_id")

    @property
    def documents(self):
        return self._load("documents", config.DOCUMENTS_JSONL, "document_id")


# ------------------------------------------------- record -> Document
def doc_konteks_dari_chunk(chunk_id: str, parent_id: str,
                           score: float,
                           skor_fusi: float | None = None,
                           skor_bm25: float | None = None) -> Document:
    """
    SMALL-TO-BIG dalam istilah LangChain:
      page_content = teks PASAL UTUH (parent) atau window-nya,
      metadata     = identitas chunk asal + skor + sitasi + provenance.
    LangChain tidak peduli text ini dari file mana — metadata yang
    menjaga auditabilitas.
    """
    ds = dataset()
    chunk = ds.chunks.get(chunk_id, {})
    parent = ds.parents.get(parent_id, {})
    ctx_id = parent_id
    sumber = "parents"
    if parent.get("char_len", 0) > 1800:          # pasal panjang -> window w01
        wid = f"{parent_id}::w01"
        win = ds.windows.get(wid)
        if win:
            ctx_id, teks, sumber = wid, win["text"], "parents_windows"
        else:
            teks = parent.get("text", "")
    else:
        teks = parent.get("text", chunk.get("text", ""))

    prov = chunk.get("provenance") or {}
    return Document(
        page_content=teks,
        metadata={
            "chunk_id": chunk_id, "parent_id": parent_id, "context_id": ctx_id,
            "sumber": sumber, "score": round(float(score), 4),
            "skor_fusi": skor_fusi,   # hybrid RRF (None bila jalur dense saja)
            "skor_bm25": skor_bm25,   # jalur leksikal (None bila bukan kostum)
            "document_id": chunk.get("document_id", parent.get("document_id")),
            "official_title": parent.get("official_title",
                                         chunk.get("official_title", "")),
            "citation_str": chunk.get("citation_str") or parent.get("citation_str", ""),
            "doc_type": chunk.get("doc_type"), "year": chunk.get("year"),
            "content_role": chunk.get("content_role"),
            "page": prov.get("page_number"),
            "lines": f"{prov.get('line_start')}-{prov.get('line_end')}"
                     if prov.get("line_start") else None,
        })


def doc_pasal(parent_id: str) -> Document:
    """Document satu pasal utuh (untuk tool get_article)."""
    ctx = dataset().parents.get(parent_id)
    if ctx is None:
        raise KeyError(f"parent tidak ditemukan: {parent_id}")
    return Document(page_content=ctx["text"], metadata={
        "parent_id": parent_id, "sumber": "parents",
        "official_title": ctx.get("official_title", ""),
        "citation_str": ctx.get("citation_str", ""),
        "char_len": ctx.get("char_len")})


def doc_peraturan(document_id: str) -> Document | None:
    """Document metadata peraturan (untuk tool get_document)."""
    d = dataset().documents.get(document_id)
    if d is None:
        return None
    return Document(page_content="", metadata={k: d.get(k) for k in
                    ("document_id", "official_title", "doc_type", "doc_type_full",
                     "number", "year", "topic", "legal_force_axis")})


if __name__ == "__main__":
    # `python loader.py` -> lihat bentuk Document yang dihasilkan
    d = doc_konteks_dari_chunk("uud-1945::p::1::02", "uud-1945::p::1", 0.99)
    print("page_content:", d.page_content[:120], "...")
    for k, v in d.metadata.items():
        print(f"  metadata[{k}] = {v}")
    print("\ndoc_pasal  :", doc_pasal("uud-1945::p::1").metadata)
    print("doc_peraturan:", doc_peraturan("uud-1945").metadata)
