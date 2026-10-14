"""
retriever.py — Custom RETRIEVER: membungkus pencarian proyek untuk LangChain.
=============================================================================

APA ITU RETRIEVER?
  Kontrak paling sederhana di LangChain: komponen yang menerima STRING
  pertanyaan dan mengembalikan list[Document]. Banyak komponen lain
  (LCEL chain, RAG graph, tool, dsb.) hanya mensyaratkan "sesuatu yang
  bisa di-invoke seperti ini" — jadi retrieval kita bisa disambung ke
  semuanya tanpa mengubah implementasinya.

CARA MEMBUATNYA
  Subclass `BaseRetriever` + implementasikan SATU method:
      _get_relevant_documents(query, *, run_manager) -> list[Document]
  (versi async `_aget_relevant_documents` opsional — kita pakai sync.)

  Field serializable (mis. k: int) dideklarasikan seperti field pydantic
  biasa; objek BESAR yang tidak layak diserialisasi (dataset, matriks)
  disimpan lewat PrivateAttr — pola resmi LangChain.

MENGAPA internal-nya memakai kode folder tanpa-framework secara konsep
  (embedder + exact search)? Karena itu aset proyek yang sudah terukur:
  embeddings.npy + aturan besi "model index = model query". Retriever
  hanyalah ADAPTOR — justru ini pesan utama panduan: framework tidak
  menggantikan retrieval kita, ia membungkusnya.
"""

from __future__ import annotations

import json

import numpy as np
from pydantic import PrivateAttr

import config
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from loader import doc_konteks_dari_chunk   # record -> Document small-to-big
from bm25 import BM25Scorer


def fusi_rrf(daftar_jalur: list[list[str]], rrf_k: int = config.RRF_K
             ) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (lihat bm25.py utk latar).
    daftar_jalur[j] = id kandidat terurut rank (rank = 1, 2, 3, ...) jalur j;
    skor_fusi(c) = SUM_j 1/(rrf_k + rank_j(c)). Kembalikan [(id, skor), ...]
    urut turun. Kandidat yang TIDAK muncul di salah satu jalur tetap dapat
    skor dari jalur yang lain — inilah kenapa pasal yang ditemukan BM25
    (yang oleh dense dilewatkan) tetap bisa muncul di atas."""
    skor: dict[str, float] = {}
    for jalur in daftar_jalur:
        for rank, cid in enumerate(jalur, start=1):
            skor[cid] = skor.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(skor.items(), key=lambda x: -x[1])


class QwenQueryEmbedder:
    """Menyematkan pertanyaan dengan model index (aturan besi)."""

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        meta = json.loads(config.EMB_META.read_text(encoding="utf-8"))
        self.meta = meta
        self.tok = AutoTokenizer.from_pretrained(meta["model"],
                                                 local_files_only=True)
        self.model = AutoModel.from_pretrained(meta["model"],
                                               local_files_only=True)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        import torch
        prompt = f"Instruct: {self.meta['query_instruction']}\nQuery: {text}"
        batch = self.tok([prompt], padding=True, truncation=True,
                         max_length=self.meta.get("max_len", 1024),
                         return_tensors="pt", padding_side="left")
        with torch.no_grad():
            out = self.model(**{k: v for k, v in batch.items()})
            h = out.last_hidden_state
            idx = torch.arange(h.size(0), device=h.device)
            lens = batch["attention_mask"].sum(dim=1) - 1
            vec = h[idx, lens].float().cpu().numpy()[0]
        return vec / np.linalg.norm(vec)


class LegislationRetriever(BaseRetriever):
    """
    Retriever HYBRID atas embeddings.npy + BM25 leksikal + join pasal utuh.

    Alur: query string -> (a) embed -> cosine top-N; (b) token BM25 -> top-N;
    lalu fusi RRF de-dupe-kandidat, ambil k teratas, & susun Document.
    Index leksikal belum dibangun / dimatikan via HYBRID=0 -> fallback
    penuh ke dense-only (perilaku persis sebelum hybrid ada).
    """

    k: int = config.TOP_K              # field pydantic biasa (serializable)
    hybrid: bool = config.HYBRID       # ON/OFF hybrid (env HYBRID=1|0)
    _searcher: object = PrivateAttr()  # matriks vektor + ids.jsonl
    _embedder: object = PrivateAttr()  # QwenQueryEmbedder (objek berat)
    _bm25: object = PrivateAttr()      # BM25Scorer (muat sekali)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # muat sekali saat retriever dibuat (bukan per pertanyaan)
        self._searcher = _ExactSearch()
        self._embedder = QwenQueryEmbedder()
        self._bm25 = None
        if self.hybrid:
            try:
                self._bm25 = BM25Scorer()
            except FileNotFoundError:
                # index leksikal belum dibangun -> silent fallback dense-only
                print("[hybrid] index leksikal tidak ada — dense-only")
                self._bm25 = None

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> list[Document]:
        """Kontrak wajib BaseRetriever: query string -> list[Document]."""
        vec = self._embedder.embed(query)
        if self._bm25 is None:
            return self._searcher.dokumen(vec, k=self.k)
        return self._hybrid_dokumen(vec, query)

    # ----------------------------------------------------------- hybrid
    def _hybrid_dokumen(self, vec: np.ndarray, query: str) -> list[Document]:
        """Dense + BM25 -> RRF -> dedup per pasal utuh -> top-k Document.

        PENTING urutan operasinya: DUA jalur dulu difusi RRF dengan rank
        murni masing-masing, dedup per-parent dilakukan SETELAH fusi.
        Kalau salah satu jalur mendudاپкаtan rank-1 untuk pasal yang juga
        muncul di jalur lain, kontribusi itu tak boleh hilang — fusi RRF
        justru bekerja dengan MENAMBAH skor dari dua rank."""
        n = config.HYBRID_PER_JALUR

        # -- jalur 1: dense (urutan cosine pas-chunk, rank = 1,2,3,...) --
        M = np.asarray(self._searcher.matrix, dtype=np.float32)
        scores = M @ vec
        urut = np.argsort(-scores)
        dense_urut: list[str] = []
        dense_ranks: dict[str, int] = {}
        dense_skor: dict[str, float] = {}
        chunk_meta: dict[str, dict] = {}
        seen_parent_dense: set[str] = set()
        for i in urut:
            m = self._searcher.ids[int(i)]
            if m["parent_id"] in seen_parent_dense:
                continue
            seen_parent_dense.add(m["parent_id"])
            dense_urut.append(m["chunk_id"])
            dense_ranks[m["chunk_id"]] = len(dense_urut)
            dense_skor[m["chunk_id"]] = float(scores[int(i)])
            chunk_meta[m["chunk_id"]] = m
            if len(dense_urut) >= n:
                break

        # -- jalur 2: BM25 leksikal (rank = 1,2,3,... pada baris ids) -----
        lex = self._bm25.search(query, k=n)
        lex_ranks: dict[str, int] = {}
        lex_skor: dict[str, float] = {}
        for rank, (baris, s) in enumerate(lex, start=1):
            m = self._searcher.ids[baris]
            # satukan rank per CHUNK_ID: beberapa baris BM25 bisa milik
            # pasal yang sama (parent sama) — ambil rank TERBAIK (terkecil).
            if m["chunk_id"] not in lex_ranks:
                lex_ranks[m["chunk_id"]] = rank
                lex_skor[m["chunk_id"]] = round(s, 4)

        # -- fusi RRF: rank murni kedua jalur dijumlahkan ------------------
        skor_fusi: dict[str, float] = {}
        for cid, rank in dense_ranks.items():
            skor_fusi[cid] = skor_fusi.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        for cid, rank in lex_ranks.items():
            skor_fusi[cid] = skor_fusi.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        fusi = sorted(skor_fusi.items(), key=lambda x: -x[1])

        # -- dedup per pasal utuh (parent) SETELAH fusi --------------------
        hasil: list[Document] = []
        seen_parent: set[str] = set()
        cid_by_meta = {m["chunk_id"]: m for m in self._searcher.ids}
        for cid, sf in fusi:
            if len(hasil) >= self.k:
                break
            m = chunk_meta.get(cid) or cid_by_meta.get(cid)
            if m is None or m["parent_id"] in seen_parent:
                continue
            seen_parent.add(m["parent_id"])
            hasil.append(doc_konteks_dari_chunk(
                cid, m["parent_id"], dense_skor.get(cid, 0.0),
                skor_fusi=round(float(sf), 6),
                skor_bm25=lex_skor.get(cid)))
        return hasil


class _ExactSearch:
    """Matriks embeddings (mmap) + filter metadata — versi ringkas."""

    def __init__(self) -> None:
        self.matrix = np.load(config.EMB_NPY, mmap_mode="r")
        self.ids = [json.loads(l) for l in open(config.EMB_IDS,
                                                encoding="utf-8") if l.strip()]

    def dokumen(self, vec: np.ndarray, k: int) -> list[Document]:
        scores = np.asarray(self.matrix, dtype=np.float32) @ vec
        urut = np.argsort(-scores)
        hasil, seen = [], set()
        for i in urut:
            m = self.ids[int(i)]
            if m["parent_id"] in seen:        # satu pasal cukup satu kali
                continue
            seen.add(m["parent_id"])
            hasil.append(doc_konteks_dari_chunk(
                m["chunk_id"], m["parent_id"], float(scores[int(i)])))
            if len(hasil) >= k:
                break
        return hasil


if __name__ == "__main__":
    # `python retriever.py` — butuh beberapa detik utk memuat model
    r = LegislationRetriever(k=3)
    for d in r.invoke("Di tangan siapa kedaulatan berada menurut UUD 1945?"):
        print(f"- {d.metadata['official_title']} — {d.metadata['citation_str']} "
              f"(fusi {d.metadata.get('skor_fusi')} bm25 "
              f"{d.metadata.get('skor_bm25')} cosine {d.metadata['score']})")
        print("  ", d.page_content[:150].replace("\n", " "))
