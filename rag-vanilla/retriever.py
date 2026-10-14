"""
retriever.py — Mesin pencarian: pertanyaan -> kandidat pasal paling relevan.
============================================================================

TIGA JALUR PENCARIAN, SATU BENTUK HASIL
  1. ExactRetriever (DEFAULT): pencarian exact dengan numpy.
     Kelebihan: deterministik, tanpa server, tanpa indeks ANN — cocok untuk
     menguji kebenaran (baseline) dan untuk dataset 63 ribu vektor ini
     tetap cepat (<50 ms). Vektor dimuat dengan mmap (memory-mapping):
     numpy hanya membaca bagian file yang benar-benar dipakai.
  2. milvus_search() (OPSI): jalur produksi yang sama dengan eval proyek
     (HNSW, metric IP, ef=64). Dipakai bila Docker Milvus berjalan.
     Jika pymilvus/server tidak tersedia, fungsi melempar error yang jelas.
  3. hybrid_search() (HYBRID): DENSE + BM25 leksikal digabung RRF
     (Reciprocal Rank Fusion — lihat bm25.py untuk alasannya). Dense kuat
     di makna, BM25 kuat di identifier persis (nomor pasal, tahun, jenis
     dokumen); fusi peringkat mengambil kekuatan keduanya.

HASIL SERAGAM: list[dict] dengan kunci
  chunk_id, parent_id, score + semua metadata penting dari ids.jsonl
  (doc_type, year, content_role, ...) sehingga lapisan atas (prompt/agent)
  tidak perlu tahu jalur pencarian mana yang dipakai. Kandidat hasil hybrid
  tambahan membawa skor_bm25 + skor_fusi (score tetap = cosine, 0.0 bila
  tidak ikut jalur dense).

MENGAPA IP (inner product) = COSINE di proyek ini?
  Semua vektor sudah L2-normal (panjang = 1), sehingga dot product dua
  vektor = cosine similarity. Milvus metric="IP" + vektor normal = cosine.

HASIL SERAGAM: list[dict] dengan kunci
  chunk_id, parent_id, score + semua metadata penting dari ids.jsonl
  (doc_type, year, content_role, ...) sehingga lapisan atas (prompt/agent)
  tidak perlu tahu jalur pencarian mana yang dipakai.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import numpy as np

import config
from dataset import dataset
from bm25 import BM25Scorer

# ---------------------------------------------------------- jalur exact


class ExactRetriever:
    """Pencarian exact penuh (brute-force cosine) di atas embeddings.npy."""

    def __init__(self) -> None:
        meta = json.loads(config.EMB_META.read_text(encoding="utf-8"))
        self.dim: int = meta["dim"]
        self.count: int = meta["count"]
        # mmap_mode="r": file TIDAK dimuat penuh ke RAM; dibaca per-slot.
        self.matrix = np.load(config.EMB_NPY, mmap_mode="r")
        assert self.matrix.shape == (self.count, self.dim), (
            f"bentuk npy {self.matrix.shape} != meta ({self.count}, {self.dim})")

        # ids.jsonl baris-i = metadata vektor baris-i (urutan harus sama)
        self.ids: list[dict] = []
        with open(config.EMB_IDS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.ids.append(json.loads(line))
        assert len(self.ids) == self.count, "ids.jsonl tidak sinkron dengan npy"

    def search(self, query_vec: np.ndarray, k: int = config.TOP_K,
               doc_type: str | None = None, year: int | None = None,
               unique_parent: bool = True) -> list[dict]:
        """
        Kembalikan top-k kandidat.

        unique_parent=True  : dari beberapa ayat dalam pasal yang sama,
                              hanya skor tertinggi yang lolos (small-to-big
                              tidak menyuntik pasal yang sama berkali-kali).
        doc_type / year     : filter opsional, mis. hanya "UU" tahun 2017.
        """
        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[0] != self.dim:
            raise ValueError(f"dimensi query {q.shape[0]} != index {self.dim}")

        # EXACT: satu perkalian matriks. (63.231 x 1024) @ (1024,) -> skor semua
        # Baris diambil per-slice dari mmap; numpy yang mengelola caching.
        scores = np.asarray(self.matrix, dtype=np.float32) @ q

        urut = np.argsort(-scores)  # indeks skor tertinggi dulu

        hasil: list[dict] = []
        seen_parent: set[str] = set()
        for i in urut:
            meta = self.ids[int(i)]
            # --- filter metadata (dari ids.jsonl, tanpa buka chunks.jsonl)
            if doc_type and meta.get("doc_type") != doc_type:
                continue
            if year and int(meta.get("year") or 0) != year:
                continue
            if unique_parent:
                if meta["parent_id"] in seen_parent:
                    continue
                seen_parent.add(meta["parent_id"])

            hasil.append({"chunk_id": meta["chunk_id"],
                          "parent_id": meta["parent_id"],
                          "score": round(float(scores[int(i)]), 4), **meta})
            if len(hasil) >= k:
                break
        return hasil


# ---------------------------------------------------------- jalur Milvus

def milvus_search(query_vec: np.ndarray, k: int = config.TOP_K,
                  collection: str = "legislation_norma_v1",
                  method: str = "hnsw", ef: int = 64,
                  nprobe: int = 16) -> list[dict]:
    """
    Pencarian via Milvus lokal (Docker). Jalur yang SAMA dipakai eval proyek.
    Prasyarat: docker compose up (folder C:\\milvus) & koleksi sudah dimuat.
    Catatan jebakan: setelah insert, index & flush ASINKRON — tunggu sebelum
    mencari; di sini kita hanya MEMBACA index jadi aman.

    method: "hnsw" (graph ANN, param ef), "ivf" (klaster ANN, param nprobe),
    atau "flat" (brute-force exact, tanpa parameter). Metrik selalu IP
    (vektor L2-normal -> dot = cosine). Lihat pipeline/README-retrieval.md.
    """
    method = (method or "hnsw").strip().lower()
    if method == "ivf":
        search_params = {"metric_type": "IP",
                         "params": {"nprobe": max(1, int(nprobe))}}
    elif method == "flat":
        search_params = {"metric_type": "IP", "params": {}}
    else:
        method = "hnsw"
        search_params = {"metric_type": "IP",
                         "params": {"ef": max(1, int(ef))}}
    try:
        from pymilvus import MilvusClient
    except ImportError as e:
        raise RuntimeError("pymilvus belum terpasang: pip install pymilvus") from e

    try:
        client = MilvusClient(uri="http://localhost:19530")
    except Exception as e:
        raise RuntimeError(
            "Milvus tidak terjangkau di localhost:19530 — "
            "jalankan `cd C:\\milvus; docker compose up -d` dulu") from e
    if not client.has_collection(collection):
        raise RuntimeError(f"koleksi {collection} tidak ada — jalankan insert_milvus.py")

    try:
        res = client.search(
            collection_name=collection,
            data=[np.asarray(query_vec, dtype=np.float32).tolist()],
            limit=k,
            output_fields=["chunk_id", "parent_id", "doc_type", "year",
                           "content_role", "citation_str"],   # metadata utk sitasi
            search_params=search_params,
        )
    except Exception as e:
        msg = str(e)
        if "dimension mismatch" in msg or "dimension" in msg.lower():
            raise RuntimeError(
                "dimensi query != dimensi koleksi Milvus "
                "(aturan besi: model index = model query; "
                "cek file ACTIVE vs isi koleksi)") from e
        raise
    out: list[dict] = []
    for hit in res[0]:
        ent = hit["entity"]
        out.append({"chunk_id": ent["chunk_id"], "parent_id": ent["parent_id"],
                    "score": round(float(hit["distance"]), 4),
                    "citation_str": ent.get("citation_str", ""),
                    "doc_type": ent.get("doc_type"), "year": ent.get("year"),
                    "content_role": ent.get("content_role")})
    return out


# ---------------------------------------------------------- jalur hybrid

def fusi_rrf(daftar_jalur: list[list[int]], rrf_k: int = config.RRF_K
             ) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.
    daftar_jalur[j] = indeks baris terurut rank (rank-1 pertama) jalur j.
    skor_fusi(c) = SUM_j 1/(rrf_k + rank_j(c))  (rank mulai 1, bukan 0).
    Kembalikan [(baris, skor_fusi), …] urut turun."""
    skor: dict[int, float] = {}
    for jalur in daftar_jalur:
        for rank, baris in enumerate(jalur, start=1):
            skor[baris] = skor.get(baris, 0.0) + 1.0 / (rrf_k + rank)
        # (dok yang tak muncul di jalur j tak dapat kontribusi dari jalur itu)
    urut = sorted(skor.items(), key=lambda x: -x[1])
    return urut


class HybridRetriever:
    """
    Hybrid search: DENSE (ExactRetriever) + LEXICAL (BM25) -> RRF.

    Kenapa fusi peringkat dan bukan skor: skala skor kedua jalur tidak
    sebanding (cosine 0..1 vs BM25 0..20+). RRF cuma butuh peringkat,
    jadi bebas dari masalah kalibrasi.
    """

    def __init__(self) -> None:
        self.dense = ExactRetriever()
        # Muat BM25 sekali; gagal -> fallback dense penuh (fallback aman,
        # pola yang sama dengan reranker).
        try:
            self.bm25 = BM25Scorer()
        except FileNotFoundError as e:
            print(f"[hybrid] {e} — fallback ke dense-only")
            self.bm25 = None

    def search(self, query_vec: np.ndarray, query_text: str,
               k: int = config.TOP_K, doc_type: str | None = None,
               year: int | None = None) -> list[dict]:
        """Dua jalur -> RRF (per-rank, bukan skor) -> dedup parent -> top-k.

        PENTING urutan operasinya: rank murni tiap jalur dulu difusi RRF,
        dedup per-parent SETELAH fusi — kontribusi rank-1 BM25 atas pasal
        yang juga muncul di dense tidak boleh hilang."""
        n = config.HYBRID_PER_JALUR
        if self.bm25 is None:
            # index leksikal belum dibangun -> fallback penuh dense-only
            return self.dense.search(query_vec, k=k, doc_type=doc_type,
                                     year=year, unique_parent=True)

        # --- jalur 1: dense (cosine) ----------------------------------
        M = np.asarray(self.dense.matrix, dtype=np.float32)
        scores = M @ np.asarray(query_vec, dtype=np.float32)
        urut = np.argsort(-scores)
        meta_baris = self.dense.ids
        dense_urut: list[str] = []
        dense_skor: dict[str, float] = {}
        chunk_meta: dict[str, dict] = {}
        seen_parent_dense: set[str] = set()
        for i in urut:
            m = meta_baris[int(i)]
            if doc_type and m.get("doc_type") != doc_type:
                continue
            if year and int(m.get("year") or 0) != year:
                continue
            if m["parent_id"] in seen_parent_dense:
                continue
            seen_parent_dense.add(m["parent_id"])
            dense_urut.append(m["chunk_id"])
            dense_skor[m["chunk_id"]] = round(float(scores[int(i)]), 4)
            chunk_meta[m["chunk_id"]] = m
            if len(dense_urut) >= n:
                break

        # --- jalur 2: BM25 leksikal -----------------------------------
        lex = self.bm25.search(query_text, k=n)
        lex_ranks: dict[str, int] = {}
        lex_skor: dict[str, float] = {}
        for rank, (baris, s) in enumerate(lex, start=1):
            m = meta_baris[baris]
            if doc_type and m.get("doc_type") != doc_type:
                continue
            if year and int(m.get("year") or 0) != year:
                continue
            # rank per chunk terbaik (kalau dua baris BM25 parent sama,
            # ambil rank terkecil — BM25 menilai per-chunk)
            if m["chunk_id"] not in lex_ranks:
                lex_ranks[m["chunk_id"]] = rank
                lex_skor[m["chunk_id"]] = round(s, 4)

        # --- fusi RRF: SUM 1/(rrf_k + rank_jalur(c)) -------------------
        skor_fusi: dict[str, float] = {}
        for rank, cid in enumerate(dense_urut, start=1):
            skor_fusi[cid] = skor_fusi.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        for cid, rank in lex_ranks.items():
            skor_fusi[cid] = skor_fusi.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        fusi = sorted(skor_fusi.items(), key=lambda x: -x[1])

        # --- dedup per pasal utuh (parent) SETELAH fusi -> top-k -------
        hasil: list[dict] = []
        ds = dataset()
        seen_parent: set[str] = set()
        for cid, sf in fusi:
            if len(hasil) >= k:
                break
            m = chunk_meta.get(cid) or next(
                (x for x in meta_baris if x["chunk_id"] == cid), None)
            if m is None or m["parent_id"] in seen_parent:
                continue
            seen_parent.add(m["parent_id"])
            if cid in dense_skor:
                rec = {"chunk_id": cid, "parent_id": m["parent_id"],
                       "score": dense_skor[cid], **m}
            else:
                chunk = ds.get_chunk(cid) or {}
                rec = {"chunk_id": cid, "parent_id": m["parent_id"],
                       "score": 0.0, **m, **chunk}
            out = dict(rec)
            out["skor_bm25"] = lex_skor.get(cid)
            out["skor_fusi"] = round(sf, 6)
            hasil.append(out)
        return hasil


def retrieve(query_vec: np.ndarray, k: int = config.TOP_K,
             backend: str = "numpy", **filters) -> list[dict]:
    """Satu pintu pencarian: backend "numpy" (default), "hybrid",
    atau "milvus"."""
    if backend == "numpy":
        return ExactRetriever().search(query_vec, k=k, **filters)
    if backend == "hybrid":
        raise ValueError("backend hybrid butuh query_text — panggil "
                         "HybridRetriever().search(vec, text, ...) langsung")
    if backend == "milvus":
        return milvus_search(query_vec, k=k)
    raise ValueError(f"backend tidak dikenal: {backend}")


def konteks_kaya(hasil: list[dict]) -> list[dict]:
    """
    SMALL-TO-BIG: gabungkan tiap kandidat chunk dengan teks PARENT-nya.
    Output tiap item: skor + sitasi + teks pasal utuh/window — persis
    bentuk yang dibutuhkan prompt.py untuk blok konteks.
    """
    ds = dataset()
    out = []
    for r in hasil:
        chunk = ds.get_chunk(r["chunk_id"]) or {}
        ctx = ds.context_for_parent(r["parent_id"])
        prov = chunk.get("provenance") or {}
        out.append({
            "score": r["score"],
            "chunk_id": r["chunk_id"],
            "parent_id": r["parent_id"],
            "sumber": ctx["sumber"],
            "official_title": ctx["official_title"],
            "citation_str": ctx["citation_str"] or chunk.get("citation_str", ""),
            "skor_fusi": r.get("skor_fusi"),      # hybrid (RRF) bila ada
            "skor_bm25": r.get("skor_bm25"),      # jalur leksikal bila ada
            "page": prov.get("page_number"),
            "lines": (prov.get("line_start"), prov.get("line_end"))
                      if prov.get("line_start") else None,
            "text": ctx["text"],
        })
    return out


if __name__ == "__main__":
    # Demo tanpa LLM: embed satu pertanyaan -> cari (dense & hybrid) -> tampilkan
    from embedder import embed_query
    q = "Di tangan siapa kedaulatan berada menurut UUD 1945?"
    v = embed_query(q)
    print("=== DENSE saja (cosine) ===")
    for i, r in enumerate(konteks_kaya(ExactRetriever().search(v, k=3)), 1):
        print(f"\n#{i} skor={r['score']} {r['official_title']} — "
              f"{r['citation_str']} [{r['sumber']}]")
        print(r["text"][:220].replace("\n", " "))
    print("\n=== HYBRID (dense + BM25 -> RRF) ===")
    h = HybridRetriever().search(v, q, k=3)
    for i, r in enumerate(konteks_kaya(h), 1):
        print(f"\n#{i} fusi={r.get('skor_fusi')} bm25={r.get('skor_bm25')} "
              f"cosine={r['score']} {r['official_title']} — {r['citation_str']}")
        print(r["text"][:220].replace("\n", " "))
