"""
engine.py — Inti webapp: lapisan tipis membungkus modul lama + pelacak TRACE.
=============================================================================

Prinsip: tidak ada rumus retrieval yang ditulis-ulang. Semua perhitungan
(dense cosine, BM25 Okapi, fusi RRF, rerank cross-encoder, small-to-big,
prompt sistem-sitasi, loop agent) tetap 100% di modul `rag-vanilla/`.
Webapp hanya memilih jalur, mencatat trace, dan memetakan output.

PARAMETER BARU (v2):
  top_k    : jumlah kandidat yang dicari (1..50) — ukuran pool retrieval.
  inject_k : berapa dokumen yang DISUNTIK ke konteks LLM setelah fusi/rerank
             (default = top_k). Bila rerank aktif, rerank mengurutkan ulang
             pool lalu hanya inject_n teratas yang masuk prompt.
  doc_type : str ATAU list[str] — multi-jenis. Bila lebih dari 1 jenis,
             pencarian dijalankan per jenis lalu hasil digabung + dedup
             parent (retainer lama menerima 1 jenis per panggilan).

SINGLETON CACHE
  ExactRetriever / HybridRetriever / BM25Scorer / meta-ids dimuat sekali
  per proses (embedding matrix besar). Ganti embedding via
  bridge.antre_reload_embedding() WAJIB diikuti engine.reset_cache().
"""

from __future__ import annotations

import functools
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend"))

import bridge                      # noqa: E402
import config                      # noqa: E402
from embedder import embed_query   # noqa: E402
from dataset import dataset        # noqa: E402
from retriever import (ExactRetriever, HybridRetriever, konteks_kaya,  # noqa: E402
                         fusi_rrf, milvus_search)
from bm25 import BM25Scorer        # noqa: E402
from reranker import rerank_kandidat  # noqa: E402
from prompt import build_context_block, build_messages  # noqa: E402
import llm_provider                # noqa: E402


class Trace:
    """Perekam node trace (edge implisit: node i -> node i+1)."""

    def __init__(self, awal: int = 1) -> None:
        # awal: nomor node pertama (untuk LANJUTAN antar-request dalam
        # satu sesi Graf; default 1). Mis. awal=27 -> n27, n28, ...
        self.node: list[dict] = []
        self._dasar = max(1, int(awal or 1)) - 1

    def nomor_berikut(self) -> str:
        """Id node berikutnya (n27, n28, ...) — dipakai juga saat
        menggabungkan sub-trace agar Graf linear tanpa ulang."""
        return f"n{self._dasar + len(self.node) + 1}"

    def add(self, tipe: str, label: str, detail: dict | None = None) -> str:
        nid = self.nomor_berikut()
        self.node.append({"id": nid, "type": tipe,
                          "label": label, "detail": detail or {}})
        return nid

    def list(self) -> list[dict]:
        return self.node


# ------------------------------------------------------------- singleton
@functools.lru_cache(maxsize=1)
def _baris_ids_meta() -> tuple[dict, ...]:
    """Meta baris-i ids.jsonl (untuk jalur BM25-only). Tiap baris-i embeddings."""
    meta: list[dict] = []
    with open(config.EMB_IDS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return tuple(meta)


@functools.lru_cache(maxsize=1)
def _bm25() -> BM25Scorer:
    return BM25Scorer()


@functools.lru_cache(maxsize=1)
def _exact() -> ExactRetriever:
    return ExactRetriever()


@functools.lru_cache(maxsize=1)
def _hybrid() -> HybridRetriever:
    return HybridRetriever()


def reset_cache() -> None:
    """Wajib dipanggil setelah /api/embedding/switch agar singleton
    (matrix embeddings + idf BM25) dibangun ulang dari index aktif."""
    _exact.cache_clear()
    _hybrid.cache_clear()
    _bm25.cache_clear()
    _baris_ids_meta.cache_clear()


METODE_DENSE_VALID = ("exact", "hnsw", "ivf", "flat")


def _dense_cari(vektor, k: int, method: str, ef: int, nprobe: int,
                doc_type: str | None = None,
                year: int | None = None) -> list[dict]:
    """Kaki dense: exact (numpy brute-force) atau Milvus (hnsw/ivf/flat).

    Kembalian selalu bentuk ExactRetriever.search:
    [{chunk_id, parent_id, score, **meta_ids}] sehingga pemanggil
    (dedup parent, konteks_kaya, trace) tidak perlu berubah.
    """
    method = (method or "exact").strip().lower()
    if method == "exact":
        return _exact().search(vektor, k=k, doc_type=doc_type, year=year)
    # Milvus: ambil lebih (filter + dedup parent), petakan ke meta ids.jsonl
    # agar bentuknya identik dengan jalur exact (sitasi lengkap).
    mentah = milvus_search(vektor, k=max(k * 3, k + 10),
                           method=method, ef=ef, nprobe=nprobe)
    meta = {m["chunk_id"]: m for m in _exact().ids}
    hasil: list[dict] = []
    seen: set[str] = set()
    for h in mentah:
        m = meta.get(h["chunk_id"])
        if m is None:
            continue
        if doc_type and m.get("doc_type") != doc_type:
            continue
        if year and int(m.get("year") or 0) != year:
            continue
        if m["parent_id"] in seen:
            continue
        seen.add(m["parent_id"])
        hasil.append({"chunk_id": m["chunk_id"], "parent_id": m["parent_id"],
                      "score": h["score"], **m})
        if len(hasil) >= k:
            break
    return hasil


def _hybrid_cari(vektor, question: str, k: int, per_jalur: int,
                 method: str, ef: int, nprobe: int,
                 doc_type: str | None = None,
                 year: int | None = None) -> list[dict]:
    """Hybrid RRF di mana kaki dense memakai Milvus (bukan ExactRetriever).

    Dipakai hanya bila dense_method != "exact". Rumus fusi SAMA dengan
    HybridRetriever (fusi_rrf atas indeks baris, dedup parent setelah fusi);
    hanya sumber peringkat dense yang berbeda. Kembalian bentuk
    HybridRetriever.search: [{chunk_id, parent_id, score,
    skor_bm25, skor_fusi, **meta}].
    """
    ids = _exact().ids
    idx_of: dict[str, int] = {}
    for i, m in enumerate(ids):
        idx_of.setdefault(m["chunk_id"], i)
    mentah = milvus_search(vektor, k=per_jalur * 3,
                           method=method, ef=ef, nprobe=nprobe)
    dense_baris: list[int] = []
    dense_skor: dict[int, float] = {}
    for h in mentah:
        bi = idx_of.get(h["chunk_id"])
        if bi is None or bi in dense_baris:
            continue
        m = ids[bi]
        if doc_type and m.get("doc_type") != doc_type:
            continue
        if year and int(m.get("year") or 0) != year:
            continue
        dense_baris.append(bi)
        dense_skor[bi] = h["score"]
        if len(dense_baris) >= per_jalur:
            break
    lex = _bm25().search(question, k=per_jalur)
    lex_rank: dict[int, int] = {}
    lex_skor: dict[int, float] = {}
    for rank, (bi, s) in enumerate(lex, start=1):
        m = ids[bi]
        if doc_type and m.get("doc_type") != doc_type:
            continue
        if year and int(m.get("year") or 0) != year:
            continue
        if bi not in lex_rank:
            lex_rank[bi] = rank
            lex_skor[bi] = round(s, 4)
    fusi = fusi_rrf([dense_baris, list(lex_rank.keys())])
    skor_fusi = dict(fusi)
    ds = dataset()
    hasil: list[dict] = []
    for bi, sf in fusi:
        m = ids[bi]
        if bi in dense_skor:
            rec = {"chunk_id": m["chunk_id"], "parent_id": m["parent_id"],
                   "score": dense_skor[bi], **m}
        else:
            chunk = ds.get_chunk(m["chunk_id"]) or {}
            rec = {"chunk_id": m["chunk_id"], "parent_id": m["parent_id"],
                   "score": 0.0, **m, **chunk}
        rec["skor_bm25"] = lex_skor.get(bi)
        rec["skor_fusi"] = round(sf, 6)
        hasil.append(rec)
        if len(hasil) >= k:
            break
    return hasil


def _ringkas(t: str | None, n: int = 400) -> str:
    t = (t or "").strip()
    return t if len(t) <= n else t[:n] + " …"


def _types(doc_type) -> list[str]:
    """Normalisasi doc_type: str | list[str] | None -> list[str] bersih."""
    if doc_type is None:
        return []
    if isinstance(doc_type, str):
        return [doc_type] if doc_type.strip() else []
    return [d for d in (doc_type or []) if isinstance(d, str) and d.strip()]


_ISTILAH_HUKUM = (
    "pasal", "ayat", "undang", "peraturan", "hukum", "pidana", "perdata",
    "presiden", "menteri", "dpr", "kpu", "bawaslu", "pemilu", "pilkada",
    "kedaulatan", "warga", "hak", "kewajiban", "sanksi", "denda", "penjara",
    "gugatan", "pengadilan", "mahkamah", "konstitusi", "yudisial", "banding",
    "kasasi", "jaksa", "polisi", "notaris", "tanah", "waris", "nikah",
    "cerai", "pajak", "anggaran", "apbn", "desa", "otonomi", "negara",
    "pemerintah", "lembaga", "jabatan", "masa", "berlaku", "mencabut",
    "menetapkan", "dilarang", "wajib", "tata cara", "ketentuan", "bab",
    "bupati", "gubernur", "walikota", "mpr", "uu", "uud", "pp", "tap",
)
_SAPAAN_RINGAN = (
    "halo", "hai", "hello", "hi", "pagi", "siang", "sore", "malam",
    "assalamualaikum", "makasih", "terima kasih", "thanks", "thank you",
    "ok", "oke", "siap", "ya", "sip", "test", "tes", "ping", "apa kabar",
    "siapa kamu", "kamu siapa",
)


def butuh_cari(question: str) -> bool:
    """Tebakan murah (tanpa LLM): apakah pertanyaan butuh pencarian pasal?
    Sapaan/terima kasih singkat TANPA istilah hukum -> False (jawab
    langsung, hemat token). Ada istilah hukum ATAU pola sitasi
    (nomor/tahun/ayat) -> True. Selain itu default True (aman).
    """
    q = (question or "").strip().lower()
    if not q:
        return False
    if re.search(r"(pasal\s*\d|ayat\s*\(|nomor\s*\d+|tahun\s*\d{4}|no\.\s*\d+)",
                 q):
        return True
    if any(k in q for k in _ISTILAH_HUKUM):
        return True
    # bandingkan tanpa tanda baca agar "apa kabar?" cocok dengan "apa kabar"
    bersih = re.sub(r"[^\w\s]", "", q)
    pendek = len(bersih) <= 40
    if pendek and any(bersih == s or bersih.startswith(s + " ")
                      or bersih.endswith(" " + s)
                      or f" {s} " in f" {bersih} "
                      for s in _SAPAAN_RINGAN):
        return False
    return True


def _merge_best_per_parent(hasil: list[dict], kunci_skor: str) -> list[dict]:
    """Gabung hasil multi-jenis doc_type: satu pasal (parent) terbaik saja."""
    by_parent: dict[str, dict] = {}
    for r in hasil:
        pid = r["parent_id"]
        prev = by_parent.get(pid)
        if prev is None or (r.get(kunci_skor) or 0) > (prev.get(kunci_skor) or 0):
            by_parent[pid] = r
    out = list(by_parent.values())
    out.sort(key=lambda x: -(x.get(kunci_skor) or 0))
    return out


# ---------------------------------------------------------------- fixed mode
def run_fixed(question: str, *, retrieval: str = "hybrid",
              rerank: bool = True, top_k: int = 8,
              inject_k: int | None = None,
              doc_type: str | list[str] | None = None,
              year: int | None = None, dry_run: bool = False,
              llm: dict | None = None, resume_from: int | None = None,
              trace: Trace | None = None,
              riwayat_chat: list[dict] | None = None,
              dense_method: str = "exact", dense_ef: int = 64,
              dense_nprobe: int = 16, gaya: str = "standar",
              temperature: float | None = None,
              max_tokens: int | None = None,
              sumber: str = "user", nomor_awal: int = 1) -> dict:
    """
    Fixed pipeline (modul lama): embed -> retrieve -> (rerank) ->
    join-parent -> build-prompt -> LLM.
    gaya: ringkas | standar | rinci. temperature/max_tokens: None = env.
    sumber: "user" (pertanyaan manusia -> node user_ask) atau "llm"
    (sub-pencarian tool agent -> node llm_ask, agar terbedakan di Graf).
    nomor_awal: nomor sitasi pertama konteks (global per sesi chat — nomor
    yang sudah dipakai pesan sebelumnya tak boleh dipakai ulang).
    """
    t0 = time.perf_counter()
    trace = trace or Trace()
    tset = _types(doc_type)
    inject_n = max(1, int(inject_k or top_k))
    # pool retrieval: min RERANK_POOL_SIZE agar fase rerank bermakna
    per_jalur = max(top_k, config.RERANK_POOL_SIZE) if rerank else top_k

    metode = (dense_method or "exact").strip().lower()
    if metode not in METODE_DENSE_VALID:
        metode = "exact"
    settings_detail = {"retrieval": retrieval, "rerank": rerank, "top_k": top_k,
                       "inject_k": inject_n, "doc_type": tset or None,
                       "year": year, "dry_run": dry_run,
                       "dense_method": metode,
                       **({"dense_ef": dense_ef} if metode == "hnsw" else {}),
                       **({"dense_nprobe": dense_nprobe} if metode == "ivf" else {}),
                       "gaya": gaya,
                       **({"temperature": temperature}
                          if temperature is not None else {}),
                       **({"max_tokens": max_tokens}
                          if max_tokens is not None else {})}
    tipe_tanya = "llm_ask" if sumber == "llm" else "user_ask"
    awalan_tanya = "Tanya LLM: " if sumber == "llm" else "Pertanyaan: "
    if not trace.node:
        trace.add(tipe_tanya, f"{awalan_tanya}{_ringkas(question, 100)}",
                  {"mode": "fixed", "question": question,
                   "settings": settings_detail})
    if resume_from is not None and 1 < resume_from <= len(trace.node):
        trace.node = trace.node[:resume_from - 1]

    if retrieval == "bm25":
        # --- BM25-only: tanpa embed ------------------------------------
        t1 = time.perf_counter()
        scorer = _bm25()
        meta_baris = _baris_ids_meta()
        merged: dict[str, dict] = {}
        for t in (tset or [None]):
            hits = scorer.search(question, k=per_jalur)
            seen: set[str] = set()
            rank = 0
            for baris, s in hits:
                m = meta_baris[baris]
                if t and m.get("doc_type") != t:
                    continue
                if year and int(m.get("year") or 0) != year:
                    continue
                if m["parent_id"] in seen:
                    continue
                seen.add(m["parent_id"])
                rank += 1
                k = dict(m)
                k.update({"score": 0.0, "skor_bm25": round(s, 4),
                          "skor_fusi": None, "rank": rank})
                cid = m["chunk_id"]
                prev = merged.get(cid)
                if prev is None or k["skor_bm25"] > (prev.get("skor_bm25") or 0):
                    merged[cid] = k
        hasil = sorted(merged.values(), key=lambda x: -(x.get("skor_bm25") or 0))
        kandidat = konteks_kaya(hasil)
        trace.add("retrieve_bm25", f"BM25 leksikal top-{per_jalur}",
                  {"query": question, "top_k": per_jalur,
                   "detik": round(time.perf_counter() - t1, 3),
                   "hasil": [_trace_singkat(k, rank=i + 1)
                             for i, k in enumerate(kandidat)]})
    else:
        # --- embed query ------------------------------------------------
        t1 = time.perf_counter()
        trace.add("embed_query", f"Embed query ({bridge.active_tag()})",
                  {"query": question, "model": bridge.active_tag(),
                   "detik": round(time.perf_counter() - t1, 3)})
        vektor = embed_query(question)

        t2 = time.perf_counter()
        label_dense = {"exact": "Dense exact (numpy)", "hnsw": "Dense HNSW (Milvus)",
                       "ivf": "Dense IVF (Milvus)", "flat": "Dense exact (Milvus FLAT)"}[metode]
        if retrieval == "hybrid":
            hasil: list[dict] = []
            for t in (tset or [None]):
                if metode == "exact":
                    hret = _hybrid()
                    hasil.extend(hret.search(vektor, question, k=per_jalur,
                                             doc_type=t, year=year))
                else:
                    hasil.extend(_hybrid_cari(vektor, question, k=per_jalur,
                                             per_jalur=per_jalur, method=metode,
                                             ef=dense_ef, nprobe=dense_nprobe,
                                             doc_type=t, year=year))
            hasil = _merge_best_per_parent(hasil, "skor_fusi")
            kandidat = konteks_kaya(hasil)
            trace.add("retrieve_dense", f"{label_dense} top-{per_jalur}",
                      {"query": question, "index_method": metode,
                       "detik": round(time.perf_counter() - t2, 3),
                       "hasil": [_trace_singkat(k, rank=i + 1, cosine=k["score"])
                                 for i, k in enumerate(kandidat) if k.get("score")]})
            trace.add("retrieve_bm25", f"BM25 leksikal top-{per_jalur}",
                      {"query": question,
                       "hasil": [_trace_singkat(k, rank=i + 1,
                                                skor_bm25=k.get("skor_bm25"))
                                 for i, k in enumerate(kandidat) if k.get("skor_bm25")]})
            trace.add("fuse_rrf", f"Fusi RRF (K={config.RRF_K})",
                      {"rrf_k": config.RRF_K,
                       "hasil": [_trace_singkat(k, rank=i + 1,
                                                skor_fusi=k.get("skor_fusi"))
                                 for i, k in enumerate(kandidat)]})
        else:
            hasil = []
            for t in (tset or [None]):
                hasil.extend(_dense_cari(vektor, k=per_jalur, method=metode,
                                         ef=dense_ef, nprobe=dense_nprobe,
                                         doc_type=t, year=year))
            hasil = _merge_best_per_parent(hasil, "score")
            kandidat = konteks_kaya(hasil)
            trace.add("retrieve_dense",
                      f"{label_dense} top-{per_jalur}" +
                      (f" × {len(tset)} jenis" if len(tset) > 1 else ""),
                      {"query": question, "index_method": metode,
                       "detik": round(time.perf_counter() - t2, 3),
                       "hasil": [_trace_singkat(k, rank=i + 1, cosine=k["score"])
                                 for i, k in enumerate(kandidat)]})

    if not kandidat:
        return {"jawaban": "Tidak ditemukan dasar dalam korpus.",
                "kandidat": [], "trace": trace.list(),
                "usage": {"jumlah_kandidat": 0,
                          "detik": round(time.perf_counter() - t0, 3)},
                "mode": "kosong", "messages": []}

    # --- rerank (opsional): pool diurut ulang, hanya inject_n teratas ---
    if rerank:
        t3 = time.perf_counter()
        trace.add("rerank", "Rerank cross-encoder",
                  {"model": config.RERANK_MODEL, "pool": len(kandidat),
                   "top_n": inject_n})
        kandidat = rerank_kandidat(question, kandidat, top_n=inject_n)
        trace.node[-1]["detail"].update({
            "detik": round(time.perf_counter() - t3, 3),
            "hasil": [_trace_singkat(k, rank=i + 1,
                                     score_rerank=k.get("score_rerank"))
                      for i, k in enumerate(kandidat)]})

    # --- inject_k: batasi jumlah dokumen yang disuntik ke prompt ---------
    kandidat = kandidat[:inject_n]

    # --- join parent (small-to-big) --------------------------------------
    trace.add("join_parent", "Join parent (small-to-big)",
              {"kandidat": [{"parent_id": k["parent_id"],
                             "citation_str": k.get("citation_str"),
                             "official_title": k.get("official_title"),
                             "sumber": k.get("sumber"),
                             "char_len": len(k.get("text") or "")}
                            for k in kandidat]})

    # --- build prompt ------------------------------------------------------
    trace.add("build_prompt", "Build prompt",
              {"query": question, "anggaran_char": config.CONTEXT_CHAR_BUDGET})
    blok = build_context_block(kandidat, mulai=max(1, int(nomor_awal or 1)))
    messages = build_messages(question, blok, gaya=gaya)
    # --- memori: semua giliran selama muat budget 100rb token -----------
    n_memori, tok_memori = 0, 0
    if riwayat_chat:
        messages, n_memori, tok_memori = _sisip_riwayat(messages, riwayat_chat)
    trace.node[-1]["detail"]["prompt"] = {
        "system": messages[0]["content"], "context": blok,
        "user": messages[-1]["content"], "char_len": len(blok),
        "chat_memori": n_memori, "chat_memori_token": tok_memori}

    # --- LLM -----------------------------------------------------------------
    if dry_run:
        from llm_client import dry_run
        trace.add("llm_answer", "Dry-run (tidak memanggil LLM)",
                  {"payload": _ringkas(dry_run(messages), 3000)})
        jawaban = ("[DRY-RUN] Tidak memanggil LLM. Lihat konteks & prompt di "
                   "panel Graf (node build_prompt).")
        mode = "dry-run"
    else:
        trace.add("llm_answer", "LLM live",
                  {**_llm_detail(llm), "mode": "live"})
        mentah: list = []
        balasan = _chat_live(messages, llm, usage_out=mentah,
                             temperature=temperature, max_tokens=max_tokens)
        jawaban = (balasan.get("content") or "").strip() or "(jawaban kosong)"
        mentah_nalar = (balasan.get("reasoning") or "")[:6000]
        nalar = llm_provider.rapikan_nalar(balasan.get("reasoning"))
        trace.node[-1]["detail"]["jawaban_panjang"] = len(jawaban)
        tokens = llm_provider.normalisasi_usage(mentah[0] if mentah else None)
        trace.node[-1]["detail"]["tokens"] = tokens
        if nalar:
            trace.node[-1]["detail"]["penalaran"] = nalar
            trace.node[-1]["detail"]["penalaran_mentah"] = mentah_nalar
        mode = "live"

    return {"jawaban": jawaban, "kandidat": kandidat, "trace": trace.list(),
            "reasoning": nalar if mode == "live" else None,
            "reasoning_raw": mentah_nalar if mode == "live" and nalar else None,
            "usage": {"jumlah_kandidat": len(kandidat),
                      "detik": round(time.perf_counter() - t0, 3),
                      "context_chars": len(blok),
                      "tokens": tokens if mode == "live"
                      else llm_provider.normalisasi_usage(None)},
            "mode": mode, "messages": messages}


# ---------------------------------------------------------------- agent mode
def run_agent(question: str, *, retrieval: str = "hybrid",
              rerank: bool = True, top_k: int = 8,
              inject_k: int | None = None,
              doc_type: str | list[str] | None = None,
              year: int | None = None, dry_run: bool = False,
              llm: dict | None = None, trace: Trace | None = None,
              riwayat_chat: list[dict] | None = None,
              dense_method: str = "exact", dense_ef: int = 64,
              dense_nprobe: int = 16, gaya: str = "standar",
              temperature: float | None = None,
              max_tokens: int | None = None,
              paksa_cari: bool = True, nomor_awal: int = 1) -> dict:
    """Loop agent persis agent.py lama; tool search_norma diarahkan ke jalur UI.
    paksa_cari: wajibkan minimal 1× search_norma sebelum jawab final
    (mencegah jawaban dari memori tanpa sitasi valid).
    nomor_awal: nomor sitasi global pertama sesi (tak boleh pakai ulang)."""
    t0 = time.perf_counter()
    trace = trace or Trace()
    metode = (dense_method or "exact").strip().lower()
    if metode not in METODE_DENSE_VALID:
        metode = "exact"
    # paksa_cari hanya untuk pertanyaan HUKUM; sapaan ringan ("halo",
    # "makasih") tetap dijawab langsung tanpa tool (hemat token).
    wajib = bool(paksa_cari) and butuh_cari(question)
    if not trace.node:
        trace.add("user_ask", f"Pertanyaan: {_ringkas(question, 100)}",
                  {"mode": "agent", "question": question,
                   "settings": {"retrieval": retrieval, "rerank": rerank,
                                "top_k": top_k, "inject_k": inject_k,
                                "doc_type": _types(doc_type) or None,
                                "year": year, "dry_run": dry_run,
                                "dense_method": metode, "gaya": gaya,
                                "paksa_cari": paksa_cari,
                                "wajib_cari": wajib,
                                "nomor_awal": max(1, int(nomor_awal or 1))}})

    import tools as mod_tools

    def _execute_wrapped(nama: str, argumen_json: str) -> str:
        try:
            a = json.loads(argumen_json or "{}")
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"argumen bukan JSON valid: {e}"})
        # Nomor sitasi agent [S1]...: tiap hasil search_norma diberi nomor
        # URUT GLOBAL (lintas putaran). Nomor inilah yang WAJIB dipakai LLM
        # — bukan nomor karangan. Dikumpulkan ke semua_sumber untuk kartu
        # sitasi di UI (lihat _citations_dari_kandidat di app.py).
        if nama == "search_norma":
            sub = run_fixed(a.get("question") or question,
                            retrieval=retrieval, rerank=rerank,
                            top_k=int(a.get("top_k", 5) or 5),
                            inject_k=inject_k,
                            doc_type=(a.get("doc_type") or doc_type),
                            year=(a.get("year") or year),
                            dry_run=True, trace=Trace(),
                            dense_method=metode, dense_ef=dense_ef,
                            dense_nprobe=dense_nprobe, sumber="llm")
            kandidat = sub["kandidat"]
            out = _beri_nomor(kandidat)
            return json.dumps({"daftar": out,
                               "trace_reguler": sub["trace"][-3:]},
                              ensure_ascii=False)
        return mod_tools.execute_tool(nama, argumen_json)

    _execute_asli = mod_tools.execute_tool
    mod_tools.execute_tool = _execute_wrapped
    nomor_sumber = [max(1, int(nomor_awal or 1)) - 1]
    terlihat: set = set()
    semua_sumber: list = []

    def _beri_nomor(daftar: list) -> list:
        """Beri nomor sitasi global + kumpulkan (dipakai tool & fallback)."""
        keluar = []
        for k in daftar or []:
            if k.get("chunk_id") in terlihat:
                continue
            terlihat.add(k.get("chunk_id"))
            nomor_sumber[0] += 1
            item = {"n": nomor_sumber[0],
                    "chunk_id": k.get("chunk_id"),
                    "parent_id": k.get("parent_id"),
                    "citation_str": k.get("citation_str"),
                    "official_title": k.get("official_title"),
                    "page": k.get("page"), "lines": k.get("lines"),
                    "score": k.get("score"), "skor_bm25": k.get("skor_bm25"),
                    "skor_fusi": k.get("skor_fusi"),
                    "text": _ringkas(k.get("text"), 600)}
            keluar.append(item)
            semua_sumber.append(item)
        return keluar

    from prompt import instruksi_gaya as _gaya
    sistem = ("Kamu asisten hukum Indonesia. Gunakan tools yang tersedia "
              "untuk mencari pasal SEBELUM menjawab. Jawab hanya dari "
              "hasil tool. Setiap hasil tool SUDAH bernomor sitasi "
              "[Sn] sesuai field \"n\" tiap item daftar (nomor melanjutkan sesi, bisa mulai bukan dari 1) — "
              "pakai nomor itu PERSIS di akhir pernyataan hukummu, "
              "misal \"... rakyat [S1].\" DILARANG mengarang nomor "
              "sendiri dan DILARANG sitasi dalam kurung biasa. "
              "Jika tool tidak menemukan dasar, katakan jujur.\n"
              "DEKOMPOSISI: bila pertanyaan mengandung beberapa sub-pertanyaan "
              "berbeda, PECAH menjadi beberapa panggilan search_norma "
              "terpisah — satu per sub-pertanyaan, boleh sekaligus dalam "
              "satu blok tool_calls. JANGAN gabung menjadi satu query "
              "campuran. Contoh BENAR untuk pertanyaan 3-bagian: "
              "[search_norma({\"question\": \"Bentuk negara Indonesia "
              "menurut UUD 1945\"}), search_norma({\"question\": "
              "\"Pemegang kekuasaan membentuk UU menurut UUD 1945\"}), "
              "search_norma({\"question\": \"Jaminan beragama menurut "
              "UUD 1945\"})]. Contoh SALAH: satu search_norma dengan "
              "question gabungan ketiganya.\n"
              + _gaya(gaya))
    # wajib sudah dihitung di atas (pakai aturan butuh_cari).
    if wajib:
        sistem += ("\nATURAN TAMBAHAN: kamu WAJIB memanggil tool search_norma "
                   "minimal SATU KALI sebelum menjawab final, walaupun jawaban "
                   "tampaknya sudah ada di riwayat percakapan. Sitasi hanya "
                   "sah dari hasil tool putaran ini.")
    messages: list[dict] = [
        {"role": "system", "content": sistem},
        {"role": "user", "content": question},
    ]
    # --- memori: semua giliran selama muat budget 100rb token -----------
    if riwayat_chat:
        messages, _, _ = _sisip_riwayat(messages, riwayat_chat)

    MAX_ITERATIONS = 5
    jawaban = ""
    try:
        if dry_run:
            from llm_client import MockLLM
            llm_ob = MockLLM([
                {"tool_calls": [{"name": "search_norma",
                                 "arguments": {"question": question,
                                               "top_k": min(3, top_k)}}]},
                {"content": "[DRY-RUN agent] MockLLM menjawab final berdasarkan "
                            "hasil tool. Node tool_call di bawah mencatat "
                            "pencarian sungguhan yang dijalankan oleh kode."},
            ])
        else:
            llm_ob = None

        detail_extra: dict = {}
        tokens = llm_provider.normalisasi_usage(None)
        ulangi_kosong = 0
        nalar_semua: list[str] = []
        for putaran in range(1, MAX_ITERATIONS + 1):
            if llm_ob is None:
                mentah: list = []
                balasan = _chat_live(messages, llm, tools=mod_tools.TOOLS,
                                     usage_out=mentah, temperature=temperature,
                                     max_tokens=max_tokens)
                tokens = llm_provider.tambah_usage(
                    tokens, mentah[0] if mentah else None)
            else:
                balasan = llm_ob.chat(messages, tools=mod_tools.TOOLS)
            tool_calls = balasan.get("tool_calls")
            isi_balasan = (balasan.get("content") or "").strip()
            if balasan.get("reasoning"):
                nalar_semua.append(balasan["reasoning"])
            if not tool_calls and not isi_balasan and llm_ob is None:
                # Pengaman model yang balas kosong tanpa tool-call.
                if ulangi_kosong < 1:
                    ulangi_kosong += 1
                    continue
                sub = run_fixed(
                    question, retrieval=retrieval, rerank=rerank,
                    top_k=top_k, inject_k=inject_k,
                    doc_type=doc_type, year=year,
                    dry_run=dry_run, llm=llm, trace=Trace(),
                    dense_method=metode, dense_ef=dense_ef,
                    dense_nprobe=dense_nprobe, gaya=gaya,
                    temperature=temperature, max_tokens=max_tokens,
                    sumber="llm")
                jawaban = sub["jawaban"]
                # Petakan sitasi lokal fixed ([S1..] sesuai urutan
                # kandidatnya) ke nomor global agar klik cocok. Dua tahap
                # via placeholder agar [S1]->[S2] tak tertimpa [S2]->[S5].
                peta: dict[int, int] = {}
                for i, k in enumerate(sub["kandidat"] or [], 1):
                    temu = next((s for s in semua_sumber
                                 if s["chunk_id"] == k.get("chunk_id")), None)
                    if temu is None:
                        nomor_sumber[0] += 1
                        item = {"n": nomor_sumber[0],
                                "chunk_id": k.get("chunk_id"),
                                "parent_id": k.get("parent_id"),
                                "citation_str": k.get("citation_str"),
                                "official_title": k.get("official_title"),
                                "page": k.get("page"),
                                "lines": k.get("lines"),
                                "score": k.get("score"),
                                "text": _ringkas(k.get("text"), 600)}
                        semua_sumber.append(item)
                        peta[i] = nomor_sumber[0]
                    else:
                        peta[i] = temu["n"]
                for i, g in peta.items():
                    jawaban = re.sub(r"\[S" + str(i) + r"\]",
                                     f"\x00{g}\x00", jawaban)
                for g in peta.values():
                    jawaban = jawaban.replace(f"\x00{g}\x00", f"[S{g}]")
                tokens = llm_provider.tambah_usage(
                    tokens, (sub.get("usage") or {}).get("tokens"))
                trace.add("llm_answer",
                          f"Jawaban fallback fixed (putaran {putaran})",
                          {"teks": _ringkas(jawaban, 2000),
                           "jumlah_putaran": putaran, "tokens": tokens})
                break
            if not tool_calls:
                jawaban = isi_balasan or "(jawaban kosong)"
                trace.add("llm_answer", f"Jawaban final (putaran {putaran})",
                          {"teks": _ringkas(jawaban, 2000),
                           "jumlah_putaran": putaran, "tokens": tokens})
                break
            messages.append({"role": "assistant", "content": balasan.get("content"),
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                nama = tc["function"]["name"]
                argumen = tc["function"]["arguments"]
                t1 = time.perf_counter()
                try:
                    hasil_tool = mod_tools.execute_tool(nama, argumen)
                except Exception as e:
                    hasil_tool = json.dumps({"error": str(e)}, ensure_ascii=False)
                detail_extra = {"tool": nama, "argumen": _ringkas(argumen, 400),
                                "detik": round(time.perf_counter() - t1, 3)}
                try:
                    parsed = json.loads(hasil_tool)
                    if isinstance(parsed, dict) and parsed.get("trace_reguler"):
                        detail_extra["retrieval_subtrace"] = parsed["trace_reguler"]
                        detail_extra["daftar_hasil"] = parsed.get("daftar", [])
                        hasil_render = json.dumps({"daftar": parsed.get("daftar", [])},
                                                  ensure_ascii=False)
                    else:
                        hasil_render = hasil_tool
                except (json.JSONDecodeError, TypeError):
                    hasil_render = hasil_tool
                detail_extra["hasil"] = _ringkas(hasil_render, 2000)
                trace.add("tool_call", f"{nama} (putaran {putaran})", dict(detail_extra))
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": hasil_tool})
        else:
            # Batas putaran habis: JANGAN lempar pesan teknis ke user.
            # Satu panggilan pamungkas TANPA tool — jawab seadanya dari
            # hasil tool yang sudah terkumpul (atau "tidak ditemukan").
            messages.append({
                "role": "user",
                "content": "Batas pencarian tercapai. Jawab pertanyaan awal "
                           "SEADANYA dari hasil tool di atas dengan sitasi "
                           "[Sn] yang sudah ada. Jika tidak cukup untuk "
                           "menjawab, katakan: Tidak ditemukan dasar "
                           "dalam korpus."})
            if llm_ob is None:
                mentah2: list = []
                pamungkas = _chat_live(messages, llm, usage_out=mentah2,
                                       temperature=temperature,
                                       max_tokens=max_tokens)
                tokens = llm_provider.tambah_usage(
                    tokens, mentah2[0] if mentah2 else None)
                jawaban = ((pamungkas.get("content") or "").strip()
                           or "Tidak ditemukan dasar dalam korpus.")
            else:
                jawaban = ("Tidak ditemukan dasar dalam korpus "
                           "(batas putaran dry-run tercapai).")
            trace.add("llm_answer", "Jawaban pamungkas (batas putaran)",
                      {"teks": _ringkas(jawaban, 2000),
                       "jumlah_putaran": MAX_ITERATIONS, "tokens": tokens})
    finally:
        mod_tools.execute_tool = _execute_asli

    mentah_nalar = "\n\n".join(nalar_semua)[:6000]
    nalar = llm_provider.rapikan_nalar("\n\n".join(nalar_semua))
    if nalar:
        for n in reversed(trace.node):
            if n.get("type") == "llm_answer":
                n["detail"]["penalaran"] = nalar
                n["detail"]["penalaran_mentah"] = mentah_nalar
                break
    return {"jawaban": jawaban, "kandidat": semua_sumber,
            "trace": trace.list(), "reasoning": nalar,
            "reasoning_raw": mentah_nalar if nalar else None,
            "usage": {"detik": round(time.perf_counter() - t0, 3),
                      "tokens": tokens},
            "mode": "dry-run" if dry_run else "live", "messages": []}


# ------------------------------------------------------------------ helpers
def estimasi_token(teks: str | None) -> int:
    """Perkiraan kasar token teks (±4 char/token ala tokenizer GPT/Gemma).
    Bukan hitungan pabrik provider — hanya untuk budgeting memori internal.
    """
    return max(1, len(teks or "") // 4)


def _sisip_riwayat(messages: list[dict],
                   riwayat_chat: list[dict],
                   budget_token: int = 100_000) -> tuple[list, int, int]:
    """
    Sisipkan chat sebelumnya antara SYSTEM dan konteks/pertanyaan.
    riwayat_chat = [{"role": "user"|"assistant", "content": str}, ...]
    UNLIMITED giliran: ambil dari yang TERBARU mundur selama total
    (prompt saat ini + riwayat) masih muat dalam budget_token.
    Kembalian: (messages_baru, jumlah_pesan_dipakai, estimasi_token_memori).
    """
    dasar = sum(estimasi_token(m.get("content")) for m in messages)
    pakai: list[dict] = []
    total = dasar
    for m in reversed(riwayat_chat or []):
        biaya = estimasi_token(m.get("content")) + 4  # + overhead role
        if total + biaya > budget_token:
            break
        total += biaya
        pakai.append(m)
    pakai.reverse()
    if not pakai:
        return messages, 0, 0
    return [messages[0]] + pakai + messages[1:], len(pakai), total - dasar


def _trace_singkat(k: dict, rank: int, **tambahan) -> dict:
    d = {"rank": rank, "chunk_id": k.get("chunk_id"),
         "parent_id": k.get("parent_id"),
         "citation_str": k.get("citation_str"),
         "official_title": k.get("official_title")}
    d.update(tambahan)
    return d


def _llm_detail(llm: dict | None) -> dict:
    if not llm:
        return {"provider": "(default .env)"}
    return {"provider": llm.get("provider"), "model": llm.get("model"),
            "base_url": llm.get("base_url")}


def _chat_live(messages: list[dict], llm: dict | None,
               tools: list[dict] | None = None,
               usage_out: list | None = None,
               temperature: float | None = None,
               max_tokens: int | None = None):
    """Satu panggilan LL live; kredensial per-request.
    Rasa API dipilih otomatis (llm_adapter.rasa): Anthropic native lewat
    adapter, sisanya via llm_client.py lama (OpenAI-compatible).
    Bila usage_out (list) diberikan, usage mentah provider ditambahkan ke
    dalamnya agar pemanggil bisa mengakumulasi token multi-putaran."""
    import os
    from llm_client import chat_with_usage
    import llm_adapter

    provider = (llm or {}).get("provider", "custom")
    base_url, api_key, model = llm_provider.validasi(llm)
    if not base_url:
        raise RuntimeError("live LLM butuh llm.base_url (provider preset/custom)")
    if not api_key and provider != "ollama":
        raise RuntimeError("live LLM butuh llm.api_key")
    if not api_key:
        api_key = "ollama"

    if llm_adapter.rasa(provider, base_url) == "anthropic":
        msg, usage = llm_adapter.complete_anthropic(
            base_url, api_key, model, messages, tools,
            temperature, max_tokens)
        if usage_out is not None:
            usage_out.append(usage)
        return msg

    _bak = {k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}
    try:
        os.environ["LLM_BASE_URL"] = base_url
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = model
        msg, usage = chat_with_usage(messages, tools=tools,
                                     temperature=temperature,
                                     max_tokens=max_tokens)
        if usage_out is not None:
            usage_out.append(usage)
        return msg
    finally:
        for k, v in _bak.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
