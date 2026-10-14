"""
rag_chain.py — Arsitektur A di dunia LangChain: RAG dengan LCEL.
================================================================

LCEL (LangChain Expression Language) = merangkai komponen dengan `|`:

    {"konteks": ..., "pertanyaan": ...} | prompt | llm | StrOutputParser()

Setiap panah hanyalah sebuah Runnable; `|` menyusunnya menjadi Runnable
baru yang punya .invoke() / .stream() / .batch() seragam. Bandingkan
dengan answer.py di rag-vanilla: langkahnya SAMA
(retrieve → format konteks → prompt → LLM → jawaban), tetapi kerangka
aliran data ditangani LangChain — kode "lem" kita tinggal sedikit.

DUA POIN UNTUK DIBACA
  1. RunnableParallel({..}) menjalankan DUA cabang dari input yang sama:
     cabang konteks (string -> retriever -> format) dan cabang pertanyaan
     (string -> string). Hasilnya dict yang mengisi {placeholder} prompt.
  2. format_docs() menyusun blok [S1]..[Sn] dari metadata Document —
     identik dengan prompt.py versi tanpa-framework. Kualitas konteks ada
     di DATA kita (metadata provenance), bukan di framework.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from retriever import LegislationRetriever
from reranker import QwenReranker

from llm import get_llm

SYSTEM = """Kamu adalah asisten hukum Indonesia yang bekerja hanya dari \
teks peraturan yang diberikan.

Aturan keras:
1. Jawab HANYA berdasarkan konteks pasal yang diberikan.
2. Setiap pernyataan hukum WAJIB diakhiri sitasi [S1], [S2], ... \
PERSIS seperti itu (kurung siku + S + nomor), sesuai nomor konteks yang dipakai.
   BENAR: "Kedaulatan di tangan rakyat [S1]."
   SALAH (JANGAN PERNAH TULIS BEGITU): "Kedaulatan di tangan rakyat \
(UUD — Bab I Pasal 1 — hal. 1)."
   DILARANG menulis sitasi dalam tanda kurung biasa — hanya [S1], [S2], dst.
3. Jika konteks tidak memuat jawaban, jawab persis: "Tidak ditemukan dasar \
dalam korpus." — JANGAN mengarang.
4. Jawab dalam Bahasa Indonesia, ringkas dan terstruktur."""


def format_docs(docs) -> str:
    """list[Document] -> blok konteks bernomor + sitasi + anggaran char."""
    blok, total = [], 0
    for i, d in enumerate(docs, 1):
        m = d.metadata
        pos = f" — hal. {m['page']} baris {m['lines']}" if m.get("page") else ""
        # Tampilkan skor rerank bila ada (urutan final), fallback ke
        # skor fusi hybrid, lalu ke cosine — urutan prioritas yang sama
        # dengan urutan penilaian pipeline.
        skor = m.get("score_rerank") or m.get("skor_fusi") or m["score"]
        header = (f"[S{i}] {m['official_title']} — "
                  f"{m['citation_str'] or '(tanpa sitasi)'}{pos} "
                  f"(skor {skor})")
        total += len(header) + 1 + len(d.page_content)
        if total > config.CONTEXT_CHAR_BUDGET and blok:
            blok.append(f"[i] (kandidat dipotong: anggaran "
                        f"{config.CONTEXT_CHAR_BUDGET} char tercapai)")
            break
        blok.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blok)


def daftar_sumber(docs) -> str:
    """Ringkasan sumber utk diaudit di bawah jawaban."""
    baris = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        skor = m.get("score_rerank") or m.get("skor_fusi") or m["score"]
        # tunjukkan skor BM25 utk pasal yang muncul berkat jalur leksikal
        bm = f"/BM25 {m['skor_bm25']}" if m.get("skor_bm25") else ""
        baris.append(f"  [S{i}] {m['official_title']} — "
                     f"{m['citation_str'] or '-'} (skor {skor}{bm})")
    return "\n".join(baris)


def bangun_chain(dry: bool = False, top_k: int = config.TOP_K,
                 rerank: bool = config.RERANK,
                 hybrid: bool = config.HYBRID):
    """
    Rakit chain LCEL kanonik dan kembalikan (chain, retriever).

    Diagram alurnya:
      input string
        -> RunnableParallel { konteks:    |retriever|format_docs,
                              pertanyaan: |passthrough }
        -> ChatPromptTemplate  (mengisi {konteks} & {pertanyaan})
        -> llm                 (ChatOpenAI / ChatOllama / mock)
        -> StrOutputParser     (AMBil .content saja)
    """
    retriever = LegislationRetriever(k=max(top_k, config.RERANK_POOL_SIZE)
                                     if rerank else top_k, hybrid=hybrid)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "KONTEKS PASAL:\n{konteks}\n\nPERTANYAAN: {pertanyaan}"),
    ])
    # Cabang konteks. Tanpa rerank: retriever -> format (kanonik).
    # Dengan rerank: sisipkan QwenReranker sebagai step LCEL biasa — ia
    # menerima list[Document] dan mengembalikan list[Document] urut baru,
    # sehingga bisa diselipkan di mana saja di dalam rantai `|`.
    if rerank:
        reranker = QwenReranker(top_n=top_k)
        cabang_konteks = (
            RunnableLambda(lambda q: reranker.rerank(
                q, retriever.invoke(q)))
            | RunnableLambda(format_docs)
        )
    else:
        cabang_konteks = retriever | RunnableLambda(format_docs)
    chain = (
        RunnableParallel(
            konteks=cabang_konteks,
            pertanyaan=RunnablePassthrough(),
        )
        | prompt
        | get_llm(dry=dry)
        | StrOutputParser()
    )
    return chain, retriever


def jawab(pertanyaan: str, dry: bool = False, top_k: int = config.TOP_K,
          show_context: bool = False, rerank: bool = config.RERANK,
          hybrid: bool = config.HYBRID) -> str:
    """Satu pertanyaan -> jawaban (sumber dicetak bila diminta)."""
    chain, retriever = bangun_chain(dry=dry, top_k=top_k, rerank=rerank,
                                    hybrid=hybrid)
    if show_context or dry:
        docs = retriever.invoke(pertanyaan)
        if rerank:                       # pratinjau urutan final hasil rerank
            docs = QwenReranker(top_n=top_k).rerank(pertanyaan, docs)
        print(daftar_sumber(docs), "\n")
    return chain.invoke(pertanyaan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG LCEL — Peraturan Indonesia")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--dry-run", action="store_true",
                    help="tanpa jaringan: jawaban dari FakeListChatModel")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--no-rerank", action="store_true",
                    help="matikan reranker (urutan cosine asli)")
    ap.add_argument("--no-hybrid", action="store_true",
                    help="matikan hybrid: dense-only (cosine) saja")
    a = ap.parse_args()

    print("=" * 72)
    print("RAG LCEL" + (" — DRY-RUN" if a.dry_run else ""))
    print("=" * 72)
    print(jawab(a.question, dry=a.dry_run, top_k=a.top_k,
                show_context=a.show_context, rerank=not a.no_rerank,
                hybrid=not a.no_hybrid))
