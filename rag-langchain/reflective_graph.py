"""
reflective_graph.py — "Naik kelas": RAG REFLEKTIF sebagai state machine.
===========================================================================

Arsitektur A (rag_chain.py) tidak bisa memperbaiki diri: kalau retrieval
meleset, prompt salah dan jawaban salah. Versi REFLEKTIF memakai LLM
sebagai PENILAI (grader) di tengah alur:

                    ┌──────────────────────────────────┐
                    ▼                                  │
  pertanyaan -> retrieve -> grade ──relevan──> generate -> END
                    │            │                ▲
                    │        tidak relevan        │
                    └──> rewrite query ────────────┘
                                  (maks 2 percobaan, lalu jawab jujur)

Ini contoh "perilaku agentic" dari panduan (bab 7): LLM memutuskan
langkah. Perbedaan dengan agent tool-calling: alurnya KITA yang
merancang (bentuk graph tetap) — LLM hanya mengisi keputusan di
conditional edge. Cocok saat arah sudah jelas tetapi perlu pengendalian
kualitas di tengah jalan.

STATE: TypedDict GraphState — data yang mengalir antar-node:
    pertanyaan : str            (bisa diganti oleh node rewrite)
    dokumen    : list[Document]
    jawaban    : str
    percobaan  : int            (counter rewrite — guardrail loop)
"""

from __future__ import annotations

import argparse
import sys
from typing import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

import rag_chain
from llm import get_llm
from retriever import LegislationRetriever

MAX_PERCOBAAN = 2      # setelah 2 rewrite, berhenti mencoba & jawab jujur


class GraphState(TypedDict):
    """Skema state yang mengalir di graph (lihat docstring modul)."""
    pertanyaan: str
    dokumen: list[Document]
    jawaban: str
    percobaan: int


# ------------------------------------------------------------ NODES
import config


def buat_node_retrieve(retriever):
    def node_retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["pertanyaan"])
        # Node 'rerank' (saran roadmap): kandidat diurutkan ulang oleh
        # cross-encoder sebelum grader menilai — grader melihat kandidat
        # TERBAIK, bukan kandidat kosmik paling mirip secara cosine.
        if config.RERANK:
            from reranker import QwenReranker
            docs = QwenReranker().rerank(state["pertanyaan"], docs)
        print(f"  [retrieve] {len(docs)} kandidat "
              f"utk: {state['pertanyaan'][:60]}")
        return {"dokumen": docs}
    return node_retrieve


def buat_node_grade(llm):
    """LLM sebagai GRADER: relevan/tidak — jawaban 'YA'/'TIDAK' saja.
    Mengapa prompt segampang itu? Tugas sempit = jawaban stabil & murah."""
    grader = ChatPromptTemplate.from_messages([
        ("system", "Kamu penilai relevansi. Jawab SATU kata: YA atau TIDAK."),
        ("human", "Pertanyaan: {pertanyaan}\n\nPasal pertama:\n{pasal}\n\n"
                  "Apakah pasal itu relevan untuk menjawab pertanyaan?"),
    ]) | llm | StrOutputParser()

    def node_grade(state: GraphState) -> dict:
        if not state["dokumen"]:
            return {"jawaban": "Tidak ditemukan dasar dalam korpus."}
        pasal = state["dokumen"][0].page_content[:800]
        putusan = (grader.invoke({"pertanyaan": state["pertanyaan"],
                                  "pasal": pasal}) or "").strip().upper()
        print(f"  [grade] {putusan[:3]}")
        state["jawaban"] = "RELEVAN" if putusan.startswith("YA") else "TIDAK"
        return state   # keputusan dibaca conditional edge dari jawaban sementara
    return node_grade


def buat_node_generate(llm):
    """Jawab dari konteks yang SUDAH ada di state — tidak men-retrieve ulang
    (hemat satu kali pencarian; rag_chain.SYSTEM & format_docs dipakai ulang
    agar aturan prompt konsisten di seluruh folder)."""

    def node_generate(state: GraphState) -> dict:
        blok = rag_chain.format_docs(state["dokumen"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", rag_chain.SYSTEM),
            ("human", "KONTEKS PASAL:\n{konteks}\n\nPERTANYAAN: {pertanyaan}"),
        ])
        jawaban = (prompt | llm | StrOutputParser()).invoke(
            {"konteks": blok, "pertanyaan": state["pertanyaan"]})
        print("  [generate] jawaban tersusun")
        return {"jawaban": jawaban}
    return node_generate


def buat_node_rewrite(llm):
    """LLM merumuskan ulang pertanyaan dengan kata kunci istilah hukum."""
    rewriter = ChatPromptTemplate.from_messages([
        ("system", "Ubah pertanyaan pengguna menjadi KALIMAT PENCARIAN "
                   "singkat berisi istilah hukum kuncinya (tanpa penjelasan)."),
        ("human", "{pertanyaan}"),
    ]) | llm | StrOutputParser()

    def node_rewrite(state: GraphState) -> dict:
        baru = (rewriter.invoke({"pertanyaan": state["pertanyaan"]})
                or state["pertanyaan"]).strip()
        print(f"  [rewrite] -> {baru[:70]}")
        return {"pertanyaan": baru,
                "percobaan": state.get("percobaan", 0) + 1}
    return node_rewrite


# ------------------------------------------------- CONDITIONAL EDGES
def keputusan_setelah_grade(state: GraphState) -> str:
    if state["jawaban"] == "RELEVAN":
        return "generate"
    return "rewrite"   # tidak relevan -> coba rumuskan ulang


def keputusan_setelah_rewrite(state: GraphState) -> str:
    if state.get("percobaan", 0) >= MAX_PERCOBAAN:
        return "generate"   # habis kesabaran: jawab dari konteks (jujur)
    return "retrieve"       # coba cari lagi dengan pertanyaan baru


# ------------------------------------------------------- RAKIT GRAPH
def bangun_graph(dry: bool = False):
    llm = get_llm(dry=dry)
    retriever = LegislationRetriever()

    g = StateGraph(GraphState)
    g.add_node("retrieve", buat_node_retrieve(retriever))
    g.add_node("grade", buat_node_grade(llm))
    g.add_node("generate", buat_node_generate(llm))
    g.add_node("rewrite", buat_node_rewrite(llm))

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", keputusan_setelah_grade,
                            ["generate", "rewrite"])
    g.add_conditional_edges("rewrite", keputusan_setelah_rewrite,
                            ["retrieve", "generate"])
    g.add_edge("generate", END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG reflektif LangGraph")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    app = bangun_graph(dry=a.dry_run)
    hasil = app.invoke({"pertanyaan": a.question, "dokumen": [],
                        "jawaban": "", "percobaan": 0},
                       config={"recursion_limit": 16})
    print("\n" + "=" * 70)
    print(hasil["jawaban"])
