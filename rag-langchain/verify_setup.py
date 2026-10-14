"""
verify_setup.py — Pemeriksaan kesehatan folder LangChain/LangGraph.
===================================================================

    python verify_setup.py            # versi library + dataset + retriever
    python verify_setup.py --full     # + dry-run chain & compile graph
    python verify_setup.py --llm      # + 1 panggilan LLM asli (perlu .env)
"""

from __future__ import annotations

import argparse
import os
import sys

# Verify menguji RERANKER lewat cek khususnya sendiri; matikan rerank otomatis
# di chain/graph dry-run supaya verify tetap cepat.
os.environ.setdefault("RERANK", "0")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, BAD = "\u2705", "\u274c"
hasil: list[tuple[bool, str]] = []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    hasil.append((kondisi, f"{OK if kondisi else BAD} {nama}"
                  + (f" — {detail}" if detail else "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    # 1) versi library — penting karena API LangChain bergerak cepat.
    # Versi dibaca via importlib.metadata (modul LangGraph 1.x tidak lagi
    # mengekspos atribut __version__).
    try:
        import importlib.metadata as md
        import langchain, langchain_core, langgraph  # noqa: F401
        cek("import langchain", True,
            "langchain " + md.version("langchain") + ", core "
            + md.version("langchain-core") + ", langgraph "
            + md.version("langgraph"))
    except Exception as e:
        cek("import langchain", False, str(e)[:120])

    try:
        import langchain_openai, langchain_ollama  # noqa: F401
        cek("backend ChatOpenAI/ChatOllama", True)
    except Exception as e:
        cek("backend ChatOpenAI/ChatOllama", False, str(e)[:120])

    # 2) dataset & retriever (memuat model embedding — beberapa detik)
    try:
        from retriever import LegislationRetriever
        r = LegislationRetriever(k=3)
        docs = r.invoke("Di tangan siapa kedaulatan berada menurut UUD 1945?")
        cek("retriever.invoke", len(docs) > 0,
            f"{len(docs)} Document, [1] = "
            f"{docs[0].metadata['official_title']} — "
            f"{docs[0].metadata['citation_str']}")
    except Exception as e:
        cek("retriever", False, str(e)[:150])

    # 2b) reranker cross-encoder (opsional, --full; paling lambat)
    if args.full:
        try:
            from reranker import QwenReranker
            from langchain_core.documents import Document
            docs = [Document(page_content=t, metadata={"score": 0.0})
                    for t in ["Kedaulatan berada di tangan rakyat menurut "
                              "Undang-Undang Dasar.",
                              "Pajak kendaraan bermotor peraturan daerah."]]
            hasil_r = QwenReranker().rerank(
                "Di tangan siapa kedaulatan berada menurut UUD 1945?", docs)
            if hasil_r and "score_rerank" in hasil_r[0].metadata:
                cek("reranker Qwen3-0.6B",
                    hasil_r[0].metadata["score_rerank"] >
                    hasil_r[1].metadata["score_rerank"],
                    f"P(yes) urut: {[d.metadata['score_rerank'] for d in hasil_r]}")
            else:
                cek("reranker", True, "model tak termuat - fallback cosine")
        except Exception as e:
            cek("reranker", False, str(e)[:120])

    # 3) tools
    try:
        import tools_lc
        skema = tools_lc.search_norma.args_schema.model_json_schema()
        cek("@tool schema otomatis", len(tools_lc.TOOLS) == 4,
            f"{[t.name for t in tools_lc.TOOLS]}")
        cek("skema punya properties", "question" in
            skema.get("properties", {}))
    except Exception as e:
        cek("tools_lc", False, str(e)[:120])

    # 4) full: chain dry-run + compile kedua graph
    if args.full:
        try:
            from rag_chain import bangun_chain
            chain, _ = bangun_chain(dry=True, top_k=3, rerank=False)
            out = chain.invoke("Di tangan siapa kedaulatan menurut UUD?")
            cek("chain LCEL (dry-run)", len(out) > 0, out[:70])
        except Exception as e:
            cek("chain LCEL", False, str(e)[:150])
        try:
            from agent_graph import bangun_graph
            app = bangun_graph(dry=True)
            st = app.invoke({"messages": [("user", "kedaulatan di tangan siapa")]},
                            config={"recursion_limit": 20})
            cek("graph agent (mock)", bool(st["messages"][-1].content),
                str(st["messages"][-1].content)[:70])
        except Exception as e:
            cek("graph agent", False, str(e)[:150])
        try:
            from reflective_graph import bangun_graph as bg2
            app2 = bg2(dry=True)
            st2 = app2.invoke({"pertanyaan": "kedaulatan di tangan siapa",
                               "dokumen": [], "jawaban": "", "percobaan": 0},
                              config={"recursion_limit": 16})
            cek("graph reflektif (mock)", len(st2["jawaban"]) > 0,
                st2["jawaban"][:70])
        except Exception as e:
            cek("graph reflektif", False, str(e)[:150])

    # 5) LLM asli (opsional)
    if args.llm:
        try:
            from llm import get_llm
            jaw = get_llm().invoke("Balas satu kata: siap")
            cek("LLM reachable", bool(jaw.content), str(jaw.content)[:60])
        except Exception as e:
            cek("LLM reachable", False, str(e)[:160])

    print("\n".join(x[1] for x in hasil))
    gagal = sum(1 for ok, _ in hasil if not ok)
    print(f"\n{len(hasil) - gagal}/{len(hasil)} lolos"
          + (" — perlu dibereskan." if gagal else " — siap dipakai!"))
    raise SystemExit(1 if gagal else 0)


if __name__ == "__main__":
    main()
