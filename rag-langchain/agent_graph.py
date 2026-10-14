"""
agent_graph.py — ARSITEKTUR B di dunia LangGraph: agent = state machine.
=============================================================================

NAIK KELAS: MENGAPA LANGGRAPH?
  Di folder tanpa-framework, loop agent ditulis pakai `while`. LangGraph
  mengangkat loop yang sama menjadi GRAPH eksplisit (state machine):
      START -> agent -> (ada tool_calls?) -> tools -> agent -> ... -> END
  Manfaat nyata:
    - alur tergambar & bisa diverifikasi (bukan while tersembunyi);
    - state tersimpan terstruktur (pesan, retry, dsb.);
    - conditional edge = titik keputusan yang bisa diuji;
    - gratis dapat: checkpointing, streaming, human-in-the-loop.
  Tapi inti kerjanya SAMA dengan agent.py versi manual — bandingkan!

BAGIAN-BAGIAN
  MessagesState   : tipe state bawaan, "messages" berjalan akumulasi
                    (setiap node mengembalikan pesan yang DIAPPEND).
  agent node      : llm.bind_tools(TOOLS).invoke(state) -> AIMessage
  should_continue : kondisi cabang — "tools" jika LLM minta tool, END jika
                    sudah menjawab (persis `if not tool_calls` di agent.py)
  ToolNode        : node bawaan LangGraph yang MENJALANKAN tool & mengirim
                    hasil balik sebagai ToolMessage — padatan langkah 3-5
                    loop manual.

  Versi siap-pakai juga tersedia (lihat komentar di bawah):
      from langgraph.prebuilt import create_react_agent
      app = create_react_agent(llm, tools_lc.TOOLS)
  Kode manual di bawah dipilih supaya mesinnya terlihat.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import tools_lc
from llm import MockAgentLLM, get_llm

MAX_PUTARAN = 6   # guardrail sama dengan agent.py: batasi jumlah putaran


# ------------------------------------------------------------ NODES
def buat_node_agent(llm):
    """Tutup (closure) LLM yang sudah bind_tools menjadi fungsi node."""
    llm_dengan_tools = llm.bind_tools(tools_lc.TOOLS)

    def node_agent(state: MessagesState) -> dict:
        ai = llm_dengan_tools.invoke(state["messages"])
        label = ("minta tool: " + ", ".join(t["name"] for t in ai.tool_calls)
                 if ai.tool_calls else "jawab final")
        print(f"  [agent] {label}")
        return {"messages": [ai]}

    return node_agent


def buat_node_tools():
    node = ToolNode(tools_lc.TOOLS)

    def node_tools(state: MessagesState) -> dict:
        hasil = node.invoke(state)                      # jalankan tool asli
        last = hasil["messages"][-1]
        isi = (last.content or "")[:160].replace("\n", " ")
        print(f"  [tools] -> {isi}{' …' if len(str(last.content)) > 160 else ''}")
        return hasil

    return node_tools


# ------------------------------------------------- CONDITIONAL EDGE
def should_continue(state: MessagesState) -> str:
    """Keputusan cabang setelah node agent (persis langkah 2 loop manual)."""
    last: AIMessage = state["messages"][-1]
    if last.tool_calls:
        return "tools"            # LLM minta tool -> ke node tools
    return END                    # LLM yakin -> selesai


# ------------------------------------------------------- RAKIT GRAPH
def bangun_graph(dry: bool = False):
    """StateGraph: START -> agent -> (tools -> agent)* -> END."""
    if dry:
        # Skrip mock = multi-tool lalu jawab final (tool asli dijalankan).
        llm = MockAgentLLM(skrip=[
            {"tool_calls": [("search_norma",
                             {"question": "kedaulatan di tangan siapa",
                              "top_k": 3})]},
            {"tool_calls": [("get_document", {"document_id": "uud-1945"})]},
            "(mock) Jawaban final berdasarkan hasil tool.",
        ])
    else:
        llm = get_llm()

    g = StateGraph(MessagesState)
    g.add_node("agent", buat_node_agent(llm))
    g.add_node("tools", buat_node_tools())
    g.add_edge(START, "agent")                       # masuk -> agent
    g.add_conditional_edges("agent", should_continue,
                            ["tools", END])          # cabang keputusan
    g.add_edge("tools", "agent")                     # hasil tool -> agent lagi

    app = g.compile()
    # Versi prebuilt yang setara (satu baris, tanpa bagian edukatif):
    #   from langgraph.prebuilt import create_react_agent
    #   app = create_react_agent(llm, tools_lc.TOOLS)
    return app


def jalankan(question: str, dry: bool = False) -> str:
    app = bangun_graph(dry=dry)
    print("=" * 70)
    print("AGENT LANGGRAPH" + (" — MOCK (dry-run)" if dry else ""))
    print("=" * 70)

    # Pesan awal: aturan sistem + pertanyaan (format tuple diterima
    # LangChain dan dikonversi otomatis jadi SystemMessage/HumanMessage).
    pesan_awal = [
        ("system",
         "Kamu asisten hukum Indonesia. Gunakan tools sebelum menjawab; "
         "jawab hanya dari hasil tool; sertakan sitasi dari hasil tool. "
         "Bila tidak ada dasar, katakan jujur."),
        ("user", question),
    ]
    hasil = app.invoke(
        {"messages": pesan_awal},
        # recursion_limit: guardrail LangGraph = MAX_PUTARAN * 2 node
        config={"recursion_limit": MAX_PUTARAN * 2},
    )
    return hasil["messages"][-1].content


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent LangGraph — Peraturan Indonesia")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(jalankan(a.question, dry=a.dry_run))
