# RAG — LangChain & LangGraph

Same bridge as `rag-vanilla/`, using the LangChain ecosystem (LCEL) + LangGraph (state machine). Compare side by side: identical concepts, glue moved from hand-written code to framework abstractions. Full design: `docs/architecture-overview.md`.

> Renamed from `rag-langchain-langgraph/` → `rag-langchain/`. `agent_langgraph.py` → `agent_graph.py`, `graph_rag_reflektif.py` → `reflective_graph.py`. Old names no longer exist.

## Concept map: vanilla ↔ LangChain/LangGraph

| Concept | Vanilla folder | This folder |
|---|---|---|
| Open dataset JSONL | `dataset.py` (dict by-ID) | `loader.py` (→ `Document`) |
| Embed query (iron rule) | `embedder.py` | `retriever.QwenQueryEmbedder` |
| Top-k search + article join | `retriever.py` | `retriever.LegislationRetriever` (BaseRetriever) |
| **Hybrid dense+BM25 (RRF)** | `bm25.py` + `retriever.py` | `bm25.py` + `LegislationRetriever` (fallback inside) |
| **2-phase rerank (cross-encoder)** | `reranker.py` | `reranker.QwenReranker` (LCEL step) |
| Citation prompt assembly | `prompt.py` | `rag_chain.format_docs` + `ChatPromptTemplate` |
| LLM client | `llm_client.py` (requests) | `llm.py` (ChatOpenAI/ChatOllama) |
| Architecture A (fixed RAG) | `answer.py` | `rag_chain.py` (LCEL) |
| Tool definitions | `tools.py` (manual JSON schema) | `tools_lc.py` (`@tool` decorator) |
| Architecture B (agent loop) | `agent.py` (`while` + dispatch) | `agent_graph.py` (StateGraph + ToolNode) |
| Reflective RAG (grade/rewrite) | — (none) | `reflective_graph.py` |

## Hybrid search (dense + BM25) — `bm25.py` inside the retriever

Identical behavior to vanilla, different wrapper:

- `LegislationRetriever` (BaseRetriever) runs TWO paths: cosine (`embeddings.npy`) + BM25 (`bm25.py`, offline index `python bm25.py --build`), fused by **RRF** then de-duplicated per whole article.
- Missing lexical index / `HYBRID=0` / `hybrid=false` (tool) → clean dense-only fallback.
- Result metadata: `skor_fusi`, `skor_bm25`, `score` (cosine), `score_rerank` when reranked — all auditable.
- Real effect: Article 1 (missed by cosine) rises to #1 on the sovereignty question via BM25 31.3.

## Reranker (2-phase search) — `reranker.py`

Cosine candidates re-ordered by **cross-encoder Qwen3-Reranker-0.6B**. LangChain twist: the reranker is just an ordinary **LCEL step** — `RunnableLambda(lambda q: reranker.rerank(q, retriever.invoke(q)))` — slotted between retriever and `format_docs`.

- **On by default** in chain, `search_norma` tool (`rerank` arg), and reflective-graph `retrieve` node; safe fallback to cosine when unloaded.
- Disable: `--no-rerank` (chain), `RERANK=0` in `.env`, or `rerank=false` on the tool.
- Model loads OFFLINE from HuggingFace cache; rerank score in `metadata["score_rerank"]` beside `score` (cosine).

## Install & check

```bash
pip install -r requirements.txt
# copy .env.example -> .env (pick LLM_BACKEND api|ollama)
python verify_setup.py          # library versions + dataset + retriever
python verify_setup.py --full   # + reranker, dry-run chain, agent graph, reflective graph
python verify_setup.py --llm    # + live LLM ping
```

## Run

```bash
# LCEL RAG with no network (mock answer, real retrieval):
python rag_chain.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context          # hybrid + rerank (default)
python rag_chain.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context --no-hybrid   # classic dense-only

# Live LCEL RAG (needs .env):
python rag_chain.py -q "Who holds sovereignty under the 1945 Constitution?"

# Tool-calling agent as state machine (mock → visible loop):
python agent_graph.py -q "who holds sovereignty" --dry-run

# Reflective RAG: retrieve → grade → (rewrite)* → generate:
python reflective_graph.py -q "one-term presidency phrase" --dry-run
```

## When to pick this folder?

- You need **agentic behavior**: graders/reflection, branching multi-tool flows, LLM-chosen steps.
- You want LangGraph built-ins: checkpointing, streaming, human-in-the-loop.
- You are already comfortable with the basics (see the vanilla folder first!).

## Common issues

| Symptom | Cause & fix |
|---|---|
| `LLM_API_KEY empty` | fill `.env` or use `--dry-run` |
| API version drift | LangChain moves fast — run `python verify_setup.py` (prints versions) |
| retriever slow on first call | one-time embedding-model load per process — normal |
| `GraphRecursionError` | raise `recursion_limit` or fix the loop scenario |
