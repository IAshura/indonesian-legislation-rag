# RAG — Vanilla (Pure Python, No Framework)

Fixed-pipeline RAG (**Architecture A**) + agentic tool-calling (**Architecture B**) for the Indonesian Legislation RAG project — no LangChain, LlamaIndex, or other framework. All LLM networking is plain `requests` so the protocol stays visible. Concept guide: `docs/llm-injection-tool-calling.html` (ch. 1–9). Full design: `docs/architecture-overview.md`.

> Renamed from `rag-tanpa-framework/` → `rag-vanilla/`. Entry script renamed `rag_jawab.py` → `answer.py`. Old names no longer exist.

## Learn in order (each file runs standalone)

| # | File | What you learn | Quick check |
|---|------|---------------|-----------|
| 1 | `config.py` | single-door config + iron rules | `python config.py` |
| 2 | `dataset.py` | JSONL, 3-level IDs, **small-to-big** | `python dataset.py` |
| 3 | `embedder.py` | query vectors, last-token pooling, iron rule | `python embedder.py` |
| 4 | `retriever.py` | hybrid dense+BM25 (RRF), exact (mmap), optional Milvus | `python retriever.py` |
| 4b | `bm25.py` | lexical BM25 path + offline index builder | `python bm25.py --build` |
| 5 | `prompt.py` | 3-block prompt anatomy + `[S1]` citations | `python prompt.py` |
| 6 | `llm_client.py` | HTTP POST to LLM, `tool_calls`, dry-run | `python llm_client.py` |
| 7 | `tools.py` | tool schemas + validated dispatcher | `python tools.py` |
| 8 | `answer.py` | Architecture A: retrieve → **rerank** → prompt → LLM | see below |
| 9 | `agent.py` | Architecture B: tool-calling loop | see below |
| 10 | `verify_setup.py` | environment health check | `python verify_setup.py` |

## Hybrid search (dense + BM25) — `bm25.py` + `retriever.py`

Two paths fused by **RRF (Reciprocal Rank Fusion)** — rank fusion, not score addition (cosine 0–1 vs BM25 0–20+ are incommensurable):

- **Dense:** cosine over embeddings.npy — strong on meaning.
- **Lexical (BM25):** Okapi BM25 (k1=1.2, b=0.75) over article tokens + citation + title — strong on exact identifiers (article numbers, years, doc types).
- Fusion: `score(c) = SUM_paths 1/(60 + rank_path(c))`.
- Build the lexical index once: `python bm25.py --build` → `pipeline/result/lexical/norma_tokens.txt` (63,231 lines ≡ `ids.jsonl` lines). Missing → safe dense-only fallback.
- Real effect: sovereignty question — Article 1 missed by cosine (rank > 10) rises to #1 on BM25 31.3.
- Disable: `--no-hybrid`, `HYBRID=0` in `.env`, or `hybrid=false` on tool `search_norma`.
- Auditable scores: `skor_fusi` (RRF), `skor_bm25`, `score` (cosine) / `score_rerank` when reranked.

## Reranker (2-phase search) — `reranker.py`

After retrieval, **cross-encoder Qwen3-Reranker-0.6B** (query & article read *together*) re-orders candidates. Phase 1 takes 12 cheap candidates, phase 2 reranks to top-k.

- **On by default**; safe **fallback** to cosine order when the model is absent (loaded offline from HuggingFace cache).
- Disable: `--no-rerank`, `RERANK=0` in `.env`, or `rerank=false` on `search_norma`.
- Rerank score kept separately: `score_rerank` (P(yes) 0–1) beside `score` (cosine).

## Setup

```bash
pip install -r requirements.txt
# copy .env.example -> .env, fill ONE backend:
#   OpenAI     : LLM_BASE_URL=https://api.openai.com/v1          LLM_MODEL=gpt-4o-mini
#   OpenRouter : LLM_BASE_URL=https://openrouter.ai/api/v1       LLM_MODEL=...
#   Ollama     : LLM_BASE_URL=http://localhost:11434/v1          LLM_MODEL=qwen3:4b
python verify_setup.py          # files & retrieval
python verify_setup.py --full   # + Qwen3-Reranker embedder/reranker (~1-2 min, one pass)
```

## Run

```bash
# See the prompt that would be sent (NO key, NO network):
python answer.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context

# Live answer (needs filled .env):
python answer.py -q "Who holds sovereignty under the 1945 Constitution?"

# Filter: only 2017 laws
python answer.py -q "presidential candidacy requirements" --doc-type UU --year 2017

# Reranker & hybrid are on by default; compare with them off:
python answer.py -q "..." --dry-run --show-context                     # hybrid + rerank (default)
python answer.py -q "..." --dry-run --show-context --no-rerank         # hybrid only
python answer.py -q "..." --dry-run --show-context --no-hybrid         # classic dense-only

# Agent loop demo with NO key: MockLLM requests tools, real tools run
python agent.py -q "who holds sovereignty" --dry-run

# Real agent (multi-hop: search -> fetch document -> check status):
python agent.py -q "Is the 1945 Constitution still valid and who holds sovereignty?"
```

## File → concept map

- `answer.py` = **Architecture A**: 1 LLM call, deterministic.
- `agent.py` = **Architecture B**: LLM picks steps, max 5 rounds.
- `tools.py` = 4 tools (`search_norma`, `get_article`, `get_document`, `check_validity`) — narrow, validated, dataset-cited.
- Guardrails enforced: only `part=="norma"` enters context (embeddings hold norma only), citations from `provenance`, `check_validity` never guesses legal status.

## Common issues

| Symptom | Cause & fix |
|---|---|
| `LLM_API_KEY empty` | fill `.env` (see `.env.example`) or use `--dry-run` |
| `HTTP 401/404 from LLM` | base_url/model wrong for that provider |
| embedder slow on first call | one-time model load per process — normal (~10–30 s) |
| low scores on all candidates | don't change the `Instruct:` format/pooling in `embedder.py` (iron rule) |
| Milvus connect fail | Docker/Milvus down — default `numpy` path is used |
| `FileNotFoundError ... norma_tokens.txt` | run `python bm25.py --build` once (hybrid lexical index) |
