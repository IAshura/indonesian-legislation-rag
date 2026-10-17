# Indonesian Legislation RAG

**Retrieval-Augmented Generation over Indonesian law** — from 378 official regulations (1945–2024) to 74,218 embed-ready chunks with verified provenance (page + line), parent–child small-to-big architecture, and an automated QA gate.

> Data source: public corpus [*Indonesian legislation as data: 90,588 norm units*](https://data.mendeley.com/datasets/8wpjvhh7g2/1) — DOI `10.17632/8wpjvhh7g2.1`, CC BY 4.0, by Qausar, Prabowo, Arifin & Isa (Binus University). Chunking pipeline, dataset build, evaluation, and validity layer in this repo are our own work.
>
> **Attribution (required by CC BY 4.0):** *Qausar, Prabowo, Arifin & Isa (Binus University) — Indonesian Legislation Norm-Unit Corpus, via Mendeley Data, DOI 10.17632/8wpjvhh7g2.1.*

> [!IMPORTANT]
> **Learning project — not for real-world legal use.** This repo was built for
> personal learning about RAG / retrieval systems. It is **not** legal advice,
> not a verified legal database, and must not be relied upon for real cases,
> court filings, or professional decisions. Corpus ends at 2024, the
> legal-validity layer (`legal_status`) is only partially verified, and
> LLM answers can be wrong despite citations. Always consult official sources
> (JDIH, peraturan.bpk.go.id) and a qualified lawyer for any real legal matter.

## Key numbers

| Metric | Value |
|---|---|
| Source | 378 regulations (Constitution → institutional rules), 1945–2024, JDIH portal |
| Segmented norm units | 90,588 |
| Embed-ready chunks | 74,218 (66,292 with role `norma`; 4,704 flagged fragments) |
| Parents & windows | 11,413 parents · 14,689 windows (≤1,800 chars, 150 overlap) |
| Reasoned quarantine | 2,209 (587 duplicates, 1,622 heading-like) |
| QA provenance | 300 samples → **98% match**, gate PASS · 0 orphans · 0 duplicate IDs |
| Accountability | 74,218 + 2,209 + 14,161 headings = **90,588 — zero rows lost** |
| Embeddings | 63,231 vectors × 2 models (`qwen3-06b` 1024-d active · `qwen3-4b` 2560-d) |
| Retrieval eval | recall@8 = **0.75** on 12 golden queries |

## Repository layout

```
├── README.md                  ← this file (single entry point)
├── start.bat                  ← one-click launcher (Docker/Milvus → backend :8000 → frontend :3000)
├── requirements.txt           ← single-door Python deps
├── data/corpus/               ← untouched Mendeley v1 corpus (checksums.md5 = integrity)
│   ├── units.csv              ← 90,588 norm units + provenance + IndoBERT audit
│   ├── documents.csv          ← 378 regulations, title-block identity + UU 12/2011 classes
│   ├── texts/                 ← 378 page-delimited texts (page/line addresses from units.csv)
│   └── README-corpus.md
├── pipeline/                  ← canonical build v4.1 (the only active pipeline)
│   ├── README.md              ← build spec: 10 design decisions + funnel + QA
│   ├── build_dataset.py       ← raw → chunks/parents/windows/quarantine/manifest (deterministic)
│   ├── embed_example.py       ← embedding feed (local HF / OpenAI-compatible API)
│   ├── embed_query_qwen.py    ← query-side Qwen3 encoder (last-token pooling)
│   ├── enrich_validity.py     ← validity layer (offline default, --live best-effort)
│   ├── insert_milvus.py       ← Milvus create/index/insert/demo
│   ├── smoke_milvus.py        ← self-query health check (no model needed)
│   ├── result/                ← 7 output files (chunks.jsonl, parents.jsonl, …)
│   └── eval/                  ← eval_retrieval.py + 12 golden queries
├── rag-vanilla/               ← pure-Python RAG, no framework (learn the machinery)
│   ├── answer.py              ← fixed-pipeline RAG (was rag_jawab.py)
│   ├── agent.py               ← tool-calling loop (max 5 iterations)
│   ├── dataset.py / embedder.py / retriever.py / bm25.py / reranker.py
│   ├── prompt.py / llm_client.py / tools.py / config.py
├── rag-langchain/               ← LangChain LCEL + LangGraph version (was rag-langchain-langgraph)
│   ├── rag_chain.py           ← LCEL chain (fixed RAG)
│   ├── agent_graph.py         ← StateGraph agent (was agent_langgraph.py)
│   ├── reflective_graph.py    ← corrective-reflective RAG (was graph_rag_reflektif.py)
├── webapp/                    ← production UI (FastAPI :8000 + Next.js :3000)
│   ├── backend/ (app.py, engine.py, bridge.py, …)
│   └── frontend/ (Next.js 14 + React 18)
├── docs/                      ← consolidated documentation (English)
│   ├── pipeline/              ← per-stage strategy docs (Mermaid + real numbers)
│   ├── chatbot/               ← per-strategy chatbot docs (pipeline/retrieval/rerank/agent)
│   ├── architecture-overview.md  ← RAG design, both stacks, tool-calling, pitfalls
│   ├── pipeline-guide.md         ← dataset build end-to-end
│   ├── retrieval-indexes.md      ← HNSW vs IVF vs FLAT
│   ├── operations-runbook.md     ← start-up + daily use + troubleshooting
│   └── examples/sample-questions.txt
├── apps/corpus-viewer/index.html ← Units Viewer (dependency-free browser table)
├── tools/                        ← download_dataset.py (pull artifacts from HF)
│                                  · hf_upload_dataset.ps1 (push artifacts to HF)
│                                  · gemma-smoke-test.py (minimal LLM smoke test)
└── archive/experiments/        ← pre-v4 archived experiments (not canonical)
```

## Prerequisites (what you need installed)

| Tool | Needed for | Without it |
|---|---|---|
| **Python 3.11+** | everything | nothing runs |
| **.venv** (project-local) | `start.bat` & any pip install | `start.bat` aborts — create with `python -m venv .venv` |
| **Node.js LTS** (npm) | webapp frontend :3000 | webapp only; RAG CLI works fine |
| **Docker Desktop** | Milvus (HNSW/IVF/FLAT vector search) | OPTIONAL — exact-numpy + BM25 + hybrid-exact still work; `start.bat` warns and continues |
| **GPU + CUDA** (recommended) | local embedding (torch) & reranker | works on CPU, slower |
| pip packages | see `requirements.txt` (single door: numpy, torch, transformers, requests, pymilvus, fastapi, uvicorn, pydantic) | `start.bat` auto-installs on first run |
| **LLM API key** (`.env`) | live answers (OpenAI / OpenRouter / Groq / DeepSeek / Ollama) | everything still works with `--dry-run` (no LLM calls) |
| `huggingface_hub` | one-time dataset download via `tools/download_dataset.py` | auto-installed by `start.bat` when prompted |

Notes:
- `start.bat` handles deps + missing dataset automatically; Docker/Milvus is the
  only manual optional step (warnings instead of failure).
- Milvus is a separate container (Docker Compose) on port 19530 — used by
  `pipeline/insert_milvus.py` and the Milvus retrieval path. The corpus viewer
  (`apps/corpus-viewer/`) needs nothing but a browser.
- For no-GPU hosts: install CPU torch first to stay light —
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Quickstart

```powershell
# 0. One-click (recommended): Docker/Milvus → backend → frontend
.\start.bat
#    frontend: http://localhost:3000
#    backend:  http://127.0.0.1:8000/api/health  (must be ok:true)

# 1. Rebuild dataset from corpus (deterministic, stdlib only)
#    NOTE: needs data/corpus/ first — download via 'Data & artifacts' below.
cd pipeline
python build_dataset.py            # → result/chunks.jsonl + manifest.json (QA gate PASS)

# 2. Browse the corpus (alternative to webapp)
#    from repo root: python -m http.server 8000 → open apps/corpus-viewer/index.html

# 3. Embed (needs torch + transformers, GPU recommended)
python embed_example.py                                   # norma only, fragments skipped
python embed_example.py --model intfloat/multilingual-e5-large --batch-size 32

# 3b. Load into Milvus (local Docker; pip install pymilvus first)
python insert_milvus.py                                   # create collection + insert
python insert_milvus.py --demo "who holds sovereignty under the 1945 Constitution?"

# 4. Measure retrieval (recall@8; >=0.8 go, <0.6 change model/top-k)
python eval/eval_retrieval.py

# 5. Fill in validity status (required before production — never invent legal status)
python enrich_validity.py --live --delay 3    # candidate URLs from peraturan.bpk.go.id → manual verify
```

## Data & artifacts (not shipped in this repo)

Large artifacts are NOT stored on GitHub. Get them from Hugging Face:
[`Fayizz/indonesian-legislation-norms`](https://huggingface.co/datasets/Fayizz/indonesian-legislation-norms)
(dataset card there: provenance, processing steps, QA numbers, license).

```powershell
pip install huggingface_hub        # no auth needed — public dataset

# A. Processed artifacts → pipeline/result/  (~1.1 GB)
python tools/download_dataset.py

#    only one embedding set:
python tools/download_dataset.py --model qwen3-06b

# B. + raw Mendeley corpus → data/corpus/  (+40 MB)
python tools/download_dataset.py --corpus
```

After step A you can run everything that depends on `pipeline/result/`
(RAG CLI, webapp, eval). After step B you can also rebuild from scratch:

```powershell
cd pipeline
python build_dataset.py            # deterministic, stdlib only — verifiable against manifest.json
```

C. Rebuild the embeddings instead of downloading (needs torch + transformers, GPU):
`python pipeline/embed_example.py` (see the pipeline README).

## Daily RAG use (hybrid, the path we actually use)

```powershell
cd rag-vanilla
..\.venv\Scripts\python.exe answer.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context
```

Flags: default = hybrid (dense + BM25) + rerank (best). `--no-hybrid` = dense only. `--no-rerank` = fast but worse ranking. `--doc-type UUD` = type filter. Drop `--dry-run` for live LLM answers (needs `.env` with API key).

LangChain mirror:

```powershell
cd ..\rag-langchain
..\.venv\Scripts\python.exe rag_chain.py -q "..." --dry-run
..\.venv\Scripts\python.exe agent_graph.py -q "..." --dry-run
..\.venv\Scripts\python.exe reflective_graph.py -q "..." --dry-run
```

## Chunking design (why it matters)

- **Structure-based**: chunk boundary = legal hierarchy (Article/verse), not token slicing. 1 unit = 1 chunk, no auto-merge — page+line provenance never breaks.
- **Hierarchical parent–child**: retrieve on child (verse, precise), answer with full article from `parents.jsonl` (small-to-big); long parents split into deterministic windows ≤1,800 chars with 150 overlap.
- **Self-contained vectors**: `embedding_input = official title — citation: body`.
- **No silent loss**: 2,209 reasoned quarantines; fragments kept with `is_fragment: true`.
- **Iron rule**: `model index = model query` — switching models means re-embedding (`meta.json` records the version).

## Retrieval (hybrid + rerank)

```
dense path   (meaning) : cosine over embeddings.npy      — strong on synonyms/paraphrase
lexical path (words)   : BM25 over text+citation+title   — strong on article nos., years, types
fusion: RRF (Reciprocal Rank Fusion) = SUM_paths 1/(60 + rank_path(c))
stage 2: Qwen3-Reranker-0.6B cross-encoder re-scores top-12 → top-k
```

Toggle: `HYBRID=1|0`, `RERANK=1|0` (.env), `--no-hybrid` / `--no-rerank` (CLI). Missing lexical index → safe dense-only fallback. Auditable scores: `skor_fusi`, `skor_bm25`, `score` (cosine), `score_rerank`.

## Which RAG stack?

| | `rag-vanilla/` | `rag-langchain/` |
|---|---|---|
| Philosophy | Write everything yourself — see the bare machine | Framework tidies it up |
| Core | RAG CLI + 60-line agent loop | LCEL chain + agent graph + reflective graph |
| Transparency | 100% — every prompt byte visible | Depends on framework depth |
| Learning curve | Gentle — requests + numpy only | Steep — LCEL, state graphs, pydantic attrs |
| Tools | Manual JSON Schema (educational) | `@tool` auto-schema from type hints |
| Best for | Learning, audit, simple needs | Complex agentic systems, fast production |

**Recommendation:** start with `rag-vanilla` to understand; move to `rag-langchain` when agentic behavior is needed.

## Docs

**Strategy documentation (per stage / per strategy, Mermaid diagrams + real build numbers):**

- `docs/pipeline/` — dataset pipeline from raw corpus to QA gate
  - [index](docs/pipeline/index.md) · [source corpus](docs/pipeline/01-source-corpus.md) · [chunking](docs/pipeline/02-chunking.md) · [parents & windows](docs/pipeline/03-parents-windows.md) · [embedding](docs/pipeline/04-embedding.md) · [storage & indexes](docs/pipeline/05-storage-indexes.md) · [QA & evaluation](docs/pipeline/06-qa-eval.md)
- `docs/chatbot/` — how a question becomes a cited answer
  - [index](docs/chatbot/index.md) · [query pipeline](docs/chatbot/01-query-pipeline.md) · [retrieval strategies](docs/chatbot/02-retrieval-strategies.md) · [rerank](docs/chatbot/03-rerank-strategy.md) · [agent & tool calling](docs/chatbot/04-agent-tool-calling.md) · [validity & safety](docs/chatbot/05-validity-safety.md)

**General guides:**

- `docs/architecture-overview.md` — full RAG explanation (both stacks, tool-calling, pitfalls)
- `docs/pipeline-guide.md` — dataset build spec
- `docs/retrieval-indexes.md` — HNSW vs IVF vs FLAT
- `docs/operations-runbook.md` — start-up, daily use, troubleshooting
- `docs/examples/sample-questions.txt` — 5 sample questions

## Roadmap

1. Full embedding + FAISS index (`IndexFlatIP`)
2. recall@8 ≥ 0.8 on golden queries
3. Hybrid dense + BM25 + rerank — **all implemented** (see above)
4. Validity layer filled & verified → vector-DB payload

## License

Source corpus: **CC BY 4.0** — must cite: *Qausar, Prabowo, Arifin & Isa (Binus University) — Indonesian Legislation Norm-Unit Corpus, via Mendeley Data, DOI 10.17632/8wpjvhh7g2.1*. Source PDFs from Indonesian government JDIH portals, not redistributed here.
