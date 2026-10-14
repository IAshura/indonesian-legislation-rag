# Architecture Overview — Indonesian Legislation RAG (Vanilla vs LangChain)

> Consolidated English guide. Replaces the former scattered notes
> (`PENJELASAN-LENGKAP-RAG.md` and both per-stack READMEs — now removed).
> All function/class names verified against the actual code; all numbers
> come from `pipeline/result/manifest.json`.

## 1. The idea in one minute

**Problem:** an LLM is fluent but does not know Indonesian regulations, hallucinates, and offers no auditable source.
**Solution (RAG):** before answering, **retrieve** the relevant regulation chunks from our dataset, **inject** them into the prompt, and force the LLM to answer **only from that text**. Answers become accurate, cited, auditable.

**Analogy:** the LLM is a smart lecturer who has not memorized the law. RAG slides photocopies of the relevant articles across the desk — page numbers included — before they may speak.

| | `rag-vanilla/` | `rag-langchain/` |
|---|---|---|
| Philosophy | Write everything yourself | Framework tidies it up |
| Core | RAG CLI + 60-line agent loop | LCEL chain + agent graph + reflective graph |
| For | Understanding how it works | Building complex agentic systems |

Both are tested: vanilla passes `verify_setup.py` 11/11, LangChain passes `verify_setup.py --full` 8/8, both run fully in `--dry-run` (no API key) with real retrieval over 63,231 embedding vectors (top-8, best score ~0.63).

## 2. RAG in five steps

```
user question
   │
   ▼
[1] EMBED    : question → vector (1024 numbers)
[2] SEARCH   : nearest chunk vectors (dot product / cosine)
[3] INJECT   : chunk texts into prompt as [S1], [S2], …
[4] GENERATE : LLM answers ONLY from [S1]..[S8]
[5] CITE     : answer must name sources; source list printed separately
```

Vector similarity ≈ meaning similarity. "Who holds sovereignty?" and the sovereignty article land close together even with different wording.

Two patterns from the concept guide (`docs/llm-injection-tool-calling.html`):

- **Fixed pipeline:** deterministic retrieve → inject → generate. For ordinary RAG. Implemented in `rag-vanilla/answer.py` and `rag-langchain/rag_chain.py`.
- **Agent / tool calling:** the LLM gets *tools* (callable functions) and decides the next step itself: search first? open an article? check validity? Loop until it says "done". Implemented in `rag-vanilla/agent.py` and `rag-langchain/agent_graph.py` + `reflective_graph.py`.

Use the pipeline when the pattern is fixed ("find article X"); use the agent when steps are unpredictable (e.g. "is the law I quoted still valid?" → needs `check_validity`, not retrieval).

## 3. Data: 90,588 norm units

All pipeline output lives in `pipeline/result/`:

| File | Content | Rows |
|---|---|---|
| `chunks.jsonl` | Searchable pieces (1 line = 1 chunk) | 74,218 |
| `parents.jsonl` | Whole articles/sections | 11,413 |
| `parents_windows.jsonl` | Long parents split into windows | 14,689 |
| `documents.jsonl` | Regulation metadata | 378 |
| `embeddings/qwen3-06b/embeddings.npy` | float32 vector matrix | 63,231 × 1024 (259 MB) |
| `embeddings/qwen3-06b/ids.jsonl` | row-i vector ↔ chunk_id | 63,231 |
| `embeddings/qwen3-06b/meta.json` | Model + encoding info | — |

One `chunks.jsonl` row (keys the code actually uses):

```
chunk_id      "uud-1945::p::1::02"
parent_id     "uud-1945::p::1"
document_id   "uud-1945"
official_title "Undang-Undang Dasar 1945"
citation      "Pasal 1 UUD 1945"
content_role  "norma" | "penjelasan" | …   (pipeline keeps norma for embeddings)
text          "…Sovereignty lies in the hands of the people…"
provenance    {"page_number": 3, "line_start": 12, "line_end": 14}
```

ID pattern (memorize this):

```
uud-1945                  ← document
    └── uud-1945::p::1     ← parent (whole article)
           └── uud-1945::p::1::02      ← chunk
           └── uud-1945::p::1::w01     ← window (only if parent > ~1800 chars)
```

**Search with chunks, answer with parents** (parent-document retrieval): small chunks = sharp search; full articles = untruncated answers for the LLM.

## 4. Embeddings

Model: `Qwen/Qwen3-Embedding-0.6B` (in local HuggingFace cache → **works offline**). Dim **1024**, **last-token** pooling, L2-normalized, max 1024 input tokens.

- **Corpus embeddings** — precomputed, in `embeddings.npy`.
- **Query embeddings** — computed at runtime by `embedder.py` / `QwenQueryEmbedder`.

Iron rule (break it and search collapses): the query must use the **exact same** format as the corpus:

```
Instruct: Given a user question about Indonesian legislation, retrieve the most relevant legal provisions.
Query: <user question>
```

…plus left padding, last-token pooling, L2 normalize. Locked in one function (`embed_query()`).

Because all vectors are L2-normalized, **dot product = cosine similarity**. Scoring 63,231 × 1024 in NumPy takes a fraction of a second — no vector DB needed at this scale.

## 5. Stack A — `rag-vanilla/` (pure Python, only `requests` for networking)

Read order follows data flow:

```
config.py → dataset.py → embedder.py → retriever.py → prompt.py → llm_client.py → answer.py
                                                                              ↘ tools.py → agent.py
```

| File | Job |
|---|---|
| `config.py` | Single door for constants. `active_embedding_dir()` reads `ACTIVE`; `get_llm_config()` reads `.env` (own loader, no extra dep): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (default `gpt-4o-mini`), `TEMPERATURE=0.2`, `MAX_TOKENS`, `TIMEOUT`. `TOP_K=8` (eval recall@8 = 0.75 proves it). |
| `dataset.py` | Single door to data. Lazy `chunks/parents/windows/documents` dicts; `context_for_parent()` turns "vector row 17,482" into "Article 1, page 3"; singleton `dataset()`. |
| `embedder.py` | Question → vector. `load_meta()`, lazy model load (`local_files_only=True`), `embed_query()` (Instruct/Query + left-pad + last-token + L2; fixes BFloat16→NumPy via `vec.float()` first). |
| `retriever.py` | `ExactRetriever` (brute-force, `mmap_mode="r"`, `M @ q`), optional `milvus_search`, `retrieve()` dispatcher, `konteks_kaya()` → parent-joined, de-duplicated contexts. `HybridRetriever` adds BM25+RRF (below). |
| `bm25.py` | Lexical BM25 Okapi (k1=1.2, b=0.75). `python bm25.py --build` → `pipeline/result/lexical/norma_tokens.txt` (63,231 lines aligned with `ids.jsonl`). `fusi_rrf()` merges dense+lexical **ranks** (scale-free). Missing index → safe dense-only fallback. |
| `reranker.py` | Phase-2 cross-encoder `Qwen3-Reranker-0.6B`: reads query+article *together*, score `P(yes)`. Phase 1: cosine takes 12; phase 2: rerank → top-k. Fails to load → safe cosine fallback. Toggle `RERANK=1|0`, `--no-rerank`. |
| `prompt.py` | `SYSTEM_PROMPT` ("work only from given context; cite articles; say you don't know if absent" — the anti-hallucination shield). `build_context_block()` → `[S1] (score 0.63) <title + citation>\n<text>`, capped at 12,000 chars. `build_messages()` → OpenAI-style messages. |
| `llm_client.py` | Single door to LLMs: POST `{BASE_URL}/chat/completions` via `requests`, tools support, timeout/HTTP handling. `dry_run()` prints what *would* be sent. `MockLLM` (scripted replies) tests the agent loop with no key. |
| `answer.py` | Fixed architecture: embed → retrieve → rich context → build prompt → chat. CLI: `-q`, `--top-k 8`, `--doc-type`, `--year`, `--dry-run`, `--show-context`, `--no-hybrid`, `--no-rerank`. |
| `tools.py` | 4 narrow tools: `search_norma` / `get_article` / `get_document` / `check_validity` (reads `legal_status`). Manual OpenAI-style JSON Schema + `execute_tool()` dispatcher (validates names, clamps `top_k` 1–10, truncates to 600 chars). |
| `agent.py` | The loop (the whole "framework" in ~40 lines): send messages+tool schemas → if `tool_calls`, run `execute_tool`, append `role:"tool"` → repeat; else final answer. `MAX_ITERATIONS=5`. |
| `verify_setup.py` | 11 health checks (data, meta, embedder, retrieval scores, tools, mock agent loop). Run this first when something breaks. |

Real effect: for the sovereignty question, cosine alone misses Article 1 (rank > 10); BM25 scores it 31.3 and RRF lifts it to #1.

## 6. Stack B — `rag-langchain/` (same bridge, framework glue)

| Vanilla has… | …becomes in LangChain |
|---|---|
| `dataset.LegislationDataset` | `loader.DatasetRingkas` + `Document` wrappers |
| `embedder` + `retriever` | `retriever.QwenQueryEmbedder` + `LegislationRetriever(BaseRetriever)` |
| manual `prompt` + `chat` | LCEL chain `retriever \| prompt \| llm \| parser` |
| manual JSON tool schemas | `@tool` (schema auto from type hints) |
| `while` agent loop | LangGraph `StateGraph` |
| manual rerank call | `QwenReranker` as LCEL step / tool / graph node |
| — | bonus `reflective_graph.py` (self-critiquing agent) |

Notable files: `config.py` (+`LLM_BACKEND=api|ollama`), `loader.py` (`Document(page_content, metadata)`), `retriever.py` (`PrivateAttr` for heavy state — required in LangChain 1.x), `llm.py` (`get_llm()` factory + `MockAgentLLM`), `rag_chain.py` (LCEL + `RunnableParallel`; `batch`/streaming/retry free), `tools_lc.py`, `agent_graph.py` (`START → agent ⇄ tools → END`, `MAX_PUTARAN=6`), `graph_rag_reflektif.py`/`reflective_graph.py` (CRAG: `retrieve → grade → rewrite (max 2) → generate`).

## 7. Tool calling in one paragraph

**Tools are not run by the LLM — the LLM only sends an execution request; your code runs it.** Argument JSON must be `json.loads`-ed defensively; results return as `role:"tool"` with matching `tool_call_id`; the LLM decides when to stop — hence mandatory iteration caps.

## 8. Pitfalls already fixed (memorize these)

| # | Trap | Symptom | Fix in place |
|---|---|---|---|
| 1 | BFloat16 → NumPy | `TypeError: Got unsupported ScalarType BFloat16` | `vec.float()` before `.numpy()` |
| 2 | Windows cp1252 | `UnicodeEncodeError` on emoji/non-Latin print | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` first line |
| 3 | Extra fields in assistant messages | Provider rejects replayed history | Normalize to `{role, content, tool_calls}` before append |
| 4 | Raw dict as graph input | LangGraph rejects | Send `(role, text)` tuples |
| 5 | `RunnableParallel` not in `import *` | `NameError` in LCEL chain | Explicit per-name import |
| 6 | AST ClassDef has no `.args` | PDF generator crash on classes | Branch: ClassDef → bases, FunctionDef → params |

## 9. Run

```bash
cd rag-vanilla        # or rag-langchain
copy .env.example .env
# fill LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (langchain also: LLM_BACKEND=api|ollama)
python verify_setup.py           # vanilla: 11 checks
python verify_setup.py --full    # langchain: 8 checks incl. chain & graph mocks
python answer.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context
python agent.py  -q "Is the Job Creation Law still valid?" --dry-run
```

## 10. FAQ (short)

- **Why cite articles?** Provenance (page/line) is this dataset's biggest edge — answers without citations waste it.
- **Local LLM (Ollama)?** LangChain: `LLM_BACKEND=ollama`. Vanilla: `LLM_BASE_URL=http://localhost:11434/v1`.
- **Why no vector DB?** 63k × 1024 = 259 MB; NumPy dot product < 1 s. Milvus path (`milvus_search()`) ready for millions.
- **`--dry-run` vs mock?** `--dry-run` = real code, only LLM call faked (retrieval real). Mock = scripted fake LLM for deterministic loop tests.
- **Top-1 score only 0.63 — is retrieval bad?** No. Legal cosine rarely exceeds 0.7; eval recall@8 = 0.75. Relative order matters, not absolute score.
