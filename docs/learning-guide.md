# Learning Guide — From Raw Data to Retrieval

> Consolidated English guide. Replaces `PROMPT-BELAJAR-PIPELINE.md` (removed).
> Rule: read the real project files; make new HTML only under `docs/learning/`. Never modify project files while studying.

## Goal

A static, dark-mode, multi-page site explaining **every stage of this project in detail for total beginners** — raw data to Milvus retrieval. Language: English. No internet dependencies (pure HTML/CSS/JS, self-made CSS/SVG visuals).

Root: `docs/learning/` (renamed from `docs/belajar/`; filenames now English — see mapping below).

Start by reading `README.md` and `pipeline/README.md` for the big picture, then the files each chapter names. Quote real snippets + file/line numbers. Never invent numbers, field names, or behavior — `pipeline/result/manifest.json` is the official source of numbers. Input/output examples must use **real project data**, not invented samples.

## Chapter → file → required content

| Ch | File | Required content |
|---|---|---|
| 00 | `index.html` | Full pipeline map (flow diagram), chapter list, how to use the site, corpus attribution (CC BY 4.0, Qausar et al. — Binus, DOI 10.17632/8wpjvhh7g2.1) |
| 01 | `01-rag-basics.html` (was `01-dasar-rag.html`) | RAG for beginners: data → embedding → vector → similarity search → answer; why law needs it; small-to-big diagram (verse retrieval → whole-article answer) |
| 02 | `02-raw-corpus.html` (was `02-data-mentah.html`) | Source corpus deep-dive: `data/corpus/units.csv` (**every column**), `documents.csv`, `texts/` (`=== PAGE n ===` markers), `checksums.md5`. Real example: 1 raw `units.csv` row (e.g. UUD 1945), show how (file, page_number, line_start–line_end) points into `texts/` |
| 03 | `03-build-dataset.html` | `pipeline/build_dataset.py` **function-by-function**: constants & rules (WINDOW_MAX_CHARS 1800, overlap 150, fragments <8 tokens…), `build_doc_meta` (title-block identity restore), `role_of` + `looks_like_admin_artifact` (`content_role` + `role_source`), class `B` (chapter/article carry-forward — **verses never forwarded**, global `parent_id` `{document_id}::p::{article}` & `::adm::` blocks, dedup + reasoned quarantine), deterministic windows, 300-sample provenance QA, manifest writing. Must show **1 units.csv row → 1 chunks.jsonl record** (real JSON) → parents.jsonl slice → parents_windows.jsonl slice |
| 04 | `04-build-output.html` (was `04-hasil-build.html`) | `pipeline/result/`: 7 files field-by-field with real records; funnel 90,588 → 74,218 chunks (4,704 fragments) → 11,413 parents → 14,689 windows → 2,209 quarantine → QA 98% PASS (read `manifest.json`) |
| 05 | `05-embedding.html` | Embeddings for beginners (text → vector → cosine); mean-pooling e5 (`embed_example.py`) vs last-token pooling + Qwen3 instructions (`pipeline/colab/`); `embeddings.npy` + `ids.jsonl` + `meta.json`; folder convention & `ACTIVE` file (`emb_paths.py`); `verify_embeddings.py`; iron rule **model index = model query** |
| 06 | `06-milvus-retrieval.html` | Local Milvus (Docker: etcd+minio+standalone, data in `C:\milvus\volumes`); `insert_milvus.py` (collection schema, HNSW/IP index, batched insert, anti-double guard), `smoke_milvus.py` (self-query top-1), separate 4B/0.6B collections, `--demo` |
| 07 | `07-query-eval.html` | `embed_query_qwen.py` (`Instruct:/Query:`, last-token pooling, MRL keep_dim), `eval/golden_queries.jsonl` (12 golden queries) & `eval/eval_retrieval.py` (recall@8, matcher details), how to read `golden_report.json` (recall@8 = 0.750) |
| 08 | `08-hnsw-ivf-flat.html` | Condensed from `docs/retrieval-indexes.md`: HNSW vs IVF vs FLAT, what changes in create_index vs search params, async-index trap, exact-numpy baseline |
| 09 | `09-tracing-one-norm.html` (was `09-pelacakan-satu-norma.html`) | **Capstone**: trace ONE example upstream→downstream with real data — "Sovereignty lies in the hands of the people…" (Constitution Art. 1(2)): units.csv row → chunk uud-1945::p::1::02 → embedding_input → 1024-d vector → ids.jsonl entry → Milvus entity → top-8 results → parent article (small-to-big). Real data at every hop |
| 10 | `10-glossary.html` | Every term (RAG, chunk, parent, window, embedding, cosine, IP, HNSW, IVF, FLAT, pooling, fragment, quarantine, provenance, legal_force_axis, …) |

## Site design (required)

- Elegant tidy dark mode, warm accent, good contrast, readable type, console-styled code blocks with file & line markers.
- Zero internet dependencies; CSS/SVG/HTML visuals only.
- Each chapter: flow diagram (input → transform → output) + schema/field tables.
- `index.html` = illustrated TOC (whole pipeline map + per-stage status), prev/next nav on every page.
- **Real project data** as examples (e.g. real JSON for uud-1945::p::1::02).
- Patient teaching tone, simple analogies for abstract concepts.

## How to work

1. Read `README.md` → `pipeline/README.md` → `pipeline/result/manifest.json` for frame + official numbers.
2. Read each chapter's named files **before writing it** (read, don't guess).
3. Verify numbers from result files; note corrections in the chapter if anything mismatches.
4. Never run data-mutating commands; read only.
5. Final output: `docs/learning/` with `index.html` + one HTML per chapter. Report file list + how to open.

Mandatory per-chapter "input → process → output" with a real case: raw CSV row → chunk record; `embedding_input` → L2-normal float32 vector → ids.jsonl row → Milvus entity; user question → query vector → top-k scores → whole parent article → LLM-ready context.
