# Dataset Build Guide — Combined v4.1 (Canonical)

> Consolidated English guide. Replaces `pipeline/README-build-v4.md` (moved to `pipeline/README.md`).
> Source numbers: `pipeline/result/manifest.json`. Corpus: Mendeley DOI `10.17632/8wpjvhh7g2.1`, **CC BY 4.0** — credit *Qausar, Prabowo, Arifin & Isa — Binus University*.
> This folder is the only active pipeline (v4.1 combines three archived experiments in `archive/experiments/` + 10 agreed design decisions).

## 10 design decisions

1. **Combined layout kept** — chunk / parent / windows / documents / quarantine / manifest, one deterministic pipeline.
2. **Unit preservation:** 1 row of `units.csv` = 1 chunk; IDs and provenance never mutated.
3. **Parent–child:** retrieve on child (unit/verse), answer with full-article context from `parents.jsonl`.
4. **Audit + manifest:** SHA256 of sources, reasoned disposition for every row, usage guide in `manifest.json`.
5. **No auto-merge between units (schema 4).** Short fragments (< 8 tokens / `needs_review_short`) are **kept with `is_fragment: true`**. The embedder skips them by default — clean index, zero rows lost.
6. **Global stable `parent_id`:** `{document_id}::p::{article}` or `{document_id}::adm::{nn}`. Unique corpus-wide — safe as vector-DB payload.
7. **Long parents split into deterministic windows.** `parents.jsonl` keeps full text (citation/audit); `parents_windows.jsonl` keeps ≤1,800-char slices with 150-char overlap (`window_id = {parent_id}::w{nn}`) — fits LLM context limits.
8. **Granular `content_role`** replaces the binary flag: `norma`, `pertimbangan`, `dasar_hukum`, `preamble`, `administratif`, `ketentuan_peralihan`, `ketentuan_penutup`, `penjelasan` + audit field `role_source` (marker | flag+artifact | flag+keyword | keyword | default). Legacy `part` (norma/administratif) kept as compatibility alias.
9. **Validity status** via `validity.jsonl` + `enrich_validity.py` — *offline default, never guessed*; fill `legal_status` only with evidence. Without it, legal RAG risks citing repealed norms.
10. **Old experiments archived** — canonical dataset only here.

## Raw → embedding-ready pipeline

`python build_dataset.py` reads `../data/corpus/units.csv` + `documents.csv` + `texts/` and:

1. **Schema-validates** both CSVs (fail-fast) and records source **SHA256** in the manifest.
2. **Restores regulation identity** from the `documents.csv` title block → official title + topic (`… Nomor 7 Tahun 2017 tentang …`) — the most valuable embedding context.
3. **Per-document streaming** ordered (`page → line → unit_id`) with carry-forward of **only chapter & article** from heading units — *verses never carry forward*; `citation.ayat = null` means a single-norm article (valid per UU 12/2011 appendix & Akoma Ntoso).
4. Headings (`heading_*`) become context labels, **never embedded**.
5. Selection: ok/administrative bodies → chunk; `structural_heading` < 200 chars → quarantine (title debris); exact duplicates → reasoned quarantine.
6. Short fragments → **kept as chunks** with `is_fragment: true`.
7. Granular `content_role` + audit `role_source`.
8. **Global parents** per article/admin block + deterministic **windows**.
9. `embedding_input = official_title - citation: body` — self-contained per vector.
10. **Automatic provenance QA** (300 samples checked against source lines in `texts/`; 90% threshold — failed gate breaks the build) + cross-validation: 0 orphan chunks, 0 duplicate `chunk_id`s.

**This build's funnel:** 90,588 rows → 74,218 chunks (4,704 flagged fragments; 66,292 norma) · 11,413 parents · 14,689 windows · 2,209 reasoned quarantines · **QA 98% — PASS**.

## `result/` contents

| File | Content |
|---|---|
| `chunks.jsonl` | 74,218 embed-ready records (`embedding_input`, `content_role`, `is_fragment`, provenance) |
| `parents.jsonl` | Full article texts + `window_ids` (LLM context) |
| `parents_windows.jsonl` | ≤1,800-char slices (+overlap) for token-bounded context |
| `documents.jsonl` | 378-regulation register + empty `legal_status` columns ready for enrichment |
| `validity.jsonl` | Validity layer (default `pending_enrichment`) |
| `quarantine.jsonl` | Discarded rows + measured reasons |
| `manifest.json` | Source hashes, funnel, policies, schema, QA outcome |

## Feeding an embedding model (next stage)

One iron rule: **model index = model query** — switching models means re-embedding (model metadata in `embeddings/meta.json`).

```powershell
cd pipeline
python embed_example.py                       # default: part=norma, fragments SKIPPED
python embed_example.py --part all --include-fragments   # most complete index
python embed_example.py --model intfloat/multilingual-e5-large --batch-size 32
```

External API mode: `python embed_example.py --engine api --base-url https://api.openai.com/v1 --model text-embedding-3-large` (key via `EMBED_API_KEY`).

Output: `result/embeddings/embeddings.npy` (L2-normal float32, ordered ⇄ `ids.jsonl`), `ids.jsonl` (pointer: chunk_id/parent_id/part/content_role/year), `meta.json`. Serve with `faiss.IndexFlatIP(dim)` (dot = cosine). e5 models auto-prefix `passage:`/`query:`.

**Answer flow (small-to-big):** retrieve candidates → join `parent_id` → take full article text (or its window) → LLM → cite `citation_str` + `official_title` + provenance page. Pure filters: `content_role == "norma"` (and/or `part == "norma"`), `doc_type`, `legal_force_axis`, `year`.

## Next stages (in order)

1. **Validity:** `python enrich_validity.py --live` (best-effort candidate search on peraturan.bpk.go.id; URLs stored) → verify → fill `legal_status` & join into vector-DB payload. **Never invent status.**
2. **Measure retrieval once Qwen embeddings exist:** `python eval/eval_retrieval.py --k 8` vs `eval/golden_queries.jsonl` (recall@8; report in `eval/golden_report.json`; ≥0.8 go; <0.6 change model/top-k first).
3. **Pre-production mix:** dense embeddings + BM25 over `text_lexical`, then rerank — usually cheaper recall than swapping models.
4. **Full embedding:** run `embed_example.py` in full (GPU recommended).

## Regenerate & audit

```
python build_dataset.py            # deterministic (fixed seed, fixed sort)
# manifest.validation: qa_gate PASS = 0 orphans, 0 duplicates, match_rate >= 0.90
```

Current model recommendations: `intfloat/multilingual-e5-base|large`, `paraphrase-multilingual-mpnet-base-v2`, or Indonesia-specific models. Keep `meta.json` as the vector-version marker.
