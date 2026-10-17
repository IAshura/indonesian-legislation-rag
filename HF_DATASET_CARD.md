---
license: cc-by-4.0
language:
  - id
task_categories:
  - text-retrieval
  - question-answering
tags:
  - legal
  - indonesia
  - legislation
  - rag
  - retrieval
  - norm-units
  - bm25
  - qwen3
size_categories:
  - 10K<100K
pretty_name: Indonesian Legislation RAG Dataset (378 regulations, 1945-2024)
---

# Indonesian Legislation RAG Dataset

Chunked, embed-ready, provenance-verified dataset built from the public corpus
[*Indonesian legislation as data: 90,588 norm units*](https://data.mendeley.com/datasets/8wpjvhh7g2/1)
(DOI `10.17632/8wpjvhh7g2.1`) for Retrieval-Augmented Generation over Indonesian law.

> **Attribution (required by CC BY 4.0):**
> *Qausar, Prabowo, Arifin & Isa (Binus University) — Indonesian Legislation
> Norm-Unit Corpus, via Mendeley Data, DOI 10.17632/8wpjvhh7g2.1.*
> The chunking pipeline, dataset build, embeddings, evaluation, and validity
> layer are our own work.

## What is in this dataset

This repo contains **two layers**:

**Layer 1 — processed RAG artifacts** (built by this project's deterministic
pipeline, v4.1):

| File | Content | Count |
|---|---|---|
| `chunks.jsonl` | Embed-ready chunks; one line = one chunk with `embedding_input`, `content_role`, `is_fragment`, citation & page/line provenance | **74,218** (66,292 `norma` · 4,704 fragments) |
| `parents.jsonl` | Whole articles / administrative blocks (small-to-big context for the LLM) | **11,413** |
| `parents_windows.jsonl` | Deterministic ≤1,800-char windows (150-char overlap) of long parents | **14,689** |
| `documents.jsonl` | Register of 378 regulations with official titles + UU 12/2011 classes | **378** |
| `validity.jsonl` | Legal-validity layer (status default `pending_enrichment` — never guessed) | per-document |
| `quarantine.jsonl` | Discarded rows + machine-measurable reasons | **2,209** |
| `manifest.json` | Source hashes, build config, funnel counts, QA results | — |
| `embeddings/qwen3-06b/` | float32 L2-normalized vectors, 1024-d (`Qwen/Qwen3-Embedding-0.6B`) + `ids.jsonl` row-mapping | **63,231** |
| `embeddings/qwen3-4b/` | Same, 2560-d (`Qwen/Qwen3-Embedding-4B`) | **63,231** |
| `lexical/norma_tokens.txt` | Pre-tokenized corpus for BM25 (lexical retrieval) | — |

**Layer 2 — original source corpus** (as distributed by Mendeley):

| File | Content |
|---|---|
| `corpus/units.csv` | Main table: 90,588 norm units + provenance + IndoBERT audit columns |
| `corpus/documents.csv` | 378 regulations; identity from title blocks, legal-force classification |
| `corpus/texts/` | 378 page-delimited plain-text files that every provenance address resolves into |
| `corpus/checksums.md5` | MD5 integrity checksums |

## Where the data comes from

- 378 official Indonesian regulations (1945–2024) sourced from Indonesian
  government legal-documentation portals (JDIH), segmented into norm units by
  the corpus authors (Binus University).
- Source PDFs are official public documents and are not redistributed;
  `documents.csv` records each file's identity so extraction can be reproduced
  from the originals.

## What processing was applied

All steps are deterministic (seed `20260907`) and scripted in `pipeline/build_dataset.py`
in the companion code repository (link added once the GitHub repo is public);

1. **Schema validation & hashing** — fail-fast on `units.csv`/`documents.csv`;
   SHA256 of every source recorded in `manifest.json`.
2. **Identity restoration** — official titles + topics parsed from title
   blocks (`Undang-Undang Nomor 7 Tahun 2017 tentang ...`), not filenames.
3. **Structure-based chunking** — 1 norm unit = 1 chunk; boundaries follow the
   legal hierarchy (chapter/article/verse), never token slicing. Heading units become
   context labels, never chunks; no auto-merge between units.
4. **Granular content roles** — `norma`, `pertimbangan`, `dasar_hukum`,
   `preamble`, `administratif`, `penjelasan`, … with `role_source` audit field;
   legacy `part` (norma/administratif) kept as alias.
5. **Short fragments kept, not merged** — 4,704 chunks flagged
   `is_fragment: true` (skipped by embedding by default). Zero silent loss.
6. **Quarantine with reasons** — 2,209 rows (587 exact duplicates,
   1,622 heading-like short bodies), each with a written reason.
7. **Parent–child hierarchy** — retrieval on child (verse, precise), answers
   use whole articles from `parents.jsonl`; long parents split into
   deterministic ≤1,800-char windows with 150-char overlap.
8. **Self-contained embedding input** — `embedding_input = official title —
   citation: body` so every vector is understandable in isolation.
9. **Embedding** — two complete index sets: Qwen3-Embedding-0.6B (1024-d,
   last-token pooling) and Qwen3-Embedding-4B (2560-d). Iron rule: *the model
   that indexes must equal the model that queries*.
10. **BM25 lexical index** — pre-tokenized text+citation+title for hybrid
    (dense + lexical) retrieval fused with Reciprocal Rank Fusion (RRF k=60).

## Quality assurance

- **Provenance gate:** 300 random chunks re-verified against page + line
  ranges in the original `texts/` → **98% match**, PASS (threshold 90%);
  the 6 unverified samples are trimmed structural markers, not legal content.
- **No orphan chunks, no duplicate chunk IDs.**
- **Accountability:** `74,218 chunks + 2,209 quarantine + 14,161 headings = 90,588` — zero rows lost.
- **Retrieval eval:** recall@8 = **0.75** on 12 hand-written golden queries
  (Qwen3-Embedding-0.6B, hybrid + rerank in the companion code raises this).
- **Legal-validity layer ships empty (`pending_enrichment`) by design** —
  status is only filled with verified evidence, never invented.

## Intended use

- Building Indonesian legal RAG / retrieval systems (dense, BM25, or hybrid).
- Legal-domain chunking research (structure-based vs token-based).
- Fine-tuning or evaluating legal text embeddings for Indonesian.

Example loading:

```python
import json, urllib.request
url = "https://huggingface.co/datasets/Fayizz/indonesian-legislation-norms/resolve/main/chunks.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(url).read().splitlines()]
```

## Limitations

- Chunk boundaries, not full PDFs — rebuild documents needs the Mendeley originals.
- `legal_status` is `null` until the validity layer is enriched (run with care:
  never present unverified status in a legal answer).
- Corpus ends at 2024 regulations; newer law is absent.
- `documents.jsonl` / `validity.jsonl` gap fields may exist pending enrichment.

## Citation

```bibtex
@dataset{qausar_indonesian_legislation_corpus,
  title  = {Indonesian legislation as data: 90,588 norm units},
  author = {Qausar, Prabowo, Arifin and Isa},
  year   = {2024},
  doi    = {10.17632/8wpjvhh7g2.1},
  url    = {https://data.mendeley.com/datasets/8wpjvhh7g2/1},
  note   = {Binus University. License: CC BY 4.0}
}
```

## License

The processed artifacts derive from the CC BY 4.0 corpus, so this dataset is
also **CC BY 4.0** — share with the attribution above. The associated code
repository is MIT-licensed (see the GitHub repo).
