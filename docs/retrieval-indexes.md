# Vector Search: HNSW vs IVF vs Brute Force (Exact)

> Consolidated English guide. Replaces `pipeline/README-retrieval.md` (removed).
> Pure explanation — no code changed.

## 1. Where search happens in this pipeline

| Place | Current search type | File |
|---|---|---|
| **Milvus (local Docker)** | `HNSW` + metric `IP` (index built by `insert_milvus.py` at collection creation; search uses `ef: 64`) | `insert_milvus.py`, `smoke_milvus.py` |
| **Eval (pure numpy)** | **Brute force / exact** — `emb @ q` over all vectors | `eval/eval_retrieval.py` (`scores = emb @ vecs.T`) and Colab cell 7 |

So an **exact-search baseline already exists** and always serves as ground truth. HNSW/IVF only qualify if they approach this baseline with cheaper latency.

## 2. Three search types in brief

| Type | How it works | Recall | Speed on our data (63k × 1024/2560) | When to use |
|---|---|---|---|---|
| **FLAT (brute force / exact)** | Compare the query against **every** vector, take top-k. Absolute truth. | **100%** | ~10–50 ms (still very fast) | Datasets ≤ hundreds of thousands; official baseline |
| **HNSW** | Layered nearest-neighbor **graph**; greedy walk from entry point. | ~99–100% if `ef` large enough | ~1–5 ms | Default production pick; reliable at all sizes |
| **IVF (IVF_FLAT etc.)** | Space split into `nlist` **clusters**. Query checks only `nprobe` nearest clusters. | Depends on `nprobe/nlist` | ~2–10 ms | Pays off at **millions** of vectors |

Metric **`IP` (inner product)** everywhere, because our vectors are L2-normalized — dot = cosine. Never changes with the index type.

## 3. What changes when you switch types

Only **two touchpoints** — everything else (collection schema, insert, `ids.jsonl`, query encoder) stays identical:

### a) At collection creation — `insert_milvus.py`, `create_index`

| Type | `index_params` | Notes |
|---|---|---|
| HNSW *(active)* | `{"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}` | `M` = graph width; `efConstruction` = build quality |
| IVF_FLAT | `{"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 512}}` | Rule of thumb `nlist ≈ 4·√N` (63k → ±1,000; commonly 128–1024). Milvus refuses `nlist` > vector count |
| FLAT (exact) | `{"index_type": "FLAT", "metric_type": "IP", "params": {}}` | No params; also the automatic result of creating a collection **with no index** |

> Index is **bound at collection creation**. Changing type = drop (`--drop`) + re-insert, **or** a side-by-side collection for benchmarking (same pattern as the 4B/0.6B split — see `emb_paths.py`).

### b) At search time — `search` params

| Type | Search `param` | Raising it… |
|---|---|---|
| HNSW *(active: `ef: 64`)* | `{"metric_type": "IP", "params": {"ef": 64}}` | `ef` 64 → 128 → 256: recall up, latency up |
| IVF | `{"metric_type": "IP", "params": {"nprobe": 32}}` | more clusters checked → recall up |
| FLAT | `{"metric_type": "IP", "params": {}}` | no params — always exact |

Search params change freely **without re-inserting** — only the index is bound to the collection.

## 4. Exact brute force — two paths already available

1. **Pure numpy** (no Docker/Milvus): `emb @ q` in local eval and Colab cell 7. 63k × 1024 on CPU is milliseconds per query — why official eval uses it as ground truth.
2. **Milvus with no index / FLAT index**: the same work inside Milvus.

Evidence: Milvus HNSW top-5 (`ef` 64 and 256) was **identical in score and order** to numpy exact for the UUD-1945 sovereignty query — our ANN is healthy.

## 5. Traps we have hit (important!)

- **Index builds asynchronously.** Seconds after insert/flush, HNSW is not ready; queries can return wrong results (real case: first demo returned a different chunk with lower score; next run matched exact). Fix: wait for index / run `smoke_milvus.py` first.
- **IVF is fragile on small data.** At 63k vectors IVF shows no benefit and more fragile recall vs `nprobe`; FLAT is simpler.
- **Never mix metrics.** All collections are `IP` + L2-normalized vectors. Searching with `L2` on normalized vectors inverts the ranking scale.

## 6. Recommendations for this dataset

| Need | Pick |
|---|---|
| Benchmark / ground truth | **Numpy exact** (local eval & Colab cell 7) — keep as is |
| Current dataset (63k) | **FLAT** or HNSW both fast; HNSW `ef` 64–128 safe for production |
| Later millions of vectors / tight latency | Larger-`ef` HNSW, or start testing `IVF_FLAT` (`nlist` ±1,000, `nprobe` from 64) |
| ANN pass rule | recall top-8 vs exact ≥ 0.99 before trusting latency numbers |

Measure ANN-vs-exact recall by running exact eval (numpy) and Milvus search over the same golden set and comparing per-query `top_ids` — exactly the comparison done in this project.

## 7. One-sentence summary

**Only `index_params` at collection creation and `params` at search time change (`ef` for HNSW, `nprobe` for IVF, empty for FLAT) — schema, insert, encoder, and the "model index = model query" rule are untouched.**
