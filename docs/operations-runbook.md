# Operations Runbook — Start, Daily Use, Troubleshoot

> Consolidated English guide. Replaces `README-CARA-MENYALAKAN.md` (removed).
> Machine in this guide: lightweight `qwen3-06b` model on CPU. All commands run in PowerShell. Quote paths (they contain spaces).

## 1. Start Docker + Milvus

1. Open **Docker Desktop**, wait for `Engine running` (bottom-left).
2. In PowerShell:

```powershell
cd C:\milvus
docker compose up -d
docker ps
```

Expect 3 `Up` lines: `milvus-standalone`, `etcd`, `minio`. First start needs ~1 min (Milvus loading).

Stop later (data is safe):

```powershell
cd C:\milvus
docker compose down
```

Or one-click everything (Docker → backend → frontend): `.\run_all.bat` from the repo root.

## 2. Enter the project + venv

```powershell
cd "<repo-root>"
.\.venv\Scripts\Activate.ps1
```

## 3. Check ACTIVE (mandatory, 5 seconds)

```powershell
Get-Content "pipeline\result\embeddings\ACTIVE"
```

Must read `qwen3-06b` (this laptop uses it, not `qwen3-4b`). Fix if wrong:

```powershell
Set-Content -LiteralPath "pipeline\result\embeddings\ACTIVE" -Value "qwen3-06b"
```

## 4. Test Milvus connectivity (no model, fast)

```powershell
cd "<repo-root>\pipeline"
..\.venv\Scripts\python.exe smoke_milvus.py --n 5 --top 3
```

Good output: `63,231 entities` + `score ~1.0`. Ignore `PyMilvusDeprecationWarning`.

## 5. Test one question (uses the model, slow on first load)

```powershell
# still in pipeline/
..\.venv\Scripts\python.exe insert_milvus.py --demo "who holds sovereignty under the 1945 Constitution?"
```

Expect top-1 `chunk=uud-1945::p::1::02`.

Switch exact vs HNSW (optional):

```powershell
..\.venv\Scripts\python.exe insert_milvus.py --rebuild-index --index flat
..\.venv\Scripts\python.exe insert_milvus.py --rebuild-index --index hnsw
```

## 6. Daily use (hybrid RAG — the path we actually use)

```powershell
cd "<repo-root>\rag-vanilla"
..\.venv\Scripts\python.exe answer.py -q "Who holds sovereignty under the 1945 Constitution?" --dry-run --show-context
```

Flags:

- default = hybrid (dense + BM25) + rerank (best)
- `--no-hybrid` = dense only (for comparison)
- `--no-rerank` = no rerank (fast, worse ranking)
- `--doc-type UUD` = type filter
- drop `--dry-run` for live LLM answers (needs `.env` with API key)

More examples:

```powershell
..\.venv\Scripts\python.exe answer.py -q "How long is the presidential term and how often may they be re-elected?" --dry-run --show-context
..\.venv\Scripts\python.exe answer.py -q "Who holds governmental power under the 1945 Constitution?" --dry-run --show-context --doc-type UUD
```

## How to read results

```
[S1] score 0.48 UUD — Chapter I Article 1 (parents)
```

- `S1` = rank 1, `score` = rerank P(yes) (not cosine)
- `parents` = full article text (for the LLM), `parents_windows` = window slices
- Big gap (0.48 vs 0.08) = confident. Tight (0.86 vs 0.79) = uncertain, add filters.

Inspect raw chunk/parent text:

```powershell
# from pipeline/
$line = (Select-String -LiteralPath "result\chunks.jsonl" -Pattern "uud-1945::p::1::02" | Select-Object -First 1).Line
($line | ConvertFrom-Json).embedding_input
$pline = (Select-String -LiteralPath "result\parents.jsonl" -Pattern "uud-1945::p::1" | Select-Object -First 1).Line
($pline | ConvertFrom-Json).text
```

## Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| `Connection refused … 19530` | Milvus down | Back to step 1 (`docker compose up -d`) |
| `dimension mismatch 10240 vs 4096` | ACTIVE vs collection model mismatch (2560=4b, 1024=06b) | Back to step 3, unify on `qwen3-06b`. To switch models: `insert_milvus.py --drop` (re-insert 63k) |
| `No module named 'torch'` | ML venv incomplete | `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu transformers>=4.40 numpy` or `pip install -r requirements.txt` |
| All scores low (<0.2) + off-topic | Question outside the 378 docs (e.g. PP 13/2009 is not in the corpus) | Check `data\corpus\documents.csv`; correct answer = `No basis found in corpus.` |
| `norma_tokens.txt missing` | BM25 index not built | From `rag-vanilla`: `..\.venv\Scripts\python.exe bm25.py --build` (once) |

## Memory hook

```
Docker up → ACTIVE qwen3-06b → smoke → demo → answer
```
