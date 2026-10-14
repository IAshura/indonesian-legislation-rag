# RAG Peraturan Indonesia — Tanpa Framework (Python murni)

Implementasi **Arsitektur A (fixed-pipeline RAG)** dan **Arsitektur B (agentic
+ tool calling)** untuk proyek *Indonesian Legislation RAG* — tanpa LangChain,
LlamaIndex, atau framework lain. Semua jaringan LLM ditulis dengan `requests`
agar anatomi protokolnya terlihat. Panduan konsepnya ada di
`docs/llm-injection-tool-calling.html` (bab 1–9).

## Urutan belajar (tiap file bisa dijalankan sendiri)

| # | File | Belajar apa | Uji cepat |
|---|------|-------------|-----------|
| 1 | `config.py` | satu pintu konfigurasi + aturan besi proyek | `python config.py` (tidak ada efek, atau diimpor) |
| 2 | `dataset.py` | JSONL, id 3 tingkat, **small-to-big** | `python dataset.py` |
| 3 | `embedder.py` | vektor query, last-token pooling, aturan besi | `python embedder.py` |
| 4 | `retriever.py` | hybrid dense+BM25 (RRF), exact (mmap), Milvus opsional | `python retriever.py` |
| 4b | `bm25.py` | jalur leksikal BM25 + builder index offline | `python bm25.py --build` |
| 5 | `prompt.py` | anatomi prompt 3 blok + sitasi `[S1]` | `python prompt.py` |
| 6 | `llm_client.py` | POST HTTP ke LLM, `tool_calls`, dry-run | `python llm_client.py` |
| 7 | `tools.py` | skema tool + dispatcher tervalidasi | `python tools.py` |
| 8 | `rag_jawab.py` | Arsitektur A: retrieve → **rerank** → prompt → LLM | lihat bawah |
| 9 | `agent.py` | Arsitektur B: loop tool-calling | lihat bawah |
| 10 | `verify_setup.py` | cek kesehatan lingkungan | `python verify_setup.py` |

## Hybrid search (dense + BM25) — `bm25.py` + `retriever.py`

Pencarian 2 jalur yang digabung **RRF (Reciprocal Rank Fusion)** — fusi
PERINGKAT, bukan penjumlahan skor (skala cosine 0–1 vs BM25 0–20+ tidak
sebanding):

- **Jalur dense**: cosine embeddings.npy — kuat di makna (sinonim, parafrase).
- **Jalur lexical (BM25)**: skor Okapi BM25 (k1=1.2, b=0.75) atas token pasal
  + sitasi + judul — kuat di identifier persis (nomor pasal, tahun, jenis
  dokumen, singkatan).
- Fusi: `skor(c) = Σ_jalur 1/(60 + rank_jalur(c))` (Konstanta RRF standar).
- Index leksikal dibuat SEKALI: `python bm25.py --build` →
  `pipeline/result/lexical/norma_tokens.txt` (63.231 baris ≡ baris ids.jsonl).
  Belum dibangun → fallback aman ke dense-only (tidak crash).
- Contoh efek nyata: pertanyaan kedaulatan — Pasal 1 UUD yang oleh cosine
  dilewatkan (rank > 10) naik ke #1 berkat skor BM25-nya 31,3.
- Matikan: flag `--no-hybrid`, atau `HYBRID=0` di `.env`, atau
  `hybrid=false` pada tool `search_norma`.
- Skor tercatat terpisah untuk audit: `skor_fusi` (RRF), `skor_bm25`,
  `score` (cosine) / `score_rerank` bila rerank aktif.

## Reranker (pencarian 2 fase) — `reranker.py`

Tahap tambahan setelah retrieval: kandidat cosine diurutkan ulang oleh
**cross-encoder Qwen3-Reranker-0.6B** (query & pasal dibaca BERSAMAAN per
pasangan, jauh lebih tajam dari cosine). Fase 1 ambil 12 kandidat murah,
fase 2 rerank lalu ambil top-k.

- **Default aktif**; otomatis **fallback aman** ke urutan cosine bila model
  tidak termuat (model dibaca offline dari cache HuggingFace).
- Matikan sesuai kebutuhan: flag `--no-rerank` di CLI, atau `RERANK=0` di `.env`,
  atau argumen `rerank=false` pada tool `search_norma`.
- Hasil rerank tercatat terpisah: `score_rerank` (P(yes) 0–1) di samping
  `score` (cosine) untuk audit.

## Persiapan

```bash
# dependensi (torch & transformers sudah ada di mesin ini)
pip install -r requirements.txt

# salin .env.example -> .env, lalu isi (pilih SATU backend):
#   OpenAI     : LLM_BASE_URL=https://api.openai.com/v1          LLM_MODEL=gpt-4o-mini
#   OpenRouter : LLM_BASE_URL=https://openrouter.ai/api/v1       LLM_MODEL=...
#   Ollama     : LLM_BASE_URL=http://localhost:11434/v1          LLM_MODEL=qwen3:4b
python verify_setup.py          # ✅ file & retrieval
python verify_setup.py --full   # + embedder + reranker Qwen3-Reranker (±1–2 mnt, sekali proses)
```

## Menjalankan

```bash
# Lihat prompt yang akan dikirim (TANPA API key, tanpa jaringan):
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context

# Jawaban sungguhan (butuh .env terisi):
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?"

# Filter: hanya UU tahun 2017
python rag_jawab.py -q "syarat pencalonan presiden" --doc-type UU --year 2017

# Reranker & hybrid: aktif secara default; bandingkan dengan versi mati:
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context                     # hybrid + rerank (default)
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context --no-rerank         # hybrid saja
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context --no-hybrid         # dense-only klasik

# Demo loop agent TANPA API: MockLLM meminta tool, tool asli dijalankan
python agent.py -q "kedaulatan di tangan siapa" --dry-run

# Agent sungguhan (multi-hop: cari -> ambil dokumen -> cek status):
python agent.py -q "Apakah UUD 1945 masih berlaku dan di tangan siapa kedaulatan?"
```

## Peta file → konsep panduan

- `rag_jawab.py` = **Arsitektur A** (panduan bab 7): 1 panggilan LLM, deterministik.
- `agent.py` = **Arsitektur B** (panduan bab 5): LLM memilih langkah, maks 5 putaran.
- `tools.py` = 4 tool panduan bab 6 (`search_norma`, `get_article`, `get_document`,
  `check_validity`) — sempit, tervalidasi, sitasi dari dataset.
- Guardrails panduan bab 8 ditegakkan: hanya `part=="norma"` masuk konteks
  (embeddings memang hanya berisi norma), sitasi dari `provenance`,
  `check_validity` tidak pernah menebak status hukum.

## Masalah umum

| Gejala | Penyebab & obat |
|--------|-----------------|
| `LLM_API_KEY kosong` | isi `.env` (lihat `.env.example`) atau pakai `--dry-run` |
| `HTTP 401/404 dari LLM` | base_url/model salah untuk provider tsb. |
| embedder lambat pertama kali | muat model sekali per proses — wajar (±10–30 dtk) |
| skor rendah di semua kandidat | pastikan tak mengubah format `Instruct:`/pooling di `embedder.py` (aturan besi) |
| Milvus gagal konek | Docker/Milvus belum jalan — pakai jalur default `numpy` |
| `FileNotFoundError ... norma_tokens.txt` | jalankan `python bm25.py --build` sekali (index leksikal hybrid) |
