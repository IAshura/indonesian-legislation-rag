# RAG Peraturan Indonesia — LangChain & LangGraph

Implementasi **jembatani yang sama** dengan folder `rag-tanpa-framework/`,
tetapi menggunakan ekosistem LangChain (LCEL) + LangGraph (state machine).
Cocok dibandingkan berdampingan: konsepnya identik, "lem"-nya berpindah
dari kode tulisan tangan ke abstraksi framework.

## Pemetaan konsep: tanpa framework ↔ LangChain/LangGraph

| Konsep | Folder tanpa-framework | Folder ini |
|--------|------------------------|------------|
| Buka dataset JSONL | `dataset.py` (dict by-ID) | `loader.py` (→ `Document`) |
| Embed query (aturan besi) | `embedder.py` | `retriever.QwenQueryEmbedder` |
| Cari top-k + join pasal | `retriever.py` | `retriever.LegislationRetriever` (BaseRetriever) |
| **Hybrid dense+BM25 (RRF)** | `bm25.py` + `retriever.py` | `bm25.py` + `LegislationRetriever` (fallback di dalamnya) |
| **Rerank 2 fase (cross-encoder)** | `reranker.py` | `reranker.QwenReranker` (step LCEL) |
| Susun prompt ber-sitasi | `prompt.py` | `rag_chain.format_docs` + `ChatPromptTemplate` |
| Klien LLM | `llm_client.py` (requests) | `llm.py` (ChatOpenAI/ChatOllama) |
| Arsitektur A (fixed RAG) | `rag_jawab.py` | `rag_chain.py` (LCEL) |
| Definisi tools | `tools.py` (skema JSON manual) | `tools_lc.py` (dekorator `@tool`) |
| Arsitektur B (loop agent) | `agent.py` (`while` + dispatch) | `agent_langgraph.py` (StateGraph + ToolNode) |
| RAG reflektif (grade/rewrite) | — (belum ada) | `graph_rag_reflektif.py` |

## Hybrid search (dense + BM25) — `bm25.py` dalam retriever

Perilaku identik dengan folder tanpa-framework, hanya
pembungkusnya yang berbeda:

- `LegislationRetriever` (BaseRetriever) sekarang menjalankan DUA jalur:
  cosine (embeddings.npy) + BM25 (`bm25.py`, index offline `python bm25.py
  --build`), digabung **RRF** (`skor = Σ 1/(60 + rank)`) lalu dedup
  per-pasal utuh.
- Index leksikal hilang / `HYBRID=0` / `hybrid=false` (tool) → fallback
  bersih ke dense-only.
- Metadata hasil: `skor_fusi`, `skor_bm25`, `score` (cosine),
  `score_rerank` bila reranker jalan — semua bisa diaudit.
- Contoh efek: Pasal 1 UUD (dilewatkan cosine) naik ke #1 untuk pertanyaan
  kedaulatan berkat skor BM25 31,3.

## Reranker (pencarian 2 fase) — `reranker.py`

Kandidat cosine diurutkan ulang oleh **cross-encoder Qwen3-Reranker-0.6B**.
Keistimewaan versi LangChain: reranker hanyalah **step LCEL biasa** —
`RunnableLambda(lambda q: reranker.rerank(q, retriever.invoke(q)))` —
diselipkan di antara retriever dan `format_docs` di `rag_chain.py`.

- **Default aktif** di chain, tool `search_norma` (argumen `rerank`), dan node
  `retrieve` graph reflektif; **fallback aman** ke cosine bila model tak termuat.
- Matikan: `--no-rerank` (chain), `RERANK=0` di `.env`, atau `rerank=false` di tool.
- Model dibaca OFFLINE dari cache HuggingFace (sudah diunduh); skor rerank
  tersimpan di `metadata["score_rerank"]` di samping `score` (cosine).

## Instalasi & pemeriksaan

```bash
pip install -r requirements.txt
# salin .env.example -> .env lalu isi (pilih LLM_BACKEND api|ollama)
python verify_setup.py          # versi library + dataset + retriever
python verify_setup.py --full   # + reranker, dry-run chain, agent graph, graph reflektif
python verify_setup.py --llm    # + ping LLM asli
```

## Menjalankan

```bash
# RAG LCEL tanpa jaringan (jawaban mock, retrieval asli):
python rag_chain.py -q "Di tangan siapa kedaulatan menurut UUD 1945?" --dry-run --show-context          # hybrid + rerank (default)
python rag_chain.py -q "Di tangan siapa kedaulatan menurut UUD 1945?" --dry-run --show-context --no-hybrid   # dense-only klasik

# RAG LCEL asli (butuh .env):
python rag_chain.py -q "Di tangan siapa kedaulatan menurut UUD 1945?"

# Agent tool-calling sebagai state machine (mock → loop terlihat):
python agent_langgraph.py -q "kedaulatan di tangan siapa" --dry-run

# RAG reflektif: retrieve → grade → (rewrite)* → generate:
python graph_rag_reflektif.py -q "istilah presiden satu periode" --dry-run
```

## Kapan pilih folder ini?

- Butuh **perilaku agentic**: grader/refleksi, multi-tool bercabang, langkah
  yang dipilih LLM (panduan bab 7 — Arsitektur B).
- Butuh fitur bawaan LangGraph: checkpointing, streaming, human-in-the-loop.
- Sudah nyaman dengan lapisan dasar (lihat folder tanpa-framework dulu!).

## Masalah umum

| Gejala | Penyebab & obat |
|--------|-----------------|
| `LLM_API_KEY kosong` | isi `.env` atau pakai `--dry-run` |
| versi API berubah | API LangChain cepat bergerak — cek `python verify_setup.py` (versi dicetak) |
| retriever lambat pertama kali | memuat model embedding sekali per proses — wajar |
| `GraphRecursionError` | naikkan `recursion_limit` atau perbaiki skenario loop |
