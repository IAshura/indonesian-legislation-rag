# ⚖️ Indonesian Legislation RAG

**Sistem RAG hukum Indonesia** — dari 378 peraturan resmi (1945–2024) menjadi 74.218 chunk siap-embed dengan provenance terverifikasi (halaman & baris), arsitektur parent–child *small-to-big*, dan QA gate otomatis.

> Dasar data: korpus publik [*Indonesian legislation as data: 90,588 norm units*](https://data.mendeley.com/datasets/8wpjvhh7g2/1) — DOI `10.17632/8wpjvhh7g2.1`, CC BY 4.0, oleh Qausar, Prabowo, Arifin & Isa (Binus University). Pipeline chunking, dataset build, evaluasi, dan lapisan validity di bawahnya adalah pekerjaan di repositori ini.

## Angka kunci

| Metrik | Nilai |
|---|---|
| Sumber | 378 peraturan (UUD → Perlembaga), 1945–2024, portal JDIH |
| Unit norma tersegmentasi | 90.588 |
| Chunk siap-embed | 74.218 (66.292 role `norma`; 4.704 fragmen ber-flag) |
| Parent pasal/blok & window | 11.413 parent · 14.689 window (≤1.800 char, overlap 150) |
| Quarantine beralasan | 2.209 (duplikat 587, heading-like 1.622) |
| QA provenance | 300 sampel → **98% match**, gate PASS · 0 orphan · 0 duplikat ID |
| Akuntabilitas | 74.218 + 2.209 + 14.161 heading = **90.588 — 0 baris hilang** |

## Struktur

```
├── README.md                  ← file ini
├── index.html                 ← Units Viewer (browser korpus interaktif, tanpa dependensi)
├── data/corpus/               ← korpus Mendeley v1 utuh (jangan diubah; checksums.md5 = integritas)
│   ├── units.csv              ← 90.588 unit norma + provenance + audit IndoBERT
│   ├── documents.csv          ← 378 peraturan, identitas dari blok judul + klasifikasi UU 12/2011
│   ├── texts/                 ← 378 teks per-halaman (alamat page/line dari units.csv)
│   ├── checksums.md5
│   └── README-corpus.md       ← dokumentasi asli korpus
├── pipeline/                  ← build v4.1 kanonik (satu-satunya pipeline aktif)
│   ├── build_dataset.py       ← raw → chunks/parents/windows/quarantine/manifest (deterministik)
│   ├── embed_example.py       ← feed embedding (HF lokal / API OpenAI-compatible)
│   ├── enrich_validity.py     ← lapisan status keberlakuan (offline default, --live best-effort)
│   ├── README-build-v4.md     ← pedoman build: 10 keputusan desain + funnel + QA
│   ├── result/                ← 7 file output (chunks.jsonl, parents.jsonl, dst.)
│   └── eval/                  ← eval_retrieval.py + 12 golden queries
├── experiments/               ← arsip eksperimen pembanding (GPT, muse_spark, qwen)
├── reference/corpus-scripts/  ← skrip asli penulis korpus (referensi, butuh proyek asli)
└── docs/                      ← penjelasan pipeline end-to-end (HTML visual)
```

## Quickstart

```powershell
# 1. Bangun dataset siap-embed dari korpus (deterministik, stdlib saja)
cd pipeline
python build_dataset.py            # → result/chunks.jsonl + manifest.json (QA gate PASS)

# 2. (Opsional) Lihat korpus di browser
#    dari root: python -m http.server 8000 → http://localhost:8000/

# 3. Embed (butuh torch + transformers, GPU disarankan)
python embed_example.py                                   # norma saja, fragmen dilewati
python embed_example.py --model intfloat/multilingual-e5-large --batch-size 32
#    Alternatif GPU Google Colab — Qwen3-Embedding-4B (dim 2560):
#    colab/embed_qwen3_4b_colab.ipynb → hasil 3 file → result/embeddings/

# 3b. Masukkan ke Milvus (Docker lokal; pip install pymilvus dulu)
python insert_milvus.py                                   # buat koleksi + insert
python insert_milvus.py --demo "siapa pemegang kedaulatan menurut uud 1945?"

# 4. Ukur retrieval (recall@8; ≥0.8 lanjut, <0.6 ganti model/top-k)
python eval/eval_retrieval.py        # ikut meta.json: e5 mean-pool ATAU Qwen last-token

# 5. Isi status keberlakuan (wajib sebelum produksi — jangan mengarang status hukum)
python enrich_validity.py --live --delay 3    # kandidat URL dari peraturan.bpk.go.id → verifikasi manual
```

## Desain chunking (mengapa bernilai)

- **Structure-based**: batas chunk = hierarki hukum (Pasal/ayat), bukan potongan token. 1 unit = 1 chunk, tanpa merge otomatis — provenance halaman+baris tak pernah rusak.
- **Hierarchical parent–child**: retrieval di child (ayat, presisi), jawaban memakai pasal utuh dari `parents.jsonl` (small-to-big); parent panjang dipecah window deterministik ≤1.800 char dengan overlap 150.
- **Self-contained vector**: `embedding_input = judul resmi — sitasi: isi`.
- **Tanpa data hilang diam-diam**: 2.209 quarantine beralasan terukur; fragmen dipertahankan `is_fragment: true`.

## Rencana (roadmap)

1. Embedding penuh + indeks FAISS (`IndexFlatIP`, dot = cosine)
2. Eval recall@8 atas golden queries (≥0.8)
3. Hybrid dense + BM25 + rerank — **semuanya sudah terimplementasi**:
   - Hybrid: `rag-tanpa-framework/bm25.py` (BM25 leksikal, index offline `python bm25.py --build` → `pipeline/result/lexical/norma_tokens.txt`) + fusi RRF (Reciprocal Rank Fusion) di `retriever.py` kedua folder. ON/OFF via `HYBRID=1|0`; indeks leksikal hilang → fallback dense-only.
   - Rerank: `rag-tanpa-framework/reranker.py` & `rag-langchain-langgraph/reranker.py` pakai cross-encoder Qwen3-Reranker-0.6B (pencarian 2 fase; fallback aman ke urutan hybrid/cosine bila model tidak termuat; ON/OFF via `RERANK=1|0`).
4. Lapisan validity terisi & terverifikasi → payload vector DB

## Atribusi & lisensi

Korpus sumber: **CC BY 4.0** — wajib cantumkan: *Qausar, Prabowo, Arifin & Isa (Binus University) — Indonesian Legislation Norm-Unit Corpus, via Mendeley Data, DOI 10.17632/8wpjvhh7g2.1*. PDF sumber dari portal JDIH pemerintah RI, tidak didistribusikan ulang di sini.
