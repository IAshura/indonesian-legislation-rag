# PENJELASAN LENGKAP — RAG Peraturan Indonesia (Tanpa Framework vs LangChain/LangGraph)

> **Satu dokumen, dari nol sampai paham.** Dokumen ini merangkum: (1) konsep dasar RAG & tool calling,
> (2) data proyekmu dan bentuknya, (3) isi kedua folder implementasi file-per-file, fungsi-per-fungsi,
> (4) perbandingan keduanya, (5) jebakan teknis yang sudah diatasi, dan (6) peta belajar.
> Sumber teori aslinya: [docs/llm-injection-tool-calling.html](docs/llm-injection-tool-calling.html).
> Penjelasan per folder versi PDF: [rag-tanpa-framework/Penjelasan-Kode-rag-tanpa-framework.pdf](rag-tanpa-framework/Penjelasan-Kode-rag-tanpa-framework.pdf)
> dan [rag-langchain-langgraph/Penjelasan-Kode-rag-langchain-langgraph.pdf](rag-langchain-langgraph/Penjelasan-Kode-rag-langchain-langgraph.pdf).

---

## Daftar Isi

1. [Esensi dalam 1 Menit](#1-esensi-dalam-1-menit)
2. [Konsep Dasar: Apa Itu RAG dan Kenapa](#2-konsep-dasar)
3. [Data Proyekmu: 90.588 Norm Unit dan Bentuknya](#3-data-proyekmu)
4. [Embedding: Dari Teks ke Vektor](#4-embedding)
5. [Arsitektur A — Folder `rag-tanpa-framework/`](#5-arsitektur-a)
6. [Arsitektur B — Folder `rag-langchain-langgraph/`](#6-arsitektur-b)
7. [Perbandingan A vs B: Kapan Pakai Mana](#7-perbandingan)
8. [Tool Calling: Cara Kerja Agent](#8-tool-calling)
9. [Jebakan Teknis yang Sudah Diatasi (dan Pelajarannya)](#9-jebakan-teknis)
10. [Cara Menjalankan Langkah demi Langkah](#10-cara-menjalankan)
11. [Peta Belajar: Urutan Membaca yang Disarankan](#11-peta-belajar)
12. [FAQ & Troubleshooting](#12-faq)

---

## 1. Esensi dalam 1 Menit

**Masalah:** LLM (GPT dll.) pintar bicara tapi tidak tahu isi peraturan Indonesia, mudah "mengarang" (halusinasi),
dan tidak punya sumber yang bisa diaudit.

**Solusi (RAG — Retrieval-Augmented Generation):** sebelum LLM menjawab, kita **cari dulu** potongan peraturan
yang relevan dari dataset milikmu, lalu **suntikkan** potongan itu ke prompt, dan suruh LLM menjawab **hanya dari
teks itu**. Jawaban jadi: akurat, ada kutipan pasal, bisa diaudit.

**Analogi:** LLM = dosen pinter tapi tidak hafal peraturan. RAG = kamu menyodorkan fotokopi pasal yang relevan
ke mejanya sebelum dia menjawab, lengkap dengan nomor halaman.

**Dua folder yang kamu minta (sesuai panduan `llm-injection-tool-calling.html`):**

| | Folder A `rag-tanpa-framework/` | Folder B `rag-langchain-langgraph/` |
|---|---|---|
| Filosofi | "Tulis sendiri semuanya" — lihat mesinnya telanjang | "Naik kelas" — framework yang merapikan | 
| Isi inti | RAG CLI + agent loop 60 baris | RAG chain LCEL + agent graph + graph reflektif |
| Untuk | Memahami cara kerja | Membangun sistem agentic yang kompleks |

Keduanya **sudah teruji**: Folder A lulus `verify_setup.py` **11/11**, Folder B lulus `verify_setup.py --full` **8/8**,
dan keduanya jalan penuh dalam mode *dry-run* (tanpa API key) — retrieval benar-benar berjalan di data aslimu
(63.231 vektor embedding, top-8, skor tertinggi 0.63).

---

## 2. Konsep Dasar

### 2.1 RAG dalam satu kalimat per langkah

```
Pertanyaan user
   │
   ▼
[1] EMBED     : ubah pertanyaan jadi vektor (daftar 1024 angka)
[2] SEARCH    : cari vektor chunk yang paling mirip (dot product / cosine)
[3] INJECT    : masukkan teks chunk ke prompt dengan label [S1], [S2], ...
[4] GENERATE  : LLM menjawab HANYA dari [S1]..[S8]
[5] CITATE    : jawaban wajib menyebut sumber; daftar sumber dicetak terpisah
```

Kenapa ini bekerja? Karena **kemiripan vektor ≈ kemiripan makna**. Pasal tentang "kedaulatan" dan pertanyaan
"di tangan siapa kedaulatan berada?" — meski katanya tidak identik — akan punya vektor yang berdekatan.

### 2.2 Dua arsitektur dari panduan

Panduan membedakan dua pola kerja:

- **Arsitektur A (pipeline tetap):** alurnya deterministik — retrieval → inject → generate. Cocok untuk RAG biasa.
  Diimplementasikan di `rag_jawab.py` (Folder A) dan `rag_chain.py` (Folder B).
- **Arsitektur B (agent / tool calling):** LLM diberi *tool* (fungsi yang bisa dipanggil), lalu LLM sendiri yang
  **memutuskan** langkah berikutnya: cari dulu? buka pasal? cek status berlaku? Loop terus sampai LLM bilang "cukup,
  ini jawabannya". Diimplementasikan di `agent.py` (Folder A) dan `agent_langgraph.py` + `graph_rag_reflektif.py` (Folder B).

> **Kapan butuh agent, kapan cukup pipeline?** Kalau pertanyaannya selalu pola sama ("cari pasal X"), pipeline tetap
> lebih murah, cepat, dan mudah diuji. Agent dibutuhkan saat langkahnya **tidak bisa diprediksi** — misal user
> bertanya "apakah UU yang saya kutip ini masih berlaku?" → butuh tool `check_validity`, bukan retrieval.

### 2.3 Istilah yang akan muncul terus

| Istilah | Arti singkat |
|---|---|
| **chunk** | Potongan teks kecil (misal 1 pasal) yang jadi unit pencarian |
| **parent** | "Bapak" dari chunk — pasal utuh; chunk dipakai untuk cari, parent untuk dibaca LLM |
| **window** | Versi memanjang dari parent yang sangat besar (dipecah dengan tumpang-tindih) |
| **document** | 1 peraturan utuh (mis. `uud-1945`), ada 378 dokumen |
| **embedding** | Representasi teks sebagai vektor angka (di sini: 1024 dimensi) |
| **rerank/filter** | Menyaring/mengurut ulang hasil pencarian (filter metadata, atau cross-encoder — lihat reranker.py) |
| **dry-run** | Mode uji tanpa memanggil API LLM — retrieval tetap jalan sungguhan |
| **mock** | Pengganti LLM palsu dengan skrip jawaban tetap, untuk menguji alur tanpa jaringan |

---

## 3. Data Proyekmu

Semua data hasil pipeline ada di `pipeline/result/`. Angka pentingnya:

| File | Isi | Jumlah baris |
|---|---|---|
| `chunks.jsonl` | Potongan teks (1 baris = 1 chunk, JSON) | **90.588** |
| `parents.jsonl` | Pasal/bagian utuh (unit hukum) | **11.413** |
| `parents_windows.jsonl` | Parent panjang yang dipecah window | **14.689** |
| `documents.jsonl` | Metadata peraturan | **378** |
| `embeddings/qwen3-06b/embeddings.npy` | Matriks vektor float32 | **63.231 × 1024** (259 MB) |
| `embeddings/qwen3-06b/ids.jsonl` | Kunci baris ke-i vektor ↔ chunk_id | **63.230** |
| `embeddings/qwen3-06b/meta.json` | Info model & cara embed | — |

> Catatan kecil: `ids.jsonl` berisi 63.230 baris vs `meta.json` mengklaim 63.231. Selisih 1 tidak mengganggu
> pencarian (baris tanpa id hanya tidak pernah ter-retrieve); kode tidak meng-assert kesetaraan keduanya.

### 3.1 Anatomi satu baris `chunks.jsonl`

Kunci-kunci yang dipakai kode (yang lain dibiarkan lewat saja):

```
chunk_id      "uud-1945::p::1::02"      ← unik; ::p:: = pasal; ::02 = chunk ke-2
parent_id     "uud-1945::p::1"          ← pasal utuh pemilik chunk ini
document_id   "uud-1945"                ← peraturannya
official_title "Undang-Undang Dasar 1945"
citation      "Pasal 1 UUD 1945"
content_role  "norma" | "penjelasan" | ...   (pipeline menyaring hanya norma)
doc_type      "uud" | "uu" | "pp" | ...
year          1945
legal_status  "berlaku" | ...           ← dipakai tool check_validity
text          "Menurut UUD 1945 Pasal 1 (2): Kedaulatan berada di tangan rakyat..."
provenance    {"page_number": 3, "line_start": 12, "line_end": 14}   ← jejak halaman asli
char_len, token_estimate, quality_flags                 ← statistik & penanda kualitas
```

### 3.2 Pola ID — hafalkan ini, kuncinya segalanya

```
uud-1945                  ← document
   └── uud-1945::p::1     ← parent (pasal 1; contoh nyata: char_len 199)
          └── uud-1945::p::1::02      ← chunk (potongan ke-2 dari pasal 1)
          └── uud-1945::p::1::w01     ← window (muncul kalau parent > ~1800 karakter)
```

Karena ID deterministik, dari sebuah chunk kita bisa **naik ke parent** (teks utuh untuk LLM) dan
**naik ke document** (judul resmi + status). Inilah trik `context_for_parent()` dan `doc_pasal()`.

### 3.3 Kenapa cari pakai chunk tapi jawab pakai parent?

Chunk kecil = pencarian tajam (vektor tidak "encer"). Tapi kalau yang diberikan ke LLM hanya potongan,
jawabannya gampang terpotong konteks. Solusinya: **cari dengan chunk, baca dengan parent** —
chunk hanya alamat, parent isinya. Konsep ini bernama *parent-document retrieval*.

---

## 4. Embedding

### 4.1 Model yang dipakai

`Qwen/Qwen3-Embedding-0.6B` — sudah terunduh di cache HuggingFace milikmu, jadi **jalan offline** (tanpa internet).
Spesifikasi dari `meta.json`: dimensi **1024**, pooling **last_token**, `normalized_l2` (panjang vektor = 1),
maks 1024 token input. Embedding terdiri dari dua jenis:

1. **Embedding korpus** — sudah dihitung pipeline lama, tersimpan di `embeddings.npy` (63.231 baris).
2. **Embedding query** — dihitung *saat runtime* setiap ada pertanyaan, oleh `embedder.py` (A) / `QwenQueryEmbedder` (B).

### 4.2 Aturan main query embedding (WAJIB sama persis dengan korpus)

Model Qwen3-Embedding memakai **instruksi** untuk query. Formatnya:

```
Instruct: Given a user question about Indonesian legislation, retrieve the most relevant legal provisions.
Query: <pertanyaan user>
```

Plus tiga aturan teknis: **left padding** (bukan right), **last-token pooling** (ambil vektor token terakhir),
dan **L2 normalize**. Kalau salah satu berbeda dari saat korpus di-embed, hasil pencarian hancur.
Itulah kenapa format ini dikunci di satu fungsi saja (`embed_query()`), tidak disebar ke mana-mana.

### 4.3 Kenapa cari dengan dot product?

Karena semua vektor sudah ternormalisasi L2 (panjang = 1), maka **dot product = cosine similarity** —
nilai antara -1 dan 1, makin besar makin mirip. Menghitung dot product 63.231 × 1024 di NumPy butuh
**kurang dari sepersekian detik** — tidak perlu database khusus untuk skala ini.

---

## 5. Arsitektur A

Folder: [rag-tanpa-framework/](rag-tanpa-framework/) — Python murni; satu-satunya dependensi jaringan adalah
pustaka `requests`. Urutan baca kode mengikuti urutan aliran data:

```
config.py → dataset.py → embedder.py → retriever.py → prompt.py → llm_client.py → rag_jawab.py
                                                                              ↘ tools.py → agent.py
```

### 5.1 `config.py` — semua konstanta di satu tempat

| Entitas | Tugas |
|---|---|
| `active_embedding_dir()` | Baca file `ACTIVE` di folder embeddings untuk tahu model mana yang dipakai (mis. `qwen3-06b`), lalu kembalikan path lengkapnya. Ini membuat ganti model embed cukup ganti satu file. |
| `get_llm_config()` | Baca `.env` **dengan loader buatan sendiri** (tanpa pustaka tambahan): `LLM_BASE_URL` (default `https://api.openai.com/v1`), `LLM_API_KEY` (cadangan: `OPENAI_API_KEY`), `LLM_MODEL` (default `gpt-4o-mini`), `LLM_TEMPERATURE=0.2`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`. |
| `TOP_K = 8` | Jumlah kandidat retrieval. Kenapa 8? Karena evaluasi retrieval di `eval/` mengukur **recall@8 = 0,75** — 8 adalah titik yang sudah terbukti di data ini. |

**Kenapa temperature 0.2?** Untuk jawaban hukum kita ingin konsisten, bukan kreatif. Nilai kecil = jawaban stabil.

### 5.2 `dataset.py` — pintu satu-satunya ke data

| Entitas | Tugas |
|---|---|
| `load_jsonl(path)` | Generator membaca JSON Lines baris-per-baris. Baris-per-baris penting: file chunks berukuran ratusan MB, tidak boleh di-`json.load` sekaligus. |
| `LegislationDataset` | Kelas induk data. Punya 4 properti **lazy**: `chunks`, `parents`, `windows`, `documents` — baru dimuat ke memori saat pertama kali diakses, lalu dicache sebagai `{id: baris}`. Lazy loading membuat `python rag_jawab.py --help` tidak perlu menunggu data dimuat. |
| `_load(attr)` | Mekanisme umum di balik keempat properti (hindari 4× kode yang sama). |
| `get_chunk(id)` / `get_document(id)` | Akses kamus langsung — O(1). |
| `context_for_parent(parent_id)` | **Fungsi terpenting di file ini.** Dari parent_id, kumpulkan pasal utuh + dokumen induk + provenance + chunk sumber. Inilah yang mengubah "vektor baris ke-17.482" menjadi "Pasal 1 UUD 1945, halaman 3". |
| `citation_line(ctx)` | Format satu baris sitasi rapi, mis. `Pasal 1 UUD 1945 (UUD 1945, hal. 3)`. |
| `dataset()` | Singleton — semua file berbagi satu instance, data tak pernah dimuat dua kali. |

### 5.3 `embedder.py` — pertanyaan jadi vektor

| Entitas | Tugas |
|---|---|
| `load_meta()` | Baca `meta.json` → tahu nama model, instruksi query, dimensi. |
| `_lazy_load()` | Muat model transformers sekali saja (`local_files_only=True` → pasti jalan offline). Memuatnya berlangsung beberapa detik (progress bar muncul — itu wajar). |
| `embed_query(text)` | **Fungsi paling sering salah dibuat ulang.** Membungkus teks dengan `Instruct: ...\nQuery: ...`, left padding, last-token pooling, L2 normalize. Punya jebakan yang sudah diperbaiki — lihat [Bab 9, jebakan #1](#9-jebakan-teknis). |

### 5.4 `retriever.py` — mesin pencari vektor

| Entitas | Tugas |
|---|---|
| `ExactRetriever` | Pencarian **exact** (brute-force) memakai `np.load(..., mmap_mode="r")` — matriks 259 MB tidak dimuat utuh ke RAM, hanya bagian yang dibaca saat dihitung. `search(query_vec, k)` menghitung `M @ q` (dot product semua baris sekaligus, itu cepat), ambil k skor terbesar, kembalikan `{chunk_id, skor, rank}`. |
| `milvus_search(...)` | Jalur alternatif kalau kelak data bermilyaran dan kamu pakai Milvus (butuh Docker). Saat ini **tidak dipakai** karena Docker tidak jalan — dan memang tidak perlu pada skala 63 ribu. |
| `retrieve(query_vec, k)` | Dispatcher: pilih backend exact atau milvus. |
| `konteks_kaya(hasil)` | Terjemahkan hasil mentah (`chunk_id`, skor) menjadi konteks siap-pakai lewat `context_for_parent()`, deduplikasi (dua chunk dari parent yang sama → satu konteks), sertakan skor & rank. |

### 5.5 `prompt.py` — tempat "suntik konteks" sesuai nama panduan

| Entitas | Tugas |
|---|---|
| `SYSTEM_PROMPT` | Kalimat peran untuk LLM: *"Kamu asisten hukum yang bekerja hanya dari konteks yang diberikan; kutip pasal; jika konteks tidak memuat jawaban, katakan tidak tahu."* Perintah terakhir inilah tameng utama halusinasi. |
| `build_context_block(kandidat, ...)` | Bangun blok teks berlabel: `[S1] (skor 0.63) <judul + sitasi>\n<teks>`, dibatasi **anggaran 12.000 karakter** (biar tidak menjebol context window dan biaya tetap terkendali). Ini inti "injection" dari judul panduan. |
| `build_messages(question, context_block)` | Susun pesan gaya OpenAI: `[{"role":"system",...},{"role":"user",...}]`. |
| `daftar_sumber(kandidat)` | Daftar sitasi untuk dicetak di bawah jawaban — jejak audit. |

### 5.6 `llm_client.py` — satu pintu ke LLM

| Entitas | Tugas |
|---|---|
| `chat(messages, tools=None, ...)` | POST `requests` ke `{LLM_BASE_URL}/chat/completions`. Mendukung `tools` (untuk agent). Menangani timeout & error HTTP dengan pesan jelas. Semua panggilan LLM di seluruh folder A **wajib** lewat sini — satu tempat untuk ganti provider. |
| `dry_run(messages, tools)` | Mode uji: tidak memanggil jaringan; mencetak apa yang *akan* dikirim (panjang prompt, jumlah sumber) dan mengembalikan jawaban tiruan. |
| `MockLLM` | Kelas LLM palsu dengan **skrip jawaban terjadwal** (`skrip=[...]`): panggilan ke-1 mengembalikan keputusan tool, ke-2 jawaban final. Dipakai `verify_setup.py` dan `agent.py --dry-run` untuk menguji loop agent **tanpa API key**. |

### 5.7 `rag_jawab.py` — Arsitektur A (pipeline tetap)

| Entitas | Tugas |
|---|---|
| `jawab(question, top_k, ...)` | Alur lengkap 6 langkah: embed → retrieve → konteks_kaya → build_context_block → build_messages → chat. Mengembalikan jawaban + daftar sumber. |
| `main()` | CLI: `-q "pertanyaan"`, `--top-k 8`, `--doc-type uu`, `--year 2020`, `--dry-run`, `--show-context` (lihat konteks yang disuntik — bagus untuk belajar). |

Cobalah: `python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context`.
Hasil terverifikasi: `[S1]` = Pasal 9 Perubahan Pertama (skor 0.6336), dan Pasal 1 muncul di `[S4]` (0.6094).

### 5.8 `tools.py` — 4 tangan untuk sang agent

| Tool | Kapan LLM memakainya | Implementasi |
|---|---|---|
| `search_norma(question, doc_type?, year?, top_k?)` | Perlu mencari peraturan | embed + retrieve + konteks_kaya (versi ringkas: teks dipotong 600 char) |
| `get_article(parent_id)` | Sudah tahu pasal spesifik, mau teks utuh | `context_for_parent()` |
| `get_document(document_id)` | Perlu metadata peraturan | `get_document()` |
| `check_validity(document_id)` | "Apakah masih berlaku?" | baca `legal_status` |

Skema tiap tool ditulis dalam format JSON Schema gaya OpenAI (daftar `TOOLS`), dan `execute_tool(name, arguments)`
adalah **dispatcher terpusat**: validasi nama (tool ngawur → error JSON rapi, bukan crash), clamp `top_k` ke 1–10,
potong teks ke 600 karakter. Guardrail kecil seperti ini mencegah prompt meledak biaya.

### 5.9 `agent.py` — Arsitektur B (loop tool calling)

| Entitas | Tugas |
|---|---|
| `jalankan(question, dry)` | **The loop** (perhatikan betapa pendeknya — ini seluruh "framework" yang sebenarnya): kirim pesan + skema tools → kalau balasan berisi `tool_calls` → jalankan `execute_tool` → appen dhasil sebagai pesan `role:"tool"` → ulangi. Kalau tidak ada tool_calls → itulah jawaban final. |
| `MAX_ITERATIONS = 5` | Guardrail: maksimal 5 putaran, mencegah agent muter-muter membakar token. |
| `main()` | CLI `python agent.py -q "..." [--dry-run]`. |

Terbukti jalan (mock): putaran 1 → `search_norma`, putaran 2 → `get_document`, putaran 3 → jawaban final.

### 5.10 `bm25.py` — hybrid search: jalur leksikal + fusi RRF


Pencarian sekarang **hybrid — dua jalur independen lalu difusi**:

```
jalur dense (makna) : cosine embeddings.npy      — kuat di sinonim/parafrase
jalur lexical (kata): BM25 teks+sitasi+judul     — kuat di nomor pasal, tahun,
                                                   jenis dok, singkatan, angka
fusi: RRF (Reciprocal Rank Fusion) = Σ_jalur 1/(60 + rank_jalur(c))
```

**Kenapa fusi peringkat, bukan penjumlahan skor?** Skala cosine (0–1) dan BM25
(0–20+) tidak sebanding; RRF hanya melihat RANK di tiap jalur, jadi bebas dari
masalah kalibrasi (rumus standar Cormack et al. 2009, k=60).

| Entitas | Tugas |
|---|---|
| `tokenisasi_bm25(teks)` | Case-fold + token alfanumerik ≥2 char + stopwords dasar; SATU fungsi untuk builder & query (padanan "model index = model query"). | |
| `build_lexical_index()` | Jalankan SEKALI offline (`python bm25.py --build`) → `pipeline/result/lexical/norma_tokens.txt`; baris-i file = baris-i ids.jsonl = baris-i embeddings.npy. 63.231 baris, avgdl ≈ 42 token. |
| `BM25Scorer` | Muat index + postings (token→dok,tf) + precompute idf; `.search(q,k)` → `[(baris, skor)]` dalam ±2–130 ms (tanpa jaringan, tanpa dependency). |
| `fusi_rrf()` + `HybridRetriever` (di retriever.py) | Rank murni tiap jalur → RRF → dedup per pasal setelah fusi (kontribusi rank-1 kedua jalur tidak pernah hilang). |

Efek nyata terverifikasi: untuk pertanyaan kedaulatan, **Pasal 1 UUD yang
dilewatkan cosine (rank > 10) naik ke #1** berkat skor BM25-nya 31,3. Kontrol:
default aktif; `--no-hybrid` (CLI), `HYBRID=0` (.env), `hybrid=false` (tool);
index belum dibangun → fallback aman dense-only. Skor audit: `skor_fusi`,
`skor_bm25`, di samping `score` cosine / `score_rerank`.

### 5.10b `reranker.py` — rerank cross-encoder (tahap akhir di atas hybrid)

Di atas hasil hybrid, pencarian ditambah **tahap penilaian ulang**:

```
fase 1 (murah)  : cosine ambil 12 kandidat  →  fase 2 (akurat): rerank → top-k
```

Modelnya **Qwen3-Reranker-0.6B** (cross-encoder — beda jenis dengan embedding):
query dan pasal DIGABUNG jadi satu prompt yes/no, dan model membaca keduanya
bersamaan. Skor = `P(yes) = softmax(logit_yes, logit_no)` — jauh lebih tajam
dari cosine, tapi mahal (1 forward-pass per pasangan), karena itu hanya untuk
kandidat teratas.

| Entitas | Tugas |
|---|---|
| `muat_reranker()` | Singleton lazy, OFFLINE dari cache HuggingFace; `None` bila gagal → fallback aman ke cosine. |
| `skor_rerank(query, teks_list)` | P(yes) per pasangan, per batch (template resmi: system judge + `<Instruct>/<Query>/<Document>` + footer `shed thought/shed answer`). |
| `rerank_kandidat(query, kandidat, top_n)` | Urut ulang berdasarkan `score_rerank`; `score` cosine lama tetap disimpan untuk audit. |

Kontrol: default aktif; `--no-rerank` (CLI), `RERANK=0` (.env), `rerank=false`
(tool `search_norma`). Smoke test terverifikasi: pasal relevan P=0.67 vs tak
relevan P=0.008. Catatan CPU: scoring ~1 menit per batch 4 pasangan — normal.

### 5.11 Dua file utilitas

- `verify_setup.py` — `cek()` + `main()`: **11 pemeriksaan** (data ada, meta cocok, embedder bisa jalan, retrieval
  menghasilkan skor wajar, tools bisa dipanggil, loop agent mock selesai, dsb.). Anggap ini "kunci engsel" — kalau
  suatu hari ada yang rusak setelah update paket, jalankan ini dulu.
- `pdf_engine.py` + `buat_pdf_penjelasan.py` — generator PDF (reportlab, A4, daftar isi via `multiBuild`,
  footer, `tabel_entitas()` yang membaca kode lewat **AST Python** lalu menampilkan class/fungsi + signature +
  docstring). File penjelasan PDF A dihasilkan dari sini. `DocTOC.afterFlowable` yang mengumpulkan judul bab
  selama render; `_signature()` menangani perbedaan AST antara FunctionDef dan ClassDef.

---

## 6. Arsitektur B

Folder: [rag-langchain-langgraph/](rag-langchain-langgraph/) — versi "naik kelas" dengan pekerjaan yang sama
tetapi dipetakan ke konsep framework. **Terjemahan konsepnya:**

| Karena di Folder A ada... | Di Folder B menjadi... |
|---|---|
| `dataset.LegislationDataset` | `loader.DatasetRingkas` + fungsi pembungkus `Document` (`doc_pasal`, `doc_peraturan`) |
| `embedder.embed_query` + `retriever.ExactRetriever` | `retriever.QwenQueryEmbedder` + `_ExactSearch`, dibungkus `LegislationRetriever(BaseRetriever)` |
| `prompt.build_context_block` + `llm_client.chat` dipasangkan manual | **LCEL chain** di `rag_chain.py`: `retriever \| prompt \| llm \| parser` |
| `tools.TOOLS` skema JSON manual | `tools_lc.py` — cukup dekorator `@tool`, skema dibuat otomatis |
| `agent.jalankan` loop while manual | **LangGraph StateGraph** di `agent_langgraph.py` |
| `reranker.rerank_kandidat` (dipanggil manual) | `QwenReranker` sebagai **step LCEL** di rantai `\|`, tool `@tool`, dan node graph |
| (tidak ada padanannya) | **Bonus:** `graph_rag_reflektif.py` — agent yang *mengkritik hasilnya sendiri* |

### 6.1 `config.py`

Sama seperti A, plus satu variabel: `LLM_BACKEND=api|ollama` → pilih ChatOpenAI-compatible atau ChatOllama
tanpa mengubah kode lain. Pola "konfigurasi mengganti kode".

### 6.2 `loader.py` — data → `Document`

| Entitas | Tugas |
|---|---|
| `DatasetRingkas` | Pasangan lazy-load dari `LegislationDataset` (4 properti: chunks/parents/windows/documents). |
| `doc_konteks_dari_chunk(...)` / `doc_pasal(parent_id)` / `doc_peraturan(document_id)` | Membuat objek `langchain_core.documents.Document` — standar framework: `page_content` (teks) + `metadata` (sitasi, skor, dsb.). Konversi inilah tikus pintu masuk ke ekosistem LangChain: semua retriever/chain di framework mengalirkan `Document`. |

### 6.3 `retriever.py` — subclass `BaseRetriever` (hybrid di dalam)

Perilaku identik dengan folder tanpa-framework (bab 5.10): dua jalur dense
+ BM25 (`bm25.py`) difusi RRF di dalam `LegislationRetriever` — rank murni
tiap jalur disatukan, dedup per pasal SETELAH fusi. Fallback aman ke
dense-only bila index leksikal belum dibangun (`python bm25.py --build`),
`HYBRID=0`, atau `hybrid=false` pada tool. Metadata hasil memuat
`skor_fusi`, `skor_bm25`, `score` — semua bisa diaudit.

| Entitas | Tugas |
|---|---|
| `QwenQueryEmbedder.embed(text)` | Identik dengan `embed_query()` di A (aturan Instruct/Query, left padding, last-token pooling, L2) — **dipindah ke kelas** supaya bisa di-inject ke retriever lain. |
| `_ExactSearch.dokumen(vec, k)` | Mesin mmap + dot product yang sama, tapi mengembalikan `Document` ber-metadata. |
| `LegislationRetriever(BaseRetriever)` | **Integrasi framework yang benar.** Subclass `BaseRetriever`, implementasi `_get_relevant_documents(query, ...)`. Setelah ini, retrievermu bisa dipakai di *semua* fitur LangChain (chain, tool, dsb.) tanpa penyesuaian. Catatan: state berat (model, matriks) disimpan via `pydantic.PrivateAttr` karena BaseRetriever adalah model Pydantic — pola wajib di LangChain 1.x. |

### 6.4 `llm.py` — factory + mock

| Entitas | Tugas |
|---|---|
| `get_llm(dry, **kwargs)` | Factory: `dry=True` → `FakeListChatModel` (jawaban tetap dari daftar); `LLM_BACKEND=ollama` → `ChatOllama`; selain itu → `ChatOpenAI(base_url=..., api_key=...)`. Karena semua memenuhi kontrak yang sama, sisa kode tidak peduli backend apa yang dipakai. |
| `MockAgentLLM(BaseChatModel)` | Mock yang paling menarik. Memenuhi kontrak `BaseChatModel`: wajib `_llm_type` (properti) + `_generate(messages,...)`. Karena agent akan memanggil `bind_tools(tools)`, mock meng-override-nya mengembalikan `self`. Skrip terjadwal memungkinkan menguji graph agent **persis seperti dengan LLM asli**. |

### 6.5 `rag_chain.py` — LCEL, Arsitektur A versi framework

| Entitas | Tugas |
|---|---|
| `format_docs(docs)` | Gabungkan `Document` jadi blok konteks berlabel `[S1]...` (padanan `build_context_block`). |
| `daftar_sumber(docs)` | Padanan persis namanya di A. |
| `bangun_chain(dry, top_k)` | **Rantai LCEL**: `{"konteks": retriever, "pertanyaan": RunnablePassthrough()} → ChatPromptTemplate → llm → StrOutputParser`. Operator `\|` menyambung step seperti pipa. `RunnableParallel` memastikan pertanyaan user mengalir ke dua cabang sekaligus (ke retriever dan ke prompt) — dan harus **di-import eksplisit** (tidak ikut `import *`). |
| `jawab(pertanyaan, ...)` | Panggil chain + cetak sumber; CLI dengan `--dry-run`. |

Keunggulan nyata LCEL: `batch()`, streaming, dan retry datang **gratis** dari framework, tanpa mengubah satu baris definisi chain.

### 6.6 `tools_lc.py` — `@tool` menggantikan skema JSON manual

Empat tool dengan nama & fungsi sama (`search_norma`, `get_article`, `get_document`, `check_validity`),
tetapi cukup diberi dekorator `@tool` + docstring — **skema argumen dibuat otomatis dari type hints**.
Itu sebabnya docstring di file ini penting: docstring = instruksi ke LLM tentang kapan memakai tool.
`TOOLS = [...]` adalah daftar yang diberikan ke agent. Clamp top_k dan pemotongan teks dipertahankan
(lesson dari Folder A tidak hilang).

### 6.7 `agent_langgraph.py` — loop agent jadi state machine

| Entitas | Tugas |
|---|---|
| `buat_node_agent(llm)` | Node "otak": panggil LLM dengan tools. Closure membawa `llm` tanpa variabel global. |
| `buat_node_tools()` | Node "tangan": `ToolNode(TOOLS)` milik LangGraph mengeksekusi semua tool_calls sekaligus. |
| `should_continue(state)` | **Conditional edge**: ada tool_calls → lanjut ke node tools; tidak ada → END. |
| `bangun_graph(dry)` | `StateGraph(MessagesState)`: `START → agent ⇄ tools → END` (agent selalu kembali dihubungkan). Komentar di file juga menunjukkan alternatif satu barisnya: `create_react_agent(llm, tools)` — dibuat manual justru untuk belajar apa yang framework lakukan. |
| `MAX_PUTARAN = 6` + `recursion_limit` | Guardrail kembar: batas logika di kode + batas engine graph. |
| `jalankan(question, dry)` | CLI. |

Pesan awal dikirim sebagai **tuple** `(role, teks)` — format yang diterima LangGraph. Pesan balasan dari LLM
yang berisi tool_calls dinormalisasi sebelum di-append (lihat [Bab 9, jebakan #3](#9-jebakan-teknis)).

### 6.8 `graph_rag_reflektif.py` — bonus: agent yang mengkritik dirinya

Pola CRAG (Corrective RAG) — penambahan yang tidak ada di Folder A:

```
retrieve ──→ grade ──(relevan)──→ generate ──→ END
                └─(tidak relevan)─→ rewrite ──(coba lagi)──→ retrieve
                                       └─(sudah 2×)──→ generate (jawab jujur)
```

| Entitas | Tugas |
|---|---|
| `GraphState(TypedDict)` | Skema state yang mengalir antar node: pertanyaan, dokumen, keputusan grade, jumlah percobaan, jawaban. |
| `buat_node_retrieve/grade/generate/rewrite` | Grade = LLM menilai "apakah dokumen menjawab pertanyaan? yes/no". Rewrite = LLM menulis ulang pertanyaan jadi kata kunci retrieval yang lebih baik. |
| `keputusan_setelah_grade` / `keputusan_setelah_rewrite` | Dua fungsi conditional edge (node menulis keputusan ke state; edge membacanya — pemisahan ini pola resmi LangGraph). |
| `MAX_PERCOBAAN = 2` | Setelah 2 rewrite gagal, berhenti dan jawab jujur "tidak menemukan" — guardrail ketiga. |

Ini menjawab kelemahan RAG biasa: kalau retrieval buruk, pipeline tetap menjawab dari konteks yang jelek.
Di sini, kualitas konteks **diputus dulu**, dan retrieval diperbaiki jika perlu.

---

## 7. Perbandingan

| Aspek | A — Tanpa framework | B — LangChain/LangGraph |
|---|---|---|
| Jumlah kode | Lebih panjang (semua ditulis sendiri) | Lebih pendek (abstraksi framework) |
| Transparansi | **100%** — setiap byte prompt terlihat | Bergantung pada kedalaman framework |
| Kurva belajar | Landai — hanya requests + numpy | Curam — konsep LCEL, state graph, pydantic attrs |
| Skema tool | JSON Schema manual (repot tapi mendidik) | `@tool` otomatis dari type hints |
| Loop agent | `while` 40 baris, mudah diaudit | StateGraph: visual, conditional edges, teruji |
| Refleksi korektif | Tidak ada (bisa dibuat manual, tapi ribet) | **Ada** (graph_rag_reflektif) — node & edge tambahan saja |
| Ganti backend LLM | Edit `llm_client.py` | Set `LLM_BACKEND=ollama` |
| Risiko | Semua bug milikmu sendiri | Bug/upstream framework versi baru (kamu pakai langchain 1.4.3 + langgraph 1.2.14) |
| Cocok untuk | Belajar, audit, kebutuhan sederhana | Sistem agentic kompleks, produksi cepat |

**Rekomendasi panduan (dan setuju dengan kode):** mulai dari A untuk memahami; pindah ke B saat butuh perilaku
agentic yang kompleks. Kamu sekarang punya **keduanya bersebelahan**, sehingga setiap konsep bisa dibaca dua kali
dalam dua bahasa kode.

---

## 8. Tool Calling

Prinsipnya satu kalimat: **tool tidak dijalankan oleh LLM — LLM hanya mengirimkan "permintaan eksekusi",
kode kamu yang menjalankannya.**

Alur pesan lengkap satu putaran tool call:

```
1. user      : "Apakah UU Cipta Kerja masih berlaku?"
2. assistant : { "tool_calls": [ {name:"check_validity", arguments:'{"document_id":"uu-11-2020"}'} ] }
3. tool      : '{"legal_status": "berlaku", ...}'        ← hasil execute_tool()
4. assistant : "Ya, UU Cipta Kerja masih berlaku... (kutipan)"   ← jawaban final, tidak ada tool_calls
```

Poin penting:

1. **Argumen berupa string JSON** — kode harus `json.loads` dan siap gagal rapi (dispatcher A mengembalikan
   error JSON terstruktur, bukan exception telanjang; LLM lalu bisa mencoba memperbaiki panggilannya sendiri).
2. **Hasil tool masuk sebagai pesan `role:"tool"`** dengan `tool_call_id` yang cocok — protokol API OpenAI.
3. **Keputusan berhenti ada di tangan LLM** — itulah kenapa guardrail jumlah putaran (`MAX_ITERATIONS` /
   `MAX_PUTARAN`) wajib ada di kode, bukan di "kepercayaan pada LLM".
4. Di Folder B, `ToolNode` menangani langkah 2–3 otomatis; di Folder A kamu menulisnya sendiri di `agent.py` —
   dan itulah nilai edukasinya: kamu tahu persis apa yang framework sembunyikan.

---

## 9. Jebakan Teknis

Enam jebakan yang sudah terjadi (atau nyaris terjadi) di proyek ini — hafalkan, karena akan muncul lagi di proyek lain:

| # | Jebakan | Gejala | Solusi yang dipakai |
|---|---|---|---|
| 1 | **BFloat16 → NumPy** | `TypeError: Got unsupported ScalarType BFloat16` saat model Qwen mengeluarkan tensor bfloat16 | `vec.float()` **sebelum** `.numpy()` — sudah dibenahi di `embedder.py` (A) dan `QwenQueryEmbedder` (B) |
| 2 | **Windows cp1252** | `UnicodeEncodeError` saat `print()` emoji/ karakter non-latin di konsol Windows | Baris pertama tiap skrip: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` |
| 3 | **Pesan assistant dengan field ekstra** | API provider menolak pesan balasan LLM yang memuat field non-protokol saat di-append balik ke riwayat | Normalisasi ke `{role, content, tool_calls}` sebelum append (`agent.py` A, `agent_langgraph.py` B) |
| 4 | **Pesan awal graph** | LangGraph menolak dict mentah sebagai pesan awal | Kirim tuple `(role, teks)` — format yang diterima konversi internal |
| 5 | **`RunnableParallel` tidak ikut `import *`** | `NameError` saat menyusun chain LCEL meski `from langchain_core.runnables import *` | Import eksplisit per nama; pola aman untuk semua modul framework |
| 6 | **AST ClassDef tidak punya `.args`** | Generator PDF crash saat membaca signature class | `_signature()` bercabang: ClassDef → tampilkan bases, FunctionDef → tampilkan parameter |

Plus dua fakta lingkungan yang berguna: `langgraph` tidak menyediakan `__version__` (pakai
`importlib.metadata.version("langgraph")`), dan memuat model embedding menampilkan progress bar bobot — itu wajar, bukan error.

---

## 10. Cara Menjalankan

### Langkah 0 — persyaratan
Python 3.12 di mesin ini sudah lengkap: requests, numpy, torch, transformers (untuk A);
langchain 1.4.3 + langchain-core 1.6.7 + langgraph 1.2.14 + langchain-openai (untuk B).

### Langkah 1 — konfigurasi API (sekali)
```bash
cd rag-tanpa-framework        # atau rag-langchain-langgraph
copy .env.example .env        # Windows (di Git Bash: cp .env.example .env)
# buka .env, isi:
#   LLM_API_KEY=sk-...          ← wajib untuk mode live
#   LLM_BASE_URL=https://api.openai.com/v1   ← ganti kalau pakai provider lain yang kompatibel
#   LLM_MODEL=gpt-4o-mini       ← sesuaikan
# (Folder B tambahan: LLM_BACKEND=api atau ollama)
```

### Langkah 2 — verifikasi
```bash
python verify_setup.py           # Folder A: 11 cek
python verify_setup.py --full    # Folder B: 8 cek (termasuk chain & graph mock)
```

### Langkah 3 — jalankan (mode live, butuh API key)
```bash
# Folder A
python rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?"
python agent.py    -q "Apakah UU Cipta Kerja masih berlaku?"

# Folder B
python rag_chain.py          -q "Di tangan siapa kedaulatan berada menurut UUD 1945?"
python agent_langgraph.py    -q "Apakah UU Cipta Kerja masih berlaku?"
python graph_rag_reflektif.py -q "..."          # mode reflektif
```

### Langkah 3b — tanpa API key (tetap bisa, retrieval sungguhan)
Tambahkan `--dry-run` (dan `--show-context` di Folder A untuk melihat konteks tersuntik).

### Langkah 4 — regenerasi PDF penjelasan
```bash
python buat_pdf_penjelasan.py   # di tiap folder; PDF muncul di folder yang sama
```

---

## 11. Peta Belajar

**Hari 1 — jalankan dan amati (tanpa baca kode):**
- `rag_jawab.py --dry-run --show-context` → lihat blok `[S1]..[S8]`. Pertanyaan: kenapa [S1] paling tinggi?
- Ubah pertanyaan jadi bahasa sehari-hari ("boleh gak demo di jalan raya?") dan lihat retrieval tetap menemukan pasal UU Demo.

**Hari 2 — pahami data:** buka `chunks.jsonl` 5 baris pertama; cocokkan pola ID di [Bab 3.2](#32-pola-id--hafalkan-ini-kuncinya-segalanya);
telusuri satu parent dari chunk ke pasal utuh.

**Hari 3 — Folder A berurutan:** `config → dataset → embedder → retriever → prompt → llm_client → rag_jawab`.
Setiap file, tugasnya satu: jelaskan apa fungsi utamanya dengan kata-katamu sendiri.

**Hari 4 — agent:** `tools.py` dulu (skema JSON), lalu `agent.py` — gambar ulang loop-nya di kertas.
Lalu jalankan `agent.py --dry-run` dan cocokkan urutan putarannya dengan gambarmu.

**Hari 5 — Folder B sebagai "kamus penerjemah":** bandingkan berpasangan:
`dataset↔loader`, `retriever↔retriever`, `prompt+llm_client↔rag_chain`, `tools↔tools_lc`, `agent↔agent_langgraph`.
Setiap pasangan: "apa yang framework sembunyikan dari saya?"

**Hari 6 — refleksi:** baca `graph_rag_reflektif.py` dan tonton `--dry-run`-nya. Pertanyaan pemantik:
apa yang terjadi kalau grade selalu "tidak relevan"? (Jawab: rewrite 2× lalu jawab jujur — `MAX_PERCOBAAN`.)

**Hari 7 — eksperimen mandiri:** turunkan `TOP_K` ke 3 lalu naikkan ke 20; amati kualitas jawaban dan panjang prompt.
Inilah intuisi *trade-off* retrieval yang tidak bisa diajarkan teori.

---

## 12. FAQ

**Q: Kenapa jawaban agent/RAG harus kutip pasal?**
A: Karena datasetmu punya provenance (halaman, baris). Jawaban tanpa sitasi membuang keunggulan terbesar data ini — auditabilitas.

**Q: Bisakah pakai LLM lokal (Ollama)?**
A: Folder B sudah siap (`LLM_BACKEND=ollama`); Folder A tinggal isi `LLM_BASE_URL=http://localhost:11434/v1` + `LLM_MODEL`
sesuai model Ollama — keduanya bicara protokol OpenAI-compatible. Saat ini Ollama belum terpasang di mesinmu.

**Q: Kenapa tidak pakai Milvus/vektor DB?**
A: 63 ribu vektor × 1024 dim = 259 MB; dot product NumPy butuh < 1 detik. Database vektor masuk akal saat data
jutaan baris atau butuh filter terdistribusi — jalur `milvus_search()` sudah disiapkan untuk saat itu.

**Q: Bedanya `--dry-run` dan mock?**
A: `--dry-run` = kode asli, hanya panggilan LLM diganti (retrieval tetap nyata). Mock (`MockLLM`/`MockAgentLLM`) =
LLM palsu dengan skrip, untuk menguji *logika loop* deterministik di `verify_setup.py`.

**Q: Kenapa score top-1 cuma 0.63 — apakah retrieval jelek?**
A: Tidak. Cosine similarity untuk teks hukum jarang melewati 0.7; eval resmi di `eval/` mengukur recall@8 = 0,75,
artinya 75% kasus uji pasal yang benar masuk 8 teratas. Skor absolut memang tidak punya ambang universal —
yang penting *urutan relatif*.

**Q: Kenapa ada reranker dan kapan mematikannya?**
A: Reranker cross-encoder (Qwen3-Reranker-0.6B) menilai ulang 12 kandidat
atasan dengan membaca query+pasal bersamaan — lebih tajam dari cosine tapi
mahal (CPU ~1 menit per batch). Matikan dengan `--no-rerank` / `RERANK=0`
saat iterasi cepat; biarkan aktif untuk kualitas jawaban final. Model gagal
termuat? Sistem otomatis fallback ke urutan cosine.

**Q: Kapan harus menambah tool baru?**
A: Saat LLM butuh **data/aksi di luar dataset + teks** (hitung masa berlaku, bandingkan 2 UU, ambil tanggal hari ini).
Alur menambah: tulis fungsi → bungkus skema (A) atau `@tool` (B) → tambah ke `TOOLS` → `verify_setup.py` → uji dry-run.

---

*Dokumen ini dihasilkan dari kode aktual kedua folder (nama fungsi/class diverifikasi via AST & grep) dan hasil
verifikasi yang benar-benar dijalankan: verify A 11/11, verify B --full 8/8, dry-run retrieval top-8 dengan skor
0.60–0.63 pada data asli 63.231 vektor.*
