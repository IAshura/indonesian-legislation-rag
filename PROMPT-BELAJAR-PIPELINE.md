# PROMPT TUGAS — Situs Pembelajaran "Dari Data Mentah sampai Retrieval" (Proyek Indonesian Legislation RAG)

Kamu adalah pengajar teknik yang sangat teliti. Tugasmu: **pelajari proyek di workspace ini dengan membaca file aslinya**, lalu **buat situs web statis (HTML, dark mode, multi-halaman)** yang menjelaskan SEMUA proses proyek secara sangat detail untuk PEMULA TOTAL — dari data mentah sampai retrieval di Milvus. Bahasa: **Indonesia**.

## ATURAN KERAS

1. **JANGAN mengubah/menghapus file proyek apa pun.** Kamu hanya boleh MEMBACA file proyek dan MEMBUAT file HTML baru di folder `docs/belajar/` (buat foldernya jika belum ada).
2. Semua penjelasan **wajib berbasis kode asli**: kutip cuplikan kode sungguhan + sebut nama file dan nomor baris. **Tidak boleh mengarang** angka, nama field, atau perilaku — semua diverifikasi ke file aslinya (`pipeline/result/manifest.json` adalah sumber angka resmi).
3. Contoh input/output di setiap tahap **harus data nyata** yang kamu ambil dari file proyek (bukan contoh karangan).
4. Bahasa manusia dulu, baru teknis: setiap istilah asing (RAG, embedding, vektor, cosine, pooling, HNSW, IVF, FLAT, JSONL, provenance, dst.) dijelaskan saat pertama muncul, dan dikumpulkan di bab glosarium.
5. Bahasa pemrograman/library (Python, NumPy, PyTorch, transformers, pymilvus, Milvus, Docker) dijelaskan seperlunya untuk pemula — jangan anggap pembaca paham.

## LOKASI PROYEK

Root proyek: `C:\Users\Administrator\Downloads\Indonesian legislation as data 90,588 norm units e\Indonesian legislation as data 90,588 norm units e`
Mulai dengan membaca `README.md` dan `pipeline/README-build-v4.md` untuk gambaran, lalu baca file lain sesuai bab.

## TUGAS: jelaskan ALUR DATA END-TO-END ini, bab per bab, satu file HTML per bab

| Bab | File HTML | Materi wajib |
|---|---|---|
| 00 | `index.html` | Peta pipeline lengkap (diagram alur), daftar bab, cara memakai situs, atribusi korpus (CC BY 4.0, Qausar dkk. — Binus, DOI 10.17632/8wpjvhh7g2.1) |
| 01 | `01-dasar-rag.html` | Konsep RAG untuk pemula: data → embedding → vektor → pencarian mirip → jawaban; kenapa hukum butuh pendekatan ini; diagram small-to-big (retrieval ayat → jawaban pakai pasal utuh) |
| 02 | `02-data-mentah.html` | Bedah korpus sumber: `data/corpus/units.csv` (jelaskan **setiap kolom**), `documents.csv`, `texts/` (format penanda `=== PAGE n ===`), `checksums.md5`. Contoh nyata: ambil 1 baris units.csv asli (mis. UUD 1945) dan tampilkan isinya, tunjukkan bagaimana alamat (file, page_number, line_start–line_end) menunjuk ke file `texts/` |
| 03 | `03-build-dataset.html` | Bedah `pipeline/build_dataset.py` **fungsi-per-fungsi, baris penting per baris**: konstanta & aturan (WINDOW_MAX_CHARS 1800, overlap 150, fragmen <8 token, dst.), utilitas, `build_doc_meta` (restorasi identitas dari blok judul), `role_of` + `looks_like_admin_artifact` (klasifikasi `content_role`: norma/pertimbangan/dasar_hukum/preamble/administratif/penjelasan + `role_source`), class `B` (carry-forward bab/pasal — **ayat tidak pernah di-forward**, `parent_id` global `{document_id}::p::{pasal}` & blok `::adm::`, dedup + quarantine beralasan), pembentukan window deterministik, QA provenance 300 sampel, penulisan manifest. Wajib: contoh nyata **1 baris units.csv → 1 record chunks.jsonl** (JSON asli dari `pipeline/result/chunks.jsonl`) → potongan parents.jsonl → parents_windows.jsonl |
| 04 | `04-hasil-build.html` | Isi `pipeline/result/`: 7 file (chunks, parents, parents_windows, documents, quarantine, validity, manifest) — tampilkan record asli dari tiap file, jelaskan field per field, funnel angka: 90.588 → 74.218 chunks (4.704 fragmen) → 11.413 parents → 14.689 windows → 2.209 quarantine → QA 98% PASS (baca `manifest.json`) |
| 05 | `05-embedding.html` | Konsep embedding untuk pemula (teks → vektor → kemiripan cosine); beda mean-pooling e5 (`embed_example.py`) vs last-token pooling + instruksi Qwen3 (notebook `pipeline/colab/`); artefak `embeddings.npy` + `ids.jsonl` + `meta.json`; konvensi folder & file `ACTIVE` (`emb_paths.py`); verifikasi `verify_embeddings.py`; aturan besi **model index = model query** |
| 06 | `06-milvus-retrieval.html` | Milvus lokal (Docker: etcd+minio+standalone, data di `C:\milvus\volumes`); bedah `insert_milvus.py` (skema field koleksi, index HNSW/IP, insert batch, guard anti-dobel), `smoke_milvus.py` (self-query top-1 dirinya sendiri), koleksi terpisah 4B/0.6B (`legislation_norma_v1`, `legislation_norma_q3e06b`), demo `--demo` |
| 07 | `07-query-eval.html` | `embed_query_qwen.py` (format `Instruct:/Query:`, last-token pooling, MRL keep_dim), `eval/golden_queries.jsonl` (12 query emas) & `eval/eval_retrieval.py` (recall@8, matcher: `doc_regex` dicek ke `official_title` ATAU `doc_type_full`; `pas_num` aman untuk pasal tanpa sufiks), cara baca `golden_report.json` (recall@8 = 0.750) |
| 08 | `08-hnsw-ivf-flat.html` | Ringkas dari `pipeline/README-retrieval.md`: HNSW vs IVF vs FLAT, apa yang berubah di create_index vs param search, jebakan index asinkron, baseline exact numpy |
| 09 | `09-pelacakan-satu-norma.html` | **Bab puncak**: lacak SATU contoh dari hulu ke hilir dengan data asli — "Kedaulatan berada di tangan rakyat..." (UUD Pasal 1 Ayat 2): baris units.csv → chunk uud-1945::p::1::02 → embedding_input → vektor 1024-d → entri ids.jsonl → entitas Milvus → hasil pencarian top-8 → parent pasal (small-to-big). Tampilkan cuplikan data asli di setiap lompatan |
| 10 | `10-glosarium.html` | Semua istilah (RAG, chunk, parent, window, embedding, cosine, IP, HNSW, IVF, FLAT, pooling, fragment, quarantine, provenance, legal_force_axis, dll.) |

## DESAIN SITUS (wajib)

- **Dark mode elegan & rapi**: latar gelap berlapis, aksen warna hangat, kontras baik, tipografi enak dibaca, blok kode ber-gaya konsol dengan penanda file & nomor baris.
- **Tanpa dependensi internet** (tanpa CDN) — HTML/CSS/JS murni agar tetap bagus offline; visualisasi pakai CSS/SVG/HTML buatan sendiri.
- Setiap bab punya **diagram alur** prosesnya (input → transformasi → output) dan **tabel** skema/field.
- `index.html` = daftar isi bergambar (peta pipeline utuh + status tiap tahap), navigasi antar-bab (prev/next) di setiap halaman.
- Gunakan **data asli** dari file proyek sebagai contoh (jangan dummy) — mis. kutip JSON nyata dari `chunks.jsonl` untuk uud-1945::p::1::02.
- Bahasa Indonesia, nada mengajar yang sabar, analogi sederhana untuk konsep abstrak.

## CARA KERJAMU

1. Baca `README.md` → `pipeline/README-build-v4.md` → `pipeline/result/manifest.json` untuk kerangka dan angka resmi.
2. Baca setiap file yang disebut di tabel bab **sebelum menulis babnya** (kode dibaca, tidak ditebak).
3. Verifikasi angka dari file hasil; jika ada yang tak cocok, catat koreksinya di bab terkait.
4. Jangan menjalankan perintah yang mengubah data; hanya baca file.
5. Hasil akhir: folder `docs/belajar/` berisi `index.html` + satu HTML per bab (00–10). Selesai, laporkan daftar file + cara membukanya.

Contoh standar "input → proses → output" yang wajib ada per bab (pakai kasus nyata): baris CSV mentah → record chunk; `embedding_input` → vektor float32 L2-normal → baris ids.jsonl → entitas Milvus; pertanyaan pengguna → vektor query → top-k skor → parent pasal utuh → siap jadi konteks LLM.
