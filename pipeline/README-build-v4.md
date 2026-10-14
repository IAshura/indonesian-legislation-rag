# Dataset RAG Peraturan Indonesia — Build "Combined" v4.1 (KANONIK)

Dataset siap-embed peraturan Indonesia (378 peraturan, 1945–2024; korpus
Mendeley Data DOI `10.17632/8wpjvhh7g2.1`, **CC BY 4.0** — cantumkan atribusi:
*Qausar, Prabowo, Arifin & Isa — Binus University*).

Versi 4.1 memakai kebijakan gabungan hasil kurasi tiga eksperimen arsip
(`../_archive/`: muse_spark, GPT, qwen) + 10 keputusan desain yang telah
disepakati. **Folder ini satu-satunya pipeline aktif.**

## Prinsip build v4.1 (10 keputusan desain)

1. **Struktur combined dipertahankan** — chunk / parent / windows / documents /
   quarantine / manifest, satu pipeline deterministik.
2. **Preservasi unit (prinsip GPT):** 1 baris `units.csv` = 1 chunk; ID dan
   provenance tak pernah berubah oleh pemrosesan.
3. **Parent–child (qwen):** retrieval di child (unit/ayat), jawaban memakai
   konteks pasal utuh dari `parents.jsonl`.
4. **Audit + manifest (muse):** hash SHA256 sumber, disposition beralasan
   untuk semua baris, guide pemakaian di `manifest.json`.
5. ** Tidak ada merge otomatis antar-unit (skema 4).** Fragmen pendek
   (< 8 token / `needs_review_short`) **dipertahankan dengan
   `is_fragment: true`**. Skrip embedding melewatinya secara default —
   jadi indeks bersih tanpa kehilangan satu baris pun dari korpus.
6. **`parent_id` global & stable:** `{document_id}::p::{pasal}` (pasal) atau
   `{document_id}::adm::{nn}` (blok administratif non-pasal). Berlaku unik
   lintas seluruh korpus — aman dijadikan payload di vector database apa pun.
7. **Parent panjang dipecah window deterministik.** `parents.jsonl` menyimpan
   teks pasal penuh (untuk sitasi/audit); `parents_windows.jsonl` menyimpan
   penggalan maksimal 1.800 char secara keras dengan tumpang tindih 150 char
   (`window_id = {parent_id}::w{nn}`) — dikirim ke LLM tanpa menembus batas
   token konteks.
8. **`content_role` granular** menggantikan biner: `norma`, `pertimbangan`,
   `dasar_hukum`, `preamble`, `administratif`, `ketentuan_peralihan`,
   `ketentuan_penutup`, `penjelasan` + field audit `role_source`
    (marker | flag+artifact | flag+keyword | keyword | default). Kolom `part` lama
   (norma/administratif) tetap diturunkan sebagai alias kompatibilitas.
9. **Status keberlakuan** disediakan lewat lapisan `validity.jsonl` +
   `enrich_validity.py` — *default offline, tidak pernah menebak*; isi
   `legal_status` hanya dengan bukti (lihat bawah). Tanpa ini, RAG hukum
   berisiko mengutip norma yang sudah dicabut.
10. **Eksperimen lama diarsipkan** (`../_archive/`) — dataset kanonik hanya
    di sini.

## Pipeline raw → siap feed embedding

`python build_dataset.py` membaca `../data/corpus/units.csv` + `documents.csv` +
`texts/`, melakukan:

1. **Validasi skema** kolom kedua CSV (fail-fast) dan pencatatan **SHA256**
   sumber di manifest.
2. **Restorasi identitas peraturan** dari blok judul `documents.csv` →
   judul resmi + topik (`Undang-Undang Nomor 7 Tahun 2017 tentang ...`) —
   konteks termahal untuk embedding.
3. **Per-document streaming** urut (`page → line → unit_id`) dengan status
   carry-forward **hanya bab & pasal** dari unit heading — *ayat tidak pernah
   carry-forward* dan tidak pernah dikarang; `citation.ayat = null` berarti
   pasal satu-norma (sah menurut Lampiran UU 12/2011 & Akoma Ntoso).
4. Heading (`heading_*`) menjadi label konteks, **tidak di-embed**.
5. Seleksi: body ok/administratif → chunk; `structural_heading` < 200 char →
   quarantine (serpihan judul); duplikat exact → quarantine beralasan.
6. Fragmen pendek → **tetap chunk** dengan `is_fragment: true`.
7. Peran granular `content_role` + audit `role_source`.
8. Pembentukan **parent global** per pasal/blok admin + **windows** deterministik.
9. `embedding_input = judul_resmi - sitasi: isi` — self-contained per vektor.
10. **QA provenance otomatis** (300 sampel diverifikasi ke baris asli di
    `texts/`; ambang 90% — gate gagal-build) + validasi silang: 0 chunk yatim,
    0 `chunk_id` duplikat.

**Funnel build ini:** 90.588 baris → 74.218 chunks (4.704 fragmen
ber-flag; 66.292 norma) • 11.413 parents • 14.689 windows • 2.209 quarantine
beralasan • **QA match 98% — PASS**.

## Isi `result/`

| File | Isi |
|---|---|
| `chunks.jsonl` | 74.218 record siap-embed (`embedding_input`, `content_role`, `is_fragment`, provenance) |
| `parents.jsonl` | Teks pasal utuh + `window_ids` (konteks LLM) |
| `parents_windows.jsonl` | Penggalan maksimal 1.800 char (+overlap) untuk konteks berbatas token |
| `documents.jsonl` | Register 378 peraturan + kolom kosong `legal_status` dst. siap pengayaan |
| `validity.jsonl` | Lapisan status keberlakuan (status default `pending_enrichment`) |
| `quarantine.jsonl` | Baris terbuang + alasan terukur |
| `manifest.json` | Hash sumber, funnel, kebijakan, skema, hasil QA |

## Cara feed ke model embedding (tahap berikutnya)

Satu aturan besi: **model index = model query** — ganti model berarti embed
ulang (metadata model tercatat di `embeddings/meta.json`).

Build ini sengaja belum menjalankan embedding dan belum membuat folder
`result/embeddings/`. Karena target Anda Qwen, gunakan `chunks.jsonl` sebagai
input dan field `embedding_input` sebagai teks embedding; ganti loader/model
di tahap embedding sesuai checkpoint Qwen yang dipilih. Struktur JSONL,
`chunk_id`, `parent_id`, dan metadata tidak perlu diubah.

```powershell
cd pipeline
python embed_example.py                       # default: part=norma, fragmen DILEWATI
python embed_example.py --part all --include-fragments   # indeks paling lengkap
python embed_example.py --model intfloat/multilingual-e5-large --batch-size 32
```

*Mode API eksternal:*
`python embed_example.py --engine api --base-url https://api.openai.com/v1 --model text-embedding-3-large`
(key via env `EMBED_API_KEY`).

Output: `result/embeddings/embeddings.npy` (vektor float32 L2-normal, urut ⇄
`ids.jsonl`), `ids.jsonl` (pointer: chunk_id/parent_id/part/content_role/year),
`meta.json`. Pasang: `faiss.IndexFlatIP(dim)` (dot = cosine). Model e5 memakai
awalan `passage:`/`query:` otomatis.

**Alur jawaban (small-to-big):**retrieval kandidat → join `parent_id` → ambil
`text` pasal utuh (atau window-nya) → LLM → kutip `citation_str` +
`official_title` + halaman dari provenance. Saringan murni: `content_role ==
"norma"` (dan/atau `part == "norma"`), `doc_type`, `legal_force_axis`, `year`.

## Tahap selanjutnya (berurutan)

1. **Status keberlakuan:** `python enrich_validity.py --live` (best-effort
   cari kandidat di peraturan.bpk.go.id; URL tersimpan) → verifikasi → isi
   `legal_status` & gabungkan ke payload vector DB. **Jangan mengarang status.**
2. **Ukur kualitas retrieval setelah embedding Qwen tersedia:** `python
   eval/eval_retrieval.py --k 8` terhadap `eval/golden_queries.jsonl`
   (recall@8; laporan di `eval/golden_report.json`; recall ≥ 0,8 baru lanjut;
   < 0,6 ganti model/top-k dulu). Tahap ini sengaja belum dijalankan.
3. **Campuran sebelumproduksi:** gabungkan embedding (dense) + BM25 atas
   `text_lexical`, lalu rerank — biasanya menaikkan recall lebih murah daripada
   tukar model.
4. **Embedding full:** jalankan `embed_example.py` penuh (GPU disarankan).

## Regenerasi & audit

```
python build_dataset.py            # deterministik (seed tetap, sort tetap)
# manifest.validation: qa_gate PASS = 0 yatim, 0 duplikat, match_rate >= 0.90
```

Rekomendasi model saat ini: `intfloat/multilingual-e5-base|large`
(aman multibahasa), `paraphrase-multilingual-mpnet-base-v2`, atau model khas
Indonesia. Simpan `meta.json` sebagai penanda versi vektor.
