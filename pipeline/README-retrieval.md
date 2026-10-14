# Pencarian Vektor: HNSW vs IVF vs Brute Force (Exact Search)

Penjelasan cara kerja tiga jenis pencarian vektor yang dipakai/dapat dipakai
di pipeline ini, apa saja yang berubah bila berganti jenis, dan rekomendasi
untuk dataset ini. **Dokumen ini murni penjelasan — tidak ada kode yang diubah.**

## 1. Posisi pencarian dalam pipeline ini

Ada **dua tempat** pencarian vektor dilakukan:

| Tempat | Jenis pencarian saat ini | File |
|---|---|---|
| **Milvus (Docker lokal)** | `HNSW` + metrik `IP` (index dibuat oleh `insert_milvus.py` saat koleksi dibuat; search memakai `ef: 64`) | `insert_milvus.py`, `smoke_milvus.py` |
| **Eval (numpy murni)** | **Brute force / exact** — `emb @ q` dihitung ke seluruh vektor | `eval/eval_retrieval.py` (baris `scores = emb @ vecs.T`) dan cell 7 notebook Colab |

Artinya: **baseline exact search sudah ada** dan selalu bisa dipakai sebagai
"jawaban benar" pembanding. HNSW/IVF dianggap layak hanya bila hasilnya
mendekati baseline exact ini dengan latensi lebih murah.

## 2. Tiga jenis pencarian — konsep singkat

| Jenis | Cara kerja | Recall | Kecepatan di data kita (63 rb × 1024/2560) | Kapan layak |
|---|---|---|---|---|
| **FLAT (brute force / exact)** | Query dibandingkan ke **semua** vektor satu per satu, ambil top-k. Kebenaran mutlak. | **100%** — selalu sama dengan baseline | ~10–50 ms (masih sangat cepat) | Dataset ≤ ratusan ribu; sebagai pembanding resmi |
| **HNSW** | Membangun **graph** tetangga-terdekat berlapis; pencarian menelusuri graph secara greedy dari titik masuk. | ~99–100% bila `ef` cukup besar | ~1–5 ms | Pilihan produksi default; andal di semua ukuran |
| **IVF (IVF_FLAT dll.)** | Ruang vektor dibagi jadi `nlist` **kluster** (centroid). Query hanya mengecek `nprobe` kluster terdekat, bukan semua data. | Bergantung rasio `nprobe/nlist` | ~2–10 ms | Baru terasa manfaatnya di **jutaan** vektor |

Metrik **`IP` (inner product)** dipakai semua jenis karena vektor kita sudah
L2-normal — dot product = cosine. Ini **tidak berubah** kapan pun index
diganti.

## 3. Yang berubah di mana kalau berganti jenis pencarian

Hanya **dua titik** yang menyentuh jenis index — sisanya (skema koleksi,
insert, `ids.jsonl`, encoder query) **tetap sama persis**:

### a) Saat membuat koleksi — `insert_milvus.py`, bagian `create_index`

| Jenis | `index_params` | Catatan |
|---|---|---|
| HNSW *(aktif sekarang)* | `{"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}` | `M` = lebar graph; `efConstruction` = kualitas build |
| IVF_FLAT | `{"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 512}}` | Aturan kasar `nlist ≈ 4·√N` (63 rb → ±1.000; umum dipakai 128–1024). Milvus menolak build bila `nlist` > jumlah vektor |
| FLAT (exact) | `{"index_type": "FLAT", "metric_type": "IP", "params": {}}` | Tanpa parameter; juga terjadi otomatis bila koleksi dibuat **tanpa index** sama sekali |

> Index **terikat saat koleksi dibuat**. Ganti index type = buang koleksi
> (`--drop`) dan insert ulang, **atau** buat koleksi kedua berdampingan untuk
> benchmark (pola yang sama dengan koleksi terpisah 4B/0.6B — lihat
> `emb_paths.py`).

### b) Saat mencari — parameter `search`

| Jenis | `param` search | Efek menaikkan nilainya |
|---|---|---|
| HNSW *(aktif: `ef: 64`)* | `{"metric_type": "IP", "params": {"ef": 64}}` | `ef` 64 → 128 → 256: recall naik, latensi naik |
| IVF | `{"metric_type": "IP", "params": {"nprobe": 32}}` | `nprobe` naik → makin banyak kluster dicek → recall naik |
| FLAT | `{"metric_type": "IP", "params": {}}` | Tidak ada parameter — hasil selalu exact |

Parameter search bebas diubah **tanpa insert ulang** — hanya parameter
index-lah yang terikat di koleksi.

## 4. Brute force exact — dua jalur yang sudah tersedia

1. **Numpy murni** (tanpa Docker/Milvus): `emb @ q` di eval lokal dan cell 7
   Colab. 63 rb × 1024 di CPU masih hitungan milidetik per query — inilah
   alasan eval resmi memakai ini sebagai kebenaran acuan.
2. **Milvus tanpa index / index FLAT**: pembagian kerja yang sama dilakukan di
   dalam Milvus.

Bukti dari sesi ini: top-5 Milvus HNSW (`ef` 64 maupun 256) **identik skor dan
urutan** dengan numpy exact untuk query kedaulatan UUD 1945 — ANN kita sehat.

## 5. Jebakan yang pernah kami alami (penting!)

- **Index dibangun belakangan (asinkron).** Beberapa detik setelah
  insert/flush, HNSW belum siap; query bisa mengembalikan hasil yang salah
  (kasus nyata: demo pertama mengembalikan chunk berbeda dengan skor lebih
  rendah; run berikutnya sudah identik dengan exact). Solusi: tunggu indeks
  selesai / jalankan `smoke_milvus.py` dulu.
- **IVF sensitif data kecil.** Dengan 63 rb vektor, manfaat IVF belum terasa
  dan recall lebih rapuh terhadap `nprobe`; FLAT lebih sederhana.
- **Jangan campur metrik.** Semua koleksi kita `IP` + vektor L2-normal. Search
  dengan `L2` pada vektor yang dinormalisasi akan memberi peringkat yang
  terbalik secara skala.

## 6. Rekomendasi untuk dataset ini

| Kebutuhan | Pilihan |
|---|---|
| Benchmark / kebenaran acuan | **Numpy exact** (eval lokal & cell 7 Colab) — sudah ada, jangan diubah |
| Dataset saat ini (63 rb) | **FLAT** atau HNSW sama-sama cepat; HNSW `ef` 64–128 aman untuk produksi |
| Nanti jutaan vektor / latensi ketat | HNSW `ef` lebih besar, atau mulai uji `IVF_FLAT` (`nlist` ±1.000, `nprobe` mulai 64) |
| Aturan lulus ANN | recall top-8 vs exact ≥ 0.99 sebelum percaya angka latensi |

Cara mengukur recall ANN vs exact: jalankan eval exact (numpy) dan pencarian
Milvus untuk golden set yang sama, lalu bandingkan `top_ids` per query —
persis seperti perbandingan yang pernah dilakukan di sesi ini.

## 7. Ringkasan satu kalimat

**Yang berubah hanya `index_params` saat membuat koleksi dan `params` saat
search (`ef` untuk HNSW, `nprobe` untuk IVF, kosong untuk FLAT) — schema,
insert, encoder, dan aturan "model index = model query" tidak tersentuh.**
