# Cara Menyalakan + Penggunaan (setelah restart laptop)

Tutorial ini untuk laptop kamu (model ringan `qwen3-06b`, CPU).
Semua perintah dijalankan di PowerShell. Path pakai tanda kutip karena ada spasi.

## 1. Nyalakan Docker + Milvus

1. Buka **Docker Desktop**, tunggu kiri bawah `Engine running`.
2. Di PowerShell:
```powershell
cd C:\milvus
docker compose up -d
docker ps
```
Harus muncul 3 baris `Up`: `milvus-standalone`, `etcd`, `minio`.
Tunggu ~1 menit pertama (Milvus loading).

Matikan nanti (data aman):
```powershell
cd C:\milvus
docker compose down
```

## 2. Masuk project + venv

```powershell
cd "C:\Users\Administrator\Downloads\Indonesian legislation as data 90,588 norm units e\Indonesian legislation as data 90,588 norm units e"
.\.venv\Scripts\Activate.ps1
```

## 3. Cek ACTIVE (wajib, 5 detik)

```powershell
Get-Content "pipeline\result\embeddings\ACTIVE"
```

Harus: `qwen3-06b` (laptop pakai ini, jangan `qwen3-4b`).
Kalau salah, kembalikan:
```powershell
Set-Content -LiteralPath "pipeline\result\embeddings\ACTIVE" -Value "qwen3-06b"
```

## 4. Test Milvus nyambung (tanpa model, cepat)

```powershell
cd "C:\Users\Administrator\Downloads\Indonesian legislation as data 90,588 norm units e\Indonesian legislation as data 90,588 norm units e\pipeline"
..\.venv\Scripts\python.exe smoke_milvus.py --n 5 --top 3
```

Bagus kalau: `63,231 entitas` + `score ~1.0`.
Warning `PyMilvusDeprecationWarning` abaikan saja.

## 5. Test cari 1 pertanyaan (pakai model, agak lama pertama kali)

```powershell
# masih di folder pipeline
..\.venv\Scripts\python.exe insert_milvus.py --demo "siapa pemegang kedaulatan menurut uud 1945?"
```

Harus: top-1 `chunk=uud-1945::p::1::02`.

Ganti exact vs HNSW (opsional):
```powershell
..\.venv\Scripts\python.exe insert_milvus.py --rebuild-index --index flat
..\.venv\Scripts\python.exe insert_milvus.py --rebuild-index --index hnsw
```

## 6. Pemakaian sehari-hari (RAG hybrid, ini yang dipakai)

```powershell
cd "C:\Users\Administrator\Downloads\Indonesian legislation as data 90,588 norm units e\Indonesian legislation as data 90,588 norm units e\rag-tanpa-framework"
..\.venv\Scripts\python.exe rag_jawab.py -q "Di tangan siapa kedaulatan berada menurut UUD 1945?" --dry-run --show-context
```

Arti flag:
- default = hybrid (dense + BM25) + rerank (paling bagus)
- `--no-hybrid` = dense saja (untuk banding)
- `--no-rerank` = tanpa rerank (cepat, ranking jelek)
- `--doc-type UUD` = filter jenis, contoh: `--doc-type UUD`
- hapus `--dry-run` kalau mau jawab beneran pakai LLM (butuh file `.env` berisi API key)

Contoh lain:
```powershell
..\.venv\Scripts\python.exe rag_jawab.py -q "Berapa lama masa jabatan presiden dan berapa kali dapat dipilih kembali?" --dry-run --show-context
..\.venv\Scripts\python.exe rag_jawab.py -q "Siapa yang memegang kekuasaan pemerintahan menurut UUD 1945?" --dry-run --show-context --doc-type UUD
```

## Cara baca hasil

```
[S1] skor 0.48 UUD — Bab I Pasal 1 (parents)
```
- `S1` = peringkat 1, `skor` = rerank P(yes) (bukan cosine)
- `parents` = teks pasal utuh (untuk LLM), `parents_windows` = potongan window
- Gap besar (0.48 vs 0.08) = yakin. Rapat (0.86 vs 0.79) = ragu, tambah filter.

Melihat teks asli chunk/parent:
```powershell
# dari folder pipeline
$line = (Select-String -LiteralPath "result\chunks.jsonl" -Pattern "uud-1945::p::1::02" | Select-Object -First 1).Line
($line | ConvertFrom-Json).embedding_input
$pline = (Select-String -LiteralPath "result\parents.jsonl" -Pattern "uud-1945::p::1" | Select-Object -First 1).Line
($pline | ConvertFrom-Json).text
```

## Troubleshooting

| Gejala | Artinya | Obat |
|---|---|---|
| `Connection refused ... 19530` | Milvus mati | Balik ke Langkah 1 (`docker compose up -d`) |
| `dimension mismatch 10240 vs 4096` | ACTIVE vs koleksi beda model (2560=4b, 1024=06b) | Balik ke Langkah 3, samakan ke `qwen3-06b`. Kalau mau pindah model: `insert_milvus.py --drop` (insert ulang 63 ribu) |
| `No module named 'torch'` | venv belum install ML | `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu transformers>=4.40 numpy` atau `pip install -r requirements.txt` |
| Skor semua rendah (<0.2) + isi ngawur | Pertanyaan di luar 378 dokumen (contoh: PP 13/2009 tidak ada di korpus) | Cek dulu `data\corpus\documents.csv`, jawaban benar = `Tidak ditemukan dasar dalam korpus.` |
| `norma_tokens.txt tidak ada` | Index BM25 belum dibangun | Dari folder `rag-tanpa-framework`: `..\.venv\Scripts\python.exe bm25.py --build` (sekali saja) |

## Hafalan urutan

```
Docker up → ACTIVE qwen3-06b → smoke → demo → rag_jawab
```
