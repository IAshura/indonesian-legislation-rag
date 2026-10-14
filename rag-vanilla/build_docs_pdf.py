"""
build_docs_pdf.py — Membangun `code-guide-rag-vanilla.pdf`
============================================================================

File ini adalah ISI; cara mencetaknya ada di pdf_engine.py.
Setiap bab = satu file kode di folder ini, dengan struktur tetap:
  1) untuk apa file itu ada   4) penjelasan tiap fungsi/class
  2) konsep kunci             5) jebakan umum
  3) tabel entitas (ast)      6) contoh menjalankan

Jalankan:  python build_docs_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pdf_engine as E

FOLDER = Path(__file__).resolve().parent


def bab_file(nomor: str, nama: str, tujuan: str, konsep: list[str],
             catatan: list[tuple[str, str]], jebakan: list[str],
             contoh: str) -> list:
    """Satu bab PDF utk satu file kode (struktur tetap, lihat docstring)."""
    isi = E.bab(nomor, nama)
    isi += E.para(tujuan)
    isi += E.sub("Konsep kunci") + E.bullets(konsep)
    isi += E.sub("Fungsi & class (terbaca otomatis dari kode)")
    isi += E.tabel_entitas(FOLDER / nama)
    isi += E.sub("Penjelasan tiap entitas")
    for nama_ent, jel in catatan:
        isi += E.para(f"*{nama_ent}* — {jel}")
    isi += E.sub("Jebakan umum") + E.bullets(jebakan)
    isi += E.sub("Cara mencoba") + E.kode(contoh)
    return isi + [E.PageBreak()]


# ===================================================================== ISI
ISI: list = []

# ------------------------------------------------------------- BAB 1
ISI += E.bab("1", "Ringkasan arsitektur & cara memakai folder ini")
ISI += E.para("Folder ini menghadirkan dua arsitektur LLM untuk proyek "
              "*Indonesian Legislation RAG* — TANPA framework apa pun. "
              "Dataset (90.588 unit norma, 378 peraturan) dan retrieval "
              "(Qwen3-Embedding-0.6B, dim 1024) adalah aset proyek yang "
              "sudah terukur; kode di sini hanya 'lem' yang menyambungkan "
              "pertanyaan pengguna ke aset itu, lalu ke LLM.")
ISI += E.sub("Arsitektur A — fixed-pipeline RAG (answer.py)")
ISI += E.kode(
    "pertanyaan -> embed query (Qwen3-0.6B, last-token, L2)\n"
    "           -> exact cosine search top-k (embeddings.npy, mmap)\n"
    "           -> join parent_id -> pasal utuh / window (small-to-big)\n"
    "           -> prompt 3 blok ([S1]..[Sn] ber-sitasi)\n"
    "           -> LLM (1 panggilan) -> jawaban + daftar sumber")
ISI += E.para("Deterministik, 1 panggilan LLM, mudah diaudit — inilah titik "
              "mulai yang direkomendasikan panduan proyek (bab 7).")
ISI += E.sub("Arsitektur B — agentic + tool calling (agent.py)")
ISI += E.kode(
    "pertanyaan + daftar tools -> LLM\n"
    "   <- tool_calls {name, arguments}          (LLM meminta, bukan menghitung)\n"
    "   kode KITA eksekusi tool -> hasil JSON -> kirim balik (role=tool)\n"
    "   <- (ulang maks 5 putaran) atau jawaban final + sitasi")
ISI += E.sub("Aturan besi proyek yang dijaga di seluruh folder")
ISI += E.bullets([
    "Model embedding query = model embedding index (Qwen3-Embedding-0.6B).",
    "Hanya part == 'norma' masuk konteks (embeddings memang hanya berisi norma).",
    "Sitasi selalu dari field dataset (citation_str, provenance halaman/baris) "
    "— tidak pernah dikarang LLM.",
    "Status hukum tidak pernah ditebak: check_validity melapor "
    "pending_enrichment bila belum diverifikasi.",
])
ISI += E.sub("Langkah pertama")
ISI += E.kode(
    "python verify_setup.py            # cek dataset + retrieval (offline)\n"
    "python verify_setup.py --full     # + uji embedder\n"
    "copy .env.example -> .env         # isi backend LLM (API / Ollama)\n"
    "python answer.py -q \"...\" --dry-run   # lihat prompt tanpa jaringan")
ISI += [E.PageBreak()]

# ------------------------------------------------------------- BAB 2
ISI += bab_file("2", "config.py",
    "Satu pintu semua konfigurasi: path dataset, konstanta retrieval, dan "
    "konfigurasi LLM (dibaca dari environment atau file .env). Mengganti "
    "backend LLM cukup di sini — file lain tidak tersentuh.",
    ["Path dihitung dari lokasi file (parents[1] = root proyek) sehingga "
     "skrip jalan dari folder mana pun.",
     "Loader .env kecil tanpa python-dotenv: baris KEY=VALUE di-set via "
     "os.environ.setdefault — env asli tetap lebih diutamakan.",
     "Folder embeddings aktif dibaca dari file ACTIVE (isinya 'qwen3-06b') "
     "sehingga beberapa model embedding bisa hidup berdampingan."],
    [("get_llm_config()", "mengembalikan dict base_url/api_key/model/temperature "
      "dari LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (plus fallback OPENAI_API_KEY). "
      "Default: api.openai.com + gpt-4o-mini; ganti untuk OpenRouter/Groq/" "DeepSeek/Ollama.")],
    ["Lupa menyalin .env.example ke .env -> api_key kosong; answer "
     "menolak jalan dengan pesan yang jelas (fail-fast).",
     "Jangan commit .env — berisi kunci API."],
    "python -c \"import config; print(config.get_llm_config()['model'])\"")

# ------------------------------------------------------------- BAB 3
ISI += bab_file("3", "dataset.py",
    "Membuka 'database' JSONL hasil pipeline: chunks (74,218), parents "
    "(11.413 pasal utuh), windows (14.689), documents (378). Semua diindeks "
    "kamus by-ID agar lookup O(1), dimuat lazy saat pertama dipakai.",
    ["Pola ID tiga tingkat: document uud-1945 -> parent uud-1945::p::1 -> "
     "chunk uud-1945::p::1::02; window pasal panjang: ...::w01.",
     "SMALL-TO-BIG: pencarian memakai chunk kecil (presisi), konteks LLM "
     "memakai parent (pasal utuh) — makna hukum butuh ayat tetangga.",
     "Aturan baca konteks: parent <= 1.800 char pakai parents.jsonl; "
     "lebih panjang pakai window pertama (::w01)."],
    [("load_jsonl()", "generator pembaca JSONL baris-per-baris — hemat "
      "memori untuk 90 ribu record."),
     ("dataset()", "singleton via lru_cache(maxsize=1): instance sama "
      "dipakai semua modul tanpa memuat ulang file."),
     ("LegislationDataset", "kelas pusat; properti chunks/parents/windows/" "documents memuat lazy."),
     ("context_for_parent()", "jantung small-to-big: pilih teks pasal utuh "
      "atau window, kembalikan dict seragam ber-sitasi."),
     ("citation_line()", "susun baris sitasi human-readable dari provenance "
      "— LLM hanya mengutip baris ini, bukan mengarang.")],
    ["KeyError dari context_for_parent = sengaja fail-fast: parent hilang "
     "tanda data rusak, jangan didiamkan."],
    "python dataset.py")

# ------------------------------------------------------------- BAB 4
ISI += bab_file("4", "embedder.py",
    "Mengubah pertanyaan pengguna menjadi vektor 1024-dim dengan model yang "
    "SAMA seperti index (aturan besi). Tiga detail yang membedakan "
    "Qwen3-Embedding: format instruksi, last-token pooling, left padding.",
    ["Format query: 'Instruct: {tugas}\\nQuery: {teks}' — instruksi diambil "
     "dari meta.json (bukan hardcode). Dokumen saat indexing TIDAK diberi prefix.",
     "Last-token pooling: model decoder; representasi kalimat diambil dari "
     "vektor token terakhir yang diperhatikan (EOS).",
     "Left padding: token terakhir semua baris sejajar di posisi -1.",
     "L2-normalisasi: dot product = cosine similarity — cocok dengan "
     "metric IP di Milvus."],
    [("load_meta()", "baca meta.json embeddings aktif — sumber kebenaran "
      "parameter (model, dim, pooling, instruksi)."),
     ("_lazy_load()", "muat tokenizer+model SEKALI per proses dengan "
      "local_files_only=True (cache HF lokal; offline)."),
     ("embed_query()", "teks -> np.float32 (1024,) L2-normal; tahap 1-5 "
      "diberi komentar di kode.")],
    ["Model keluarkan BFloat16 — numpy tidak mendukung; wajib .float() "
     "sebelum .numpy() (sudah ditangani).",
     "Mengubah format Instruct/pooling = skor hancur diam-diam."],
    "python embedder.py   # dim 1024, norma 1.0")

# ------------------------------------------------------------- BAB 5
ISI += bab_file("5", "retriever.py",
    "Mesin pencarian dengan dua jalur berbentuk hasil yang sama: exact "
    "cosine via numpy (default, deterministik, tanpa server) dan Milvus "
    "HNSW (opsional, jalur produksi eval proyek).",
    ["Vektor dimuat mmap (memory-mapping): file 259 MB tidak menumpuk RAM.",
     "Exact search = satu perkalian matriks (63.231 x 1024) @ (1024,) — "
     "di skala ini tetap puluhan milidetik.",
     "unique_parent=True: dari ayat-ayat dalam pasal yang sama hanya skor "
     "tertinggi lolos — konteks tidak diisi pasal yang sama berkali-kali.",
     "Filter doc_type/year membaca ids.jsonl (metadata ringan), tanpa buka "
     "chunks.jsonl."],
    [("ExactRetriever", "kelas pencarian exact; assert bentuk matriks vs "
      "meta (fail-fast kalau index lama vs meta baru)."),
     ("milvus_search()", "pymilvus MilvusClient; metric IP + ef 64; melempar "
      "error jelas bila server/koleksi tidak ada."),
     ("retrieve()", "pintu satu: backend 'numpy' | 'milvus'."),
     ("konteks_kaya()", "join tiap kandidat ke teks parent/window + sitasi "
      "— bentuk yang dikonsumsi prompt.py.")],
    ["Index Milvus terbentuk ASINKRON setelah insert — tunggu flush/index "
     "sebelum search (jebakan klasik, dijelaskan di komentar)."],
    "python retriever.py   # top-3 pasal utk satu pertanyaan")

# ------------------------------------------------------------- BAB 5a
ISI += bab_file("5a", "bm25.py",
    "HYBRID SEARCH — jalur LEXICAL. BM25 (Okapi; k1=1.2, b=0.75) atas "
    "token pasal + sitasi + judul. Dense embedding kuat di MAKNA tapi "
    "lemah di IDENTIFIER PERSIS (nomor pasal, tahun, jenis dok, singkatan) "
    "— justru di situ BM25 paling tajam (idf tinggi utk kata jarang).",
    ["Index leksikal dibuat SEKALI offline: `python bm25.py --build` -> "
     "pipeline/result/lexical/norma_tokens.txt; baris-i file = baris-i "
     "embeddings.npy / ids.jsonl (kontrak ketatkatan urutan).",
     "Tokenisasi gaya bahasa (bukan word-piece model): case-fold + token "
     "alfanumerik >= 2 char + stopwords Indonesia dasar; TANPA stemming — "
     "makna diserahkan ke jalur dense.",
     "Satu fungsi tokenisasi utk builder & query (aturan besi satunya: "
     "token index = token query).",
     "BM25Scorer: postings index token->[(bari, tf)] + precompute idf; "
     "query hanya menykor dok yang memuat >=1 token query (puluhan ms).",
     "Fallback aman: index belum dibangun -> FileNotFoundError yang jelas "
     "di-score, dan HybridRetriever otomatis pakai dense-only.",
     "FUSI RRF (Reciprocal Rank Fusion) di retriever.py: skor(c) = sum "
     "1/(60 + rank_jalur) — fusi PERINGKAT, bukan penjumlahan skor, karena "
     "skala cosine (0-1) vs BM25 (0-20+) tidak sebanding."],
    [("tokenisasi_bm25()", "teks -> tokens siap-BM25 (satu fungsi dipakai "
      "builder & query)."),
     ("build_lexical_index()", "baca ids.jsonl + chunks.jsonl -> tulis "
      "norma_tokens.txt (63.231 baris, avgdl ~42 token)."),
     ("BM25Scorer", "muat index + idf + postings; .search(q, k) -> "
      "[(baris_index, skor_BM25)] urut turun."),
     ("fusi_rrf() / HybridRetriever", "di retriever.py: dua jalur difusi "
      "RRF lalu dedup per pasal utuh (kontribusi rank-1 kedua jalur "
      "KEHILANGAN bukan); fallback dense-only bila index hilang.")],
    ["Pengaruh nyata: pertanyaan kedaulatan — Pasal 1 UUD yang oleh cosine "
     "dilewatkan (rank > 10) naik ke #1 berkat skor BM25-nya 31,3.",
     "Matikan via --no-hybrid (CLI), HYBRID=0 (.env), atau hybrid=false "
     "pada tool search_norma; skor_fusi & skor_bm25 tersedia utk audit."],
    "python bm25.py --build   # sekali, offline")

# ------------------------------------------------------------- BAB 5b
ISI += bab_file("5b", "reranker.py",
    "RERANKER CROSS-ENCODER Qwen3-Reranker-0.6B — tahap 2 pencarian. Fase 1 "
    "(retriever.py) ambil 12 kandidat murah via cosine; fase 2 model ini "
    "membaca query+pasal BERSAMAAN per pasangan dan mengurut ulang. Cross-"
    "encoder jauh lebih tajam (ada interaksi kata-per-kata) tapi mahal — "
    "karena itu hanya untuk kandidat teratas.",
    ["Skor = P(yes): prompt dibingkai yes/no (system judge + <Instruct>/"
     "<Query>/<Document> + footer shed thought/answer), lalu softmax dua "
     "token 'yes' vs 'no' pada posisi terakhir.",
     "Model LLM biasa dipakai sebagai penilai: AutoModelForCausalLM + logits "
     "token — bukan classifier terpisah.",
     "BF16 -> .float() sebelum dibaca (jebakan yang sama dgn embedder.py); "
     "LEFT padding wajib; truncation 1024 token per pasangan.",
     "Fallback aman: model gagal termuat -> kandidat kembali dgn urutan "
     "cosine, alur tidak pernah gagal keras.",
     "Kontrol: default aktif; --no-rerank (CLI), RERANK=0 (.env), atau "
     "rerank=false pada tool search_norma. score_rerank tersimpan terpisah "
     "dari score cosine utk audit."],
    [("muat_reranker()", "singleton lazy, OFFLINE dari cache HuggingFace "
      "(local_files_only=True)."),
     ("skor_rerank()", "P(yes) per pasangan, per batch (default 4 — CPU "
      "friendly); urutan keluaran = urutan masukan."),
     ("rerank_kandidat()", "tambahkan score_rerank ke tiap kandidat, sort "
      "turun, ambil top_n.")],
    ["CPU scoring ~1 menit per batch 4 pasangan — normal, bukan hang; "
     "progress per batch dicetak.",
     "Model harus ada di cache HF: Qwen/Qwen3-Reranker-0.6B (sudah diunduh)."],
    "python verify_setup.py --full   # cek reranker dgn pasangan relevan vs tidak")

# ------------------------------------------------------------- BAB 6
ISI += bab_file("6", "prompt.py",
    "Anatomi prompt RAG hukum: blok SISTEM (aturan keras anti-halusinasi), "
    "blok KONTEKS ([S1]..[Sn] ber-sitasi + anggaran karakter), blok "
    "PERTANYAAN. Format sitasi [S1] menaut jawaban ke daftar sumber.",
    ["Aturan sistem: jawab HANYA dari konteks; sitasi wajib; bila tidak ada "
     "dasar jawab 'Tidak ditemukan dasar dalam korpus.'",
     "Anggaran karakter (12.000): kandidat berikutnya dipotong bila lewat — "
     "kualitas di atas kuantitas.",
     "Konteks self-contained: tiap pasal dibawa dengan judul resmi + "
     "halaman/baris dari provenance."],
    [("SYSTEM_PROMPT", "teks aturan keras — salinan idenya dari panduan bab 3."),
     ("build_context_block()", "susun [S1]..[Sn] dari hasil konteks_kaya() "
      "dengan anggaran char."),
     ("build_messages()", "bentuk list messages OpenAI-compatible (system+user)."),
     ("daftar_sumber()", "ringkasan sumber utk ditampilkan di bawah jawaban.")],
    ["Jangan longgarkan aturan 'hanya dari konteks' — itu satu-satunya "
     "penghalang halusinasi."],
    "python prompt.py   # demo konteks dengan kandidat palsu")

# ------------------------------------------------------------- BAB 7
ISI += bab_file("7", "llm_client.py",
    "Klien LLM 'telanjang' dengan requests SAJA — anatomi satu POST HTTP ke "
    "endpoint OpenAI-compatible terlihat utuh (dipahami juga oleh Ollama, "
    "OpenRouter, Groq, DeepSeek). Termasuk mode dry-run dan MockLLM terjadwal.",
    ["messages: system -> user -> (assistant.tool_calls -> tool) — inilah "
     "seluruh rahasia tool calling.",
     "ARGUMEN tool dari LLM datang sebagai STRING JSON; hasil tool balik "
     "dengan role='tool' + tool_call_id yang cocok.",
     "MockLLM berantarmuka sama dengan chat() — agent.py tidak tahu "
     "bedanya; itulah cara menguji loop tanpa biaya API."],
    [("chat()", "POST /chat/completions; payload model/messages/temperature/" "max_tokens (+ tools bila ada); error HTTP dilempar dengan pesan "
      "perbaikan konkret."),
     ("dry_run()", "cetak payload yang AKAN dikirim — belajar prompt tanpa jaringan."),
     ("MockLLM", "LLM palsu terjadwal: tiap putaran berupa tool_calls atau "
      "jawaban final; arguments diserialisasi jadi string JSON persis API asli.")],
    ["401/404 -> cek base_url/model untuk provider itu (tiap provider punya "
     "nama model berbeda)."],
    "python llm_client.py   # demo dry-run + mock")

# ------------------------------------------------------------- BAB 8
ISI += bab_file("8", "answer.py",
    "Arsitektur A lengkap dalam satu file CLI: pertanyaan -> vektor -> "
    "top-k -> pasal utuh -> prompt -> LLM -> jawaban + sitasi. Tiap tahap "
    "bisa diperiksa lewat --dry-run / --show-context.",
    ["Alur tetap (deterministik): bagus utk audit; LLM tidak bisa memutuskan "
     "mencari ulang (itu tugas agent.py / folder LangGraph).",
     "--dry-run: seluruh pipeline jalan KECUALI panggilan LLM — payload "
     "dicetak; cocok sebelum membayar API.",
     "Filter --doc-type UU --year 2017 menegakkan guardrail filter panduan."],
    [("jawab()", "fungsi inti 6 tahap; mengembalikan dict jawaban+kandidat+" "messages+mode (live/dry-run/kosong)."),
     ("main()", "argparse CLI; mencetak jawaban lalu daftar sumber.")],
    ["Jawaban 'Tidak ditemukan...' bukan bug — aturan sistem bekerja ketika "
     "korpus memang tidak memuat pasalnya (contoh eval: APBN 20% Pasal 31(4))."],
    "python answer.py -q \"Di tangan siapa kedaulatan menurut UUD?\" --dry-run")

# ------------------------------------------------------------- BAB 9
ISI += bab_file("9", "tools.py",
    "Empat tool sempit bertanggung jawab (panduan bab 6): skema JSON OpenAI "
    "+ fungsi eksekusi asli + dispatcher tervalidasi. LLM hanya MEMINTA; "
    "kode inilah yang menyentuh data.",
    ["Deskripsi tool = prompt bagi LLM: tulis kapan dipakai & apa hasilnya.",
     "CLAMP argumen (top_k 1-10): LLM bisa minta 999 — jangan percaya.",
     "teks dipotong 600 char pada search_norma — hemat token; get_article "
     "yang membawa teks utuh.",
     "Error selalu dibungkus JSON {error:...} — LLM bisa menyesuaikan, "
     "sesi tidak crash."],
    [("TOOLS", "daftar 4 skema: search_norma, get_article, get_document, "
      "check_validity (status hukum tidak pernah ditebak)."),
     ("_search_norma()", "embed + exact search + konteks_kaya — memanggil "
      "aset yang sudah terbukti."),
     ("execute_tool()", "dispatcher: json.loads argumen, panggil fungsi, "
      "serialize hasil; TypeError -> pesan argumen tidak valid.")],
    ["Enum doc_type di skema bukan hiasan — nilai di luar daftar ditolak "
     "provider yang menegakkan skema; dispatcher tetap memvalidasi."],
    "python tools.py   # uji dispatcher tanpa LLM")

# ------------------------------------------------------------- BAB 10
ISI += bab_file("10", "agent.py",
    "Arsitektur B: loop tool-calling ditulis pakai while TELANJANG — tidak "
    "ada sihir. Setiap putaran dicetak (ikon) supaya keputusan LLM diaudit.",
    ["Langkah inti: (1) kirim messages+tools (2) tool_calls? -> eksekusi "
     "tiap call via tools.execute_tool (3) append assistant+tool messages "
     "(4) ulang; tanpa tool_calls = jawaban final.",
     "Pesan assistant yang meminta tool WAJIB ikut disimpan (tool berikutnya "
     "menjawab id-nya) — normalisasi 3 kunci inti.",
     "Guardrail: MAX_ITERATIONS=5 + PesanSistem 'gunakan tools sebelum menjawab'."],
    [("jalankan()", "satu pertanyaan -> jawaban final; trace dicetak."),
     ("main()", "CLI; --dry-run memakai MockLLM: tool ASLI dijalankan "
      "sungguhan, hanya LLM-nya palsu.")],
    ["Lupa append pesan assistant sebelum pesan tool -> provider menolak "
     "(tool_call_id tidak berpasangan)."],
    "python agent.py -q \"kedaulatan di tangan siapa\" --dry-run")

# ------------------------------------------------------------- BAB 11
ISI += bab_file("11", "verify_setup.py",
    "Fail-fast ala proyek: 11 cek kesehatan (file, bentuk matriks, join "
    "parent, lookup dokumen) + opsi --full (embedder), --llm (ping LLM), "
    "--milvus (ping server). Exit code 1 bila ada yang gagal.",
    ["Cek konsistensi: bentuk npy == meta (count,dim) == jumlah ids.jsonl.",
     "Join parent diuji dengan data nyata: uud-1945::p::1 = 199 char."],
    [("main()", "kumpulkan hasil cek (nama, ok, detail), cetak ringkas, "
      "exit code sesuai.")],
    ["--llm butuh .env terisi; tanpa itu pakai --dry-run di skrip lain."],
    "python verify_setup.py --full")

# ------------------------------------------------------------- BAB 12
ISI += E.bab("12", "Peta belajar, glosarium, dan kredit")
ISI += E.sub("Urutan belajar yang disarankan")
ISI += E.bullets([
    "config -> dataset -> embedder -> retriever : fondasi data & vektor",
    "prompt -> llm_client                       : anatomi prompt & HTTP",
    "answer (A) -> tools -> agent (B)        : dua arsitektur",
    "verify_setup                              : jaga kesehatan",
])
ISI += E.sub("Glosarium singkat")
ISI += E.bullets([
    "Chunk = unit norma kecil (ayat) yang di-embed. Parent = pasal utuh. "
    "Window = potongan <=1.800 char utk pasal panjang (overlap 150).",
    "Small-to-big = cari chunk kecil, suntikkan pasal utuh ke LLM.",
    "Provenance = jejak asal teks (halaman & baris PDF asli).",
    "IP (inner product) = dot product; pada vektor L2-normal setara cosine.",
    "Tool calling = LLM meminta pemanggilan fungsi; kode yang mengeksekusi.",
    "HNSW/ef = indeks ANN di Milvus & ukuran kandidat pencariannya.",
])
ISI += E.sub("Kredit korpus")
ISI += E.para("Korpus: Indonesian Legislation as Data — 90,588 norm units, "
              "CC BY 4.0, Qausar, Prabowo, Arifin & Isa (Binus University), "
              "DOI 10.17632/8wpjvhh7g2.1. Angka resmi build dari "
              "pipeline/result/manifest.json (v4.1, built 2026-10-04).")


# ---------------------------------------------------------------- RAKIT
if __name__ == "__main__":
    E.bangun_pdf(
        FOLDER / "code-guide-rag-vanilla.pdf",
        "Penjelasan Kode — RAG Tanpa Framework",
        "Panduan fungsi-per-fungsi untuk folder rag-vanilla/ — "
        "Arsitektur A (fixed-pipeline RAG) & Arsitektur B (agent tool-calling) "
        "atas korpus Indonesian Legislation RAG.",
        [
            "Proyek: Indonesian Legislation as Data — 90.588 unit norma, "
            "378 peraturan (1945-2024), CC BY 4.0 (Binus University).",
            "Retrieval aktif: Qwen3-Embedding-0.6B (dim 1024, last-token "
            "pooling) di atas embeddings.npy exact-cosine; Milvus opsional.",
            "Dependensi: numpy, requests, torch, transformers (semua sudah "
            "ada di proyek); reportlab hanya untuk PDF ini.",
            "PDF ini di-generate otomatis: tabel fungsi/class dibaca dari "
            "file .py via modul ast — jalankan ulang bila kode berubah.",
        ],
        ISI,
    )
