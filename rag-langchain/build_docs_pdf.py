"""
build_docs_pdf.py — Membangun `code-guide-rag-langchain.pdf`
================================================================================

File ini adalah ISI; cara mencetaknya ada di pdf_engine.py (sama dengan
folder rag-vanilla). Setiap bab = satu file kode:
  1) tujuan  2) konsep kunci  3) tabel entitas (ast)
  4) penjelasan entitas  5) jebakan  6) cara mencoba

Jalankan:  python build_docs_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pdf_engine as E

FOLDER = Path(__file__).resolve().parent


def bab_file(nomor: str, nama: str, tujuan: str, konsep: list[str],
             catatan: list[tuple[str, str]], jebakan: list[str],
             contoh: str) -> list:
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
ISI += E.bab("1", "Peta folder & kapan naik ke LangChain/LangGraph")
ISI += E.para("Folder ini membangun THE SAME dua arsitektur seperti folder "
              "`rag-vanilla/` (fixed-pipeline RAG dan agent tool-"
              "calling), tetapi 'lem'-nya diserahkan ke ekosistem LangChain "
              "(LCEL) dan LangGraph (state machine). Dataset, retrieval, dan "
              "aturan besi TIDAK berubah — aset proyek yang sudah terukur "
              "dibungkus sebagai adaptor, bukan dibangun ulang.")
ISI += E.sub("Pemetaan file: tanpa framework -> folder ini")
ISI += E.kode(
    "dataset.py          -> loader.py            (record -> Document)\n"
    "embedder+retriever  -> retriever.py        (BaseRetriever custom)\n"
    "prompt.py           -> rag_chain.format_docs + ChatPromptTemplate\n"
    "llm_client.py       -> llm.py              (ChatOpenAI/ChatOllama)\n"
    "answer.py           -> rag_chain.py        (LCEL: pipeline A)\n"
    "tools.py            -> tools_lc.py         (dekorator @tool)\n"
    "agent.py            -> agent_graph.py  (StateGraph + ToolNode)\n"
    "   (baru di sini)   -> reflective_graph.py (grade/rewrite/generate)")
ISI += E.sub("Kapan framework layak dipakai?")
ISI += E.bullets([
    "Butuh perilaku agentic yang bentuk graph-nya kompleks (cabang, loop, "
    "refleksi) — LangGraph menggambarkan & menjaganya.",
    "Butuh fitur gratis dari graph: checkpointing, streaming, "
    "human-in-the-loop, replay.",
    "SEBELUM itu: folder tanpa-framework lebih mengajarkan — abstraksi "
    "menyembunyikan persis mekanisme yang ingin dikuasai (panduan bab 4).",
])
ISI += E.sub("Versi library tempat kode ini diuji")
ISI += E.para("langchain 1.4.3, langchain-core 1.6.7, langgraph 1.2.14 "
              "(Python 3.12). API LangChain bergerak cepat: bila import "
              "berubah, `python verify_setup.py` mencetak versi yang dipakai.")
ISI += [E.PageBreak()]

# ------------------------------------------------------------- BAB 2
ISI += bab_file("2", "config.py",
    "Mirror dari config.py folder tanpa-framework: path dataset + konstanta "
    "retrieval + konfigurasi LLM via .env. Bedanya, file ini TIDAK membangun "
    "klien HTTP — ia hanya menyuplai parameter untuk llm.py.",
    ["LLM_BACKEND env: 'api' (ChatOpenAI-compatible) atau 'ollama' "
     "(ChatOllama native) — dua jalur abstraksi BaseChatModel.",
     "Path & konstanta identik folder A agar hasil kedua folder mudah "
     "dibandingkan berdampingan."],
    [("get_llm_config()", "kembalikan dict backend/base_url/api_key/model/"
      "temperature/max_tokens dari environment + .env.")],
    ["Salin .env.example -> .env sebelum jalankan mode live."],
    "python -c \"import config; print(config.get_llm_config())\"")

# ------------------------------------------------------------- BAB 3
ISI += bab_file("3", "loader.py",
    "Jembatan dataset -> objek Document LangChain (page_content + metadata). "
    "LangChain hanya mengenal bentuk sederhana itu; kekayaan proyek "
    "(provenance, sitasi, peran norma) hidup di metadata.",
    ["Document = teks + kamus metadata bebas — kontrak universal LangChain.",
     "SMALL-TO-BIG tetap ditegakkan: page_content = pasal utuh / window; "
     "metadata membawa identitas chunk asal + skor + sitasi + halaman.",
     "Mengapa bukan InMemoryVectorStore/FAISS? Embeddings 63.231 vektor "
     "SUDAH ada di embeddings.npy — vector store akan me-re-embed semua; "
     "jalan benar: custom retriever (bab 4)."],
    [("load_jsonl()", "baca JSONL jadi list[dict]."),
     ("dataset()", "singleton DatasetRingkas: 4 koleksi diindeks by-ID, lazy."),
     ("doc_konteks_dari_chunk()", "chunk+skor -> Document pasal utuh/window "
      "dengan metadata lengkap (jantung small-to-big)."),
     ("doc_pasal() / doc_peraturan()", "Document utk tool get_article dan "
      "get_document.")],
    ["Metadata yang tidak lengkap = sitasi hilang di prompt — jangan buang "
     "field provenance."],
    "python loader.py   # lihat Document yang dihasilkan")

# ------------------------------------------------------------- BAB 4
ISI += bab_file("4", "retriever.py",
    "Custom RETRIEVER: subclass BaseRetriever yang membungkus embedder "
    "Qwen3 + exact cosine search menjadi komponen LangChain standar "
    "(query string -> list[Document]).",
    ["Kontrak wajib: implement _get_relevant_documents(query, run_manager).",
     "Field pydantic (k: int) untuk data kecil; objek berat (matriks, model) "
     "via PrivateAttr — pola resmi LangChain.",
     "Aturan besi tetap: QwenQueryEmbedder memakai meta.json (format "
     "Instruct, last-token pooling, L2) — sama persis dengan index.",
     "Dedup per parent: satu pasal cukup sekali di konteks."],
    [("QwenQueryEmbedder", "muat model sekali (local_files_only); embed() "
      "mengikuti meta.json."),
     ("LegislationRetriever", "BaseRetriever; k field bisa diatur per-chain "
      "(LegislationRetriever(k=5))."),
     ("_ExactSearch", "matriks mmap + filter + dedup parent -> Document.")],
    ["Retriever baru dimuat ulang tiap proses — model embedding butuh "
     "beberapa detik; muat sekali, panggil berkali-kali."],
    "python retriever.py   # 3 Document untuk satu pertanyaan")

# ------------------------------------------------------------- BAB 4a
ISI += bab_file("4a", "bm25.py",
    "HYBRID SEARCH — jalur LEXICAL (BM25). Dense kuat di makna, BM25 kuat "
    "di identifier persis (nomor pasal, tahun, jenis dok, singkatan). "
    "Index dibuat sekali offline (python bm25.py --build) lalu dibaca "
    "BM25Scorer saat runtime; fusi RRF dilakukan di LegislationRetriever.",
    ["Baris-i index leksikal = baris-i ids.jsonl = baris-i embeddings.npy "
     "(kontrak urutan; scorer cuma balik indeks baris).",
     "RRF di retriever.py: skor(c) = sum 1/(60 + rank_jalur(c)); fusi "
     "peringkat MURNI — dedup per pasal SETELAH fusi sehingga kontribusi "
     "rank-1 kedua jalur tidak pernah hilang.",
     "Fallback aman: index belum dibangun / HYBRID=0 / hybrid=false pada "
     "tool -> dense-only, alur tidak pernah gagal keras.",
     "Skor audit: metadata skor_fusi (RRF), skor_bm25, score (cosine), "
     "lalu score_rerank bila reranker jalan."],
    [("tokenisasi_bm25()", "satu fungsi utk builder & query (token index = "
      "token query)."),
     ("build_lexical_index()", "tulis norma_tokens.txt (63.231 baris, "
      "avgdl ~42 token) dari chunks.jsonl + ids.jsonl."),
     ("BM25Scorer", "postings + idf; .search(query, k) -> [(baris, skor)].")],
    ["Efek nyata: Pasal 1 UUD (dilewatkan cosine) naik ke #1 utk pertanyaan "
     "kedaulatan berkat skor BM25 31,3."],
    "python bm25.py --build   # sekali, offline")

# ------------------------------------------------------------- BAB 4b
ISI += bab_file("4b", "reranker.py",
    "RERANKER CROSS-ENCODER Qwen3-Reranker-0.6B sebagai STEP LCEL. Fase 1 "
    "(retriever) ambil 12 kandidat via cosine; fase 2 QwenReranker membaca "
    "query+pasal BERSAMAAN per pasangan dan mengurut ulang. Di LangChain ia "
    "tidak perlu jadi retriever baru: callable list[Document] -> "
    "list[Document] bisa diselipkan di mana saja dalam rantai '|'.",
    ["Skor = P(yes) dari logits token 'yes' vs 'no' posisi terakhir "
     "(template resmi: system judge + <Instruct>/<Query>/<Document> + "
     "footer shed thought/answer).",
     "BF16 -> .float() sebelum dibaca; LEFT padding wajib; truncation per "
     "pasangan; singleton lazy OFFLINE (local_files_only=True).",
     "Fallback aman: model gagal termuat -> docs kembali urutan cosine.",
     "Kontrol: default aktif; --no-rerank (rag_chain), RERANK=0 (.env), "
     "rerank=false pada tool search_norma; node retrieve graph reflektif "
     "ikut mererank bila config.RERANK.",
     "metadata['score_rerank'] tersimpan di samping metadata['score'] "
     "(cosine) untuk audit; format_docs menampilkan score_rerank bila ada."],
    [("muat_reranker()", "muat sekali; None bila cache tidak ada."),
     ("skor_rerank()", "P(yes) per teks, batch 4 (CPU friendly), "
      "progress per batch dicetak."),
     ("QwenReranker", "kelas callable utk LCEL: .rerank(query, docs) -> "
      "docs urut baru + metadata skor; top_n opsional.")],
    ["CPU scoring ~1 menit per batch — normal, bukan hang.",
     "Model harus ada di cache HF: Qwen/Qwen3-Reranker-0.6B (sudah diunduh)."],
    "python verify_setup.py --full   # cek reranker dgn pasangan relevan vs tidak")

# ------------------------------------------------------------- BAB 5
ISI += bab_file("5", "llm.py",
    "Pabrik model chat: get_llm() mengembalikan BaseChatModel sesuai .env "
    "(ChatOpenAI-compatible / ChatOllama / mock dry-run). Ganti backend "
    "tanpa menyentuh kode lain — inilah manfaat abstraksi framework.",
    ["MockAgentLLM = BaseChatModel mini terjadwal yang bisa mengeluarkan "
     "AIMessage dengan tool_calls — memungkinkan UNIT-TEST loop agent "
     "tanpa API (FakeListChatModel bawaan hanya bisa teks).",
     "bind_tools() milik agent di-mock mengembalikan diri sendiri."],
    [("get_llm()", "pilih backend; dry=True -> FakeListChatModel (teks)."),
     ("MockAgentLLM", "_generate() pop skrip: str = jawaban final, dict = "
      "tool_calls [(nama, args)]; format tool_calls standar LangChain "
      "{name, args, id, type}."),
     ("_llm_type / _generate", "dua kontrak wajib BaseChatModel.")],
    ["Respons model asli kadang membawa field ekstra — graph prebuilt "
     "terima; bila memakai provider lain dan error, kurangi ke field inti."],
    "python llm.py   # uji dua mock tanpa jaringan")

# ------------------------------------------------------------- BAB 6
ISI += bab_file("6", "rag_chain.py",
    "Arsitektur A gaya LCEL: RunnableParallel({konteks, pertanyaan}) -> "
    "ChatPromptTemplate -> llm -> StrOutputParser, dirangkai dengan tanda "
    "pipe. Kerangka aliran data ditangani LangChain; yang tersisa hanya "
    "logika domain (format konteks ber-sitasi).",
    ["LCEL: setiap komponen adalah Runnable; `|` menyusunnya menjadi "
     "Runnable baru dengan .invoke()/.stream()/.batch() seragam.",
     "RunnableParallel menjalankan dua cabang dari input sama: konteks "
     "(retriever | format_docs) dan pertanyaan (passthrough).",
     "format_docs() menyusun [S1]..[Sn] + anggaran char — identik dengan "
     "prompt.py folder tanpa-framework: kualitas ada di DATA, bukan framework."],
    [("format_docs()", "list[Document] -> blok konteks bernomor; metadata "
      "Document dipakai untuk header sitasi."),
     ("daftar_sumber()", "ringkasan sumber utk audit di bawah jawaban."),
     ("bangun_chain()", "rakiti chain kanonik; kembalikan (chain, retriever) "
      "agar sumber bisa ditampilkan terpisah."),
     ("jawab()", "satu pertanyaan -> jawaban; --dry-run memakai "
      "FakeListChatModel (retrieval tetap asli).")],
    ["FakeListChatModel jawabannya urut (skrip) — cocok utk chain, bukan utk "
     "agent; untuk agent pakai MockAgentLLM.",
     "Jangan panggil retriever dua kali per pertanyaan (hemat model load & "
     "latensi) — lihat jawab()."],
    "python rag_chain.py -q \"Di tangan siapa kedaulatan?\" --dry-run")

# ------------------------------------------------------------- BAB 7
ISI += bab_file("7", "tools_lc.py",
    "Empat tool gaya LangChain: fungsi Python + docstring + dekorator @tool "
    "— skema JSON dikeluarkan OTOMATIS dari type hints. Bandingkan dengan "
    "tools.py folder A yang menulis skema manual.",
    ["Docstring = prompt bagi LLM: jelas kapan dipakai & apa hasilnya.",
     "Hasil tool harus JSON-serializable (dict) — ToolNode mengirim balik "
     "sebagai ToolMessage.",
     "Retriever di-cache level modul: model embedding dimuat sekali per "
     "proses, bukan per pemanggilan tool.",
     "check_validity tetap tidak pernah menebak status hukum."],
    [("@tool", "dekorator pembuat Tool: .name, .description, "
      ".args_schema (pydantic) otomatis."),
     ("search_norma()", "invoke retriever + filter doc_type/year + clamp "
      "top_k + ringkas teks 600 char."),
     ("get_article / get_document / check_validity", "pembungkus loader + "
      "validity.jsonl, sama semangatnya dengan folder A."),
     ("TOOLS", "daftar 4 Tool — di-bind ke LLM dan diberikan ke ToolNode.")],
    ["Argumen salah dari LLM ditangkap ToolNode (error dikirim balik ke "
     "LLM sebagai ToolMessage) — loop tetap hidup."],
    "python tools_lc.py   # lihat skema otomatis + uji fungsi")

# ------------------------------------------------------------- BAB 8
ISI += bab_file("8", "agent_graph.py",
    "Arsitektur B sebagai state machine eksplisit: START -> agent -> "
    "(ada tool_calls?) -> tools -> agent ... -> END. Loop `while` folder A "
    "menjadi graph yang tergambar dan bisa diverifikasi.",
    ["MessagesState: state bawaan; setiap node mengembalikan pesan yang "
     "DI-APPEND (akumulasi otomatis).",
     "should_continue = conditional edge — persis langkah 2 loop manual.",
     "ToolNode = node bawaan yang mengeksekusi tool & mengirim ToolMessage "
     "— padatan langkah 3-5 loop manual.",
     "Guardrail: recursion_limit (MAX_PUTARAN * 2 node) — analogi "
     "MAX_ITERATIONS folder A.",
     "Versi prebuilt satu baris juga ada: create_react_agent(llm, TOOLS) "
     "(lihat komentar kode)."],
    [("buat_node_agent()", "closure LLM ber-bind_tools; pencetak trace "
      "[agent] minta tool / jawab final."),
     ("buat_node_tools()", "pembungkus ToolNode + pencetak [tools]."),
     ("should_continue()", "tool_calls? -> 'tools' : END."),
     ("bangun_graph()", "StateGraph + 2 node + conditional edges + compile; "
      "dry-run memakai MockAgentLLM (tool ASLI jalan)."),
     ("jalankan()", "invoke graph dgn pesan awal (system+user).")],
    ["GraphRecursionError = loop tak selesai: periksa skenario atau naikkan "
     "recursion_limit.",
     "Pesan awal format tuple (role, teks) dikonversi otomatis — tidak perlu "
     "import SystemMessage manual."],
    "python agent_graph.py -q \"kedaulatan di tangan siapa\" --dry-run")

# ------------------------------------------------------------- BAB 9
ISI += bab_file("9", "reflective_graph.py",
    "RAG yang bisa MEMPERBAIKI DIRI: retrieve -> grade (LLM menilai "
    "relevan/tidak) -> generate, atau rewrite query lalu mencari ulang "
    "(maks 2 percobaan). Bentuk graph TETAP milik kita; LLM hanya mengisi "
    "keputusan di conditional edge — 'agentic' dengan kontrol terarah.",
    ["GraphState(TypedDict): pertanyaan, dokumen, jawaban, percobaan — "
     "skema data yang mengalir antar-node.",
     "Tugas grader dibuat SEMPIT (jawab YA/TIDAK) agar stabil & murah.",
     "generate memakai ulang format_docs + SYSTEM dari rag_chain.py — satu "
     "sumber kebenaran aturan prompt.",
     "Setelah MAX_PERCOBAAN rewrite, jawab dari konteks yang ada (jujur), "
     "bukan berhenti diam."],
    [("buat_node_retrieve()", "simpan list[Document] ke state."),
     ("buat_node_grade()", "grader prompt kecil -> 'YA'/'TIDAK' -> jawaban "
      "sementara 'RELEVAN'/'TIDAK'."),
     ("buat_node_rewrite()", "rumuskan ulang pertanyaan jadi kalimat "
      "pencarian beristilah hukum; counter percobaan++."),
     ("keputusan_setelah_grade/rewrite()", "dua conditional edges — logika "
      "cabang terbaca seperti kode biasa."),
     ("bangun_graph()", "4 node + edges; recursion_limit 16.")],
    ["Grade memakai pasal PERTAMA saja (skor tertinggi) — cukup sebagai "
     "sinyal; grader lebih rumit = biaya lebih besar."],
    "python reflective_graph.py -q \"istilah presiden satu periode\" --dry-run")

# ------------------------------------------------------------- BAB 10
ISI += bab_file("10", "verify_setup.py",
    "Cek kesehatan folder: versi library (via importlib.metadata — langgraph "
    "1.x tidak punya __version__), retriever.invoke dengan data nyata, skema "
    "@tool, plus --full (dry-run chain LCEL, compile graph agent & graph "
    "reflektif dengan mock) dan --llm (ping asli).",
    ["Dry-run full = seluruh graph diuji TANPA jaringan: retrieval asli, "
     "LLM mock — cara cepat yakin alur benar sebelum bayar API."],
    [("main()", "8 cek + opsi; exit code 1 bila ada gagal.")],
    ["Versi API LangChain bergerak — jalankan dulu file ini setelah upgrade."],
    "python verify_setup.py --full")

# ------------------------------------------------------------- BAB 11
ISI += E.bab("11", "Latihan lanjutan & kredit")
ISI += E.sub("Latihan untuk menguatkan pemahaman")
ISI += E.bullets([
    "Tambah tool 'list_documents(doc_type, year)' lalu jalurkan agent untuk "
    "pertanyaan multi-dokumen.",
    "Tambahkan node 'rerank' di graph reflektif (skor cosine + heuristik "
    "citation match) — tanpa mengubah node lain.",
    "Simpan riwayat percakapan dengan MemorySaver (checkpointer LangGraph) "
    "dan bandingkan dengan folder tanpa-framework.",
    "Uji kedua folder dengan 12 golden_queries.jsonl proyek: sama tidak "
    "hasil retrievalnya? (harus — retrieval identik).",
])
ISI += E.sub("Kredit korpus")
ISI += E.para("Korpus: Indonesian Legislation as Data — 90,588 norm units, "
              "CC BY 4.0, Qausar, Prabowo, Arifin & Isa (Binus University), "
              "DOI 10.17632/8wpjvhh7g2.1. Angka resmi dari "
              "pipeline/result/manifest.json (v4.1, built 2026-10-04).")


# ---------------------------------------------------------------- RAKIT
if __name__ == "__main__":
    E.bangun_pdf(
        FOLDER / "code-guide-rag-langchain.pdf",
        "Penjelasan Kode — RAG LangChain & LangGraph",
        "Panduan fungsi-per-fungsi untuk folder rag-langchain/ — "
        "LCEL (Arsitektur A), agent tool-calling & RAG reflektif (LangGraph) "
        "atas korpus Indonesian Legislation RAG.",
        [
            "Proyek: Indonesian Legislation as Data — 90.588 unit norma, "
            "378 peraturan (1945-2024), CC BY 4.0 (Binus University).",
            "Retrieval: custom BaseRetriever di atas embeddings.npy "
            "(Qwen3-Embedding-0.6B, dim 1024) — bukan vector store baru.",
            "Diuji pada langchain 1.4.3 / langchain-core 1.6.7 / langgraph "
            "1.2.14 (Python 3.12).",
            "PDF di-generate otomatis (tabel entitas via ast) — jalankan "
            "ulang bila kode berubah.",
        ],
        ISI,
    )
