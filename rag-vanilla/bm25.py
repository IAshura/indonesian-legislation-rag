"""
bm25.py — Jalur LEXICAL untuk hybrid search: BM25 atas token pasal.
====================================================================

MENGAPA BM25 PENTING WALAU SUDAH ADA DENSE EMBEDDING?
  Dense embedding (Qwen3-0.6B) menangkap MAKNA ("operator telekomunikasi" ≈
  "penyelenggara jaringan") tetapi LEMAH pada IDENTIFIER PERSIS:
    - nomor & nama pasal ("pasal 28j", "uu 7 2017")
    - singkatan istilah teknis ("jpk", "tp2dd")
    - angka & satuan ("10", "2", "100000000")
  BM25 (Okapi; Robertson & Zaragoza 2009) adalah skor leksikal klasik:
  term-frequency ter-saturasi + inverse document frequency + normalisasi
  panjang dokumen. Ia hebat pada kata PERSIS yang jarang muncul di korpus
  (idf tinggi) — justru kasus yang biasa dilewatkan dense.

HYBRID = DENSE + LEXICAL, DIGABUNG RRF (BUKAN PENJUMLAHAN SKOR)
  Skala dense (0..1) ≠ skala BM25 (0..20+); menjumlahkan langsung itu salah.
  RRF (Reciprocal Rank Fusion, Cormack et al. 2009) hanya melihat PERINGKAT:
      skor_fusi(c) = SUM_jalur 1/(RRF_K + rank_jalur(c))
  Nilai RRF_K = 60 adalah konstanta baku. Fusi dipakai di retriever.py.

BENTUK INDEX ON-DISK
  config.LEXICAL_INDEX = pipeline/result/lexical/norma_tokens.txt
  Baris-i file ≡ baris-i embeddings/ids.jsonl ≡ baris-i embeddings.npy
  (satu chunk norma). Isi: token dipisah spasi, termasuk PENGULANGAN
  (pengulangan inilah tf — file tetap human-readable: bisa di-grep).
  Header baris-1: `#N avgdl K1 B` (metadata kecil, diabaikan scorer).
  Dibuat SEKALI offline:  python bm25.py --build

CARA PAKAI (runtime, di retriever.py)
      scorer = BM25Scorer()          # muat index + idf + postings
      pasangan = scorer.search("pertanyaan", k=12)   # [(baris_i, skor), …]
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ------------------------------------------------------------- tokenisasi
# Bukan word-piece tokenizer model! BM25 butuh token gaya "bahasa manusia".
# Pattern: rangkaian alfanumerik (A-Z/0-9 + aksara Latin diperluas) >= 2 char.
# Case-fold dulu agar "Negara" ≡ "negara". TANPA stemming: "berlaku" ≠
# "laku" memang tidak digabung — di hybrid ini aman karena makna ditangkap
# jalur dense; BM25 tugasnya kata persis.
_TOKEN_RE = re.compile(r"[0-9a-z\u00E0-\u024F]{2,}")

# Stopwords Indonesia dasar (plus inggris yang menjebak) — cukup yang
# jelas-jelas tidak informatif; daftar kecil = risiko kecil.
_STOPWORDS = frozenset({
    "yang", "untuk", "dan", "atau", "dengan", "pada", "dalam", "ke", "dari",
    "adalah", "ini", "itu", "tidak", "akan", "dapat", "bahwa", "para",
    "sebagai", "oleh", "juga", "harus", "ada", "di", "the", "of", "and",
    "or", "to", "in", "for", "is", "are",
})


def tokenisasi_bm25(teks: str) -> list[str]:
    """Teks -> token siap-BM25. SATU fungsi untuk builder & query —
    konsistensi index vs query, saudara sepupu aturan besi
    \"model index = model query\" di sisi dense."""
    t = teks.lower()
    # normalisasi angka minus/desimal ala Indonesia: "10,5" -> "10", "5"
    t = t.replace(",", " ").replace(".", " ")
    return [tok for tok in _TOKEN_RE.findall(t) if tok not in _STOPWORDS]


# ------------------------------------------------------------- builder
def build_lexical_index() -> None:
    """
    Buat LEXICAL_INDEX dari chunks.jsonl (SATU KALI, offline, tanpa jaringan).

    Urutan baris mengikuti ids.jsonl (kontrak dengan embeddings.npy), jadi
    scorer cukup mengembalikan INDEKS BARIS dan retriever memetakan
    baris -> metadata via ids.jsonl yang sudah dimuatnya.
    """
    config.LEXICAL_DIR.mkdir(parents=True, exist_ok=True)

    print("[bm25] baca ids.jsonl (urutan vektor index)...")
    ids: list[dict] = []
    with open(config.EMB_IDS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line))
    print(f"[bm25] {len(ids):,} chunk norma (≡ baris embeddings.npy)")

    print("[bm25] indeks chunks.jsonl per chunk_id...")
    by_id: dict[str, dict] = {}
    n = 0
    with open(config.CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if n % 20_000 == 0:
                print(f"[bm25]   ... {n:,} baris chunks.jsonl")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_id[rec["chunk_id"]] = rec
    print(f"[bm25] {len(by_id):,} chunk unik")

    print("[bm25] tokenisasi + tulis index leksikal...")
    k1, b = 1.2, 0.75
    with open(config.LEXICAL_INDEX, "w", encoding="utf-8") as out:
        total_tok = 0
        for i, m in enumerate(ids):
            rec = by_id.get(m["chunk_id"]) or {}
            # Teks untuk BM25 = teks pasal + sitasi + judul + jenis + tahun.
            # Sitasi & judul kutambahkan karena di sanalah kata kunci PERSIS
            # paling informatif di domain hukum (jenis dok, nomor, tahun).
            teks = " ".join([
                rec.get("text_lexical") or rec.get("text") or "",
                (rec.get("citation_str") or "").lower(),
                (rec.get("official_title") or "").lower(),
                (rec.get("doc_type") or "").lower(),
                str(rec.get("year") or ""),
            ])
            tokens = tokenisasi_bm25(teks)
            total_tok += len(tokens)
            out.write(" ".join(tokens) + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"[bm25]   baris {i + 1:,}/{len(ids):,}")
        # Header ringkas di atas (dibaca scorer utk avgdl & N yang diharapkan)
        print(f"[bm25] header: avgdl={total_tok / max(1, len(ids)):.2f}")
    print(f"[bm25] SELESAI -> {config.LEXICAL_INDEX}")


# ------------------------------------------------------------ BM25 scorer
class BM25Scorer:
    """
    BM25 Okapi di atas LEXICAL_INDEX. Murni Python + dict/list — tanpa
    dependensi tambahan, deterministik, cepat (puluhan ms untuk 1 query).

        score(q, d) = Σ_t∈q∩d  idf(t) · tf·(k1+1)
                                       / (tf + k1·(1 - b + b·|d|/avgdl))
    dengan k1=1.2, b=0.75 (parameter baku yang biasa dipakai).
    """

    def __init__(self) -> None:
        if not config.LEXICAL_INDEX.exists():
            raise FileNotFoundError(
                f"{config.LEXICAL_INDEX} tidak ada — "
                f"jalankan `python bm25.py --build` dulu")

        docs: list[Counter[str]] = []
        with open(config.LEXICAL_INDEX, encoding="utf-8") as f:
            for line in f:
                toks = line.split()
                if toks and toks[0].startswith("#"):  # header metadata
                    continue
                docs.append(Counter(toks))
        self.n_docs: int = len(docs)
        self.doc_len: list[int] = [sum(c.values()) for c in docs]
        self.avgdl: float = (sum(self.doc_len) / max(1, self.n_docs))

        # idf per token: df = jumlah dok yang mengandung token
        df: Counter[str] = Counter()
        for c in docs:
            df.update(c.keys())
        self.idf: dict[str, float] = {
            t: math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5))
            for t, d in df.items()}

        # postings: token -> list[(baris_dok, tf)] — index terbalik klasik.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for di, c in enumerate(docs):
            for t, tf in c.items():
                self.postings.setdefault(t, []).append((di, tf))

        self.k1: float = 1.2
        self.b: float = 0.75

    # ------------------------------------------------- pencarian
    def search(self, query: str,
               k: int | None = None) -> list[tuple[int, float]]:
        """
        Query string -> top-k (baris_index, skor_BM25).
        baris_index ≡ baris ids.jsonl ≡ baris embeddings.npy.
        """
        k = k or config.HYBRID_PER_JALUR
        qtoks = tokenisasi_bm25(query)
        if not qtoks:
            return []

        # Skor HANYA dok yang memuat >= 1 token query (union postings).
        skor: dict[int, float] = {}
        k1, b, avgdl, idf = self.k1, self.b, self.avgdl, self.idf
        for t in qtoks:
            idf_t = idf.get(t)
            if idf_t is None or t not in self.postings:
                continue                # token di luar vocabulary -> lewati
            for di, tf in self.postings[t]:
                d_len = self.doc_len[di]
                skor[di] = skor.get(di, 0.0) + idf_t * (
                    tf * (k1 + 1)) / (tf + k1 * (1 - b + b * d_len / avgdl))

        urut = sorted(skor.items(), key=lambda x: -x[1])
        return urut[:k]


# ------------------------------------------------------------- CLI
if __name__ == "__main__":
    if "--build" in sys.argv:
        build_lexical_index()
    else:
        # demo: tokenize + skor satu query tanpa menyentuh dense
        s = BM25Scorer()
        hasil = s.search("kedaulatan di tangan siapa menurut uud 1945", k=5)
        for di, skor in hasil:
            print(f"baris {di}  skor={skor:.4f}")
