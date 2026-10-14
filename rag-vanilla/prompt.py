"""
prompt.py — Menyusun PROMPT yang disuntikkan ke LLM.
====================================================

TIGA BLOK PROMPT (dari dokumen panduan, bab 3)
  1. SISTEM   : peran + aturan keras (anti-halusinasi, format sitasi).
  2. KONTEKS  : pasal-pasal hasil retrieval, bernomor [S1]..[Sn], tiap
                pasal dibawa LENGKAP dengan judul resmi + sitasi +
                provenance (halaman & baris) dari dataset.
  3. PERTANYAAN: pertanyaan asli pengguna.

MENGAPA STRUKTUR INI BEKERJA
  - "Jawab HANYA berdasarkan konteks" menutup celah halusinasi.
  - Sitasi = pengulangan field dataset (bukan karangan LLM) -> bisa
    diverifikasi ke PDF asli lewat provenance halaman/baris.
  - Anggaran karakter (CONTEXT_CHAR_BUDGET) menjaga total prompt tetap
    dalam batas token model meskipun kandidat banyak.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ------------------------------------------------------------- blok sistem
SYSTEM_PROMPT = """Kamu adalah asisten hukum Indonesia yang bekerja hanya dari \
teks peraturan yang diberikan.

Aturan keras:
1. Jawab HANYA berdasarkan konteks pasal yang diberikan.
2. Setiap pernyataan hukum WAJIB diakhiri sitasi [S1], [S2], ... \
PERSIS seperti itu (kurung siku + S + nomor), sesuai nomor konteks yang dipakai.
   BENAR: "Kedaulatan di tangan rakyat [S1]."
   SALAH (JANGAN PERNAH TULIS BEGITU): "Kedaulatan di tangan rakyat \
(UUD — Bab I Pasal 1 — hal. 1)." atau "(Perubahan Pertama — Pasal 7 — hal. 1)."
   DILARANG menulis sitasi dalam tanda kurung biasa — hanya [S1], [S2], dst.
   Satu kalimat boleh memakai beberapa sitasi: "... rakyat [S1] dan MPR \
menetapkan UUD [S2]."
3. Jika konteks tidak memuat jawaban, jawab persis: "Tidak ditemukan dasar \
dalam korpus." — JANGAN mengarang, JANGAN memakai pengetahuan luar.
4. Kutip teks pasal apa adanya; jangan mengubah angka, huruf, atau istilah.
5. Jawab dalam Bahasa Indonesia, ringkas dan terstruktur."""

# ------------------------------------------------- preset gaya jawaban
# Dipilih dari UI web (radio Ringkas/Standar/Rinci). Hanya menambah SATU
# kalimat instruksi — aturan keras sitasi di SYSTEM_PROMPT tidak berubah.
GAYA_PRESET = {
    "ringkas": "Jawab MAKSIMAL 3 kalimat pendek plus sitasi. Tanpa pengantar.",
    "standar": "Jawab ringkas dan terstruktur.",
    "rinci": "Jawab rinci dan terstruktur: kutip tiap pasal relevan "
             "apa adanya per [S-nomornya], lalu simpulkan di akhir.",
}


def instruksi_gaya(gaya: str | None) -> str:
    """Kembalikan kalimat instruksi gaya ('' bila tak dikenal)."""
    return GAYA_PRESET.get((gaya or "standar").strip().lower(), "")


# ----------------------------------------------------------- blok konteks


def build_context_block(kandidat: list[dict],
                         budget: int = config.CONTEXT_CHAR_BUDGET,
                         mulai: int = 1) -> str:
    """
    Ubah hasil retrieval (output konteks_kaya()) menjadi teks konteks:

      [S8] UUD — Bab I Pasal 1 Ayat (2) — hal. 1 baris 31-32 (skor 1.0)
      <teks pasal utuh>
      [S9] ...

    budget: batas total karakter. Jika kandidat berikutnya melebihi anggaran,
    berhenti (kandidat ber skor rendah dikorbankan — kualitas > kuantitas).
    mulai: nomor sitasi pertama (global per sesi chat — S1 yang sudah dipakai
    pesan sebelumnya tak boleh dipakai ulang).
    """
    blok: list[str] = []
    total = 0
    for i, k in enumerate(kandidat, mulai):     # i = mulai, mulai+1, ...
        judul = k.get("official_title") or "-"
        sitasi = k.get("citation_str") or "(tanpa sitasi)"
        posisi = ""
        if k.get("page"):
            ln = k.get("lines") or ("?", "?")
            posisi = f" — hal. {k['page']} baris {ln[0]}-{ln[1]}"
        # Tampilkan skor rerank (urutan final) bila ada; fallback cosine.
        skor = k.get("score_rerank", k["score"])
        header = f"[S{i}] {judul} — {sitasi}{posisi} (skor {skor})"
        panjang = len(header) + 1 + len(k["text"])
        if total + panjang > budget and blok:    # budget habis & sudah ada isi
            blok.append(f"[i] (kandidat berikutnya dipotong: anggaran konteks "
                        f"{budget} char tercapai)")
            break
        blok.append(f"{header}\n{k['text']}")
        total += panjang
    return "\n\n".join(blok)


def build_messages(question: str, context_block: str,
                   gaya: str | None = "standar") -> list[dict]:
    """
    Susun daftar messages lengkap (format OpenAI-compatible).
    Inilah "konteks yang disuntikkan" ke LLM secara harfiah: teks pasal
    diletakkan di pesan user, setelah aturan sistem.
    gaya: ringkas | standar | rinci (lihat GAYA_PRESET).
    """
    return [
        {"role": "system",
         "content": SYSTEM_PROMPT + "\n" + instruksi_gaya(gaya)},
        {"role": "user",
         "content": f"KONTEKS PASAL:\n{context_block}\n\n"
                    f"PERTANYAAN: {question}\n\n"
                    f"Ingat: setiap pernyataan hukum diakhiri [S1]/[S2]/... "
                    f"(kurung siku). Jangan tulis sitasi dalam kurung biasa."},
    ]


def daftar_sumber(kandidat: list[dict]) -> str:
    """Ringkasan sumber utk ditampilkan di bawah jawaban (auditabilitas)."""
    baris = []
    for i, k in enumerate(kandidat, 1):
        skor = k.get("score_rerank", k["score"])
        baris.append(f"  [S{i}] skor {skor:<6} {k['official_title']}"
                     f" — {k['citation_str'] or '-'} ({k['sumber']})")
    return "\n".join(baris)


if __name__ == "__main__":
    # Demo dengan kandidat palsu (tanpa retrieval) — lihat bentuk konteks
    k = [{"score": 0.99, "sumber": "parents", "official_title": "UUD",
          "citation_str": "Bab I Pasal 1 Ayat (2)", "page": 1,
          "lines": (31, 32),
          "text": "(2) Kedaulatan adalah di tangan rakyat, dan dilakukan "
                  "sepenuhnya oleh Majelis Permusyawaratan Rakyat."}]
    cb = build_context_block(k)
    print(cb, "\n")
    for m in build_messages("Di tangan siapa kedaulatan?", cb):
        print(f"[{m['role']}]\n{m['content'][:400]}\n")
