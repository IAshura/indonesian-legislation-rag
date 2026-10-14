"""
dataset.py — Membuka "database" proyek: file JSONL hasil pipeline build.
===========================================================================

LATAR BELAKANG
  Pipeline build (v4.1) sudah menghasilkan dataset berprovenance:
      chunks.jsonl           90.588 unit norma (ayat, blok admin, fragmen)
      parents.jsonl         11.413 pasal/blok utuh (gabungan child-nya)
      parents_windows.jsonl 14.689 window <=1.800 char untuk pasal panjang
      documents.jsonl          378 peraturan (judul resmi, tahun, axis)
  Semua berformat JSON Lines: satu objek JSON per baris — mudah dibaca
  streaming tanpa memuat seluruh file ke memori.

KONSEP KUNCI: SMALL-TO-BIG
  Retrieval mencari CHUNK kecil (ayat — presisi tinggi), tetapi yang
  dikirim ke LLM adalah PARENT-nya (pasal utuh) supaya LLM punya konteks
  hukum yang utuh. Jika pasal > 1.800 karakter, dipakai WINDOW-nya
  (irisan 150 char) agar tetap aman untuk batas token model.

PEMETAAN ID (perhatikan polanya!)
  document_id : uud-1945
  parent_id   : uud-1945::p::1            <- Pasal 1 utuh
  chunk_id    : uud-1945::p::1::02        <- ayat (2) di Pasal 1
  window_id   : uud-1945::p::1::w01       <- window pertama dari pasal panjang
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from functools import lru_cache
from typing import Any, Iterator

import config


# ---------------------------------------------------------------- UTILITAS
def load_jsonl(path) -> Iterator[dict]:
    """
    Generator pembaca JSONL: yield satu dict per baris.
    `yield` = fungsi jadi generator -> file dibaca baris demi baris,
    hemat memori untuk file besar (chunks.jsonl ~ ratusan ribu baris).
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@lru_cache(maxsize=1)
def dataset() -> "LegislationDataset":
    """
    Satu instance dataset dipakai bersama (cache).
    lru_cache(maxsize=1) = "singleton murah": pemanggilan kedua & seterusnya
    langsung mengembalikan objek yang sama tanpa memuat ulang file.
    """
    return LegislationDataset()


# ------------------------------------------------------------------ KELAS
class LegislationDataset:
    """
    Menampung 4 koleksi JSONL + indeks kamus (dict) agar lookup by-ID
    menjadi O(1). Semua dimuat lazy (saat pertama diakses) supaya skrip
    yang hanya butuh documents tidak membayar biaya memuat chunks.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, dict] | None = None
        self._parents: dict[str, dict] | None = None
        self._windows: dict[str, dict] | None = None
        self._documents: dict[str, dict] | None = None

    # -------------------------------------------------------- pemuatan lazy
    _MAP = {
        # nama attr internal : (konstanta path di config.py, kunci ID record)
        "_chunks": ("CHUNKS_JSONL", "chunk_id"),
        "_parents": ("PARENTS_JSONL", "parent_id"),
        "_windows": ("PARENTS_WINDOWS_JSONL", "window_id"),
        "_documents": ("DOCUMENTS_JSONL", "document_id"),
    }

    def _load(self, attr: str) -> dict[str, dict]:
        """Muat satu koleksi saat PERTAMA diakses (lazy) lalu indeks by-ID."""
        if getattr(self, attr) is None:
            config_name, key = self._MAP[attr]
            data: dict[str, dict] = {}
            for rec in load_jsonl(getattr(config, config_name)):
                data[rec[key]] = rec
            setattr(self, attr, data)
        return getattr(self, attr)

    @property
    def chunks(self) -> dict[str, dict]:
        """Kamus chunk_id -> record chunk (90.588 entri)."""
        return self._load("_chunks")

    @property
    def parents(self) -> dict[str, dict]:
        """Kamus parent_id -> record pasal/blok utuh (11.413 entri)."""
        return self._load("_parents")

    @property
    def windows(self) -> dict[str, dict]:
        """Kamus window_id -> record window <=1.800 char (14.689 entri)."""
        return self._load("_windows")

    @property
    def documents(self) -> dict[str, dict]:
        """Kamus document_id -> record peraturan (378 entri)."""
        return self._load("_documents")

    # ------------------------------------------------------------- lookup
    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunks.get(chunk_id)

    def get_document(self, document_id: str) -> dict | None:
        return self.documents.get(document_id)

    def context_for_parent(self, parent_id: str) -> dict:
        """
        Ambil TEKS KONTEKS terbaik untuk satu parent (pasal).

        Aturan baca konteks (sama dengan manifest.json / rag_guide):
          - jika parent <= 1.800 char  -> pakai parents.jsonl.text (pasal utuh)
          - jika lebih panjang         -> pakai window pertamanya
                                          (parents_windows.jsonl, id ...::w01)
        Kembalikan dict seragam: {id, sumber, citation_str, official_title,
        text, document_id, provenance?} — dipakai prompt.py untuk menyusun
        blok konteks ber-sitasi.
        """
        parent = self.parents.get(parent_id)
        if parent is None:  # parent hilang = data rusak; gagal-cepat, jangan diam
            raise KeyError(f"parent tidak ditemukan: {parent_id}")

        if parent["char_len"] <= 1800:
            return {
                "id": parent_id, "sumber": "parents",
                "citation_str": parent.get("citation_str", ""),
                "official_title": parent.get("official_title", ""),
                "document_id": parent["document_id"],
                "text": parent["text"],
                "char_len": parent["char_len"],
            }

        # Pasal panjang: window pertama (suffix ::w01) memuat awal pasal +
        # overlap; window berikutnya w02... tersedia bila butuh sisa teks.
        window_id = f"{parent_id}::w01"
        window = self.windows.get(window_id)
        if window is None:
            raise KeyError(f"window tidak ditemukan: {window_id}")
        return {
            "id": window_id, "sumber": "parents_windows",
            "citation_str": window.get("citation_str", ""),
            "official_title": window.get("official_title",
                                         parent.get("official_title", "")),
            "document_id": window["document_id"],
            "text": window["text"],
            "char_len": window["char_len"],
        }

    # --------------------------------------------------------- sitasi teks
    def citation_line(self, ctx: dict, page: int | None = None,
                      lines: tuple | None = None) -> str:
        """
        Susun satu baris sitasi HUMAN-READABLE dari field provenance dataset.
        Contoh: "[UUD — Bab I Pasal 1 Ayat (2) — hal. 1 baris 31-32]"
        Sitasi TIDAK pernah dikarang LLM; ia hanya mengutip baris ini.
        """
        judul = ctx.get("official_title") or \
            (self.get_document(ctx["document_id"]) or {}).get("official_title", "?")
        sitasi = ctx.get("citation_str") or "(tanpa sub-bagian)"
        pos = f" — hal. {page} baris {lines[0]}-{lines[1]}" if page else ""
        return f"[{judul} — {sitasi}{pos}]"


# ------------------------------------------------------ uji cepat mandiri
if __name__ == "__main__":
    # `python dataset.py` = demo mini: buka dataset & jalur small-to-big
    ds = dataset()
    print(f"chunks   : {len(ds.chunks):,}")
    print(f"parents  : {len(ds.parents):,}")
    print(f"windows  : {len(ds.windows):,}")
    print(f"documents: {len(ds.documents):,}")

    contoh = "uud-1945::p::1"
    ctx = ds.context_for_parent(contoh)
    print("\nKonteks untuk", contoh, "->", ctx["sumber"], ctx["char_len"], "char")
    print(ds.citation_line(ctx, page=1, lines=(31, 32)))
    print(ctx["text"][:200])
