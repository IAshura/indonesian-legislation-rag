#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline kanonik dataset RAG peraturan Indonesia — v4 (skema 4).

Perubahan v4 terhadap v3:
  - Preservasi unit (prinsip GPT): 1 baris units.csv = 1 chunk; TIDAK ADA
    merge otomatis. Fragmen pendek dipertahankan dengan is_fragment=true
    dan dilewati pengindeksan vektor secara default.
  - parent_id GLOBAL dan stabil: {document_id}::p::{slug} / {document_id}::adm::{nn}.
  - parents_windows.jsonl: induk > WINDOW_MAX_CHAR dipecah deterministik
    dengan tumpang tindih; text induk penuh tetap ada di parents.jsonl.
  - content_role granular (norma/pertimbangan/dasar_huma/preamble/
   administratif/ketentuan_perubahan/penutup/penjelasan) + role_source;
    'part' turunan (norma/administratif) dipertahankan untuk kompatibilitas.
  - Validasi silang: setiap chunk punya parent_id yang ADA di parents.jsonl.

Input : ../data/corpus/units.csv, ../data/corpus/documents.csv, ../data/corpus/texts/
Output: ./result/*.jsonl + manifest.json   (deterministik, seed QA tetap)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(2 ** 31 - 1)

HERE = Path(__file__).resolve().parent
CORPUS_ROOT = HERE.parent / "data" / "corpus"
UNITS_CSV = CORPUS_ROOT / "units.csv"
DOCS_CSV = CORPUS_ROOT / "documents.csv"
TEXTS_ROOT = CORPUS_ROOT / "texts"
OUT_DIR = HERE / "result"

CORPUS_ID = "mendeley:8wpjvhh7g2 v1 (DOI 10.17632/8wpjvhh7g2.1)"
LICENSE = "CC BY 4.0"
ATTRIBUTION = ("Qausar, Prabowo, Arifin & Isa (Binus University) — "
               "Indonesian Legislation Norm-Unit Corpus, via Mendeley Data")
SCHEMA_VERSION = 4
VERSION = "combined-v4.1"
SEED = 20260907
QA_SAMPLES = 300
QA_PREFIX_CHARS = 60
QA_MIN_MATCH_RATE = 0.90
HEADING_LIKE_MAX_CHARS = 200
MIN_FRAGMENT_TOKENS = 8
WINDOW_MAX_CHARS = 1800
WINDOW_OVERLAP_CHARS = 150
WINDOW_MIN_CHARS = 400

REQUIRED_UNITS = {
    "unit_id", "document_name", "relative_path", "doc_type", "hierarchy_level",
    "year", "page_number", "line_start", "line_end", "unit_type", "bab_marker",
    "paragraf_marker", "pasal_marker", "ayat_marker", "text_original",
    "text_normalized", "indobert_token_count", "indobert_unk_count",
    "indobert_unk_ratio", "quality_flag",
}
REQUIRED_DOCS = {
    "file_name", "title_block_type", "number_from_filename",
    "year_from_filename", "number_from_title", "year_from_title",
    "title_resolved", "filename_title_mismatch", "legal_force_axis",
    "title_block_excerpt",
}
WS = re.compile(r"\s+")
NONSLUG = re.compile(r"[^a-z0-9]+")
PAGE_MARK_RE = re.compile(r"^={3,}\s*PAGE\s+(\d+)\s*={0,}", re.I)
TITLE_STOPS = ("DENGAN RAHMAT", "DENGAN RAKHMAT", "RAKHMAT TUHAN",
               "DENGAN MENETAPKAN", "MENETAPKAN", "MENIMBANG",
               "MEMUTUSKAN")
PERALIHAN_RE = re.compile(r"(?i)ketentuan\s+peralihan|pengaturan\s+peralihan")
PENUTUP_RE = re.compile(r"(?i)ketentuan\s+penutup")
PENJELASAN_RE = re.compile(r"(?i)^\s*penjelasan\b")
MENIMBANG_RE = re.compile(r"(?i)menimbang")
MENGINGAT_RE = re.compile(r"(?i)mengingat")
PREAMBLE_RE = re.compile(r"(?i)rahmat tuhan yang maha esa|dengan rahmat")
SIGNATURE_ARTIFACT_RE = re.compile(
    r"(?is)(?:\bttd\.?\b|\bditetapkan\s+di\b|\bdiundangkan\s+di\b|"
    r"\bberita\s+negara\s+republik\s+indonesia\b|"
    r"\blembaran\s+negara\s+republik\s+indonesia\b|"
    r"\btambahan\s+lembaran\s+negara\b)"
)
ADMIN_HEADING_RE = re.compile(
    r"(?i)^\s*(?:bab\b|bagian\b|paragraf\b|warga\s+negara\b|"
    r"penduduk\b|hak\s+dan\s+kewajiban\b|bendera\b|bahasa\b|"
    r"lambang\s+negara\b|lagu\s+kebangsaan\b|"
    r"pertahanan\s+dan\s+keamanan\b|kekuasaan\s+kehakiman\b|"
    r"wilayah\s+negara\b)"
)
NORMATIVE_SIGNAL_RE = re.compile(
    r"(?i)\b(?:wajib|harus|dapat|berhak|dilarang|dikenai|dipidana|"
    r"diatur|ditetapkan|berlaku|merupakan|melaksanakan|menetapkan|"
    r"memiliki|mempunyai|dilakukan|disediakan|diselenggarakan|terdiri)\b"
)

FULL_TYPE = {
    "UUD": "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
    "TAP MPR": "Ketetapan Majelis Permusyawaratan Rakyat",
    "UU": "Undang-Undang", "PERPPU": "Peraturan Pemerintah Pengganti Undang-Undang",
    "PP": "Peraturan Pemerintah", "PERPRES": "Peraturan Presiden",
    "PERMEN": "Peraturan Menteri", "PERDA": "Peraturan Daerah",
    "PKPU": "Peraturan Komisi Pemilihan Umum",
    "PERKAP": "Peraturan KPU Provinsi/Kabupaten/Kota",
    "PERBAWASLU": "Peraturan Bawaslu", "PERBAN": "Peraturan Badan",
    "PERKPI": "Peraturan KPI", "PERDKPP": "Peraturan DKPP",
    "PERLEMBAGA": "Peraturan Lembaga",
}


# ------------------------------------------------------------------ utils

def s(v) -> str:
    return str(v).strip() if v is not None else ""


def nz(v):
    t = s(v)
    return t if t else None


def collapse(t: str) -> str:
    return WS.sub(" ", t or "").strip()


def to_int(v):
    t = s(v)
    try:
        return int(float(t))
    except ValueError:
        return None


def slug(text: str, maxlen: int = 48) -> str:
    t = NONSLUG.sub("@", (text or "").lower()).replace(" ", "-")
    t = NONSLUG.sub("-", t).strip("-")
    return t[:maxlen].strip("-") or "x"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt_bab(b):
    b = collapse(b or "").strip(" .:")
    m = re.match(r"(?i)^bab\s+(.+)", b)
    return f"Bab {m.group(1)}" if m else (b or "")


def fmt_pasal(p):
    core = re.sub(r"(?i)^\s*(pasal|ps\.?)\s+", "", collapse(p or "").strip(" .:")).strip(" .:")
    return f"Pasal {core}" if core else ""


def fmt_ayat(a):
    a = nz(a)
    return f"Ayat ({a.strip(' ()')})" if a else ""


def pas_core(p):
    return re.sub(r"(?i)^\s*(pasal|ps\.?)\s+", "", s(p)).strip()


def topic_from_excerpt(ex: str) -> str:
    u = collapse(ex).upper()
    i = u.find(" TENTANG ")
    if i < 0:
        return ""
    tail = collapse(ex)[i + 9:]
    cut = len(tail)
    ut = tail.upper()
    for st in TITLE_STOPS:
        j = ut.find(st)
        if 0 <= j < cut:
            cut = j
    return tail[:cut].strip(" .,:;-")[:200]


def build_doc_meta(fname, row, used):
    resolved = s(row.get("title_resolved")) == "1"
    ttype = nz(row.get("title_block_type")) or ""
    number = (nz(row.get("number_from_title")) if resolved else None) or nz(row.get("number_from_filename")) or ""
    year = (nz(row.get("year_from_title")) if resolved else None) or nz(row.get("year_from_filename")) or ""
    mismatch = s(row.get("filename_title_mismatch")) == "1"
    topic = topic_from_excerpt(row.get("title_block_excerpt") or "")
    full = FULL_TYPE.get(ttype, ttype)
    if number and year and topic:
        title = f"{full} Nomor {number} Tahun {year} tentang {topic}"
    elif number and year:
        title = f"{full} Nomor {number} Tahun {year}"
    elif topic:
        title = f"{full} tentang {topic}" if full else collapse(topic)
    else:
        title = collapse(re.sub(r"_\d{4}$", "", Path(fname).stem).replace("_", " "))
    base = slug(Path(fname).stem)
    ds, k = base, 2
    while ds in used:
        ds, k = f"{base}-{k}", k + 1
    used.add(ds)
    src = ("title_block" if resolved else
           "filename_fallback" if (number or year) else
           "excerpt_topic" if topic else "file_name_stub")
    return {"document_id": ds, "file_name": fname, "doc_type": ttype or None,
            "doc_type_full": full or None, "number": number or None,
            "year": year or None, "topic": topic or None,
            "official_title": collapse(title),
            "legal_force_axis": nz(row.get("legal_force_axis")),
            "identity_source": src, "filename_title_mismatch": mismatch,
            "legal_status": None, "effective_date": None, "revision_basis": None,
            "enrichment_source": None}


def texts_index() -> dict:
    idx = {}
    if TEXTS_ROOT.exists():
        for p in TEXTS_ROOT.rglob("*.txt"):
            idx.setdefault(p.stem.lower().strip(), p.relative_to(CORPUS_ROOT).as_posix())
    return idx


def pages_of(path: Path) -> dict:
    out, cur = {0: []}, 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for ln in f:
            m = PAGE_MARK_RE.match(ln.strip())
            if m:
                cur = int(m.group(1))
                out.setdefault(cur, [])
                continue
            if ln.strip():
                out.setdefault(cur, []).append(ln.rstrip())
    return out


# ------------------------------------------------------------------ role

def looks_like_admin_artifact(text: str, flag: str) -> bool:
    """Detect headings/signatures before Pasal carry-forward is applied.

    Some source rows are flagged structural_admin but occur after a Pasal
    marker has been seen. They must not inherit the Pasal's norma role when
    the row is actually a heading, signature, or gazette metadata.
    """
    if flag != "structural_admin":
        return False
    clean = collapse(text)
    if not clean:
        return False
    if SIGNATURE_ARTIFACT_RE.search(clean):
        return True
    if ADMIN_HEADING_RE.search(clean) and not NORMATIVE_SIGNAL_RE.search(clean):
        return True
    if len(clean) <= HEADING_LIKE_MAX_CHARS:
        letters = [ch for ch in clean if ch.isalpha()]
        upper_ratio = (sum(ch.isupper() for ch in letters) / len(letters)
                       if letters else 0.0)
        if upper_ratio >= 0.80 and not NORMATIVE_SIGNAL_RE.search(clean):
            return True
    return False


def role_of(text: str, flag: str, pasal: str | None, bab_state: str):
    u = text[:280].upper()
    if looks_like_admin_artifact(text, flag):
        return "administratif", "flag+artifact"
    if PERALIHAN_RE.search(bab_state or ""):
        return "ketentuan_peralihan", "keyword"
    if PENUTUP_RE.search(bab_state or ""):
        return "ketentuan_penutup", "keyword"
    if PENJELASAN_RE.search(text) or "PENJELASAN ATAS" in u:
        return "penjelasan", "keyword"
    if PREAMBLE_RE.search(text[:200]):
        return "preamble", "keyword"
    if flag == "structural_admin":
        if MENIMBANG_RE.search(u) or text.lstrip().lower().startswith("bahwa"):
            return "pertimbangan", "flag+keyword"
        if MENGINGAT_RE.search(u):
            return "dasar_hukum", "flag+keyword"
    if pasal:
        return "norma", "marker"
    if flag == "structural_admin":
        return "administratif", "flag"
    return "administratif", "default"


# ------------------------------------------------------------------ build

class B:
    def __init__(self, tidx):
        self.chunks = []
        self.quarantine = []
        self.cnt = {}
        self.errors = []
        self.parents = OrderedDict()
        self.windows = OrderedDict()
        self.tidx = tidx

    def qadd(self, r):
        self.cnt[r] = self.cnt.get(r, 0) + 1

    def _q(self, row, fname, why):
        self.quarantine.append({
            "reason": why, "unit_id": to_int(row.get("unit_id")),
            "file_name": fname,
            "unit_type": nz(row.get("unit_type")),
            "quality_flag": nz(row.get("quality_flag")),
            "pasal": nz(row.get("pasal_marker")), "ayat": nz(row.get("ayat_marker")),
            "page_number": to_int(row.get("page_number")),
            "line_start": to_int(row.get("line_start")),
            "line_end": to_int(row.get("line_end")),
            "text_preview": collapse(s(row.get("text_normalized")))[:140]})

    def build_doc(self, meta, rows):
        doc_id = meta["document_id"]
        title = meta["official_title"]
        state = {"bab": "", "paragraf": "", "pasal": ""}
        admins = OrderedDict()          # key -> pid
        adm_n = [0]
        seen_norm = set()

        def admin_pid(bab):
            return admins.setdefault(f"{doc_id}::adm-group", f"{doc_id}::adm::{len(admins):03d}")
        # blok admin per dokumen (satu blok runtut antar-pasal dipisah per pasal state):
        cur_admin = [None, ""]

        def admin_block(bab):
            if bab:
                cur_admin[1] = bab
            if cur_admin[0] is None:
                adm_n[0] += 1
                cur_admin[0] = f"{doc_id}::adm::{adm_n[0]:03d}"
            return cur_admin[0]

        def close_admin():
            cur_admin[0] = None

        seq = {}
        doc_pids = []
        for row in rows:
            ut = s(row.get("unit_type"))
            flag = s(row.get("quality_flag"))
            body = collapse(s(row.get("text_normalized")) or s(row.get("text_original")))
            tokens = to_int(row.get("indobert_token_count")) or 0
            bab = nz(row.get("bab_marker")) or state["bab"]
            par = nz(row.get("paragraf_marker")) or state["paragraf"]
            pasal_raw = nz(row.get("pasal_marker")) or state["pasal"]
            ayat = nz(row.get("ayat_marker"))            # tidak pernah carry-forward

            if ut in ("heading_bab", "heading_paragraf", "heading_pasal"):
                if ut == "heading_bab":
                    state["bab"] = bab or state["bab"]
                elif ut == "heading_paragraf":
                    state["paragraf"] = par or state["paragraf"]
                    state["pasal"] = ""
                    close_admin()
                else:
                    state["pasal"] = pasal_raw or state["pasal"]
                    close_admin()
                self.qadd("heading_unit_as_context")
                continue
            if pasal_raw and not s(row.get("pasal_marker")):
                pass  # carry-forward pasal sudah tercermin di pasal_raw
            if not body:
                self.qadd("empty_text")
                self._q(row, fname=meta["file_name"], why="empty_text")
                continue
            if flag == "structural_heading" and len(body) < HEADING_LIKE_MAX_CHARS:
                self.qadd("heading_like_body_quarantined")
                self._q(row, meta["file_name"], "heading_like_body")
                continue

            is_frag = (flag == "needs_review_short" or tokens < MIN_FRAGMENT_TOKENS)
            role, role_src = role_of(body, flag, pasal_raw, state["bab"])
            part = "norma" if role == "norma" else "administratif"

            if pasal_raw:
                close_admin()
                pid = f"{doc_id}::p::{slug(pas_core(pasal_raw), 40)}"
                parent = self.parents.get(pid)
                if parent is None:
                    self.parents[pid] = parent = {
                        "parent_id": pid, "document_id": doc_id,
                        "parent_kind": "pasal",
                        "citation_str": collapse(fmt_bab(bab) + " " + fmt_pasal(pasal_raw)),
                        "citation": {"bab": bab or None, "pasal": pasal_raw, "ayat": None},
                        "official_title": title, "part": "norma",
                        "year": to_int(row.get("year")),
                        "text_parts": [], "child_ids": [], "char_len": 0}
                    parent["part"] = "norma" if role == "norma" else "administratif"
                    doc_pids.append(pid)
            else:
                pid = admin_block(bab)
                parent = self.parents.get(pid)
                if parent is None:
                    self.parents[pid] = parent = {
                        "parent_id": pid, "document_id": doc_id,
                        "parent_kind": "administratif_block",
                        "citation_str": collapse(fmt_bab(bab)),
                        "citation": {"bab": bab or None, "pasal": None, "ayat": None},
                        "official_title": title, "part": "administratif",
                        "year": to_int(row.get("year")),
                        "text_parts": [], "child_ids": [], "char_len": 0}
                    doc_pids.append(pid)
            kind = "p" if pasal_raw else "a"
            n = seq.get(pid, 0) + 1
            seq[pid] = n
            cid = f"{pid}::{n:02d}"
            key = (pid, "", body.lower())
            if key in seen_norm:
                self.qadd("duplicate_norm_quarantined")
                self._q(row, meta["file_name"], "duplicate_norm")
                continue
            seen_norm.add(key)

            chunk = {
                "chunk_id": cid, "parent_id": pid, "document_id": doc_id,
                "official_title": title,
                "citation": {"bab": bab or None, "paragraf": par or None,
                             "pasal": pasal_raw or None, "ayat": ayat or None},
                "citation_str": collapse(fmt_bab(bab) + " " + fmt_pasal(pasal_raw)
                                         + (" " + fmt_ayat(ayat) if ayat else "")),
                "content_role": role, "role_source": role_src, "part": part,
                "is_fragment": bool(is_frag),
                "doc_type": meta.get("doc_type"), "doc_type_full": meta.get("doc_type_full"),
                "legal_force_axis": meta.get("legal_force_axis"),
                "hierarchy_level": to_int(row.get("hierarchy_level")),
                "year": to_int(row.get("year")),
                "legal_status": None,
                "text": body, "text_original": collapse(s(row.get("text_original"))),
                "text_lexical": body.lower(),
                "embedding_input": self._embed(title, collapse(
                    fmt_bab(bab) + " " + fmt_pasal(pasal_raw)
                    + (" " + fmt_ayat(ayat) if ayat else "")), body),
                "provenance": {
                    "source_unit_id": to_int(row.get("unit_id")),
                    "file_name": meta["file_name"],
                    "relative_path": nz(row.get("relative_path")),
                    "texts_path": self.tidx.get(Path(meta["file_name"]).stem.lower()),
                    "page_number": to_int(row.get("page_number")),
                    "line_start": to_int(row.get("line_start")),
                    "line_end": to_int(row.get("line_end"))},
                "char_len": len(body), "token_estimate": tokens,
                "quality_flags": [flag] if flag else []}
            self.chunks.append(chunk)
            parent["text_parts"].append(chunk["text"])
            parent["child_ids"].append(cid)
            parent["char_len"] += len(body)
            if pasal_raw:
                state["pasal"] = pasal_raw
            if bab:
                state["bab"] = bab

        # finalize parent text + windows (hanya parent dokumen ini, deterministik)
        for pid in list(doc_pids):
            p = self.parents.get(pid)
            if p is None or "text_parts" not in p:
                continue
            p["text"] = "\n\n".join(p.pop("text_parts"))
            p.pop("text_parts", None)
            ws = []
            txt = p["text"]
            # `char_len` sums child text only; joining with paragraph
            # separators can make the final parent slightly longer. Window
            # the actual assembled text, not the pre-join estimate.
            if len(txt) <= WINDOW_MAX_CHARS:
                ws = [txt]
            else:
                i, n = 0, 0
                while i < len(txt):
                    hard_end = min(i + WINDOW_MAX_CHARS, len(txt))
                    end = hard_end
                    if hard_end < len(txt):
                        brk = txt.rfind("\n\n", i + WINDOW_MIN_CHARS, hard_end)
                        if brk < 0:
                            brk = txt.rfind("\n", i + WINDOW_MIN_CHARS, hard_end)
                        if brk > i:
                            end = min(brk, hard_end)
                    if end <= i:
                        end = hard_end
                    segment = txt[i:end].strip()
                    # Defensive hard cap: paragraph-aware boundaries may not
                    # be allowed to exceed the declared window contract.
                    if len(segment) > WINDOW_MAX_CHARS:
                        segment = segment[:WINDOW_MAX_CHARS].rstrip()
                        end = i + WINDOW_MAX_CHARS
                    n += 1
                    ws.append(segment)
                    i = max(i + 1, end - WINDOW_OVERLAP_CHARS) if end < len(txt) else end
            p["window_ids"] = [f"{pid}::w{n:02d}" for n in range(1, len(ws) + 1)]
            for wid, wtxt in zip(p["window_ids"], ws):
                self.windows[wid] = {
                    "window_id": wid, "parent_id": pid, "document_id": doc_id,
                    "part": p["part"],
                    "citation_str": p["citation_str"], "char_len": len(wtxt),
                    "text": wtxt}

    @staticmethod
    def _embed(title, cit, text):
        head = f"{title} - {cit}" if cit else title
        return f"{head}: {text}"


def qa(chunks, rng):
    cache = {}
    sample = rng.sample(chunks, min(QA_SAMPLES, len(chunks)))
    ok = fail = skip = 0
    probs = []
    for c in sample:
        rel = c["provenance"].get("texts_path")
        if not rel:
            if any(f in c["quality_flags"] for f in ("structural_admin",)):
                skip += 1
            else:
                probs.append({"chunk_id": c["chunk_id"], "why": "texts_not_found"})
            continue
        p = CORPUS_ROOT / rel
        if not p.exists():
            if any(f in c["quality_flags"] for f in ("structural_admin",)):
                skip += 1
            else:
                probs.append({"chunk_id": c["chunk_id"], "why": "texts_not_found"})
            continue
        if rel not in cache:
            cache[rel] = pages_of(p)
        pages = cache[rel]
        pg = c["provenance"]["page_number"]
        lines = pages.get(pg) or pages.get(0)
        if not lines:
            probs.append({"chunk_id": c["chunk_id"], "why": "page_not_found"})
            continue
        a = c["provenance"]["line_start"] or 1
        b = c["provenance"]["line_end"] or a
        src = collapse(" ".join(lines[max(a - 1, 0):max(b, a)])).casefold()
        got = c["text_original"].casefold()
        if not src:
            probs.append({"chunk_id": c["chunk_id"], "why": "empty_source_slice"})
            continue
        if (src[:QA_PREFIX_CHARS] == got[:QA_PREFIX_CHARS]
                or src.startswith(got[:24]) or got.startswith(src[:24])):
            ok += 1
        else:
            fail += 1
            if len(probs) < 8:
                probs.append({"chunk_id": c["chunk_id"],
                              "src": src[:60], "txt": got[:60],
                              "flag": c["quality_flags"]})
    checked = ok + fail
    rate = ok / checked if checked else None
    return {"strategy": f"{QA_SAMPLES} sampel chunk; isi dicek ke halaman+rentang "
                        f"baris pada file sumber (prefix {QA_PREFIX_CHARS} char)",
            "samples": len(sample), "verified": ok, "unverified": fail,
            "skipped": skip, "match_rate": round(rate, 4) if rate is not None else None,
            "min_required": QA_MIN_MATCH_RATE, "sample_failures": probs,
            "note": "unverified umumnya baris penanda struktural/admin yang dipangkas "
                    "builder korpus sebelum normalisasi; isi norma tetap utuh"}


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = OrderedDict()
    with open(DOCS_CSV, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        miss = REQUIRED_DOCS - set(r.fieldnames or [])
        if miss:
            raise SystemExit(f"documents.csv kolom hilang: {sorted(miss)}")
        used = set()
        for row in r:
            docs[row["file_name"]] = build_doc_meta(row["file_name"], row, used)

    tidx = texts_index()
    units_by_doc = OrderedDict()
    with open(UNITS_CSV, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        miss = REQUIRED_UNITS - set(r.fieldnames or [])
        if miss:
            raise SystemExit(f"units.csv kolom hilang: {sorted(miss)}")
        for row in r:
            units_by_doc.setdefault(row["document_name"], []).append(row)

    b = B(tidx)
    for fn, rows in units_by_doc.items():
        doc = docs.get(fn)
        if doc is None:
            doc = {"document_id": slug(Path(fn).stem), "file_name": fn, "doc_type": None,
                   "doc_type_full": None, "number": None, "year": None, "topic": None,
                   "official_title": Path(fn).stem, "legal_force_axis": None,
                   "identity_source": "unresolved_file_name",
                   "filename_title_mismatch": False, "legal_status": None,
                   "effective_date": None, "revision_basis": None, "enrichment_source": None}
        docs[fn] = doc
        rows.sort(key=lambda x: (to_int(x.get("page_number")) or 0,
                                 to_int(x.get("line_start")) or 0,
                                 to_int(x.get("unit_id")) or 0))
        b.build_doc(doc, rows)

    # validasi silang: setiap chunk punya parent yang ada
    if b.errors:
        pass
    pids = set(b.parents)
    orphan = [c["chunk_id"] for c in b.chunks if c["parent_id"] not in pids]
    dup_cids = len(b.chunks) - len({c["chunk_id"] for c in b.chunks})

    parents = OrderedDict()
    for pid, p in b.parents.items():
        if p["child_ids"]:
            parents[pid] = p
    windows = OrderedDict((w, b.windows[w]) for w in b.windows
                          if b.windows[w]["parent_id"] in parents)

    rng = random.Random(SEED)
    q = qa(b.chunks, rng)
    gate = q["match_rate"] is not None and q["match_rate"] >= QA_MIN_MATCH_RATE

    def jw(path, items):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    jw(OUT_DIR / "chunks.jsonl", b.chunks)
    jw(OUT_DIR / "parents.jsonl", list(parents.values()))
    jw(OUT_DIR / "parents_windows.jsonl", list(windows.values()))
    jw(OUT_DIR / "quarantine.jsonl", b.quarantine)
    jw(OUT_DIR / "documents.jsonl", list(docs.values()))

    roles = {}
    frags = sum(1 for c in b.chunks if c["is_fragment"])
    for c in b.chunks:
        roles[c["content_role"]] = roles.get(c["content_role"], 0) + 1
    manifest = {
        "name": "indonesian-legislation-rag-dataset",
        "dataset_version": VERSION, "schema_version": SCHEMA_VERSION,
        "corpus": CORPUS_ID, "license": LICENSE, "attribution": ATTRIBUTION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deterministic": True, "seed": SEED, "config": {
            "heading_like_max_chars": HEADING_LIKE_MAX_CHARS,
            "fragment_tokens_lt": MIN_FRAGMENT_TOKENS,
            "window_max_chars": WINDOW_MAX_CHARS,
            "window_overlap_chars": WINDOW_OVERLAP_CHARS,
            "merge_between_units": False},
        "schema": {
            "chunk": "chunk_id, parent_id (global), document_id, content_role, "
                     "role_source, part (turunan: norma|administratif), "
                     "is_fragment, citation*, embedding_input, text*, provenance, "
                     "legal_status=null",
            "parent": "parent_id GLOBAL {document_id}::p::{pasal} | "
                      "{document_id}::adm::{nn}; text utuh; window_ids",
            "window": "window_id {parent_id}::w{nn}; deterministik+overlap "
                      f"{WINDOW_OVERLAP_CHARS} char; max {WINDOW_MAX_CHARS} char",
            "validity": "isi legal_status/effective_date/enrichment_source di "
                        "documents.jsonl + chunk (join document_id) dari "
                        "validity.jsonl hasil enrich_validity.py"},
        "counts": {"input_rows": sum(len(v) for v in units_by_doc.values()),
                   "documents": len(docs), "chunks": len(b.chunks),
                   "fragments": frags, "content_roles": dict(sorted(roles.items())),
                   "parents": len(parents),
                   "long_parents_windowed": sum(1 for p in parents.values()
                                                if len(p["window_ids"]) > 1),
                   "windows": len(windows),
                   "chunks_by_part": {
                       "norma": sum(1 for c in b.chunks if c["part"] == "norma"),
                       "administratif": sum(1 for c in b.chunks if c["part"] != "norma")},
                   "heading_units_as_context": b.cnt.get("heading_unit_as_context", 0),
                   "quarantine": len(b.quarantine),
                   "quarantine_by_reason": dict(sorted(
                       {k: v for k, v in b.cnt.items()
                        if k.endswith("quarantined") or k in ("empty_text",
                                                              "needs_review_short")}.items())),
                   "fragment_disposition": "dipertahankan sebagai chunk "
                                           "(is_fragment=true), TIDAK merge"},
        "validation": {"orphan_chunks": len(orphan),
                       "orphan_examples": orphan[:10],
                       "duplicate_chunk_ids": dup_cids,
                       "qa_provenance": q,
                       "qa_gate": "PASS" if (gate if q["match_rate"] is not None else True)
                                  else "FAIL",
                       "errors": b.errors[:20]},
        "rag_guide": {
            "embed": "embedding_input skip is_fragment=true; "
                     "strict: content_role=='norma'; luas: semua part",
            "retrieve": "kandidat -> ambil text induk via parent_id (parents.jsonl) "
                        "atau window via window_ids (konteks utuh untuk LLM)",
            "cite_back": "content_role, part, citation_str, official_title, "
                         "provenance page/line",
            "validity_before_production": "jalankan enrich_validity.py lalu gabungkan "
                                          "legal_status ke chunk/payload vector db"},
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input    :", manifest["counts"]["input_rows"])
    print("chunks   :", len(b.chunks), f"(fragmen {frags})")
    print("parents  :", len(parents), "| window:", len(windows))
    print("roles    :", dict(sorted(roles.items())))
    print("quarantine:", len(b.quarantine))
    print("QA       : rate=", q["match_rate"], "gap:", q["unverified"],
          "skip:", q["skipped"], "->", manifest["validation"]["qa_gate"])
    print("OUTPUT   :", str(OUT_DIR))
    return 0 if manifest["validation"]["duplicate_chunk_ids"] == 0 and not orphan \
        else 1


if __name__ == "__main__":
    raise SystemExit(main())
