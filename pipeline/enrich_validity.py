#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrichment status keberlakuan hukum (berlaku / diubah / dicabut).

Korpus sumber TIDAK memuat status ini — script ini melapisinya.
PRINSIP: JANGAN PERNAH menebak. Tidak yakin -> tetap null, catat alasan.

Mode default (offline, aman, deterministik):
    python enrich_validity.py
  -> membuat result/validity.jsonl dari documents.jsonl dengan
     status "pending_enrichment" (siap diisi manual/API nanti).

Mode --live (butuh internet, best-effort, hormati aturan portal):
    python enrich_validity.py --live --delay 3
  -> mencari kandidat peraturan di peraturan.bpk.go.id (pencarian publik),
     menyimpan URL kandidat + kecocokan identitas ke validity.jsonl.
     Status hukum TETAP null sampai halaman detail terverifikasi manusia
     (parsing status HTML portal rapuh; kita tidak mengarang).
"""
import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "result" / "documents.jsonl"
OUT = HERE / "result" / "validity.jsonl"

UA = "Mozilla/5.0 (legal-rag-enrichment; contact: operator Anda) "


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def live_lookup(doc, delay):
    q = f"{doc.get('doc_type') or ''} {doc.get('number') or ''} " \
        f"{doc.get('year') or ''} {doc.get('topic') or ''}".strip()
    url = ("https://peraturan.bpk.go.id/Home/ResultsPeraturan?Keyword="
           + urllib.parse.quote(q))
    try:
        page = fetch(url)
        time.sleep(delay)
        cand = None
        pat = re.compile(r'href="(/Details/\d+/[^"]+)"', re.I)
        want = [x for x in
                (doc.get("number"), doc.get("year")) if x]
        for href in pat.findall(page):
            slug = norm(urllib.parse.unquote(href.split("/")[-1]))
            if all(str(norm(w)) in slug for w in want if w):
                cand = "https://peraturan.bpk.go.id" + href
                break
        state = "candidate_found" if cand else "no_match_verify_manually"
        return state, cand, None
    except Exception as e:  # noqa: BLE001 — jaringan: catat, jangan menebak
        return "fetch_failed", None, str(e)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    docs = [json.loads(l) for l in open(DOCS, encoding="utf-8")]
    if a.limit:
        docs = docs[:a.limit]
    rows = []
    for d in docs:
        row = {"file_name": d["file_name"], "document_id": d["document_id"],
               "doc_type": d.get("doc_type"), "number": d.get("number"),
               "year": d.get("year"), "identity_source": d.get("identity_source"),
               "legal_status": None, "effective_date": None,
               "repealed_by": None, "registry_url": None,
               "status_state": "pending_enrichment",
               "note": "isi legal_status hanya dengan bukti; jangan menebak",
               "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if a.live and d.get("number") and d.get("year"):
            st, url, err = live_lookup(d, a.delay)
            row.update({"status_state": st, "registry_url": url, "note": err or ""})
        rows.append(row)

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    st = {}
    for r in rows:
        st[r["status_state"]] = st.get(r["status_state"], 0) + 1
    print(f"validity -> {OUT}\nstate: {st}")


if __name__ == "__main__":
    main()
