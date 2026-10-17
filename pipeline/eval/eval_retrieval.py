#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ukur retrieval pada set emas eval/golden_queries.jsonl (recall@k).

Prasyarat: jalankan ../../embed_example.py dulu (menghasilkan
result/embeddings/embeddings.npy + ids.jsonl + meta.json).

Pemakaian:
    python eval_retrieval.py                 # k=8, model dari meta.json
    python eval_retrieval.py --k 12 --engine api --base-url ... --model ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent
sys.path.insert(0, str(COMBINED))
from emb_paths import active_embeddings_dir  # noqa: E402

EMB_DIR = active_embeddings_dir()
CHUNKS = COMBINED / "result" / "chunks.jsonl"
GOLDEN = HERE / "golden_queries.jsonl"
REPORT = HERE / "golden_report.json"


def load_chunks():
    out = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["chunk_id"]] = r
    return out


def pas_num(p):
    if not p:
        return ""
    m = re.search(r"(\d+)\s*([a-z])?", str(p), re.I)
    if not m:
        return str(p).lower()
    return "%d%s" % (int(m.group(1)), (m.group(2) or "").lower())


def matches(ch, exp):
    if exp.get("doc_regex"):
        # official_title UUD bisa berupa stub ("UUD 1945 Perubahan Kedua") —
        # doc_type_full tetap memuat identitas resmi, jadi cek keduanya.
        t = (ch.get("official_title") or "").lower()
        tf = (ch.get("doc_type_full") or "").lower()
        if not (re.search(exp["doc_regex"], t, re.I)
                or re.search(exp["doc_regex"], tf, re.I)):
            return False
    if "pasal" in exp and exp["pasal"]:
        if pas_num((ch.get("citation") or {}).get("pasal")) != pas_num(exp["pasal"]):
            return False
    if "ayat" in exp and exp["ayat"]:
        ay = str((ch.get("citation") or {}).get("ayat") or "").strip(" ()")
        if ay != str(exp["ayat"]):
            return False
    kws = exp.get("kw") or []
    if kws:
        txt = ch["text_lexical"]
        if not any(k.lower() in txt for k in kws):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--engine", choices=["hf", "api"], default="hf")
    ap.add_argument("--model", default=None, help="default: dari embeddings/meta.json")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import numpy as np

    meta = json.loads((EMB_DIR / "meta.json").read_text(encoding="utf-8"))
    model = a.model or meta.get("model")
    eng = meta.get("engine", a.engine)
    if a.engine == "api":
        eng = "api"
    ids = [json.loads(l) for l in open(EMB_DIR / "ids.jsonl", encoding="utf-8")]
    emb = np.load(EMB_DIR / "embeddings.npy")
    assert len(ids) == emb.shape[0], "ids.jsonl tidak sejajar embeddings.npy"
    chunks = load_chunks()
    golden = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]

    texts = [g["question"] for g in golden]
    if eng == "api":
        from embed_example import encode_api
        import urllib.request  # noqa
        vecs = np.concatenate(list(encode_api(texts, a.base_url, model, a.batch)))
    elif meta.get("pooling") == "last_token":
        # Qwen3-Embedding & kawan-kawan: last-token pooling + instruksi tugas
        # pada sisi query — encoder harus cocok dengan meta.json, bukan mean-pool e5.
        from embed_query_qwen import encode_query
        vecs = encode_query(texts, model)
    else:
        from embed_example import encode_hf
        out = []
        for t in texts:
            out.extend(encode_hf([t], model, 1, 512, prefix="query"))
        vecs = np.concatenate(out)

    scores = emb @ vecs.T           # L2-normal => dot = cosine
    order = np.argsort(-scores, axis=0)[: a.k, :]

    per = []
    hits = 0
    for gi, g in enumerate(golden):
        top = [ids[i] for i in order[:, gi]]
        hit = any(matches(chunks[t["chunk_id"]], e)
                  for e in g["expect_any"] for t in top)
        if not hit:  # longgarkan: abaikan predikat pasal/ayat bila doc+kw cocok
            hit = any(matches(chunks[t["chunk_id"]], {**e, "pasal": None, "ayat": None})
                      for e in g["expect_any"] for t in top)
        hits += hit
        per.append({"id": g["id"], "question": g["question"], "hit": bool(hit),
                    "top_ids": [t["chunk_id"] for t in top[:5]]})
    rate = hits / len(golden)
    rep = {"model": model, "engine": eng, "k": a.k, "queries": len(golden),
           "recall_at_k": round(rate, 3), "per_query": per,
           "interpretasi": "recall>=0.8 bagus; 0.6-0.8 evaluasi ulang model/top-k; "
                           "<0.6 jangan lanjut ke agent tanpa perbaikan"}
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in per:
        print(("HIT  " if p["hit"] else "MISS ") + p["id"] + "  " + p["question"][:70])
    print(f"\nrecall@{a.k} = {rate:.3f}  ->  {REPORT}")
    if rate < 0.6:
        sys.exit(2)


if __name__ == "__main__":
    main()
