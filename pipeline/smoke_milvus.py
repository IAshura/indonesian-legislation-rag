#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test Milvus TANPA model embedding (butuh: pymilvus + numpy saja).

Ambil beberapa vektor langsung dari embeddings.npy, cari ke Milvus, dan
pastikan top-1 = vektor itu sendiri dengan skor ~1.0. Membuktikan koneksi,
koleksi, indeks HNSW, dan insert lengkap — terpisah dari kualitas model.

Pakai:
    python smoke_milvus.py                 # uji 5 vektor acak
    python smoke_milvus.py --n 20 --top 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from emb_paths import active_embeddings_dir

HERE = Path(__file__).resolve().parent
EMB_DIR = active_embeddings_dir()
COLLECTION = "legislation_norma_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--dir", default=str(EMB_DIR),
                    help="folder artefak embedding (default: result/embeddings)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    from pymilvus import Collection, connections

    emb_dir = Path(a.dir)
    vecs = np.load(emb_dir / "embeddings.npy")
    ids = [json.loads(l) for l in open(emb_dir / "ids.jsonl", encoding="utf-8")]
    if len(ids) != vecs.shape[0]:
        sys.exit("ids.jsonl tidak sejajar embeddings.npy — jalankan verify_embeddings.py")

    connections.connect(uri=a.uri)
    col = Collection(a.collection)
    col.load()
    # parameter pencarian mengikuti indeks AKTIF di koleksi:
    #   HNSW -> ef | IVF_FLAT -> nprobe | FLAT / tanpa indeks -> brute force
    itype = (col.indexes[0].params.get("index_type", "FLAT")
             if col.indexes else "FLAT")
    param = {"metric_type": "IP", "params": {}}
    if itype == "HNSW":
        param["params"]["ef"] = 32
    elif itype == "IVF_FLAT":
        param["params"]["nprobe"] = 64
    print(f"[milvus] {a.collection}: {col.num_entities:,} entitas, indeks={itype}"
          f" (param={param['params']}) — uji {a.n} self-query"
          " (top-1 harus dirinya sendiri)")

    rng = random.Random(42)
    picks = rng.sample(range(len(ids)), a.n)
    ok = 0
    for i in picks:
        res = col.search([vecs[i].tolist()], anns_field="embedding",
                         param=param, limit=a.top, output_fields=["chunk_id"])
        hits = res[0]
        hit_self = hits[0].entity.get("chunk_id")
        match = hit_self == ids[i]["chunk_id"]
        ok += match
        status = "OK " if match else "SALAH"
        print(f"  [{status}] {ids[i]['chunk_id']}")
        for j, h in enumerate(hits, 1):
            print(f"     {j}. score={h.score:.4f} {h.entity.get('chunk_id')}")

    print(f"\nhasil: {ok}/{a.n} self-query tepat")
    return 0 if ok == a.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
