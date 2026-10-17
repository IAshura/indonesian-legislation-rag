#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifikasi artefak embedding: npy ↔ ids.jsonl ↔ meta.json wajib sejajar.

Pakai:
    python verify_embeddings.py                     # cek result/embeddings/
    python verify_embeddings.py --dir result/embeddings

Yang dicek:
  1. header npy — dtype float32, shape (n, dim)
  2. ids.jsonl  — n baris, kolom kunci terisi, chunk_id unik
  3. meta.json  — count & dim konsisten
  4. (bila numpy ada) norm L2 ≈ 1.0 & tidak ada NaN/Inf — sampling 1000 baris
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from emb_paths import active_embeddings_dir

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = active_embeddings_dir()
REQUIRED_ID_KEYS = ("chunk_id", "parent_id", "document_id")


def read_npy_header(path: Path) -> tuple[str, tuple[int, ...]]:
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            sys.exit("bukan file .npy valid")
        f.read(2)  # versi
        hlen = struct.unpack("<H", f.read(2))[0]
        header = f.read(hlen).decode()
    descr = header.split("'descr': '")[1].split("'")[0]
    shape = header.split("'shape': (")[1].split(")")[0]
    dims = tuple(int(x.strip().rstrip(",")) for x in shape.split(",") if x.strip())
    return descr, dims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    a = ap.parse_args()
    d = Path(a.dir)

    npy, ids_f, meta_f = d / "embeddings.npy", d / "ids.jsonl", d / "meta.json"
    for p in (npy, ids_f, meta_f):
        if not p.exists():
            sys.exit(f"file tidak ada: {p}")

    descr, dims = read_npy_header(npy)
    print(f"[npy]       dtype={descr} shape={dims}")

    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    print(f"[meta.json] model={meta.get('model')} count={meta.get('count')} "
          f"dim={meta.get('dim')} pooling={meta.get('pooling')}")

    rows, seen, problems = [], set(), []
    with open(ids_f, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            for k in REQUIRED_ID_KEYS:
                if not r.get(k):
                    problems.append(f"baris {i}: kolom '{k}' kosong")
            if r["chunk_id"] in seen:
                problems.append(f"baris {i}: chunk_id duplikat")
            seen.add(r["chunk_id"])
            rows.append(r)
    print(f"[ids.jsonl] {len(rows):,} baris | chunk_id unik: {len(seen):,}")

    n = dims[0] if dims else 0
    dim = dims[1] if len(dims) > 1 else 0
    if descr != "<f4":
        problems.append(f"dtype npy {descr} — seharusnya <f4 (float32)")
    if len(rows) != n:
        problems.append(f"ids.jsonl ({len(rows):,}) != baris npy ({n:,})")
    if meta.get("count") not in (n, len(rows)):
        problems.append(f"meta.count ({meta.get('count')}) != n ({n:,})")
    if meta.get("keep_dim") not in (dim, None) and meta.get("dim") != dim:
        problems.append(f"meta.dim ({meta.get('dim')}) != dim npy ({dim})")

    try:
        import numpy as np
    except ImportError:
        print("[numpy] tidak terpasang — cek norm lewati")
    else:
        idx = np.linspace(0, n - 1, num=min(1000, n), dtype=int)
        vecs = np.load(npy, mmap_mode="r")[idx]
        norms = np.linalg.norm(vecs.astype("float32"), axis=1)
        bad_norm = int(((norms < 0.98) | (norms > 1.02)).sum())
        if not np.isfinite(vecs).all():
            problems.append("ada NaN/Inf di sampel vektor")
        if bad_norm:
            problems.append(f"{bad_norm}/{len(idx)} vektor sampel norm != 1.0 "
                            "(±0.02) — L2-normal bermasalah")
        else:
            print(f"[vektor]   sampel {len(idx)} norm L2: "
                  f"min={norms.min():.4f} max={norms.max():.4f} [OK]")

    if problems:
        print("\nGAGAL:")
        for p in problems:
            print("  [X]", p)
        return 1
    print("\nSEMUA COCOK - siap lanjut: insert_milvus.py / eval_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
