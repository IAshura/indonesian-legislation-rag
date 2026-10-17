#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Insert embeddings → Milvus standalone (Docker di Windows ini).

Prasyarat:
  1. Milvus berjalan — setup mesin ini: Docker Desktop + compose di C:\\milvus
        cd C:\\milvus; docker compose up -d          # start
        docker compose down                           # stop (data tetap aman)
  2. Artefak embedding sudah ada:
        pipeline/result/embeddings/<tag>/{embeddings.npy, ids.jsonl, meta.json}
  3. pip install pymilvus

Pemakaian:
  python insert_milvus.py                        # buat koleksi + insert (indeks HNSW default)
  python insert_milvus.py --drop                 # buang koleksi lama dulu
  python insert_milvus.py --demo "pertanyaan"    # uji retrieval top-5
  python insert_milvus.py --index flat           # buat koleksi dgn brute force (exact)
  python insert_milvus.py --index ivf            # buat koleksi dgn IVF_FLAT
  python insert_milvus.py --rebuild-index --index flat  # GANTI indeks TANPA insert ulang
  python insert_milvus.py --demo "..." --ef 128    # HNSW: ef lebih besar = lebih akurat
  python insert_milvus.py --demo "..." --nprobe 64 # IVF : nprobe lebih besar = lebih akurat

Mode pencarian (--index {hnsw,ivf,flat}) — metrik tetap IP (dot = cosine):
  hnsw  — graph ANN, cepat & akurat (default)  | search: ef
  ivf   — ANN klaster k-means, ringan skala besar | search: nprobe
  flat  — brute force / exact, baseline kebenaran | tanpa parameter
Ganti indeks = rebuild struktur indeks saja — vektor & ids TIDAK diubah.
Detail & benchmark: README-milvus.md

Skema koleksi: 1 vektor = 1 chunk (part=norma, non-fragmen).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from emb_paths import active_embeddings_dir

HERE = Path(__file__).resolve().parent
EMB_DIR = active_embeddings_dir()
COLLECTION = "legislation_norma_v1"
INSERT_BATCH = 1000

# Konfigurasi indeks + parameter pencarian per mode. Semua metrik IP
# (vektor L2-normal → dot-product = cosine). Yang membedakan hanya STRUKTUR
# INDEKS (dipilih saat create/rebuild) dan PARAMETER SEARCH (saat query).
INDEX_CONFIGS = {
    "hnsw": {   # graph ANN — default; ef = lebar pencarian graph saat query
        "index": {"index_type": "HNSW", "metric_type": "IP",
                  "params": {"M": 16, "efConstruction": 200}},
        "search": {"metric_type": "IP", "params": {"ef": 64}},
    },
    "ivf": {    # ANN berbasis klaster; nprobe = jumlah klaster yang diperiksa
        "index": {"index_type": "IVF_FLAT", "metric_type": "IP",
                  "params": {"nlist": 1024}},
        "search": {"metric_type": "IP", "params": {"nprobe": 16}},
    },
    "flat": {   # brute force / exact — bandingkan SEMUA vektor, selalu benar
        "index": {"index_type": "FLAT", "metric_type": "IP", "params": {}},
        "search": {"metric_type": "IP", "params": {}},
    },
}


def search_param(cfg, ef=None, nprobe=None):
    """Search param dari cfg + override --ef / --nprobe (deep copy aman)."""
    sp = json.loads(json.dumps(cfg["search"]))
    if ef is not None and "ef" in sp["params"]:
        sp["params"]["ef"] = ef
    if nprobe is not None and "nprobe" in sp["params"]:
        sp["params"]["nprobe"] = nprobe
    return sp


def index_type_of(col) -> str:
    """Baca tipe indeks AKTIF di koleksi ('FLAT' bila tanpa indeks → brute force)."""
    idx = list(getattr(col, "indexes", []))
    return idx[0].params.get("index_type", "FLAT") if idx else "FLAT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--dir", default=str(EMB_DIR),
                    help="folder artefak embedding (default: folder ACTIVE)")
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--index", choices=list(INDEX_CONFIGS), default="hnsw",
                    help="jenis indeks saat membuat koleksi / rebuild (default: hnsw)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="ganti indeks koleksi yang ada tanpa insert ulang")
    ap.add_argument("--ef", type=int, default=None,
                    help="HNSW: perbesar ef saat --demo untuk akurasi lebih tinggi")
    ap.add_argument("--nprobe", type=int, default=None,
                    help="IVF: perbesar nprobe saat --demo untuk akurasi lebih tinggi")
    ap.add_argument("--demo", default=None,
                    help="uji retrieval: pertanyaan dalam tanda kutip")
    a = ap.parse_args()

    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections, utility)

    emb_dir = Path(a.dir)
    meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    dim = int(meta.get("keep_dim") or meta["dim"])
    ids = [json.loads(l) for l in open(emb_dir / "ids.jsonl", encoding="utf-8")]
    vecs = np.load(emb_dir / "embeddings.npy")
    if len(ids) != vecs.shape[0]:
        sys.exit("ids.jsonl tidak sejajar embeddings.npy")
    if vecs.shape[1] != dim:
        sys.exit(f"dim vektor {vecs.shape[1]} != meta.dim {dim}")

    connections.connect(uri=a.uri)
    print(f"[milvus] tersambung {a.uri}")
    cfg = INDEX_CONFIGS[a.index]

    if a.drop and utility.has_collection(a.collection):
        utility.drop_collection(a.collection)
        print(f"[milvus] koleksi lama dibuang: {a.collection}")

    if utility.has_collection(a.collection):
        col = Collection(a.collection)
        itype = index_type_of(col)
        print(f"[milvus] koleksi sudah ada: {a.collection} "
              f"({col.num_entities} entitas, indeks {itype})")
        if a.rebuild_index and itype != a.index.upper():
            print(f"[indeks] ganti indeks {itype} -> {a.index} "
                  f"(vektor TIDAK disentuh)")
            col.release()
            for ix in list(col.indexes):
                ix.drop()
            col.create_index("embedding", index_params=cfg["index"])
            col.load()
            print(f"[indeks] indeks baru dibangun: {a.index}")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True,
                        auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=160),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=160),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=160),
            FieldSchema(name="part", dtype=DataType.VARCHAR, max_length=24),
            FieldSchema(name="content_role", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=24),
            FieldSchema(name="legal_force_axis", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="year", dtype=DataType.INT64),
            FieldSchema(name="is_fragment", dtype=DataType.BOOL),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        col = Collection(a.collection, schema=CollectionSchema(
            fields, description="Indonesian legislation RAG — norma chunks"))
        col.create_index("embedding", index_params=cfg["index"])
        print(f"[milvus] koleksi dibuat: {a.collection} (dim={dim}, {a.index}/IP)")

    if col.num_entities == len(ids):
        print(f"[milvus] data sudah lengkap ({len(ids)}) — tidak insert ulang")
    else:
        t0 = time.time()
        for i in range(0, len(ids), INSERT_BATCH):
            rows = ids[i:i + INSERT_BATCH]
            col.insert([
                [r["chunk_id"] for r in rows],
                [r["parent_id"] for r in rows],
                [r["document_id"] for r in rows],
                [str(r["part"] or "") for r in rows],
                [str(r["content_role"] or "") for r in rows],
                [str(r["doc_type"] or "") for r in rows],
                [str(r["legal_force_axis"] or "") for r in rows],
                [int(r["year"] or 0) for r in rows],
                [bool(r["is_fragment"]) for r in rows],
                vecs[i:i + INSERT_BATCH].tolist(),
            ])
            print(f"[insert] {min(i + INSERT_BATCH, len(ids)):>6}/{len(ids)} "
                  f"| {time.time() - t0:5.1f}s")
        col.flush()
        print(f"[milvus] flush — total entitas: {col.num_entities}")
        col.load()

    if a.demo:
        from embed_query_qwen import encode_query
        col.load()                       # pastikan koleksi termuat utk search
        # param pencarian mengikuti indeks AKTIF di koleksi (bukan --index),
        # jadi demo tetap benar apa pun --index yang diketik.
        actual = index_type_of(col)      # "HNSW" / "IVF_FLAT" / "FLAT"
        cfg_name = {"HNSW": "hnsw", "IVF_FLAT": "ivf"}.get(actual, "flat")
        sp = search_param(INDEX_CONFIGS[cfg_name], a.ef, a.nprobe)
        q = encode_query([a.demo])[0]
        res = col.search([q.tolist()], anns_field="embedding",
                         param=sp, limit=5,
                         output_fields=["chunk_id", "parent_id", "document_id",
                                        "year"])
        print(f"\n[demo] {a.demo!r}  (indeks={actual}, param={sp['params']})")
        for j, hit in enumerate(res[0], 1):
            print(f"  {j}. score={hit.score:.4f} chunk={hit.entity.get('chunk_id')}"
                  f" parent={hit.entity.get('parent_id')}")
        print("\nlangkah berikutnya: join parent_id -> parents.jsonl -> LLM")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
