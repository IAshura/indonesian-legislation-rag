#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query-side encoder untuk Qwen3-Embedding (dijalankan DI WINDOWS LOKAL).

Kenapa file ini ada: pooling Qwen3-Embedding adalah LAST-TOKEN + instruksi
tugas pada sisi query — BERBEDA dari mean-pooling + prefix e5 di
embed_example.py. eval_retrieval.py membaca meta.json dan memakai encoder
ini bila meta.pooling == "last_token".

Sumber kebenaran parameter: pipeline/result/embeddings/meta.json
(yang ditulis notebook Colab) — jangan mengarang model/dimensi di sini.

Pemakaian (impor):
    from embed_query_qwen import encode_query
    vecs = encode_query(["Di tangan siapa kedaulatan berada menurut UUD 1945?"])
    # float32 L2-normal, shape (n, dim), dim dari meta.json (Matryoshka-aware)

Pemakaian (CLI, uji cepat tanpa FAISS):
    python embed_query_qwen.py "Di tangan siapa kedaulatan berada menurut UUD 1945?"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from emb_paths import active_embeddings_dir

HERE = Path(__file__).resolve().parent
META = active_embeddings_dir() / "meta.json"


def load_meta() -> dict:
    if not META.exists():
        sys.exit(f"meta.json tidak ada: {META} — jalankan embedding (Colab) dulu")
    return json.loads(META.read_text(encoding="utf-8"))


_MODEL_CACHE: dict = {}


def _load(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval()
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    _MODEL_CACHE[model_id] = (tok, mdl)
    return tok, mdl


@torch.no_grad()
def encode_query(texts: list[str], model: str | None = None,
                 instruction: str | None = None,
                 max_length: int = 512, batch: int = 8) -> np.ndarray:
    """Encode pertanyaan → vektor float32 L2-normal (n, dim).

    Format Qwen3-Embedding untuk query: "Instruct: {instr}\nQuery: {q}".
    Instruksi diambil dari meta.json (query_instruction) bila tidak diberikan.
    """
    meta = load_meta()
    model = model or meta.get("model")
    instruction = instruction or meta.get("query_instruction", "")
    keep_dim = meta.get("keep_dim") or meta.get("dim")
    tok, mdl = _load(model)
    dev = next(mdl.parameters()).device

    out = []
    for i in range(0, len(texts), batch):
        chunk = [f"Instruct: {instruction}\nQuery: {t}" for t in texts[i:i + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(dev)
        hs = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"]
        if (mask[:, -1] == 1).all():          # left padding → posisi terakhir
            pooled = hs[:, -1]
        else:
            seq = mask.sum(dim=1) - 1
            pooled = hs[torch.arange(hs.shape[0], device=hs.device), seq]
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        out.append(pooled.cpu().numpy())
    vecs = np.concatenate(out).astype("float32")
    return vecs[:, :keep_dim]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('pemakaian: python embed_query_qwen.py "pertanyaan ..."')
    v = encode_query(sys.argv[1:])
    m = load_meta()
    print(f"model={m['model']} dim={v.shape[1]} query={v.shape[0]}")
    print("norm per vektor (harus ~1.0):", [round(float(np.linalg.norm(x)), 4) for x in v])
