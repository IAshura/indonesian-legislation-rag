#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feed result/chunks.jsonl ke model embedding mana pun (skema 4).

Default: part=norma dan fragmen pendek (is_fragment=true) DILEWATI, sesuai
prinsip unit-preservation: dataset utuh di file, indeks hanya unit bermakna.

Contoh:
    python embed_example.py                          # norma saja, non-fragmen
    python embed_example.py --include-fragments
    python embed_example.py --part all
    python embed_example.py --model intfloat/multilingual-e5-large --batch-size 32

API OpenAI-compatible:
    $env:EMBED_API_KEY="..."; python embed_example.py --engine api ^
      --base-url https://api.openai.com/v1 --model text-embedding-3-large
"""
import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "result" / "chunks.jsonl"
DEFAULT_OUT = HERE / "result" / "embeddings"
TEXT_FIELD = "embedding_input"


def load(path, part, include_fragments):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if part != "all" and r["part"] != part:
                continue
            if r.get("is_fragment") and not include_fragments:
                continue
            out.append(r)
    return out


def encode_hf(texts, model_name, batch, max_len, prefix="passage"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    is_e5 = "e5" in model_name.lower()
    print(f"[embed] device={dev} model={model_name} batches~{-(-len(texts)//batch)}")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            ch = texts[i:i + batch]
            if is_e5 and prefix:
                ch = [f"{prefix}: {t}" for t in ch]
            enc = tok(ch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            hid = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            vec = (hid * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.float().cpu().numpy())
            if (i // batch) % 20 == 0:
                print(f"  {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    return out


def encode_api(texts, base_url, model, batch):
    import urllib.request
    import numpy as np

    key = os.environ.get("EMBED_API_KEY", "")
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/embeddings",
            data=json.dumps({"model": model, "input": texts[i:i+batch]}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        vec = np.asarray([d["embedding"] for d in sorted(data["data"],
                          key=lambda d: d["index"])], dtype="float32")
        vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
        yield vec
        if (i // batch) % 20 == 0:
            print(f"  {min(i+batch, len(texts))}/{len(texts)}", flush=True)


def embed_query(text, model_name, prefix="query"):
    """Helper untuk sisi query (impor dari script ini agar model konsisten)."""
    import numpy as np
    return encode_hf([text], model_name, 1, 512, prefix=prefix)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--engine", choices=["hf", "api"], default="hf")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--part", choices=["all", "norma", "administratif"],
                    default="norma")
    ap.add_argument("--include-fragments", action="store_true",
                    help="indeks juga unit pendek ber-flag is_fragment")
    a = ap.parse_args()

    import numpy as np

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    ep, ip, mp = (od / "embeddings.npy", od / "ids.jsonl", od / "meta.json")
    recs = load(a.input, a.part, a.include_fragments)
    texts = [r[TEXT_FIELD] for r in recs]
    print(f"[embed] {len(texts)} record (part={a.part}, "
          f"fragments={'in' if a.include_fragments else 'skip'}) -> {ep}")

    t0 = time.time()
    if a.engine == "hf":
        vecs = np.concatenate(encode_hf(texts, a.model, a.batch_size, a.max_len))
    else:
        assert a.base_url, "mode API butuh --base-url"
        vecs = np.concatenate(list(encode_api(texts, a.base_url, a.model,
                                              a.batch_size)))
    np.save(ep, vecs)
    with open(ip, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: r.get(k) for k in
                                ("chunk_id", "parent_id", "document_id",
                                 "part", "content_role", "is_fragment",
                                 "doc_type", "year", "legal_force_axis")})
                    + "\n")
    json.dump({"model": a.model, "engine": a.engine, "part_filter": a.part,
               "fragments_included": bool(a.include_fragments),
               "text_field": TEXT_FIELD, "count": int(vecs.shape[0]),
               "dim": int(vecs.shape[1]), "normalized_l2": True,
               "query_note": "model e5: awali pertanyaan dengan 'query: ' "
                             "(helper embed_query tersedia)",
               "seconds": round(time.time() - t0, 1)},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[embed] selesai {vecs.shape} dalam {round(time.time()-t0)}s")
    print("Indeks: faiss.IndexFlatIP(dim) — vektor L2-normal, dot = cosine")


if __name__ == "__main__":
    main()
