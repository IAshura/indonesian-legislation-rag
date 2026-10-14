"""
verify_setup.py — Pemeriksaan kesehatan SEBELUM memakai folder ini.
===================================================================

Filosofi fail-fast proyek: lebih baik gagal di awal dengan pesan jelas
daripada gagal samar di tengah jawaban. Jalankan:

    python verify_setup.py              # cek inti (offline)
    python verify_setup.py --full       # + uji embedder (butuh ~10-30 dtk)
    python verify_setup.py --llm        # + 1 panggilan LLM (perlu API key)
    python verify_setup.py --milvus     # + ping Milvus (perlu Docker jalan)
"""

from __future__ import annotations

import argparse
import sys

# Konsol Windows kadang memakai cp1252 (tidak punya emoji); paksa UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

OK = "\u2705"   # ✅
BAD = "\u274c"  # ❌
hasil: list[tuple[bool, str]] = []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    hasil.append((kondisi, f"{OK if kondisi else BAD} {nama}" +
                  (f" — {detail}" if detail else "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="uji embedder juga")
    ap.add_argument("--llm", action="store_true", help="ping LLM (1 panggilan)")
    ap.add_argument("--milvus", action="store_true", help="ping Milvus")
    args = ap.parse_args()

    # 1) file inti ada
    import config
    for label, p in [("chunks.jsonl", config.CHUNKS_JSONL),
                     ("parents.jsonl", config.PARENTS_JSONL),
                     ("parents_windows.jsonl", config.PARENTS_WINDOWS_JSONL),
                     ("documents.jsonl", config.DOCUMENTS_JSONL),
                     ("embeddings.npy", config.EMB_NPY),
                     ("ids.jsonl", config.EMB_IDS),
                     ("meta.json", config.EMB_META)]:
        cek(f"file {label}", p.exists(), str(p.relative_to(config.PROJECT_ROOT)))

    # 2) konsistensi embeddings <-> ids
    try:
        from retriever import ExactRetriever
        r = ExactRetriever()
        cek("matriks embeddings", True,
            f"{r.count:,} x {r.dim}, mmap aktif")
        v = np.zeros(r.dim, dtype=np.float32)
        cek("pencarian exact", len(r.search(v, k=3)) == 3)
    except Exception as e:
        cek("embeddings/retriever", False, str(e)[:120])

    # 3) dataset join
    try:
        from dataset import dataset
        ds = dataset()
        ctx = ds.context_for_parent("uud-1945::p::1")
        cek("join parent utuh", ctx["char_len"] > 0,
            f"uud-1945::p::1 = {ctx['char_len']} char via {ctx['sumber']}")
        cek("lookup document", ds.get_document("uud-1945") is not None)
    except Exception as e:
        cek("dataset join", False, str(e)[:120])

    # 4) embedder (opsional — paling lambat)
    if args.full:
        try:
            from embedder import embed_query
            v = embed_query("uji koneksi model")
            cek("embedder Qwen3-0.6B", abs(np.linalg.norm(v) - 1.0) < 1e-3,
                f"dim {v.shape[0]}, norma {np.linalg.norm(v):.4f}")
        except Exception as e:
            cek("embedder", False, str(e)[:120])

    # 4b) reranker cross-encoder (opsional — paling lambat setelah embedder)
    if args.full:
        try:
            from reranker import skor_rerank
            sk = skor_rerank(
                "Di tangan siapa kedaulatan berada menurut UUD 1945?",
                ["Kedaulatan berada di tangan rakyat, dilaksanakan menurut "
                 "Undang-Undang Dasar.",
                 "Pajak kendaraan bermotor dikenakan peraturan daerah."],
                batch_size=2)
            if sk:
                cek("reranker Qwen3-0.6B",
                    sk[0] > sk[1] and 0 <= sk[1] < sk[0] <= 1,
                    f"P(yes) relevan={sk[0]:.3f} > tak-relevan={sk[1]:.3f}")
            else:
                cek("reranker", True,
                    "model tidak termuat — mode tanpa rerank (fallback aman)")
        except Exception as e:
            cek("reranker", False, str(e)[:120])

    # 5) LLM ping (opsional — perlu API key / server lokal)
    if args.llm:
        try:
            from llm_client import chat
            m = chat([{"role": "user", "content": "Balas satu kata: siap"}],
                     max_tokens=5)
            cek("LLM reachable", bool(m.get("content")),
                str(m.get("content"))[:60])
        except Exception as e:
            cek("LLM reachable", False, str(e)[:160])

    # 6) Milvus ping (opsional)
    if args.milvus:
        try:
            from pymilvus import MilvusClient
            cli = MilvusClient(uri="http://localhost:19530")
            koleksi = [c for c in cli.list_collections() if "legislation" in c]
            cek("Milvus", True, f"koleksi: {', '.join(koleksi) or '-'}")
        except Exception as e:
            cek("Milvus", False, str(e)[:120])

    print("\n".join(x[1] for x in hasil))
    gagal = sum(1 for ok, _ in hasil if not ok)
    print(f"\n{len(hasil) - gagal}/{len(hasil)} lolos"
          + (" — ada yang perlu dibereskan." if gagal else " — siap dipakai!"))
    raise SystemExit(1 if gagal else 0)


if __name__ == "__main__":
    main()
