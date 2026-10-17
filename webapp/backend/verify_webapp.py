"""
verify_webapp.py — Verifikasi webapp meniru pola verify_setup.py lama.
======================================================================

Cek bertahap (tiap cek cetak [OK]/[FAIL], exit code non-zero bila ada FAIL):
  1   struktur data (chunks/parents/lexical/embeddings meta)
  2   koneksi modul lama (import bridge -> config/retriever/bm25)
  3   BM25-only retrieval (BM25Scorer.search() langsung)
  4   Dense retrieval (ExactRetriever + embed_query)
  5   Hybrid retrieval (HybridRetriever -> RRF)
  6   engine.run_fixed dry-run (trace lengkap, tanpa LLM)
  7   engine.run_agent dry-run (tool_call trace, tanpa LLM)
  8   citations format lengkap (n/chunk_id/parent_id/skor...)
  9   endpoint HTTP /api/health, /api/models, /api/ask dry-run,
      /api/embedding/switch, /api/history (uvicorn dijalankan background)

Jalankan:   python verify_webapp.py [--skip-http]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CEK: list[tuple[str, bool]] = []


def cek(nama: str, ok: bool, detail: str = "") -> bool:
    CEK.append((nama, bool(ok)))
    print(f"{'[OK]  ' if ok else '[FAIL]'} {nama}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    print("=" * 72)
    print("VERIFIKASI WEBAPP RAG HUKUM INDONESIA")
    print("=" * 72)

    # ---------------------------------------------------------- 1. struktur
    root = HERE.parents[1]
    ok = all((root / "pipeline" / "result" / p).exists() for p in
             ("chunks.jsonl", "parents.jsonl", "documents.jsonl"))
    cek("1. file dataset pipeline ada", ok)

    lex = root / "pipeline" / "result" / "lexical" / "norma_tokens.txt"
    cek("1b. index leksikal BM25 ada", lex.exists())

    emb_root = root / "pipeline" / "result" / "embeddings"
    cek("1c. folder embeddings + ACTIVE ada",
        (emb_root / "ACTIVE").exists() and (emb_root / "qwen3-06b" / "meta.json").exists())

    # ---------------------------------------------------------- 2. bridge
    try:
        import bridge
        import config as mod_config
        cek("2. bridge import modul lama (config)", True,
            f"embedding aktif: {bridge.active_tag()}")
    except Exception as e:
        cek("2. bridge import modul lama (config)", False, str(e))
        return          # tidak bisa lanjut bila import gagal

    # ---------------------------------------------------------- 3. BM25-only
    try:
        from bm25 import BM25Scorer
        t0 = time.perf_counter()
        hits = BM25Scorer().search("kedaulatan di tangan siapa", k=5)
        cek("3. BM25Scorer.search() jalan", len(hits) == 5,
            f"{len(hits)} hit, {round((time.perf_counter() - t0) * 1000)} ms")
    except Exception as e:
        cek("3. BM25Scorer.search() jalan", False, str(e))

    # ---------------------------------------------------------- 4-5. dense/hybrid
    try:
        from embedder import embed_query
        from retriever import ExactRetriever, HybridRetriever, konteks_kaya
        v = embed_query("Di tangan siapa kedaulatan berada menurut UUD 1945?")
        d = ExactRetriever().search(v, k=3)
        cek("4. dense retrieval (ExactRetriever)", len(d) == 3)
        h = HybridRetriever().search(v, "kedaulatan", k=3)
        ctx = konteks_kaya(h)
        cek("5. hybrid retrieval (RRF)", len(ctx) == 3 and
            ctx[0].get("skor_fusi") is not None)
    except Exception as e:
        cek("4-5. dense/hybrid", False, str(e))

    # ---------------------------------------------------------- 6. fixed dry-run
    try:
        import engine
        hasil = engine.run_fixed("Di tangan siapa kedaulatan berada menurut UUD 1945?",
                                 retrieval="hybrid", dry_run=True)
        tipe = [n["type"] for n in hasil["trace"]]
        wajib = {"user_ask", "embed_query", "retrieve_dense", "retrieve_bm25",
                 "fuse_rrf", "join_parent", "build_prompt", "llm_answer"}
        cek("6. engine.run_fixed dry-run (trace lengkap)",
            wajib <= set(tipe) and hasil["kandidat"], " → ".join(tipe))
        s1 = hasil["kandidat"][0] if hasil["kandidat"] else {}
        pasal1 = "pasal" in (s1.get("citation_str") or "").lower() and \
                 "::p::1" in (s1.get("parent_id") or "")
        cek("6b. demo kedaulatan: [S1] = Pasal 1 UUD",
            pasal1, str(s1.get("citation_str")))
    except Exception as e:
        cek("6. engine.run_fixed dry-run", False, str(e))

    # ---------------------------------------------------------- 7. agent dry-run
    try:
        hasil = engine.run_agent("kedaulatan UUD 1945", dry_run=True)
        tipe = [n["type"] for n in hasil["trace"]]
        cek("7. engine.run_agent dry-run (tool_call trace)",
            "tool_call" in tipe and "llm_answer" in tipe, " → ".join(tipe))
    except Exception as e:
        cek("7. engine.run_agent dry-run", False, str(e))

    # ---------------------------------------------------------- 8. format citations
    try:
        import app as mod_app
        c = mod_app._citations_dari_kandidat(hasil_dummy(hasil))
        kunci = {"n", "chunk_id", "parent_id", "citation_str",
                 "official_title", "page", "lines", "score",
                 "skor_bm25", "skor_fusi"}
        cek("8. format citations lengkap", bool(c) and kunci <= set(c[0]))
    except Exception as e:
        cek("8. format citations", False, str(e))

    # ---------------------------------------------------------- 9. endpoint HTTP
    if "--skip-http" in sys.argv:
        cek("9. endpoint HTTP", True, "(dilewati --skip-http)")
    else:
        cek_endpoint_http()

    # ---------------------------------------------------------- ringkasan
    gagal = [n for n, ok in CEK if not ok]
    print("\n" + "=" * 72)
    if gagal:
        print(f"GAGAL {len(gagal)}/{len(CEK)}: " + "; ".join(gagal))
        sys.exit(1)
    print(f"SEMUA {len(CEK)} CEK LOLOS")


def hasil_dummy(hasil: dict) -> list[dict]:
    """Kandidat terakhir bermanfaat untuk uji format (agent dry-run)."""
    from retriever import konteks_kaya
    from embedder import embed_query
    from retriever import ExactRetriever
    v = embed_query("kedaulatan")
    return konteks_kaya(ExactRetriever().search(v, k=2))


def cek_endpoint_http() -> None:
    import requests
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", "8017"],
        cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:8017"
        # tunggu server siap maksimum 30 dtk
        for _ in range(60):
            try:
                if requests.get(f"{base}/api/health", timeout=2).ok:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            cek("9. endpoint HTTP", False, "server tidak siap")
            return
        cek("9a. GET /api/health", True)
        m = requests.get(f"{base}/api/models", timeout=10).json()
        cek("9b. GET /api/models", m.get("aktif") in ("qwen3-06b", "qwen3-4b"),
            f"aktif={m.get('aktif')}")
        ask = requests.post(f"{base}/api/ask", json={
            "question": "Di tangan siapa kedaulatan berada menurut UUD 1945?",
            "mode": "fixed", "retrieval": "hybrid", "dry_run": True,
        }, timeout=120).json()
        cek("9c. POST /api/ask dry-run", bool(ask.get("citations")) and
            ask.get("trace"),
            f"citations={len(ask.get('citations', []))}, "
            f"trace_nodes={len(ask.get('trace', []))}")
        sw = requests.post(f"{base}/api/embedding/switch",
                           json={"tag": "qwen3-06b"}, timeout=30).json()
        cek("9d. POST /api/embedding/switch", sw.get("ok") is True)
        h = requests.get(f"{base}/api/history", timeout=10).json()
        ada = bool(h.get("daftar"))
        cek("9e. GET /api/history", ada)
        if ada:
            rid = h["daftar"][0]["id"]
            ex = requests.get(f"{base}/api/history/{rid}/export?format=md",
                              timeout=10)
            cek("9f. export md", ex.ok and "[S1]" in ex.text)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
