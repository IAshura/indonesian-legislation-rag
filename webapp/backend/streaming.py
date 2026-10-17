"""
streaming.py — Varian STREAMING dari engine.py (SSE untuk webapp).
===============================================================

KENAPA FILE INI ADA?
  /api/ask biasa menunggu SEMUA selesai (embed -> retrieve -> rerank ->
  LLM) baru menjawab sekaligus. Di UI itu jelek: pesan user pun baru
  muncul setelah selesai, tidak ada indikator proses, jawaban mengagetkan.

  Generator di sini mengeluarkan event sesuai kejadian NYATA (bukan
  persen palsu):
    {"t": "status", "text": "..."}  — tahap tak-terukur mulai (embed,
                                       rerank batch, "LLM berpikir…")
    {"t": "node", "node": {...}}    — node trace baru (muncul di Graf live)
    {"t": "token", "text": "..."}   — potongan jawaban LLM (SSE asli)
    {"t": "done", ...}              — hasil final (dirakit app.py)
    {"t": "error", "message": ...}  — gagal di tengah jalan

PRINSIP ANTI-DUPLIKASI RUMUS:
  Tidak ada rumus retrieval yang ditulis ulang. Semua perhitungan tetap
  memanggil helper engine.py (dense/BM25/RRF/rerank/small-to-big/prompt)
  dan modul lama. File ini hanya MEMOTONG alur menjadi generator +
  mengganti SATU panggilan LLM blocking menjadi SSE.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend"))

import bridge  # noqa: E402
import config  # noqa: E402
import engine  # noqa: E402
import llm_provider  # noqa: E402
from dataset import dataset  # noqa: E402
from embedder import embed_query  # noqa: E402
from prompt import build_context_block, build_messages  # noqa: E402
from retriever import konteks_kaya  # noqa: E402
from reranker import rerank_kandidat  # noqa: E402


# ---------------------------------------------------------- SSE client
def chat_stream_iter(messages: list[dict], llm: dict | None,
                     tools: list[dict] | None = None,
                     temperature: float | None = None,
                     max_tokens: int | None = None):
    """Chat completion STREAMING (OpenAI-compatible SSE) sebagai generator.

    Mengeluarkan {"t": "token", "text": "..."} SETIAP potongan tiba,
    {"t": "usage", "usage": {...}} bila provider mengirimnya (butuh
    stream_options.include_usage — tanpa itu chunk usage tak ada),
    lalu SATU event {"t": "message", "message": {...}} di akhir berisi
    message LENGKAP seperti chat() biasa:
      {"role": "assistant", "content": str|None, "tool_calls": [...]|None}
    (untuk mode agent: content antar-putaran biasanya None, tool_calls ada).
    """
    import requests

    provider = (llm or {}).get("provider", "custom")
    base_url, api_key, model = llm_provider.validasi(llm)
    if not base_url:
        raise RuntimeError("live LLM butuh llm.base_url (provider preset/custom)")
    if not api_key and provider != "ollama":
        raise RuntimeError("live LLM butuh llm.api_key")
    if provider == "ollama" and not api_key:
        api_key = "ollama"

    import llm_adapter
    if llm_adapter.rasa(provider, base_url) == "anthropic":
        yield from llm_adapter.stream_anthropic_iter(
            base_url, api_key, model, messages, tools,
            temperature, max_tokens)
        return

    import os as _os
    payload: dict = {"model": model, "messages": messages,
                     "stream": True,
                     # TANPA ini provider tidak mengirim chunk usage via SSE
                     "stream_options": {"include_usage": True},
                     "temperature": (temperature if temperature is not None
                                     else float(_os.environ.get(
                                         "LLM_TEMPERATURE", "0.2"))),
                     "max_tokens": (max_tokens if max_tokens is not None
                                    else int(_os.environ.get(
                                        "LLM_MAX_TOKENS", "1024")))}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json",
               "Accept": "text/event-stream"}
    try:
        resp = requests.post(url, json=payload, headers=headers,
                             stream=True, timeout=(30, 300))
    except Exception as e:
        raise RuntimeError(f"LLM tak terjangkau ({base_url}): {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} dari LLM: "
            f"{resp.text[:300] if not resp.raw else ''}")

    content: list[str] = []
    nalar: list[str] = []
    tc_acc: dict[int, dict] = {}
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # chunk usage (biasanya TERAKHIR, choices kosong)
            if chunk.get("usage"):
                yield {"t": "usage",
                       "usage": llm_provider.normalisasi_usage(chunk["usage"])}
                continue
            try:
                delta = chunk["choices"][0].get("delta", {})
            except (KeyError, IndexError):
                continue
            # penalaran model reasoning (DeepSeek Reasoner dkk;
            # model chat biasa tidak mengirim field ini -> kosong wajar)
            ragu = (delta.get("reasoning_content")
                    or delta.get("reasoning") or "")
            if ragu:
                nalar.append(ragu)
                yield {"t": "reasoning", "text": ragu}
            teks = delta.get("content") or ""
            if teks:
                content.append(teks)
                yield {"t": "token", "text": teks}
            for tc in delta.get("tool_calls") or []:
                i = int(tc.get("index", 0))
                slot = tc_acc.setdefault(i, {"id": "", "name": "",
                                             "arguments": ""})
                slot["id"] += tc.get("id") or ""
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
    finally:
        resp.close()

    tool_calls = None
    if tc_acc:
        tool_calls = [{"id": s["id"] or f"call_{i}", "type": "function",
                       "function": {"name": s["name"],
                                    "arguments": s["arguments"]}}
                      for i, s in sorted(tc_acc.items())]
    isi = "".join(content) or None
    yield {"t": "message",
           "message": {"role": "assistant", "content": isi,
                       "tool_calls": tool_calls,
                       "reasoning": "".join(nalar) or None}}


# ---------------------------------------------------------- fixed stream
def run_fixed_stream(question: str, *, retrieval: str = "hybrid",
                     rerank: bool = True, top_k: int = 8,
                     inject_k: int | None = None,
                     doc_type=None, year: int | None = None,
                     dry_run: bool = False, llm: dict | None = None,
                     riwayat_chat=None, dense_method: str = "exact",
                     dense_ef: int = 64, dense_nprobe: int = 16,
                     gaya: str = "standar",
                     temperature: float | None = None,
                     max_tokens: int | None = None,
                     sumber: str = "user",
                     nomor_awal: int = 1,
                     node_awal: int = 1):
    """Generator event fixed-pipeline. Tahap retrieval SAMA PERSIS dengan
    engine.run_fixed (memakai helper yang sama); bedanya tiap node trace
    dikeluarkan langsung + jawaban LLM dialirkan per token.
    sumber "llm" untuk sub-pencarian tool agent (node llm_ask)."""
    t0 = time.perf_counter()
    trace = engine.Trace(awal=node_awal)
    metode = (dense_method or "exact").strip().lower()
    if metode not in engine.METODE_DENSE_VALID:
        metode = "exact"
    tset = engine._types(doc_type)
    inject_n = max(1, int(inject_k or top_k))
    per_jalur = max(top_k, config.RERANK_POOL_SIZE) if rerank else top_k

    tipe_tanya = "llm_ask" if sumber == "llm" else "user_ask"
    awalan_tanya = "Tanya LLM: " if sumber == "llm" else "Pertanyaan: "
    trace.add(tipe_tanya, f"{awalan_tanya}{engine._ringkas(question, 100)}",
              {"mode": "fixed", "question": question,
               "settings": {"retrieval": retrieval, "rerank": rerank,
                            "top_k": top_k, "inject_k": inject_n,
                            "doc_type": tset or None, "year": year,
                            "dry_run": dry_run, "dense_method": metode,
                            "gaya": gaya,
                            "nomor_awal": max(1, int(nomor_awal or 1))}})
    yield {"t": "node", "node": trace.node[-1]}

    if retrieval == "bm25":
        yield {"t": "status", "text": "Mencari kata kunci (BM25)…"}
        t1 = time.perf_counter()
        scorer = engine._bm25()
        meta_baris = engine._baris_ids_meta()
        merged: dict[str, dict] = {}
        for t in (tset or [None]):
            hits = scorer.search(question, k=per_jalur)
            seen: set[str] = set()
            rank = 0
            for baris, s in hits:
                m = meta_baris[baris]
                if t and m.get("doc_type") != t:
                    continue
                if year and int(m.get("year") or 0) != year:
                    continue
                if m["parent_id"] in seen:
                    continue
                seen.add(m["parent_id"])
                rank += 1
                k = dict(m)
                k.update({"score": 0.0, "skor_bm25": round(s, 4),
                          "skor_fusi": None, "rank": rank})
                cid = m["chunk_id"]
                prev = merged.get(cid)
                if prev is None or k["skor_bm25"] > (prev.get("skor_bm25") or 0):
                    merged[cid] = k
        hasil = sorted(merged.values(), key=lambda x: -(x.get("skor_bm25") or 0))
        kandidat = konteks_kaya(hasil)
        trace.add("retrieve_bm25", f"BM25 leksikal top-{per_jalur}",
                  {"query": question, "top_k": per_jalur,
                   "detik": round(time.perf_counter() - t1, 3),
                   "hasil": [engine._trace_singkat(k, rank=i + 1)
                             for i, k in enumerate(kandidat)]})
        yield {"t": "node", "node": trace.node[-1]}
    else:
        yield {"t": "status", "text": "Meng-embed pertanyaan…"}
        t1 = time.perf_counter()
        trace.add("embed_query", f"Embed query ({bridge.active_tag()})",
                  {"query": question, "model": bridge.active_tag(),
                   "detik": round(time.perf_counter() - t1, 3)})
        yield {"t": "node", "node": trace.node[-1]}
        vektor = embed_query(question)

        label_dense = {"exact": "Dense exact (numpy)",
                       "hnsw": "Dense HNSW (Milvus)",
                       "ivf": "Dense IVF (Milvus)",
                       "flat": "Dense exact (Milvus FLAT)"}[metode]
        yield {"t": "status",
               "text": f"Mencari kandidat ({label_dense}"
                       f"{' + BM25 → RRF' if retrieval == 'hybrid' else ''})…"}
        t2 = time.perf_counter()
        if retrieval == "hybrid":
            hasil = []
            for t in (tset or [None]):
                if metode == "exact":
                    hasil.extend(engine._hybrid().search(
                        vektor, question, k=per_jalur, doc_type=t, year=year))
                else:
                    hasil.extend(engine._hybrid_cari(
                        vektor, question, k=per_jalur, per_jalur=per_jalur,
                        method=metode, ef=dense_ef, nprobe=dense_nprobe,
                        doc_type=t, year=year))
            hasil = engine._merge_best_per_parent(hasil, "skor_fusi")
            kandidat = konteks_kaya(hasil)
            trace.add("retrieve_dense", f"{label_dense} top-{per_jalur}",
                      {"query": question, "index_method": metode,
                       "hasil": [engine._trace_singkat(k, rank=i + 1,
                                                      cosine=k["score"])
                                 for i, k in enumerate(kandidat)
                                 if k.get("score")]})
            yield {"t": "node", "node": trace.node[-1]}
            trace.add("retrieve_bm25", f"BM25 leksikal top-{per_jalur}",
                      {"query": question,
                       "hasil": [engine._trace_singkat(k, rank=i + 1,
                                                      skor_bm25=k.get("skor_bm25"))
                                 for i, k in enumerate(kandidat)
                                 if k.get("skor_bm25")]})
            yield {"t": "node", "node": trace.node[-1]}
            trace.add("fuse_rrf", f"Fusi RRF (K={config.RRF_K})",
                      {"rrf_k": config.RRF_K,
                       "hasil": [engine._trace_singkat(k, rank=i + 1,
                                                      skor_fusi=k.get("skor_fusi"))
                                 for i, k in enumerate(kandidat)]})
            yield {"t": "node", "node": trace.node[-1]}
        else:
            hasil = []
            for t in (tset or [None]):
                hasil.extend(engine._dense_cari(
                    vektor, k=per_jalur, method=metode, ef=dense_ef,
                    nprobe=dense_nprobe, doc_type=t, year=year))
            hasil = engine._merge_best_per_parent(hasil, "score")
            kandidat = konteks_kaya(hasil)
            trace.add("retrieve_dense", f"{label_dense} top-{per_jalur}",
                      {"query": question, "index_method": metode,
                       "detik": round(time.perf_counter() - t2, 3),
                       "hasil": [engine._trace_singkat(k, rank=i + 1,
                                                      cosine=k["score"])
                                 for i, k in enumerate(kandidat)]})
            yield {"t": "node", "node": trace.node[-1]}

    if not kandidat:
        yield {"t": "done", "jawaban": "Tidak ditemukan dasar dalam korpus.",
               "kandidat": [], "mode": "kosong", "reasoning": None,
               "usage": {"jumlah_kandidat": 0,
                         "detik": round(time.perf_counter() - t0, 3),
                         "tokens": llm_provider.normalisasi_usage(None)}}
        return

    if rerank:
        yield {"t": "status", "text": "Me-rerank kandidat…"}
        t3 = time.perf_counter()
        trace.add("rerank", "Rerank cross-encoder",
                  {"model": config.RERANK_MODEL, "pool": len(kandidat),
                   "top_n": inject_n})
        yield {"t": "node", "node": trace.node[-1]}
        kandidat = rerank_kandidat(question, kandidat, top_n=inject_n)
        trace.node[-1]["detail"].update({
            "detik": round(time.perf_counter() - t3, 3),
            "hasil": [engine._trace_singkat(k, rank=i + 1,
                                            score_rerank=k.get("score_rerank"))
                      for i, k in enumerate(kandidat)]})
        yield {"t": "node", "node": trace.node[-1]}

    kandidat = kandidat[:inject_n]
    trace.add("join_parent", "Join parent (small-to-big)",
              {"kandidat": [{"parent_id": k["parent_id"],
                             "citation_str": k.get("citation_str"),
                             "official_title": k.get("official_title"),
                             "sumber": k.get("sumber"),
                             "char_len": len(k.get("text") or "")}
                            for k in kandidat]})
    yield {"t": "node", "node": trace.node[-1]}

    trace.add("build_prompt", "Build prompt",
              {"query": question, "anggaran_char": config.CONTEXT_CHAR_BUDGET})
    blok = build_context_block(kandidat, mulai=max(1, int(nomor_awal or 1)))
    messages = build_messages(question, blok, gaya=gaya)
    n_memori, tok_memori = 0, 0
    if riwayat_chat:
        messages, n_memori, tok_memori = engine._sisip_riwayat(
            messages, riwayat_chat)
    trace.node[-1]["detail"]["prompt"] = {
        "system": messages[0]["content"], "context": blok,
        "user": messages[-1]["content"], "char_len": len(blok),
        "chat_memori": n_memori, "chat_memori_token": tok_memori}
    yield {"t": "node", "node": trace.node[-1]}

    if dry_run:
        from llm_client import dry_run as _dry
        trace.add("llm_answer", "Dry-run (tidak memanggil LLM)",
                  {"payload": engine._ringkas(_dry(messages), 3000)})
        yield {"t": "node", "node": trace.node[-1]}
        teks = ("[DRY-RUN] Tidak memanggil LLM. Lihat konteks & prompt di "
                "panel Graf (node build_prompt).")
        yield {"t": "token", "text": teks}
        yield {"t": "done", "jawaban": teks, "kandidat": kandidat,
               "mode": "dry-run", "reasoning": None,
               "usage": {"jumlah_kandidat": len(kandidat),
                         "detik": round(time.perf_counter() - t0, 3),
                         "context_chars": len(blok),
                         "tokens": llm_provider.normalisasi_usage(None)}}
        return

    yield {"t": "status", "text": "LLM menulis jawaban…"}
    trace.add("llm_answer", "LLM live", {**engine._llm_detail(llm),
                                         "mode": "live"})
    yield {"t": "node", "node": trace.node[-1]}
    msg: dict = {}
    tokens = llm_provider.normalisasi_usage(None)
    nalarParts: list[str] = []
    try:
        for ev in chat_stream_iter(messages, llm, temperature=temperature,
                                   max_tokens=max_tokens):
            if ev["t"] == "token":
                yield ev
            elif ev["t"] == "usage":
                tokens = llm_provider.tambah_usage(tokens, ev["usage"])
                yield ev  # teruskan agar UI bisa tampilkan live
            elif ev["t"] == "reasoning":
                nalarParts.append(ev.get("text") or "")
                yield ev  # panel penalaran live di UI
            else:
                msg = ev["message"]
    except Exception as e:
        yield {"t": "error", "message": _saring_error(str(e), llm)}
        return
    jawaban = (msg.get("content") or "").strip() or "(jawaban kosong)"
    trace.node[-1]["detail"]["jawaban_panjang"] = len(jawaban)
    trace.node[-1]["detail"]["tokens"] = tokens
    # Utamakan teks final provider (utuh & bersih); gabungan live
    # hanya cadangan bila final tak ada.
    mentah = (msg.get("reasoning") or "".join(nalarParts) or "")[:6000]
    nalar = llm_provider.rapikan_nalar(
        msg.get("reasoning") or "".join(nalarParts))
    if nalar:
        trace.node[-1]["detail"]["penalaran"] = nalar
        trace.node[-1]["detail"]["penalaran_mentah"] = mentah
    yield {"t": "done", "jawaban": jawaban, "kandidat": kandidat,
           "reasoning": nalar, "reasoning_raw": mentah if nalar else None,
           "mode": "live",
           "usage": {"jumlah_kandidat": len(kandidat),
                     "detik": round(time.perf_counter() - t0, 3),
                     "context_chars": len(blok), "tokens": tokens}}


def _saring_error(msg: str, llm: dict | None) -> str:
    if llm and llm.get("api_key") and llm["api_key"] in msg:
        msg = msg.replace(llm["api_key"], "***")
    return msg


# ---------------------------------------------------------- agent stream
def run_agent_stream(question: str, *, retrieval: str = "hybrid",
                     rerank: bool = True, top_k: int = 8,
                     inject_k: int | None = None,
                     doc_type=None, year: int | None = None,
                     dry_run: bool = False, llm: dict | None = None,
                     riwayat_chat=None, dense_method: str = "exact",
                     dense_ef: int = 64, dense_nprobe: int = 16,
                     gaya: str = "standar",
                     temperature: float | None = None,
                     max_tokens: int | None = None,
                     paksa_cari: bool = True,
                     nomor_awal: int = 1,
                     node_awal: int = 1):
    """Loop agent seperti engine.run_agent, tapi tiap tool_call disiarkan
    langsung + jawaban final dialirkan per token."""
    import tools as mod_tools

    t0 = time.perf_counter()
    trace = engine.Trace(awal=node_awal)
    metode = (dense_method or "exact").strip().lower()
    if metode not in engine.METODE_DENSE_VALID:
        metode = "exact"
    # paksa_cari hanya untuk pertanyaan HUKUM (lihat engine.butuh_cari).
    wajib = bool(paksa_cari) and engine.butuh_cari(question)
    trace.add("user_ask", f"Pertanyaan: {engine._ringkas(question, 100)}",
              {"mode": "agent", "question": question,
               "settings": {"retrieval": retrieval, "rerank": rerank,
                            "top_k": top_k, "inject_k": inject_k,
                            "doc_type": engine._types(doc_type) or None,
                            "year": year, "dry_run": dry_run,
                            "dense_method": metode, "gaya": gaya,
                            "paksa_cari": paksa_cari,
                            "wajib_cari": wajib,
                            "nomor_awal": max(1, int(nomor_awal or 1))}})
    yield {"t": "node", "node": trace.node[-1]}

    def _execute_wrapped(nama: str, argumen_json: str) -> str:
        try:
            a = json.loads(argumen_json or "{}")
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"argumen bukan JSON valid: {e}"})
        if nama == "search_norma":
            sub_trace = engine.Trace()
            sub = engine.run_fixed(
                a.get("question") or question,
                retrieval=retrieval, rerank=rerank,
                top_k=int(a.get("top_k", 5) or 5), inject_k=inject_k,
                doc_type=(a.get("doc_type") or doc_type),
                year=(a.get("year") or year),
                dry_run=True, trace=sub_trace,
                dense_method=metode, dense_ef=dense_ef,
                dense_nprobe=dense_nprobe, sumber="llm")
            # Penomoran lanjut (bukan ulang dari n1): sub-trace digabung ke
            # trace utama dengan id diteruskan, agar Graf linear n1..nN.
            # llm_ask + llm_answer (Dry-run) milik sub-run DISARING: sudah
            # diwakili node tool_call, kalau ikut Graf jadi pecah berantakan.
            for n in sub_trace.node:
                if n["type"] in ("user_ask", "llm_ask", "llm_answer"):
                    continue
                n["id"] = trace.nomor_berikut()
                trace.node.append(n)
                node_out.append({"t": "node", "node": n})
            kandidat = sub["kandidat"]
            out = _beri_nomor(kandidat)
            return json.dumps({"daftar": out,
                               "trace_reguler": sub["trace"][-3:]},
                              ensure_ascii=False)
        return mod_tools.execute_tool(nama, argumen_json)

    _execute_asli = mod_tools.execute_tool
    mod_tools.execute_tool = _execute_wrapped
    node_out: list[dict] = []
    nomor_sumber = [max(1, int(nomor_awal or 1)) - 1]
    terlihat: set = set()
    semua_sumber: list = []

    def _beri_nomor(daftar: list) -> list:
        keluar = []
        for k in daftar or []:
            if k.get("chunk_id") in terlihat:
                continue
            terlihat.add(k.get("chunk_id"))
            nomor_sumber[0] += 1
            item = {"n": nomor_sumber[0],
                    "chunk_id": k.get("chunk_id"),
                    "parent_id": k.get("parent_id"),
                    "citation_str": k.get("citation_str"),
                    "official_title": k.get("official_title"),
                    "page": k.get("page"), "lines": k.get("lines"),
                    "score": k.get("score"), "skor_bm25": k.get("skor_bm25"),
                    "skor_fusi": k.get("skor_fusi"),
                    "text": engine._ringkas(k.get("text"), 600)}
            keluar.append(item)
            semua_sumber.append(item)
        return keluar

    from prompt import instruksi_gaya as _gaya
    sistem = ("Kamu asisten hukum Indonesia. Gunakan tools yang tersedia "
              "untuk mencari pasal SEBELUM menjawab. Jawab hanya dari "
              "hasil tool. Setiap hasil tool SUDAH bernomor sitasi "
              "[Sn] sesuai field \"n\" tiap item daftar (nomor melanjutkan sesi, bisa mulai bukan dari 1) — "
              "pakai nomor itu PERSIS di akhir pernyataan hukummu, "
              "misal \"... rakyat [S1].\" DILARANG mengarang nomor "
              "sendiri dan DILARANG sitasi dalam kurung biasa. "
              "Jika tool tidak menemukan dasar, katakan jujur.\n"
              "DEKOMPOSISI: bila pertanyaan mengandung beberapa sub-pertanyaan "
              "berbeda, PECAH menjadi beberapa panggilan search_norma "
              "terpisah — satu per sub-pertanyaan, boleh sekaligus dalam "
              "satu blok tool_calls. JANGAN gabung menjadi satu query "
              "campuran. Contoh BENAR untuk pertanyaan 3-bagian: "
              "[search_norma({\"question\": \"Bentuk negara Indonesia "
              "menurut UUD 1945\"}), search_norma({\"question\": "
              "\"Pemegang kekuasaan membentuk UU menurut UUD 1945\"}), "
              "search_norma({\"question\": \"Jaminan beragama menurut "
              "UUD 1945\"})]. Contoh SALAH: satu search_norma dengan "
              "question gabungan ketiganya.\n"
              + _gaya(gaya))
    if wajib:
        sistem += ("\nATURAN TAMBAHAN: kamu WAJIB memanggil tool search_norma "
                   "minimal SATU KALI sebelum menjawab final, walaupun jawaban "
                   "tampaknya sudah ada di riwayat percakapan. Sitasi hanya "
                   "sah dari hasil tool putaran ini.")
    messages: list[dict] = [
        {"role": "system", "content": sistem},
        {"role": "user", "content": question},
    ]
    if riwayat_chat:
        messages, _, _ = engine._sisip_riwayat(messages, riwayat_chat)

    MAX_ITERATIONS = 5
    jawaban = ""
    tokens = llm_provider.normalisasi_usage(None)
    nalar_semua: list[str] = []
    try:
        if dry_run:
            from llm_client import MockLLM
            llm_ob = MockLLM([
                {"tool_calls": [{"name": "search_norma",
                                 "arguments": {"question": question,
                                               "top_k": min(3, top_k)}}]},
                {"content": "[DRY-RUN agent] MockLLM menjawab final berdasarkan "
                            "hasil tool. Node tool_call di bawah mencatat "
                            "pencarian sungguhan yang dijalankan oleh kode."},
            ])
        else:
            llm_ob = None

        ulangi_kosong = 0  # retry bila model balas kosong tanpa tool-call
        for putaran in range(1, MAX_ITERATIONS + 1):
            yield {"t": "status",
                   "text": f"LLM berpikir (putaran {putaran})…"}
            if llm_ob is None:
                balasan = {}
                try:
                    for ev in chat_stream_iter(messages, llm,
                                               tools=mod_tools.TOOLS,
                                               temperature=temperature,
                                               max_tokens=max_tokens):
                        if ev["t"] == "token":
                            yield ev
                        elif ev["t"] == "usage":
                            tokens = llm_provider.tambah_usage(
                                tokens, ev["usage"])
                            yield ev
                        elif ev["t"] == "reasoning":
                            # JANGAN diakumulasi di sini: potongan live sudah
                            # diteruskan apa adanya ke UI; yang disimpan ialah
                            # teks final per putaran di bawah (utuh & bersih).
                            yield ev
                        else:
                            balasan = ev["message"]
                except Exception as e:
                    yield {"t": "error",
                           "message": _saring_error(str(e), llm)}
                    return
            else:
                balasan = llm_ob.chat(messages, tools=mod_tools.TOOLS)
            while node_out:
                yield node_out.pop(0)
            tool_calls = balasan.get("tool_calls")
            isi_balasan = (balasan.get("content") or "").strip()
            if balasan.get("reasoning"):
                nalar_semua.append(balasan["reasoning"])
            if not tool_calls and not isi_balasan and llm_ob is None:
                # Model kadang balas kosong tanpa tool-call (teramati pada
                # model kecil via gateway). Coba sekali lagi, lalu jatuh ke
                # pencarian langsung agar tak pernah "(jawaban kosong)".
                if ulangi_kosong < 1:
                    ulangi_kosong += 1
                    yield {"t": "status",
                           "text": "Respons kosong, mencoba lagi…"}
                    continue
                yield {"t": "status",
                       "text": "Beralih ke pencarian langsung…"}
                sub_trace = engine.Trace()
                sub = engine.run_fixed(
                    question, retrieval=retrieval, rerank=rerank,
                    top_k=top_k, inject_k=inject_k,
                    doc_type=doc_type, year=year,
                    dry_run=dry_run, llm=llm, trace=sub_trace,
                    dense_method=metode, dense_ef=dense_ef,
                    dense_nprobe=dense_nprobe, gaya=gaya,
                    temperature=temperature, max_tokens=max_tokens,
                    sumber="llm")
                for n in sub_trace.node:
                    if n["type"] in ("user_ask", "llm_ask", "llm_answer"):
                        continue
                    n["id"] = trace.nomor_berikut()
                    trace.node.append(n)
                    yield {"t": "node", "node": n}
                jawaban = sub["jawaban"]
                # Petakan sitasi lokal fixed ke nomor global (lihat engine).
                import re as _re
                peta: dict[int, int] = {}
                for i, k in enumerate(sub["kandidat"] or [], 1):
                    temu = next((s for s in semua_sumber
                                 if s["chunk_id"] == k.get("chunk_id")), None)
                    if temu is None:
                        nomor_sumber[0] += 1
                        item = {"n": nomor_sumber[0],
                                "chunk_id": k.get("chunk_id"),
                                "parent_id": k.get("parent_id"),
                                "citation_str": k.get("citation_str"),
                                "official_title": k.get("official_title"),
                                "page": k.get("page"),
                                "lines": k.get("lines"),
                                "score": k.get("score"),
                                "text": engine._ringkas(k.get("text"), 600)}
                        semua_sumber.append(item)
                        peta[i] = nomor_sumber[0]
                    else:
                        peta[i] = temu["n"]
                for i, g in peta.items():
                    jawaban = _re.sub(r"\[S" + str(i) + r"\]",
                                      f"\x00{g}\x00", jawaban)
                for g in peta.values():
                    jawaban = jawaban.replace(f"\x00{g}\x00", f"[S{g}]")
                tokens = llm_provider.tambah_usage(
                    tokens, (sub.get("usage") or {}).get("tokens"))
                yield {"t": "token", "text": jawaban}
                trace.add("llm_answer",
                          f"Jawaban fallback fixed (putaran {putaran})",
                          {"teks": engine._ringkas(jawaban, 2000),
                           "jumlah_putaran": putaran})
                yield {"t": "node", "node": trace.node[-1]}
                break
            if not tool_calls:
                jawaban = isi_balasan or "(jawaban kosong)"
                if llm_ob is not None:
                    # MockLLM: tidak ada token live, kirim sekaligus
                    yield {"t": "token", "text": jawaban}
                trace.add("llm_answer", f"Jawaban final (putaran {putaran})",
                          {"teks": engine._ringkas(jawaban, 2000),
                           "jumlah_putaran": putaran})
                yield {"t": "node", "node": trace.node[-1]}
                break
            messages.append({"role": "assistant", "content": balasan.get("content"),
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                nama = tc["function"]["name"]
                argumen = tc["function"]["arguments"]
                yield {"t": "status",
                       "text": f"Memanggil tool {nama}…"}
                t1 = time.perf_counter()
                try:
                    hasil_tool = mod_tools.execute_tool(nama, argumen)
                except Exception as e:
                    hasil_tool = json.dumps({"error": str(e)},
                                            ensure_ascii=False)
                detail_extra = {"tool": nama,
                                "argumen": engine._ringkas(argumen, 400),
                                "detik": round(time.perf_counter() - t1, 3)}
                try:
                    parsed = json.loads(hasil_tool)
                    if isinstance(parsed, dict) and parsed.get("trace_reguler"):
                        detail_extra["retrieval_subtrace"] = parsed["trace_reguler"]
                        detail_extra["daftar_hasil"] = parsed.get("daftar", [])
                        hasil_render = json.dumps(
                            {"daftar": parsed.get("daftar", [])},
                            ensure_ascii=False)
                    else:
                        hasil_render = hasil_tool
                except (json.JSONDecodeError, TypeError):
                    hasil_render = hasil_tool
                detail_extra["hasil"] = engine._ringkas(hasil_render, 2000)
                # node sub-pencarian DULU (terjadi sebelum tool_call dicatat),
                # baru node tool_call — agar urutan kronologis n1,n2,...
                while node_out:
                    yield node_out.pop(0)
                trace.add("tool_call", f"{nama} (putaran {putaran})",
                          dict(detail_extra))
                yield {"t": "node", "node": trace.node[-1]}
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": hasil_tool})
        else:
            # Batas putaran habis: panggilan pamungkas TANPA tool agar
            # user tetap dapat jawaban seadanya (bukan pesan teknis).
            messages.append({
                "role": "user",
                "content": "Batas pencarian tercapai. Jawab pertanyaan awal "
                           "SEADANYA dari hasil tool di atas dengan sitasi "
                           "[Sn] yang sudah ada. Jika tidak cukup untuk "
                           "menjawab, katakan: Tidak ditemukan dasar "
                           "dalam korpus."})
            if llm_ob is None:
                yield {"t": "status",
                       "text": "Menyusun jawaban pamungkas…"}
                pamungkas: dict = {}
                try:
                    for ev in chat_stream_iter(messages, llm,
                                               temperature=temperature,
                                               max_tokens=max_tokens):
                        if ev["t"] == "token":
                            yield ev
                        elif ev["t"] == "usage":
                            tokens = llm_provider.tambah_usage(
                                tokens, ev["usage"])
                            yield ev
                        elif ev["t"] == "reasoning":
                            nalar_semua.append(ev.get("text") or "")
                            yield ev
                        else:
                            pamungkas = ev["message"]
                except Exception as e:
                    yield {"t": "error",
                           "message": _saring_error(str(e), llm)}
                    return
                if pamungkas.get("reasoning"):
                    nalar_semua.append(pamungkas["reasoning"])
                jawaban = ((pamungkas.get("content") or "").strip()
                           or "Tidak ditemukan dasar dalam korpus.")
            else:
                jawaban = ("Tidak ditemukan dasar dalam korpus "
                           "(batas putaran dry-run tercapai).")
                yield {"t": "token", "text": jawaban}
            trace.add("llm_answer", "Jawaban pamungkas (batas putaran)",
                      {"teks": engine._ringkas(jawaban, 2000),
                       "jumlah_putaran": MAX_ITERATIONS, "tokens": tokens})
            yield {"t": "node", "node": trace.node[-1]}
    except Exception as e:
        yield {"t": "error", "message": _saring_error(str(e), llm)}
        return
    finally:
        mod_tools.execute_tool = _execute_asli

    mentah = "\n\n".join(nalar_semua)[:6000]
    nalar = llm_provider.rapikan_nalar("\n\n".join(nalar_semua))
    if nalar:
        for n in reversed(trace.node):
            if n.get("type") == "llm_answer":
                n["detail"]["penalaran"] = nalar
                n["detail"]["penalaran_mentah"] = mentah
                break
    yield {"t": "done", "jawaban": jawaban, "kandidat": semua_sumber,
           "reasoning": nalar,
           "reasoning_raw": mentah if nalar else None,
           "mode": "dry-run" if dry_run else "live",
           "usage": {"detik": round(time.perf_counter() - t0, 3),
                     "tokens": tokens}}
