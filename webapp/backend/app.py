"""
app.py — Server FastAPI webapp RAG Hukum Indonesia.
====================================================

Menjalankan:
    cd webapp/backend && uvicorn app:app --port 8000

Endpoint:
  GET  /api/health                    -> {ok, model_aktif, bertahan}
  GET  /api/milvus/status             -> {ok, koleksi, entitas, dim, cocok_dim}
  GET  /api/models                   -> daftar embedding + tag aktif
  POST /api/llm/ping {llm}            -> {ok, ms, model, usage} (tanpa retrieval)
  GET  /api/llm/env-defaults          -> {ok, base_url, model, has_api_key}
                                         (prefill Setting dari .env backend)
  POST /api/embedding/switch {tag}    -> tulis ACTIVE + reload modul lama
  POST /api/ask                       -> jawaban + citations + trace + usage
    body: {question, mode:"fixed"|"agent", retrieval:"hybrid"|"dense"|"bm25",
           dense_method?, dense_ef?, dense_nprobe?,
           rerank:bool, top_k:int, doc_type?, year?,
           embedding:"qwen3-06b"|"qwen3-4b", llm:{provider, base_url,
           api_key, model}, dry_run:bool, resume_from?:int}
  POST /api/ask/stream                -> SSE: status/node/token/done/error
    (body sama; history disimpan server saat done)
  GET/POST /api/history, GET/DELETE /api/history/:id,
  GET /api/history/:id/export?format=md|txt|html

CATATAN KEAMANAN
  - api_key TIDAK PERNAH di-log dan TIDAK disimpan ke file apapun; hanya
    lewat memori proses (env var sementara saat panggil LLM).
  - riwayat disimpan ke(history.jsonl) TANPA api_key.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Muat .env (webapp/backend/.env) TANPA dependensi python-dotenv:
# format `KEY=VALUE`, baris kosong / # diabaikan; env asli lebih diutamakan.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge               # noqa: E402  (sys.path -> modul lama)
import engine               # noqa: E402
import llm_provider         # noqa: E402
import streaming            # noqa: E402  (varian SSE dari engine)
from retriever import konteks_kaya  # noqa: E402  (typing kandidat)

app = FastAPI(title="WebApp RAG Hukum Indonesia", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

RIWAYAT_FILE = Path(__file__).resolve().parent / "history.jsonl"

HISTORY_FILE = RIWAYAT_FILE


def _baca_riwayat() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    out = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(out, key=lambda r: -r.get("created_at_ts", 0))


def _tulis_riwayat(rows: list[dict]) -> None:
    HISTORY_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) or "",
        encoding="utf-8")


# ------------------------------------------------------------------ schemas
class LLMConfig(BaseModel):
    provider: str = "custom"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=64000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("fixed", pattern="^(fixed|agent)$")
    retrieval: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    dense_method: str = Field("exact", pattern="^(exact|hnsw|ivf|flat)$")
    dense_ef: int = Field(64, ge=1, le=512)
    dense_nprobe: int = Field(16, ge=1, le=1024)
    gaya: str = Field("standar", pattern="^(ringkas|standar|rinci)$")
    paksa_cari: bool = True
    nomor_awal: int = Field(1, ge=1, le=100000)
    node_awal: int = Field(1, ge=1, le=1000000)
    rerank: bool = True
    top_k: int = Field(8, ge=1, le=50)            # ukuran pool retrieval
    inject_k: int | None = Field(None, ge=1, le=50)  # dokumen disuntik ke LLM
    doc_type: str | list[str] | None = None       # multi-jenis via checklist
    year: int | None = None
    # tag embedding validasi dinamis dari disk (/api/models), bukan hardcoded
    embedding: str | None = None
    llm: LLMConfig | None = None
    dry_run: bool = False
    resume_from: int | None = None
    conversation_id: str | None = None   # memori: chat lanjutan (follow-up)
    riwayat_chat: list[dict] | None = None       # [{role, content}] dari UI


class SwitchRequest(BaseModel):
    tag: str


# Memori chat UNLIMITED giliran, dibatasi 100.000 token (estimasi ±4
# char/token, lihat engine.estimasi_token). engine._sisip_riwayat mengambil
# giliran terbaru mundur selama muat dalam budget tersebut.
PAGU_PESAN_RIWAYAT = 400  # pengaman: maksimal 400 pesan riwayat per request


def _riwayat_konversasi(cid: str | None) -> list[dict]:
    """Giliran chat sebelumnya dlm conversation_id, urut lama->baru.
    Format siap disisipkan ke messages LLM: [{role, content}, ...]"""
    if not cid:
        return []
    isi = []
    for r in _baca_riwayat():
        if r.get("conversation_id") == cid:
            isi.append((r.get("created_at_ts", 0),
                        {"role": "user", "content": r.get("question", "")}))
            isi.append((r.get("created_at_ts", 0),
                        {"role": "assistant", "content": r.get("answer", "")}))
    isi.sort(key=lambda x: x[0])
    return [m for _, m in isi][-PAGU_PESAN_RIWAYAT:]


# ------------------------------------------------------------------ mappers
def _citations_dari_kandidat(kandidat: list[dict], awal: int = 1) -> list[dict]:
    """Konversi kandidat konteks_kaya -> kontrak citations UI.
    Mode agent memberi nomor "n" sendiri saat tool dipanggil (lihat
    engine.run_agent); nomor itu DIPERTAHANKAN agar cocok dengan [S..]
    yang ditulis LLM. Mode fixed tak punya "n" -> penomoran urut mulai
    dari `awal` (global per sesi; S lama tak dipakai ulang).
    """
    out = []
    for i, k in enumerate(kandidat, awal):
        out.append({
            "n": k.get("n") or i,                  # [S8], [S9], ...
            "chunk_id": k.get("chunk_id"),
            "parent_id": k.get("parent_id"),
            "citation_str": k.get("citation_str"),
            "official_title": k.get("official_title"),
            "doc_type": k.get("doc_type"),
            "page": k.get("page"),
            "lines": k.get("lines"),
            "score": k.get("score"),               # cosine (atau rerank rank-proxy)
            "score_rerank": k.get("score_rerank"),
            "skor_bm25": k.get("skor_bm25"),
            "skor_fusi": k.get("skor_fusi"),
            "text": k.get("text"),
        })
    return out


def _sanitize_llm(llm: dict | None) -> dict | None:
    """Hilangkan api_key dari objek yang akan disimpan/di-log."""
    if not llm:
        return None
    return {k: v for k, v in llm.items() if k != "api_key" and v}


# ------------------------------------------------------------------ /api/ask
@app.post("/api/ask")
def api_ask(req: AskRequest):
    """Jalankan mode fixed/agent + trace; dry-run tidak memanggil LLM."""
    # tag embedding bebas: validasi dinamis dari disk, bukan daftar hardcode.
    if req.embedding:
        t = (req.embedding or "").strip()
        if not (bridge.ENB_ROOT / t / "meta.json").exists():
            raise HTTPException(status_code=400,
                                detail=f"embedding '{t}' tidak ditemukan di disk")
        if t != bridge.active_tag():
            bridge.antre_reload_embedding(t)
            engine.reset_cache()

    # node_awal: lanjutan penomoran Graf sesi (n27, n28, ...) — default 1.
    trace = engine.Trace(awal=req.node_awal)
    # memori percakapan: kombinasi riwayat_chat eksplisit dari UI atau
    # sinonim conversation_id dari history.jsonl. Chuaks didahulukan.
    memori = (req.riwayat_chat
              or _riwayat_konversasi(req.conversation_id))
    llm_d = req.llm.model_dump(exclude_none=True) if req.llm else None
    temp = (req.llm.temperature if req.llm and req.llm.temperature is not None
            else None)
    mtok = (req.llm.max_tokens if req.llm and req.llm.max_tokens is not None
            else None)
    try:
        if req.mode == "agent":
            hasil = engine.run_agent(
                req.question, retrieval=req.retrieval, rerank=req.rerank,
                top_k=req.top_k, inject_k=req.inject_k,
                doc_type=req.doc_type, year=req.year,
                dry_run=req.dry_run, llm=llm_d,
                trace=trace, riwayat_chat=memori,
                dense_method=req.dense_method, dense_ef=req.dense_ef,
                dense_nprobe=req.dense_nprobe, gaya=req.gaya,
                paksa_cari=req.paksa_cari, nomor_awal=req.nomor_awal,
                temperature=temp, max_tokens=mtok)
        else:
            hasil = engine.run_fixed(
                req.question, retrieval=req.retrieval, rerank=req.rerank,
                top_k=req.top_k, inject_k=req.inject_k,
                doc_type=req.doc_type, year=req.year,
                dry_run=req.dry_run, llm=llm_d,
                resume_from=req.resume_from,
                trace=trace, riwayat_chat=memori,
                dense_method=req.dense_method, dense_ef=req.dense_ef,
                dense_nprobe=req.dense_nprobe, gaya=req.gaya,
                nomor_awal=req.nomor_awal,
                temperature=temp, max_tokens=mtok)
    except RuntimeError as e:
        msg = str(e)
        # JANGAN bocorkan isi api_key bila ada di error message provider.
        if req.llm and req.llm.api_key and req.llm.api_key in msg:
            msg = msg.replace(req.llm.api_key, "***")
        raise HTTPException(status_code=502, detail=msg) from e

    citations = _citations_dari_kandidat(hasil["kandidat"],
                                         awal=req.nomor_awal)
    body = {
        "answer": hasil["jawaban"],
        "citations": citations,
        "trace": hasil["trace"],
        "usage": hasil["usage"],
        "mode": hasil["mode"],
        "reasoning": hasil.get("reasoning"),
        "reasoning_raw": hasil.get("reasoning_raw"),
    }

    # ---- simpan riwayat (tanpa api_key)
    rec = {"id": str(uuid.uuid4()),
           "created_at": datetime.now(timezone.utc).isoformat(),
           "created_at_ts": datetime.now(timezone.utc).timestamp(),
           "question": req.question, "answer": body["answer"],
           "mode": req.mode, "retrieval": req.retrieval,
           "dense_method": req.dense_method, "gaya": req.gaya, "paksa_cari": req.paksa_cari, "nomor_awal": req.nomor_awal, "node_awal": req.node_awal,
           "rerank": req.rerank, "top_k": req.top_k,
           "doc_type": req.doc_type, "year": req.year, "inject_k": req.inject_k,
           "embedding": bridge.active_tag(),
           "llm": _sanitize_llm(req.llm.model_dump(exclude_none=True)
                                if req.llm else None),
           "citations": citations, "trace": hasil["trace"],
           "usage": hasil["usage"], "reasoning": hasil.get("reasoning"),
           "reasoning_raw": hasil.get("reasoning_raw"),
           "conversation_id": req.conversation_id}
    rows = _baca_riwayat()
    rows.insert(0, rec)
    # cap 200 entri terbaru; file tidak tumbuh tanpa batas
    _tulis_riwayat(rows[:200])
    body["history_id"] = rec["id"]
    return body


# ------------------------------------------------------------- /api/ask/stream
@app.post("/api/ask/stream")
def api_ask_stream(req: AskRequest):
    """Varian STREAMING (SSE) dari /api/ask untuk UI progresif.

    Event: status (tahap mulai) -> node (trace live) -> token (jawaban
    mengalir) -> done (hasil final + history_id) | error.
    History disimpan server TEPAT SEBELUM event done dikirim.
    """
    if req.embedding:
        t = (req.embedding or "").strip()
        if not (bridge.ENB_ROOT / t / "meta.json").exists():
            raise HTTPException(status_code=400,
                                detail=f"embedding '{t}' tidak ditemukan di disk")
        if t != bridge.active_tag():
            bridge.antre_reload_embedding(t)
            engine.reset_cache()

    memori = (req.riwayat_chat
              or _riwayat_konversasi(req.conversation_id))
    llm_d = req.llm.model_dump(exclude_none=True) if req.llm else None
    kwargs = dict(retrieval=req.retrieval, rerank=req.rerank,
                  top_k=req.top_k, inject_k=req.inject_k,
                  doc_type=req.doc_type, year=req.year,
                  dry_run=req.dry_run, llm=llm_d,
                  riwayat_chat=memori, dense_method=req.dense_method,
                  dense_ef=req.dense_ef, dense_nprobe=req.dense_nprobe,
                  gaya=req.gaya, paksa_cari=req.paksa_cari,
                  nomor_awal=req.nomor_awal, node_awal=req.node_awal,
                  temperature=(req.llm.temperature if req.llm else None),
                  max_tokens=(req.llm.max_tokens if req.llm else None))

    def gen():
        try:
            # paksa_cari hanya milik mode agent (fixed selalu mencari)
            kw = dict(kwargs)
            if req.mode != "agent":
                kw.pop("paksa_cari", None)
            gen_ev = (streaming.run_agent_stream(req.question, **kw)
                      if req.mode == "agent"
                      else streaming.run_fixed_stream(req.question, **kw))
            akumulasi = ""
            kandidat = []
            trace_nodes: list[dict] = []
            mode = req.mode
            usage: dict = {}
            for ev in gen_ev:
                if ev.get("t") == "token":
                    akumulasi += ev.get("text") or ""
                    yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                elif ev.get("t") == "done":
                    # rakit final + simpan riwayat DULU, baru kirim done
                    citations = _citations_dari_kandidat(ev.get("kandidat") or [],
                                                         awal=req.nomor_awal)
                    body = {"answer": ev.get("jawaban", akumulasi),
                            "citations": citations,
                            "trace": trace_nodes,
                            "usage": ev.get("usage", {}),
                            "mode": ev.get("mode", mode),
                            "reasoning": ev.get("reasoning"),
                            "reasoning_raw": ev.get("reasoning_raw")}
                    rec = {"id": str(uuid.uuid4()),
                           "created_at": datetime.now(timezone.utc).isoformat(),
                           "created_at_ts": datetime.now(timezone.utc).timestamp(),
                           "question": req.question, "answer": body["answer"],
                           "mode": req.mode, "retrieval": req.retrieval,
                           "dense_method": req.dense_method, "gaya": req.gaya, "paksa_cari": req.paksa_cari, "nomor_awal": req.nomor_awal, "node_awal": req.node_awal,
                           "rerank": req.rerank, "top_k": req.top_k,
                           "doc_type": req.doc_type, "year": req.year,
                           "inject_k": req.inject_k,
                           "embedding": bridge.active_tag(),
                           "llm": _sanitize_llm(llm_d),
                           "citations": citations, "trace": trace_nodes,
                           "usage": body["usage"],
                           "reasoning": body["reasoning"],
                           "reasoning_raw": body.get("reasoning_raw"),
                           "conversation_id": req.conversation_id}
                    rows = _baca_riwayat()
                    rows.insert(0, rec)
                    _tulis_riwayat(rows[:200])
                    body["history_id"] = rec["id"]
                    yield "data: " + json.dumps({"t": "done", **body},
                                                ensure_ascii=False) + "\n\n"
                elif ev.get("t") == "node":
                    trace_nodes.append(ev["node"])
                    yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                else:  # status / error: teruskan apa adanya
                    yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    if ev.get("t") == "error":
                        return
        except Exception as e:
            msg = str(e)
            if req.llm and req.llm.api_key and req.llm.api_key in msg:
                msg = msg.replace(req.llm.api_key, "***")
            yield "data: " + json.dumps({"t": "error", "message": msg},
                                        ensure_ascii=False) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class PingRequest(BaseModel):
    llm: LLMConfig | None = None


@app.get("/api/llm/env-defaults")
def api_llm_env_defaults():
    """Default LLM dari env/.env backend untuk prefill Setting di browser.

    api_key TIDAK dikirim ke browser - cukup tanda has_api_key; saat request
    tanpa api_key, backend otomatis pakai LLM_API_KEY dari env (llm_provider).
    """
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    return {
        "ok": bool(base_url),
        "base_url": base_url,
        "model": model,
        "has_api_key": bool(os.environ.get("LLM_API_KEY", "").strip()),
    }


# ------------------------------------------------------------------ models
@app.post("/api/llm/ping")
def api_llm_ping(req: PingRequest):
    """Uji koneksi LLM mentah: SATU panggilan mini TANPA retrieval/RAG.

    Beda dengan /api/ask (yang menjalankan pipeline retrieval + menjawab
    pertanyaan): endpoint ini hanya memastikan base_url + api_key + model
    diterima provider. Prompt 1 kata, max_tokens 8 — biaya ~belasan token.
    api_key tidak pernah di-log/di-echo.
    """
    import time as _time

    import requests as _rq

    import llm_adapter as _ad

    llm_d = req.llm.model_dump(exclude_none=True) if req.llm else {}
    provider = ((llm_d.get("provider") or "custom").strip().lower()
                or "custom")
    try:
        base_url, api_key, model = llm_provider.validasi(llm_d)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not base_url:
        raise HTTPException(
            status_code=400,
            detail="live LLM butuh llm.base_url (isi preset provider/custom)")
    if not api_key and provider != "ollama":
        raise HTTPException(status_code=400,
                            detail="live LLM butuh llm.api_key")
    if provider == "ollama" and not api_key:
        api_key = "ollama"
    if not model:
        raise HTTPException(status_code=400,
                            detail="live LLM butuh llm.model")
    messages = [{"role": "user", "content": "Reply with exactly: ok"}]
    t0 = _time.perf_counter()
    try:
        if _ad.rasa(provider, base_url) == "anthropic":
            _msg, usage_raw = _ad.complete_anthropic(
                base_url, api_key, model, messages, None, 0, 8)
        else:
            url = base_url.rstrip("/") + "/chat/completions"
            resp = _rq.post(
                url,
                json={"model": model, "messages": messages,
                      "temperature": 0, "max_tokens": 8},
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                timeout=60)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"HTTP {resp.status_code} dari LLM: "
                           f"{resp.text[:300]}")
            try:
                data = resp.json()
                _msg = data["choices"][0]["message"]
            except (ValueError, KeyError, IndexError):
                raise HTTPException(
                    status_code=502,
                    detail="Respons LLM tak berbentuk chat completion")
            usage_raw = data.get("usage") or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM tak terjangkau ({base_url}): {e}")
    ms = int((_time.perf_counter() - t0) * 1000)
    content = ((_msg or {}).get("content") or "").strip()
    return {"ok": True, "ms": ms, "model": model, "answer": content,
            "usage": {"tokens": llm_provider.normalisasi_usage(usage_raw)}}


@app.get("/api/health")
def api_health():
    return {"ok": True,
            "embedding_aktif": bridge.active_tag(),
            "rerank_default": bool(getattr(bridge.config, "RERANK", False)),
            "waktu": datetime.now(timezone.utc).isoformat()}


@app.get("/api/milvus/status")
def api_milvus_status():
    """Cek cepat Milvus untuk opsi index HNSW/IVF/FLAT di UI.

    Dipanggil frontend SETIAP user memilih metode Milvus (bukan saat
    bertanya) agar warning muncul sebelum query dikirim. Selalu HTTP 200 —
    bedakan hidup/mati lewat field "ok" (UI tidak perlu parsing error).
    """
    mati = {"ok": False, "koleksi": "legislation_norma_v1",
            "instruksi": "cd C:\\milvus; docker compose up -d "
                         "(tunggu ±1 menit sampai 3 container Up), "
                         "lalu ulangi. Bila koleksi belum ada: "
                          "run pipeline/insert_milvus.py (see docs/operations-runbook.md step 5)."}
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return {**mati, "alasan": "pymilvus belum terpasang di backend"}
    try:
        cli = MilvusClient(uri="http://localhost:19530", timeout=3)
        nama = "legislation_norma_v1"
        if not cli.has_collection(nama):
            return {**mati, "alasan": f"koleksi {nama} belum ada — "
                    "jalankan pipeline/insert_milvus.py (tanpa --drop) sekali"}
        entitas = None
        dim_koleksi = None
        try:
            entitas = cli.get_collection_stats(nama).get("row_count")
        except Exception:
            entitas = None
        try:
            for f in cli.describe_collection(nama).get("fields", []):
                if f.get("type") == "FloatVector":
                    dim_koleksi = int(f.get("params", {}).get("dim") or 0) or None
                    break
        except Exception:
            dim_koleksi = None
    except Exception as e:
        return {**mati, "alasan": f"Milvus tidak terjangkau: {str(e)[:120]}"}
    dim_aktif = None
    try:
        meta_p = bridge.ENB_ROOT / bridge.active_tag() / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        dim_aktif = int(meta.get("keep_dim") or meta.get("dim") or 0) or None
    except Exception:
        dim_aktif = None
    cocok = (dim_koleksi is None or dim_aktif is None
             or dim_koleksi == dim_aktif)
    return {"ok": True, "koleksi": nama, "entitas": entitas,
            "dim_koleksi": dim_koleksi, "dim_aktif": dim_aktif,
            "cocok_dim": cocok,
            **({} if cocok else {
                "peringatan": f"dimensi koleksi ({dim_koleksi}) != embedding aktif "
                f"({bridge.active_tag()} dim {dim_aktif}) — ganti embedding dari "
                "panel Setting atau rebuild koleksi via insert_milvus.py --drop"})}


@app.get("/api/models")
def api_models():
    """Daftar embedding + meta dim/count + file ACTIVE."""
    tags = sorted(bridge.TAGS_VALID)
    aktif = bridge.active_tag()
    daftar = []
    for t in tags:
        meta_p = bridge.ENB_ROOT / t / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        daftar.append({"tag": t, "dim": meta.get("dim"),
                       "count": meta.get("count"), "model": meta.get("model"),
                       "aktif": t == aktif,
                       "is_gpu": t == "qwen3-4b"})
    return {"aktif": aktif, "daftar": daftar}


@app.post("/api/embedding/switch")
def api_embedding_switch(req: SwitchRequest):
    try:
        hasil = bridge.antre_reload_embedding(req.tag)
        engine.reset_cache()          # singleton matrix/idf dibangun ulang
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"ok": True, **hasil}


# ------------------------------------------------------------------ history
@app.get("/api/history")
def api_history():
    return {"daftar": [
        {"id": r["id"], "created_at": r["created_at"],
         "question": r["question"], "mode": r["mode"],
         "retrieval": r["retrieval"],
         "jawadan_panjang": len(r.get("answer") or "")}
        for r in _baca_riwayat()]}


@app.post("/api/history")
def api_history_buat(rec: dict):
    """Buat entri kosong (UI "Riwayat baru") — opsional; riwayat otomatis
    terisi dari /api/ask."""
    rec = {"id": str(uuid.uuid4()),
           "created_at": datetime.now(timezone.utc).isoformat(),
           "created_at_ts": datetime.now(timezone.utc).timestamp(),
           "question": rec.get("question", ""), "answer": rec.get("answer", ""),
           "citations": rec.get("citations", []), "trace": rec.get("trace", []),
           "usage": rec.get("usage", {}), "mode": rec.get("mode", "fixed"),
           "embedding": bridge.active_tag(), "manual": True}
    rows = _baca_riwayat()
    rows.insert(0, rec)
    _tulis_riwayat(rows[:200])
    return {"ok": True, "id": rec["id"]}


@app.get("/api/history/{rid}")
def api_history_detail(rid: str):
    for r in _baca_riwayat():
        if r["id"] == rid:
            return r
    raise HTTPException(status_code=404, detail="riwayat tidak ditemukan")


@app.delete("/api/history/{rid}")
def api_history_hapus(rid: str):
    rows = _baca_riwayat()
    sisa = [r for r in rows if r["id"] != rid]
    if len(sisa) == len(rows):
        raise HTTPException(status_code=404, detail="riwayat tidak ditemukan")
    _tulis_riwayat(sisa)
    return {"ok": True}


@app.get("/api/history/{rid}/export")
def api_history_export(rid: str, format: str = "md"):
    """Export percakapan riwayat: markdown (.md), teks (.txt), atau HTML
    print-friendly (.html — buka di browser, Ctrl+P -> Simpan sebagai file PDF)."""
    r = next((x for x in _baca_riwayat() if x["id"] == rid), None)
    if r is None:
        raise HTTPException(status_code=404, detail="riwayat tidak ditemukan")

    meta = (f'# Riwayat RAG Hukum Indonesia\n\n'
            f'- ID: `{r["id"]}`\n'
            f'- Waktu: {r["created_at"]}\n'
            f'- Mode: {r.get("mode")} | retrieval: {r.get("retrieval")} | '
            f'rerank: {r.get("rerank")} | top_k: {r.get("top_k")}\n'
            f'- Embedding: {r.get("embedding")}\n\n'
            f'## Pertanyaan\n\n{r.get("question", "")}\n\n'
            f'## Jawaban\n\n{r.get("answer", "")}\n')

    sitasi = ""
    if r.get("citations"):
        sitasi = "\n\n## Sumber sitasi\n\n"
        for c in r["citations"]:
            skor = (f"cosine {c.get('score')}"
                    + (f", BM25 {c.get('skor_bm25')}" if c.get("skor_bm25") else "")
                    + (f", fusi {c.get('skor_fusi')}" if c.get("skor_fusi") else ""))
            sitasi += (f"[S{c['n']}] {c.get('official_title')} — "
                       f"{c.get('citation_str')} ({skor})\n"
                       f"hal. {c.get('page')} baris {c.get('lines')}\n\n"
                       f"> {c.get('text', '')}\n\n")

    if format == "txt":
        txt = meta.replace("# ", "").replace("## ", "").replace("`", "")
        txt += sitasi.replace("## ", "").replace("> ", "")
        return PlainTextResponse(txt, media_type="text/plain; charset=utf-8",
                                 headers={"Content-Disposition":
                                          f'attachment; filename="rag-history-{rid[:8]}.txt"'})

    if format == "html":
        # sitasi TIDAK boleh hilang saat export — sisipkan penuh di <pre>
        html = (f"<!DOCTYPE html><html lang='id'><head><meta charset='utf-8'>"
                f"<title>Export riwayat</title><style>"
                f"body{{font-family:'Segoe UI',sans-serif;max-width:760px;"
                f"margin:2rem auto;padding:0 1rem;line-height:1.55}}"
                f"h1{{font-size:1.5rem}}h2{{margin-top:1.5rem}}"
                f"pre{{white-space:pre-wrap;background:#f6f7f9;padding:1em;"
                f"border-radius:8px;font-family:inherit}}"
                f"@media print{{body{{margin:0}}}}"
                f"</style></head><body>"
                f"<h1>Riwayat RAG Hukum Indonesia</h1>"
                f"<pre>{meta}{sitasi}</pre>"
                f"<button onclick='window.print()'>Cetak / Simpan PDF</button>"
                f"</body></html>")
        return HTMLResponse(html)

    # default .md
    md = meta + sitasi
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition":
                                      f'attachment; filename="rag-history-{rid[:8]}.md"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
