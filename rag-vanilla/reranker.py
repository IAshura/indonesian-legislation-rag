"""
reranker.py — RERANKER CROSS-ENCODER Qwen3-Reranker-0.6B (pencarian 2 fase).
============================================================================

KONSEP: KENAPA BUTUH RERANKER?
  ExactRetriever = model BI-ENCODER: query & dokumen di-embed TERPISAH lalu
  dicocokkan lewat dot product. Cepat (1 pass atas 63.231 vektor), tapi tidak
  ada interaksi kata-per-kata antara query dan isi pasal.

  Reranker = model CROSS-ENCODER: query & dokumen DIGABUNG jadi satu input
  dan dimasukkan ke LLM yang membaca keduanya bersamaan. Lebih akurat, tapi
  mahal (1 forward-pass PER pasangan) -> hanya layak untuk 10-30 kandidat
  teratas, BUKAN seluruh korpus. Itulah pola standar "retrieval 2 fase":
      fase 1 (murah) : bi-encoder ambil kandidat  ->  fase 2 (akurat): rerank.
  Semua skor lama (cosine) tetap tersimpan; urutan final dihitung dari
  score_rerank. Evaluasi lama recall@8=0.75 diukur TANPA rerank — jalankan
  eval ulang dengan rerank untuk melihat kenaikannya.

CARA KERJA MODEL (resep resmi Qwen3-Reranker):
  Input dibingkai sebagai pertanyaan yes/no:
      system : "Judge whether the Document meets the requirements ... yes or no"
      user   : <Instruct>: ...\n<Query>: ...\n<Document>: ...
      assistant : shed thought ... shed answer   (footer wajib resep resmi)
  Lalu kita membaca LOGIT token "yes" vs "no" pada posisi TERAKHIR:
      P(relevan) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
  Karena token id "yes"/"no" diambil lewat tokenizer, kode bekerja walau
  vocab berubah.

JEBAKAN YANG SUDAH DITANGANI:
  - model mengeluarkan tensor BFloat16 -> .float() sebelum dibaca (lihat
    embedder.py: jebakan yang sama);
  - LEFT padding wajib (posisi "terakhir" = posisi padding-mask terakhir);
  - teks panjang dipotong tokenizer (truncation) supaya batch tidak meledak;
  - model muat OFFLINE dari cache HuggingFace (local_files_only=True).
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import numpy as np

import config

# --------------------------------------------------------------- KONSTANTA
# Jumlah kandidat yang diberikan ke reranker. Trade-off: makin banyak makin
# akurat tapi makin lambat (setiap pasangan = 1 forward pass LLM 0.6B).
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_INSTRUCTION = (
    "Given a user question about Indonesian legislation, retrieve relevant "
    "legal provisions that answer the question."
)
# Template resmi Qwen3-Reranker (pesan system + footer assistant).
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct. Note that the answer can only be "
    "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n Giving thought\n\n\n\n Giving answer\n\n"

_STATE: dict = {"model": None, "tok": None}   # singleton lazy


def muat_reranker():
    """
    Muat model & tokenizer sekali saja (singleton), OFFLINE dari cache HF.
    Kembalikan (model, tokenizer) atau None jika cache tidak tersedia —
    pemanggil wajib bisa fallback ke 'tanpa rerank'.
    """
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["tok"]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        # CPU: fp32 ~11x lebih cepat dari bf16 (fallback lambat, diukur: 16 vs
        # 172 dtk/batch); GPU: bf16 hemat memori. Pilih berdasarkan device.
        import torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            RERANK_MODEL, dtype=dtype, local_files_only=True).eval()
        _STATE.update(model=model, tok=tok)
        return model, tok
    except Exception as e:                       # cache kosong / paket hilang
        print(f"[reranker] model tidak termuat ({e.__class__.__name__}): "
              f"rerank dimatikan, pakai urutan cosine asli.")
        return None


def skor_rerank(query: str, teks_list: list[str],
                instruction: str = RERANK_INSTRUCTION,
                batch_size: int = config.RERANK_BATCH_SIZE) -> list[float]:
    """
    Skor relevansi tiap teks terhadap query (P(yes), rentang 0..1).
    Urutan keluaran = urutan masukan. Model tidak tersedia -> kembalikan [].
    """
    muat = muat_reranker()
    if muat is None:
        return []
    model, tok = muat
    import torch
    id_yes = tok.convert_tokens_to_ids("yes")
    id_no = tok.convert_tokens_to_ids("no")
    skor: list[float] = []

    # Proses per batch kecil supaya memori GPU/CPU aman.
    for mulai in range(0, len(teks_list), batch_size):
        potongan = teks_list[mulai:mulai + batch_size]
        print(f"[reranker] batch {mulai // batch_size + 1} "
              f"({mulai + len(potongan)}/{len(teks_list)} pasangan) ...")
        prompt = [_PREFIX + f"<Instruct>: {instruction}\n<Query>: {query}"
                  f"\n<Document>: {t}" + _SUFFIX for t in potongan]
        inp = tok(prompt, return_tensors="pt", padding=True,
                  truncation=True, max_length=config.RERANK_MAX_LENGTH).to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[:, -1, :].float()   # .float(): BF16!
        for i in range(len(potongan)):
            y = logits[i, id_yes].item()
            n = logits[i, id_no].item()
            # Softmax 2-kelas: P(yes) inilah skor rerank.
            skor.append(math.exp(y) / (math.exp(y) + math.exp(n)))
    return skor


def rerank_kandidat(query: str, kandidat: list[dict],
                    top_n: int | None = None,
                    instruction: str = RERANK_INSTRUCTION) -> list[dict]:
    """
    Tahap 2 pencarian: urutkan ulang kandidat retrieval berdasarkan skor
    cross-encoder. Setiap kandidat diberi kunci baru 'score_rerank' (skor
    lama 'score' dipertahankan untuk audit). Model tak termuat -> kandidat
    kembali apa adanya (urutan cosine), jadi alur tak pernah gagal keras.
    """
    if not kandidat:
        return kandidat
    teks = [k.get("text") or k.get("snippet") or "" for k in kandidat]
    skor = skor_rerank(query, teks, instruction)
    if not skor:                                  # fallback: tanpa rerank
        return kandidat[:top_n] if top_n else kandidat
    for k, s in zip(kandidat, skor):
        k["score_rerank"] = round(s, 4)
    kandidat.sort(key=lambda k: k["score_rerank"], reverse=True)
    return kandidat[:top_n] if top_n else kandidat
