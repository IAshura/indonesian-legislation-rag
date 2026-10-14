"""
reranker.py — CROSS-ENCODER Qwen3-Reranker-0.6B, gaya LangChain (tahap 2).
==========================================================================

PERAN FILE INI DI ALUR 2 FASE
  fase 1 (murah) : LegislationRetriever — bi-encoder, 1 pass atas 63.231
                   vektor -> kandidat peringkat kasar (cosine).
  fase 2 (akurat): QwenReranker — cross-encoder membaca query+pasal
                   BERSAMAAN per pasangan, jauh lebih tajam tapi mahal,
                   jadi hanya untuk 10-20 kandidat teratas.

  Di LangChain, reranker TIDAK perlu jadi retriever baru: ia berdiri sebagai
  step LCEL biasa `RunnableLambda(reranker.rerank)` di antara retriever dan
  prompt (lihat rag_chain.py). Karena menerima & mengembalikan list[Document],
  ia bisa diselipkan di mana saja.

CARA KERJA MODEL (resep resmi Qwen3-Reranker):
  Prompt yes/no: system judge + <Instruct>/<Query>/<Document> + footer
  assistant "shed thought ... shed answer". Skor = softmax dua token:
      P(relevan) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))

JEBAKAN YANG DITANGANI: BFloat16 -> .float() sebelum dibaca; LEFT padding
  (wajib agar posisi terakhir benar); truncation per pasangan; lazy load
  offline (local_files_only=True); fallback aman bila model tak termuat.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # konsol Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import config
from langchain_core.documents import Document

RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_INSTRUCTION = (
    "Given a user question about Indonesian legislation, retrieve relevant "
    "legal provisions that answer the question."
)
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct. Note that the answer can only be "
    "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n Giving thought\n\n\n\n Giving answer\n\n"

_STATE: dict = {"model": None, "tok": None}   # singleton lazy


def muat_reranker():
    """Muat sekali (singleton) OFFLINE; None bila cache tidak tersedia."""
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
    except Exception as e:
        print(f"[reranker] model tidak termuat ({e.__class__.__name__}): "
              f"rerank dimatikan, urutan cosine dipakai.")
        return None


def skor_rerank(query: str, teks_list: list[str],
                instruction: str = RERANK_INSTRUCTION,
                batch_size: int = config.RERANK_BATCH_SIZE) -> list[float]:
    """P(yes) per teks (urutan masukan). [] bila model tak termuat."""
    muat = muat_reranker()
    if muat is None:
        return []
    model, tok = muat
    import torch
    id_yes = tok.convert_tokens_to_ids("yes")
    id_no = tok.convert_tokens_to_ids("no")
    skor: list[float] = []
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
            skor.append(math.exp(y) / (math.exp(y) + math.exp(n)))
    return skor


class QwenReranker:
    """
    Step rerank siap-sambung-LCEL: callable menerima list[Document],
    mengembalikan list[Document] urut baru berdasar score_rerank.
    (metadata['score'] cosine lama tetap dipertahankan utk audit.)
    """

    def __init__(self, top_n: int | None = None,
                 instruction: str = RERANK_INSTRUCTION) -> None:
        self.top_n = top_n
        self.instruction = instruction

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        skor = skor_rerank(query, [d.page_content for d in docs],
                           self.instruction)
        if not skor:                      # fallback aman: tanpa rerank
            return docs[:self.top_n] if self.top_n else docs
        for d, s in zip(docs, skor):
            d.metadata["score_rerank"] = round(s, 4)
        docs.sort(key=lambda d: d.metadata["score_rerank"], reverse=True)
        return docs[:self.top_n] if self.top_n else docs
