"""
llm.py — PABRIK model chat LangChain + mode demo tanpa jaringan.
================================================================

Satu fungsi `get_llm()` mengembalikan objek yang SEMUA komponen LangChain
mengerti (BaseChatModel). Ganti backend cukup lewat .env — kode lain tidak
berubah. Inilah manfaat abstraksi framework:

    LLM_BACKEND=api    -> ChatOpenAI(base_url=..., api_key=..., model=...)
                          (OpenAI / OpenRouter / Groq / DeepSeek / dsb.)
    LLM_BACKEND=ollama -> ChatOllama(model="qwen3:4b")  (protokol native)
    --dry-run          -> MockAgentLLM / FakeListChatModel (tanpa jaringan)

MOCK DENGAN TOOL CALLING
  FakeListChatModel bawaan hanya mengeluarkan TEKS — cukup untuk chain
  RAG, tetapi tidak bisa mendemokan loop agent (butuh AIMessage dengan
  tool_calls). Karena itu kita menulis MockAgentLLM: BaseChatModel mini
  yang mengembalikan AIMessage terjadwal (teks ATAU tool_calls). Antarmuka
  invoke()-nya sama persis dengan model asli — graph agent tidak sadar
  sedang bicara dengan mock. Ini juga cara yang baik untuk UNIT-TEST agent
  tanpa biaya API.
"""

from __future__ import annotations

import config
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def get_llm(dry: bool = False, **kwargs):
    """Pilih model sesuai konfigurasi / mode dry-run."""
    cfg = config.get_llm_config()
    if dry:
        # Mock teks sederhana: dipakai untuk chain RAG (bukan agent).
        return FakeListChatModel(responses=[
            "YA",  # jawaban grader reflektif (reflective_graph)
            "(dry-run) Jawaban contoh dari FakeListChatModel.",
        ])
    if cfg["backend"] == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg["model"],
                          temperature=cfg["temperature"], **kwargs)
    if not cfg["api_key"]:
        raise RuntimeError(
            "LLM_API_KEY kosong — isi .env (lihat .env.example) "
            "atau jalankan dengan --dry-run.")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg["model"], base_url=cfg["base_url"],
                      api_key=cfg["api_key"], temperature=cfg["temperature"],
                      **kwargs)


class MockAgentLLM(BaseChatModel):
    """
    BaseChatModel mini terjadwal utk demo loop agent TANPA jaringan.
    skrip: list berisi str (jawaban final) ATAU dict
           {"tool_calls": [(nama, {args...}), ...]}
    """

    skrip: list = []

    def _llm_type(self) -> str:            # properti wajib BaseChatModel
        return "mock-agent"

    def bind_tools(self, tools, **kw):     # agent memanggil ini dulu
        return self                        # mock abaikan skema — cukup balik diri

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if not self.skrip:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="(mock: skrip habis)"))])
        langkah = self.skrip.pop(0)
        if isinstance(langkah, str):
            msg = AIMessage(content=langkah)
        else:
            # Format tool_calls standar LangChain: name, args(dict), id, type.
            msg = AIMessage(content="", tool_calls=[
                {"name": n, "args": a, "id": f"call_{i:02d}", "type": "tool_call"}
                for i, (n, a) in enumerate(langkah["tool_calls"])])
        return ChatResult(generations=[ChatGeneration(message=msg)])


if __name__ == "__main__":
    # `python llm.py` -> uji dua mock tanpa jaringan
    print(get_llm(dry=True).invoke("halo").content)
    m = MockAgentLLM(skrip=[{"tool_calls": [("search_norma",
                                             {"question": "kedaulatan"})]},
                            "final"])
    print(m.invoke("x").tool_calls)
