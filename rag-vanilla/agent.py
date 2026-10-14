"""
agent.py — ARSITEKTUR B: agentic + tool calling, TANPA framework apa pun.
=========================================================================

IDE INTI (panduan bab 5)
  LLM tidak menghitung apa pun tentang database — ia BERBICARA.
  Kita beritahu dia daftar "tools". Jika butuh data, LLM membalik dengan
  PERMINTAAN pemanggilan fungsi (nama + argumen JSON). KODE KITA yang
  menjalankan pencarian sungguhan, lalu hasilnya dikirim balik. LLM
  melanjutkan: minta tool lain, atau menyusun jawaban final.

LOOP (hati-hati: inilah seluruh "keajaiban" agentic — tidak ada sihir!)
    1. kirim messages + TOOLS -> LLM
    2. LLM balas: tool_calls?  ──tidak──> selesai, itu jawaban final
              │ya
              ▼
    3. untuk TIAP tool_call: eksekusi execute_tool(), tambahkan
       pesan role="tool" (dengan tool_call_id yang cocok!)
    4. kembali ke langkah 1 — sekarang LLM melihat hasil tool di riwayat.
    Guardrail: batasi MAX_ITERATIONS (5) supaya loop tidak tak berujung.

PERBANDINGAN DENGAN answer.py
  answer : alur TETAP, 1 panggilan LLM, mudah diaudit.  <- mulai dari sini
  agent     : LLM memilih langkah, 2-5+ panggilan, fleksibel (multi-hop:
             "cari pasalnya -> ambil dokumen induknya -> cek status").
  Latensi & biaya naik; argumen tool bisa salah -> validasi di tools.py.

MODE TANPA JARINGAN (--dry-run)
  Memakai MockLLM terjadwal (llm_client.py) sehingga SELURUH LOOP tetap
  bisa didemokan & dipelajari tanpa API key: LLM palsu "meminta" tool,
  tool asli dijalankan sungguhan, hasil asli dikirim balik.
"""

from __future__ import annotations

import argparse
import json
import sys

# Emoji ikon langkah butuh stdout UTF-8 (konsol Windows default cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tools
from llm_client import MockLLM, chat, dry_run

MAX_ITERATIONS = 5          # guardrail: maksimal 5 putaran LLM
IKON = {"llm": "\U0001f9e0", "tool": "\U0001f527", "hasil": "\U0001f4c4",
        "final": "\u2705"}


def jalankan(question: str, dry: bool = False, verbose: bool = True) -> str:
    """
    Jalankan loop agent penuh untuk satu pertanyaan. Kembalikan jawaban final.
    Setiap langkah dicetak (trace) supaya keputusan LLM bisa diaudit.
    """
    # Pesan awal: aturan sistem + pertanyaan pengguna.
    messages: list[dict] = [
        {"role": "system",
         "content": "Kamu asisten hukum Indonesia. Gunakan tools yang tersedia "
                    "untuk mencari pasal SEBELUM menjawab. Jawab hanya dari "
                    "hasil tool. Setiap pernyataan hukum disertai sitasi "
                    "dari hasil tool (judul + citation_str + halaman). "
                    "Jika tool tidak menemukan dasar, katakan jujur."},
        {"role": "user", "content": question},
    ]

    if dry:   # LLM palsu terjadwal: minta search -> minta pasal -> jawab
        # Skrip mock = contoh alur multi-tool yang selalu valid:
        #   putaran 1: minta pencarian (tool asli dijalankan sungguhan)
        #   putaran 2: minta metadata dokumen (multi-hop sederhana)
        #   putaran 3: yakin -> jawab final. Loop agent-nya SAMA dengan asli.
        llm = MockLLM([
            {"tool_calls": [{"name": "search_norma",
                             "arguments": {"question": question, "top_k": 3}}]},
            {"tool_calls": [{"name": "get_document",
                             "arguments": {"document_id": "uud-1945"}}]},
            {"content": "(mock) Jawaban final berdasarkan hasil tool."},
        ])
    else:
        llm = None   # -> dipilih di loop bawah: chat() asli

    for putaran in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n{IKON['llm']} — putaran {putaran}: kirim ke LLM "
                  f"({len(messages)} pesan, {len(tools.TOOLS)} tools)")

        # ---- LANGKAH 1: panggil LLM (nyata atau mock)
        if llm is None:
            balasan = chat(messages, tools=tools.TOOLS)
        else:
            balasan = llm.chat(messages, tools=tools.TOOLS)

        tool_calls = balasan.get("tool_calls")

        # ---- LANGKAH 2: tidak ada tool_calls -> INI JAWABAN FINAL
        if not tool_calls:
            jawaban = balasan.get("content") or "(jawaban kosong)"
            if verbose:
                print(f"\n{IKON['final']} Jawaban final diterima.")
            return jawaban

        # ---- LANGKAH 3: simpan pesan assistant YANG MEMINTA tool dulu.
        # Normalisasi ke 3 kunci inti saja (respons API kadang membawa
        # field ekstra yang tidak diterima beberapa provider saat dikirim
        # balik) — dan WAJIB: pesan role="tool" berikutnya menjawab id-nya.
        messages.append({"role": "assistant",
                         "content": balasan.get("content"),
                         "tool_calls": tool_calls})

        for tc in tool_calls:
            nama = tc["function"]["name"]
            argumen = tc["function"]["arguments"]   # STRING JSON dari API
            if verbose:
                print(f"{IKON['tool']} LLM minta: {nama}({argumen})")

            # ---- LANGKAH 4: KODE KITA yang mengeksekusi (bukan LLM!)
            hasil = tools.execute_tool(nama, argumen)
            if verbose:
                print(f"{IKON['hasil']} hasil: {hasil[:180]}"
                      f"{' …' if len(hasil) > 180 else ''}")

            # ---- LANGKAH 5: hasil dikirim balik sebagai pesan role="tool"
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": hasil})

    # Guardrail kena: LLM terus minta tool tanpa selesai -> berhenti sopan.
    return "(loop berhenti: batas putaran tercapai tanpa jawaban final)"


# --------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Agent tool-calling tanpa framework — Peraturan Indonesia")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="demo loop dengan MockLLM (tool asli, tanpa API)")
    args = ap.parse_args()

    print("=" * 72)
    print("AGENT TOOL-CALLING" + (" — MODE MOCK (dry-run)" if args.dry_run else ""))
    print("=" * 72)
    jawaban = jalankan(args.question, dry=args.dry_run)
    print("\n" + "=" * 72)
    print("JAWABAN FINAL")
    print("=" * 72)
    print(jawaban)


if __name__ == "__main__":
    main()
