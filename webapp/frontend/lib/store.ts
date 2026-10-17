"use client";
import { DEFAULT_SETTINGS, Settings } from "./types";

const KEY = "rag-hukum-settings-v2";   // v2: default baru (agent/rerank on/dry-run off)

export function muatSetting(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed, llm: { ...DEFAULT_SETTINGS.llm, ...(parsed.llm || {}) } };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function simpanSetting(s: Settings) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, JSON.stringify(s));
}

const API = "http://localhost:8000";

async function res_ok(r: Response) {
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* abaikan */ }
    throw new Error(detail);
  }
  return r.json();
}

export type StreamEvent =
  | { t: "status"; text: string }
  | { t: "node"; node: object }
  | { t: "token"; text: string }
  | { t: "reasoning"; text: string }
  | { t: "usage"; usage: object }
  | { t: "done"; answer: string; citations: object[]; trace: object[];
      usage: object; mode: string; history_id: string; reasoning?: string | null;
      reasoning_raw?: string | null }
  | { t: "error"; message: string };

export const api = {
  ask: (body: object) =>
    fetch(`${API}/api/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(res_ok),

  // SSE progresif: pesan user sudah tampil duluan, lalu event mengalir.
  // onEvent dipanggil per frame; resolve saat done, throw saat error.
  // signal opsional (AbortController) untuk tombol stop.
  askStream: async (body: object, onEvent: (ev: StreamEvent) => void, signal?: AbortSignal) => {
    const r = await fetch(`${API}/api/ask/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal,
    });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch { /* abaikan */ }
      throw new Error(detail);
    }
    const reader = r.body!.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of frame.split("\n")) {
          const t = line.trim();
          if (!t.startsWith("data:")) continue;
          const payload = t.slice(5).trim();
          if (payload === "[DONE]") return;
          try {
            const ev = JSON.parse(payload) as StreamEvent;
            if (ev.t === "error") throw new Error((ev as { message: string }).message);
            onEvent(ev);
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      }
    }
  },

  health: () => fetch(`${API}/api/health`).then(res_ok),
  milvusStatus: () => fetch(`${API}/api/milvus/status`).then(res_ok),
  models: () => fetch(`${API}/api/models`).then(res_ok),
  // Default LLM dari .env backend (prefill Setting; api_key tidak dikirim)
  envDefaults: () => fetch(`${API}/api/llm/env-defaults`).then(res_ok),
  switchEmbedding: (tag: string) =>
    fetch(`${API}/api/embedding/switch`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag }),
    }).then(res_ok),

  history: () => fetch(`${API}/api/history`).then(res_ok),
  historyDetail: (id: string) => fetch(`${API}/api/history/${id}`).then(res_ok),
  historyDelete: (id: string) => fetch(`${API}/api/history/${id}`, { method: "DELETE" }).then(res_ok),
  exportUrl: (id: string, format: "md" | "txt" | "html") =>
    `${API}/api/history/${id}/export?format=${format}`,

  testKoneksi: (llm: Settings["llm"]) =>
    // Test koneksi = POST /api/llm/ping: 1 panggilan mini TANPA retrieval
    // (dulu memakai /api/ask sehingga ikut mencari di korpus + mahal).
    api.ping({
      llm: {
        provider: llm.provider === "custom" ? "custom" : llm.provider,
        base_url: llm.base_url, api_key: llm.api_key, model: llm.model,
      },
    }).then((r) => ({
      answer: String((r as { answer?: unknown }).answer ?? ""),
      ms: (r as { ms?: unknown }).ms,
      usage: (r as { usage?: unknown }).usage,
    })),

  ping: (body: object) =>
    fetch(`${API}/api/llm/ping`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(res_ok),
};
