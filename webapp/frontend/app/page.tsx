"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, muatSetting, simpanSetting } from "../lib/store";
import {
  ChatTurn, HistoryMeta, Settings, TraceNode, DEFAULT_SETTINGS,
} from "../lib/types";
import ChatPanel from "../components/ChatPanel";
import SettingPanel from "../components/SettingPanel";
import GrafPanel from "../components/GrafPanel";
import TipGlobal from "../components/TipGlobal";
import { t, type Lang } from "../lib/i18n";

type Theme = "dark" | "light";

export default function Halaman() {
  const [setting, setSetting] = useState<Settings>(DEFAULT_SETTINGS);
  const L = setting.bahasa || "id";
  const T = (k: string) => t(L, k);
  const LOC = L === "id" ? "id-ID" : "en-US";
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [historyList, setHistoryList] = useState<HistoryMeta[]>([]);
  const [aktifTurn, setAktifTurn] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // conversation memory: all turns of this session go to the LLM + graph
  const [tracesGraf, setTracesGraf] = useState<TraceNode[]>([]);
  // session token total (persist localStorage → watch provider limits)
  const [totalToken, setTotalToken] = useState(
    { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cached_tokens: 0 });
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("rag-hukum-token-total");
      if (raw) setTotalToken((t) => ({ ...t, ...JSON.parse(raw) }));
    } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    try {
      window.localStorage.setItem("rag-hukum-token-total", JSON.stringify(totalToken));
    } catch { /* ignore */ }
  }, [totalToken]);

  // ---- theme (dark/light) ----
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("rag-hukum-theme");
      if (saved === "light" || saved === "dark") setTheme(saved);
    } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { window.localStorage.setItem("rag-hukum-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  // ---- UI shell ----
  const [sidebarTutup, setSidebarTutup] = useState(false);
  const [drawer, setDrawer] = useState<null | "setting" | "graf">(null);
  const [cari, setCari] = useState("");
  // docked panel width (draggable via the handle between chat and panel)
  const [lebarKanan, setLebarKanan] = useState(400);
  const dragKanan = useRef<{ sx: number; awal: number } | null>(null);
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragKanan.current) return;
      const w = dragKanan.current.awal + (dragKanan.current.sx - e.clientX);
      setLebarKanan(Math.max(300, Math.min(640, w)));
    };
    const up = () => { dragKanan.current = null; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);
  // Esc closes the floating drawer
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDrawer(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // load settings + history
  useEffect(() => {
    setSetting(muatSetting());
    api.history().then((h) => setHistoryList(h.daftar || [])).catch(() => {});
  }, []);
  useEffect(() => { simpanSetting(setting); }, [setting]);

  const kirim = useCallback(async (question: string, override?: Partial<Settings>,
                                    replayOfNodeId?: string, resumeFrom?: number) => {
    const q = question.trim();
    if (!q || busy) return;
    const s: Settings = { ...setting, ...override };
    if (override) setSetting(s);
    // replay: trim graph from the resume point (backend re-runs fully)
    const basisGraf = (replayOfNodeId && resumeFrom != null)
      ? tracesGraf.slice(0, Math.max(0, resumeFrom - 1))
      : tracesGraf;
    if (replayOfNodeId && resumeFrom != null) setTracesGraf(basisGraf);
    // OPTIMISTIC: user message + bot bubble appear instantly, answer streams in
    setTurns((t) => [...t, { role: "user", text: q },
      { role: "bot", text: "", streaming: true, status: "Menyiapkan…",
        trace: [], mode: s.mode, dryRun: s.dryRun, replayOfNodeId }]);
    setAktifTurn((t) => t + 2);
    setBusy(true); setError("");
    // mutate the LAST bot bubble (this send; sends are busy-locked)
    const ubahBot = (fn: (b: ChatTurn) => ChatTurn) =>
      setTurns((t) => {
        if (t.length === 0) return t;
        const n = [...t];
        n[n.length - 1] = fn({ ...(n[n.length - 1] as ChatTurn) });
        return n;
      });
    try {
      // memory: all previous turns sent explicitly to the backend
      const riwayat = [...turns]
        .reduce<({ role: "user" | "assistant"; content: string } | undefined)[]>((acc, t, i) => {
          if ((t.role === "user" || (t.role === "bot" && t.text && t.historyId))
              && t.text) {
            acc.push({
              role: t.role === "user" ? "user" : "assistant",
              content: t.text,
            } as { role: "user" | "assistant"; content: string });
          }
          return acc;
        }, [])
        .filter(Boolean) as { role: "user" | "assistant"; content: string }[];
      const konvId = turns.length > 0
        ? `sesi-${Date.now()}` : undefined;   // per-session-turn UUID
      // global session citation numbers: continue from the highest n ever
      // used (old S numbers must never be reused).
      const nomorPakai = turns.flatMap((t) => (t.citations || []).map((c) => c.n));
      const nomorAwal = nomorPakai.length ? Math.max(...nomorPakai) + 1 : 1;
      const body: Record<string, unknown> = {
        question: q,
        mode: s.mode,
        paksa_cari: true,
        nomor_awal: nomorAwal,
        node_awal: basisGraf.length + 1,
        retrieval: s.retrieval,
        dense_method: s.denseMethod,
        dense_ef: s.denseEf,
        dense_nprobe: s.denseNprobe,
        gaya: s.gaya,
        rerank: s.rerank,
        top_k: s.topK,
        inject_k: Math.max(1, Math.min(s.injectK, s.topK)),
        doc_type: s.docTypes.length > 0 ? s.docTypes : null,
        year: s.year ? Number(s.year) : null,
        embedding: s.embedding,
        dry_run: s.dryRun,
        conversation_id: konvId,
        riwayat_chat: riwayat.length ? riwayat : undefined,
        llm: { provider: s.llm.provider, base_url: s.llm.base_url,
               api_key: s.llm.api_key, model: s.llm.model,
               temperature: s.llm.temperature, max_tokens: s.llm.maxTokens },
      };
      if (resumeFrom != null) body.resume_from = resumeFrom;
      await api.askStream(body, (ev) => {
        if (ev.t === "status") {
          ubahBot((b) => ({ ...b, status: (ev as { text: string }).text }));
        } else if (ev.t === "node") {
          const node = (ev as { node: TraceNode }).node;
          ubahBot((b) => ({ ...b, trace: [...(b.trace || []), node] }));
          setTracesGraf((tr) => [...tr, node]);
        } else if (ev.t === "token") {
          ubahBot((b) => ({ ...b, text: (b.text || "") + (ev as { text: string }).text }));
        } else if (ev.t === "reasoning") {
          ubahBot((b) => ({ ...b,
            reasoning: ((b.reasoning || "") + (ev as { text: string }).text) || null }));
        } else if (ev.t === "done") {
          const d = ev as { answer: string; citations: ChatTurn["citations"];
            trace: TraceNode[]; usage: ChatTurn["usage"]; mode: string; history_id: string;
            reasoning?: string | null; reasoning_raw?: string | null };
          ubahBot((b) => ({ ...b, text: d.answer, citations: d.citations,
            trace: d.trace && d.trace.length ? d.trace : b.trace,
            usage: d.usage, mode: d.mode, historyId: d.history_id,
            reasoning: d.reasoning ?? b.reasoning ?? null,
            reasoningRaw: d.reasoning_raw ?? null,
            streaming: false, status: undefined }));
          const tk = d.usage?.tokens;
          if (tk && tk.total_tokens > 0)
            setTotalToken((t) => ({
              prompt_tokens: t.prompt_tokens + (tk.prompt_tokens || 0),
              completion_tokens: t.completion_tokens + (tk.completion_tokens || 0),
              total_tokens: t.total_tokens + (tk.total_tokens || 0),
              cached_tokens: t.cached_tokens + (tk.cached_tokens || 0),
            }));
          if (replayOfNodeId) setDrawer("graf");
          api.history().then((h) => setHistoryList(h.daftar || [])).catch(() => {});
        }
        // "error" is thrown by askStream as an exception
      });
    } catch (e) {
      setError((e as Error).message || "gagal memanggil /api/ask/stream");
      ubahBot((b) => ({ ...b, streaming: false, status: "gagal" }));
    } finally {
      setBusy(false);
    }
  }, [busy, setting, turns, tracesGraf]);

  const bukaRiwayat = useCallback(async (id: string) => {
    try {
      const d = await api.historyDetail(id);
      setTurns([
        { role: "user", text: d.question },
        { role: "bot", text: d.answer, citations: d.citations,
          trace: d.trace, usage: d.usage, mode: d.mode, historyId: id,
          reasoning: (d as { reasoning?: string | null }).reasoning ?? null,
          reasoningRaw: (d as { reasoning_raw?: string | null }).reasoning_raw ?? null },
      ]);
      setAktifTurn(1);
      // graph loads from history: fill with this history's trace (not cumulative)
      setTracesGraf(d.trace || []);
    } catch (e) { setError((e as Error).message); }
  }, []);

  const hapusRiwayat = useCallback(async (id: string) => {
    await api.historyDelete(id);
    setHistoryList((h) => h.filter((x) => x.id !== id));
  }, []);

  const chatAktif = turns.length > 0 ? turns[turns.length - 1] : undefined;
  const aktifId = (chatAktif as { historyId?: string } | undefined)?.historyId;
  // graph = cumulative for this session (follow-up chats keep growing it)
  const traceAktif: TraceNode[] = tracesGraf.length > 0
    ? tracesGraf
    : ((turns.length > 0 ? turns[turns.length - 1] : undefined)?.trace || []);

  const replay = useCallback((nodeIdx: number) => {
    const q = [...turns].reverse().find((t) => t.role === "user")?.text || "";
    kirim(q, undefined, traceAktif[nodeIdx]?.id, nodeIdx + 1);
  }, [turns, traceAktif, kirim]);

  const riwayatTampil = cari.trim()
    ? historyList.filter((h) => (h.question || "").toLowerCase().includes(cari.trim().toLowerCase()))
    : historyList;

  const gantiBahasa = () => {
    const next: Lang = L === "id" ? "en" : "id";
    setSetting({ ...setting, bahasa: next });
  };

  return (
    <div className="app">
      <TipGlobal />
      {/* ================= header ================= */}
      <header className="app-header">
        <button className="icon-btn" data-tip={sidebarTutup ? T("page.buka") : T("page.tutup")}
                onClick={() => setSidebarTutup((v) => !v)}>
          ☰
        </button>
        <div className="brand">ChatBOT</div>
        {setting.dryRun && (
          <span className="tag bm25" style={{ fontSize: 11 }}>DRY-RUN</span>
        )}
        <div className="header-spacer" />
        <div className="header-status">
          <button className="icon-btn lang" data-tip={T("page.bahasa.tip")}
                  onClick={gantiBahasa}>
            <span className="lbl">{L.toUpperCase()}</span>
          </button>
          <button className="icon-btn sq" data-tip={T("page.tema.tip")}
                  onClick={() => setTheme((v) => (v === "dark" ? "light" : "dark"))}
                  aria-label={T("page.tema.tip")}>
            {theme === "dark" ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2" /><path d="M12 20v2" />
                <path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" />
                <path d="M2 12h2" /><path d="M20 12h2" />
                <path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
              </svg>
            )}
          </button>
          <button className={`icon-btn ${drawer === "graf" ? "on" : ""}`}
                  data-tip={T("page.tab.graf.tip")}
                  onClick={() => setDrawer((d) => (d === "graf" ? null : "graf"))}>
            <span className="lbl">{T("page.tab.graf")}</span>
          </button>
          <button className={`icon-btn ${drawer === "setting" ? "on" : ""}`}
                  data-tip={T("page.tab.setting.tip")}
                  onClick={() => setDrawer((d) => (d === "setting" ? null : "setting"))}>
            <span className="lbl">{T("page.tab.setting")}</span>
          </button>
        </div>
      </header>

      <div className="app-body">
        {/* ================= left: history ================= */}
        <aside className={`sidebar ${sidebarTutup ? "tutup" : ""}`} data-testid="panel-kiri">
          <div className="sidebar-head">
            <button className="btn block" data-tip={T("page.baru.tip")}
                    onClick={() => { setTurns([]); setAktifTurn(-1); setError("");
                      setTracesGraf([]);
                      setTotalToken(
                        { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cached_tokens: 0 }); }}>
              {T("page.baru")}
            </button>
          </div>
          <input className="side-search" value={cari} placeholder={T("page.cari")}
                 onChange={(e) => setCari(e.target.value)} />
          <div className="side-list">
            {riwayatTampil.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, padding: "0 12px" }}>
                {T("page.kosong")}
              </div>
            )}
            {riwayatTampil.map((h) => (
              <div key={h.id} className={`side-item ${aktifId === h.id ? "aktif" : ""}`}
                   onClick={() => bukaRiwayat(h.id)}>
                <div className="q">{h.question?.slice(0, 72) || T("page.kosong.q")}</div>
                <div className="meta">
                  <span>{new Date(h.created_at).toLocaleString(LOC)}</span>
                  <span className="dot" />
                  <span className={`chip ${h.mode === "agent" ? "agent" : ""}`}>{h.mode}</span>
                  <span className={`chip ${h.retrieval}`}>{h.retrieval}</span>
                </div>
                <div className="side-actions" onClick={(e) => e.stopPropagation()}>
                  <a className="mini-btn" href={api.exportUrl(h.id, "md")}
                     target="_blank" rel="noreferrer">MD</a>
                  <a className="mini-btn" href={api.exportUrl(h.id, "txt")}
                     target="_blank" rel="noreferrer">TXT</a>
                  <a className="mini-btn" href={api.exportUrl(h.id, "html")}
                     target="_blank" rel="noreferrer">PDF</a>
                  <button className="mini-btn" data-tip={T("page.hapus.tip")} onClick={() => hapusRiwayat(h.id)}>✕</button>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* ================= center: chat ================= */}
        <section className="chat-col">
          <ChatPanel
            turns={turns} busy={busy} error={error} dryRun={setting.dryRun}
            semuaSitasi={turns.flatMap((t) => t.citations || [])}
            bahasa={L} totalToken={totalToken}
            onKirim={(q) => kirim(q)}
            onCiteClick={(c) => alert(c.official_title
              ? `[S${c.n}] ${c.official_title} — ${c.citation_str}\n\n` +
                `${T("page.skor")}: cosine=${c.score}` +
                (c.skor_bm25 != null ? ` · BM25=${c.skor_bm25}` : "") +
                (c.skor_fusi != null ? ` · ${T("page.fusi")}=${c.skor_fusi}` : "") +
                `\n${T("chat.src.hal")}.${c.page ?? "?"} ${T("chat.src.baris")} ${c.lines?.[0] ?? "?"}-${c.lines?.[1] ?? "?"}` +
                `\n\n${c.text?.slice(0, 800)}`
              : `[S${c.n}] ${T("page.sitasi.hilang")} ` +
                T("page.sitasi.hilang2"))}
          />
        </section>

        {/* ================= right: docked panel ================= */}
        {drawer && (
          <div className="resizer-kanan" data-testid="resizer-kanan"
               title={T("page.resize.tip")}
               onMouseDown={(e) => { dragKanan.current = { sx: e.clientX, awal: lebarKanan };
                 document.body.style.cursor = "col-resize"; }} />
        )}
        <aside className={`drawer ${drawer ? "open" : ""}`} data-testid="panel-kanan"
               style={drawer ? { width: lebarKanan, minWidth: lebarKanan, maxWidth: lebarKanan } : undefined}>
          <div className="drawer-head">
            <div className="drawer-title">
              {drawer === "graf" ? T("page.tab.graf") : T("page.tab.setting")}
            </div>
            <div className="header-spacer" />
            <button className="icon-btn" onClick={() => setDrawer(null)}>✕</button>
          </div>
          <div className="kanan-body" style={{ display: drawer === "graf" ? "none" : "" }}>
            <SettingPanel setting={setting} onChange={setSetting}
              onUsage={(tk) => {
                if (!tk || (tk.total_tokens || 0) <= 0) return;
                setTotalToken((t) => ({
                  prompt_tokens: t.prompt_tokens + (tk.prompt_tokens || 0),
                  completion_tokens: t.completion_tokens + (tk.completion_tokens || 0),
                  total_tokens: t.total_tokens + (tk.total_tokens || 0),
                  cached_tokens: t.cached_tokens + (tk.cached_tokens || 0),
                }));
              }} />
          </div>
          <div className="kanan-body" style={{ display: drawer === "graf" ? "" : "none" }}>
            <GrafPanel trace={traceAktif} onReplay={replay} bahasa={L} />
          </div>
        </aside>
      </div>
    </div>
  );
}
