"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, muatSetting, simpanSetting } from "../lib/store";
import {
  ChatTurn, HistoryMeta, Settings, TokenUsage, TraceNode, DEFAULT_SETTINGS,
} from "../lib/types";
import ChatPanel from "../components/ChatPanel";
import SettingPanel from "../components/SettingPanel";
import GrafPanel from "../components/GrafPanel";
import TipGlobal from "../components/TipGlobal";
import { I } from "../components/icons";
import { t, type Lang } from "../lib/i18n";

type Theme = "dark" | "light";

function grupTanggal(h: HistoryMeta): "hari_ini" | "kemarin" | "minggu" | "lama" {
  const d = new Date(h.created_at);
  if (isNaN(+d)) return "lama";
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(new Date()) - day(d)) / 86400000);
  if (diff <= 0) return "hari_ini";
  if (diff === 1) return "kemarin";
  if (diff <= 7) return "minggu";
  return "lama";
}
const URUT_GRUP = ["hari_ini", "kemarin", "minggu", "lama"] as const;

export default function Halaman() {
  const [setting, setSetting] = useState<Settings>(DEFAULT_SETTINGS);
  const L = setting.bahasa || "id";
  const T = (k: string) => t(L, k);
  const LOC = L === "id" ? "id-ID" : "en-US";
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [historyList, setHistoryList] = useState<HistoryMeta[]>([]);
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

  // ---- theme (dark/light, follows OS when never chosen) ----
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("rag-hukum-theme");
      if (saved === "light" || saved === "dark") { setTheme(saved); return; }
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches)
        setTheme("light");
    } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { window.localStorage.setItem("rag-hukum-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  // ---- UI shell ----
  const [sidebarTutup, setSidebarTutup] = useState(false);
  const [drawer, setDrawer] = useState<null | "setting" | "graf">("setting");
  const [cari, setCari] = useState("");
  const [showAbout, setShowAbout] = useState(false);
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [confirmHapus, setConfirmHapus] = useState<string | null>(null);
  // docked panel width (draggable via the handle between chat and panel)
  const [lebarKanan, setLebarKanan] = useState(420);
  const dragKanan = useRef<{ sx: number; awal: number } | null>(null);
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragKanan.current) return;
      const w = dragKanan.current.awal + (dragKanan.current.sx - e.clientX);
      setLebarKanan(Math.max(320, Math.min(640, w)));
    };
    const up = () => { dragKanan.current = null; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);
  const searchRef = useRef<HTMLInputElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Esc closes drawer / about / menu / confirm
  useEffect(() => {
    if (!showAbout && !drawer && !openMenu && !confirmHapus) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setShowAbout(false); setDrawer(null); setOpenMenu(null); setConfirmHapus(null); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [showAbout, drawer, openMenu, confirmHapus]);
  // focus the drawer close button when it opens (keyboard users)
  useEffect(() => {
    if (drawer) closeBtnRef.current?.focus();
  }, [drawer]);
  // Cmd/Ctrl+K focuses history search
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (sidebarTutup) setSidebarTutup(false);
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [sidebarTutup]);

  // ---- autosave toast (debounced "Saved") ----
  const pertama = useRef(true);
  const [toast, setToast] = useState("");
  useEffect(() => {
    if (pertama.current) { pertama.current = false; return; }
    const id = setTimeout(() => {
      setToast(t(setting.bahasa || "id", "toast.tersimpan"));
      setTimeout(() => setToast(""), 1800);
    }, 1200);
    return () => clearTimeout(id);
  }, [setting]);

  // load settings + history
  useEffect(() => {
    const s = muatSetting();
    setSetting(s);
    // Prefill dari .env backend HANYA bila user belum punya konfigurasi sendiri
    if (!s.llm.base_url && !s.llm.api_key) {
      api.envDefaults().then((d) => {
        if (!d || !d.ok) return;
        setSetting((cur) => ({
          ...cur,
          llm: { ...cur.llm,
            base_url: d.base_url || cur.llm.base_url,
            model: d.model || cur.llm.model },
          llmEnvKey: !!d.has_api_key,
        }));
      }).catch(() => {});
    }
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
    setBusy(true); setError("");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
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
      }, ctrl.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        ubahBot((b) => ({ ...b, streaming: false,
          status: t(s.bahasa || setting.bahasa || "id", "chat.stop.status") }));
      } else {
        setError((e as Error).message || "gagal memanggil /api/ask/stream");
        ubahBot((b) => ({ ...b, streaming: false, status: "gagal" }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [busy, setting, turns, tracesGraf]);

  const stop = useCallback(() => { abortRef.current?.abort(); }, []);

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
      // graph loads from history: fill with this history's trace (not cumulative)
      setTracesGraf(d.trace || []);
      // continue the token counter from this history (don't restart at 0)
      const tk = (d as { usage?: { tokens?: TokenUsage } }).usage?.tokens;
      if (tk && (tk.total_tokens || 0) > 0)
        setTotalToken({
          prompt_tokens: tk.prompt_tokens || 0,
          completion_tokens: tk.completion_tokens || 0,
          total_tokens: tk.total_tokens || 0,
          cached_tokens: tk.cached_tokens || 0,
        });
      setOpenMenu(null);
    } catch (e) { setError((e as Error).message); }
  }, []);

  const hapusRiwayat = useCallback(async (id: string) => {
    await api.historyDelete(id);
    setHistoryList((h) => h.filter((x) => x.id !== id));
    setOpenMenu(null);
  }, []);

  const aktifId = turns.length > 0
    ? (turns[turns.length - 1] as { historyId?: string }).historyId : undefined;
  // graph = cumulative for this session (follow-up chats keep growing it)
  const traceAktif: TraceNode[] = tracesGraf.length > 0
    ? tracesGraf
    : ((turns.length > 0 ? turns[turns.length - 1] : undefined)?.trace || []);

  const replay = useCallback((nodeIdx: number) => {
    const q = [...turns].reverse().find((t) => t.role === "user")?.text || "";
    kirim(q, undefined, traceAktif[nodeIdx]?.id, nodeIdx + 1);
  }, [turns, traceAktif, kirim]);

  const riwayatCari = cari.trim()
    ? historyList.filter((h) => (h.question || "").toLowerCase().includes(cari.trim().toLowerCase()))
    : historyList;
  const grup: Record<string, HistoryMeta[]> = { hari_ini: [], kemarin: [], minggu: [], lama: [] };
  riwayatCari.forEach((h) => { grup[grupTanggal(h)].push(h); });

  return (
    <div className="app">
      <TipGlobal />
      {/* ================= header 56px ================= */}
      <header className="app-header">
        <button className="icon-btn sq" data-tip={sidebarTutup ? T("page.buka") : T("page.tutup")}
                aria-label={sidebarTutup ? T("page.buka") : T("page.tutup")}
                onClick={() => setSidebarTutup((v) => !v)}>
          <I.menu />
        </button>
        <div className="brand">ChatBOT Law ID</div>
        <div className="header-spacer" />
        <div className="header-status">
          <div className="token-head">
            <span className="tk-sub">in {totalToken.prompt_tokens.toLocaleString(LOC)} · out {totalToken.completion_tokens.toLocaleString(LOC)}</span>
            <span className="tk-total"><b>{totalToken.total_tokens.toLocaleString(LOC)}</b> {T("page.token")}</span>
          </div>
          <button className={`icon-btn sq ${drawer === "graf" ? "on" : ""}`}
                  data-tip={T("page.tab.graf.tip")} aria-label={T("page.tab.graf")}
                  onClick={() => setDrawer((d) => (d === "graf" ? null : "graf"))}>
            <I.graph />
          </button>
          <button className={`icon-btn sq ${drawer === "setting" ? "on" : ""}`}
                  data-tip={T("page.tab.setting.tip")} aria-label={T("page.tab.setting")}
                  onClick={() => setDrawer((d) => (d === "setting" ? null : "setting"))}>
            <I.gear />
          </button>
          <button className="icon-btn sq" data-tip={T("page.tema.tip")} aria-label={T("page.tema.tip")}
                  onClick={() => setTheme((v) => (v === "dark" ? "light" : "dark"))}>
            {theme === "dark" ? <I.sun /> : <I.moon />}
          </button>
          <div className="seg-lang" role="group" aria-label={T("page.bahasa.tip")}>
            {(["id", "en"] as const).map((b) => (
              <button key={b} className={L === b ? "on" : ""} data-tip={T("page.bahasa.tip")}
                      onClick={() => setSetting({ ...setting, bahasa: b })}>
                {b.toUpperCase()}
              </button>
            ))}
          </div>
          <button className="icon-btn sq" data-tip={T("page.tanya.tip")} aria-label={T("page.tanya.tip")}
                  onClick={() => setShowAbout(true)}>
            <I.help />
          </button>
        </div>
      </header>
      {showAbout && (
        <div className="about-backdrop" onClick={() => setShowAbout(false)}>
          <div className="about-card" role="dialog" aria-modal="true" aria-label={T("about.judul")}
               onClick={(e) => e.stopPropagation()}>
            <div className="about-head">
              <div className="about-title">{T("about.judul")}</div>
              <button className="icon-btn sq" onClick={() => setShowAbout(false)}
                      aria-label={T("about.tutup")}><I.x /></button>
            </div>
            <div className="about-body">
              <p>{T("about.p1")}</p>
              <p>{T("about.p2")}</p>
              <p className="about-warn">{T("about.p3")}</p>
            </div>
            <button className="btn block" onClick={() => setShowAbout(false)}>
              {T("about.tutup")}
            </button>
          </div>
        </div>
      )}

      <div className="app-body">
        {/* ================= sidebar ================= */}
        {!sidebarTutup && <div className="side-backdrop" onClick={() => setSidebarTutup(true)} />}
        <aside className={`sidebar ${sidebarTutup ? "tutup" : ""}`} data-testid="panel-kiri">
          <div className="sidebar-head">
            <button className="btn block newchat" data-tip={T("page.baru.tip")}
                    onClick={() => { setTurns([]); setError(""); setTracesGraf([]);
                      setTotalToken(
                        { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cached_tokens: 0 }); }}>
              <I.plus /> {T("page.baru")}
            </button>
          </div>
          <div className="side-search">
            <I.search />
            <input ref={searchRef} value={cari} placeholder={T("page.cari")}
                   onChange={(e) => setCari(e.target.value)} aria-label={T("page.cari")} />
            <kbd>⌘K</kbd>
          </div>
          <div className="side-list">
            {riwayatCari.length === 0 && (
              <div className="side-empty">{T("side.kosong")}</div>
            )}
            {URUT_GRUP.map((g) => grup[g].length > 0 && (
              <div key={g}>
                <div className="side-group">{T(`side.grup.${g}`)}</div>
                {grup[g].map((h) => (
                  <div key={h.id} className={`side-item ${aktifId === h.id ? "aktif" : ""}`}>
                    <div className="row">
                      <button className="txt" style={{ background: "none", border: "none", padding: 0,
                                                      textAlign: "left", cursor: "pointer", color: "inherit",
                                                      font: "inherit" }}
                              onClick={() => bukaRiwayat(h.id)}>
                        <div className="q">{h.question?.slice(0, 120) || T("page.kosong.q")}</div>
                        <div className="meta">
                          <span>{new Date(h.created_at).toLocaleTimeString(LOC, { hour: "2-digit", minute: "2-digit" })}</span>
                          <span>·</span>
                          <span style={{ textTransform: "capitalize" }}>{h.mode} · {h.retrieval}</span>
                        </div>
                      </button>
                      <button className="menu-btn" data-tip={T("side.menu")}
                              aria-label={T("side.menu")}
                              onClick={() => setOpenMenu((m) => (m === h.id ? null : h.id))}>
                        <I.dots />
                      </button>
                    </div>
                    {openMenu === h.id && (
                      <>
                        <div className="menu-backdrop"
                             onClick={(e) => { e.stopPropagation(); setOpenMenu(null); }} />
                        <div className="menu-pop">
                          <div className="menu-sec">
                            <div className="menu-label">{T("menu.ekspor")}</div>
                            <a style={{ textDecoration: "none", color: "inherit" }}
                               href={api.exportUrl(h.id, "md")} target="_blank" rel="noreferrer">
                              <button><I.file />{T("menu.md")}</button>
                            </a>
                            <a style={{ textDecoration: "none", color: "inherit" }}
                               href={api.exportUrl(h.id, "txt")} target="_blank" rel="noreferrer">
                              <button><I.file />{T("menu.txt")}</button>
                            </a>
                            <a style={{ textDecoration: "none", color: "inherit" }}
                               href={api.exportUrl(h.id, "html")} target="_blank" rel="noreferrer">
                              <button><I.file />{T("menu.pdf")}</button>
                            </a>
                          </div>
                          <div className="menu-del-box">
                            <button className="menu-del" onClick={() => {
                              setOpenMenu(null); setConfirmHapus(h.id);
                            }}><I.trash />{T("menu.hapus")}</button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </aside>

        {/* ================= center chat ================= */}
        <section className="chat-col">
          <ChatPanel
            turns={turns} busy={busy} error={error} dryRun={setting.dryRun}
            semuaSitasi={turns.flatMap((t) => t.citations || [])}
            bahasa={L} totalToken={totalToken}
            onStop={stop}
            onKirim={(q) => kirim(q)}
          />
        </section>

        {/* ================= docked panel (in-flow, pushes layout) ================= */}
        {drawer && (
          <div className="dock-resizer" data-tip={T("page.resize.tip")}
               onMouseDown={(e) => { dragKanan.current = { sx: e.clientX, awal: lebarKanan };
                 document.body.style.cursor = "col-resize"; }} />
        )}
        <aside className={`ov-panel ${drawer ? "open" : ""}`} data-testid="panel-kanan"
               style={drawer ? { width: lebarKanan, minWidth: lebarKanan, maxWidth: lebarKanan } : undefined}
               role="dialog" aria-modal="false"
               aria-label={drawer === "graf" ? T("page.tab.graf") : T("page.tab.setting")}>
          <div className="ov-head">
            <div className="ov-title">
              {drawer === "graf" ? T("page.tab.graf") : T("page.tab.setting")}
            </div>
            <button ref={closeBtnRef} className="icon-btn sq" onClick={() => setDrawer(null)}
                    aria-label={T("about.tutup")}><I.x /></button>
          </div>
          <div className="ov-body" style={{ display: drawer === "graf" ? "none" : "" }}>
            {drawer !== "graf" && (
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
            )}
          </div>
          <div className="ov-body" style={{ display: drawer === "graf" ? "" : "none" }}>
            {drawer === "graf" && (
              <GrafPanel trace={traceAktif} onReplay={replay} bahasa={L} />
            )}
          </div>
        </aside>
      </div>

      {confirmHapus && (
        <div className="about-backdrop" onClick={() => setConfirmHapus(null)}>
          <div className="about-card kecil" role="alertdialog" aria-modal="true"
               aria-label={T("hapus.yakin")}
               onClick={(e) => e.stopPropagation()}>
            <div className="about-title" style={{ marginBottom: 4 }}>{T("hapus.yakin")}</div>
            <div className="confirm-row">
              <button className="btn ghost" onClick={() => setConfirmHapus(null)}>
                {T("hapus.tidak")}
              </button>
              <button className="confirm-yes" onClick={() => {
                const id = confirmHapus; setConfirmHapus(null); hapusRiwayat(id);
              }}>
                {T("hapus.ya")}
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className="toast show" role="status">{toast}</div>}
    </div>
  );
}
