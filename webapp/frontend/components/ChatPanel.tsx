"use client";
import { Fragment, useEffect, useRef, useState } from "react";
import { Citation, ChatTurn } from "../lib/types";
import { t, type Lang } from "../lib/i18n";

const DEMO_Q = "Di tangan siapa kedaulatan berada menurut UUD 1945?";
const DEMO_Q_EN = "In whose hands does sovereignty rest under the 1945 Constitution?";

const SARAN_ID = [
  "Berapa lama masa jabatan Presiden menurut UUD 1945?",
  "Siapa pemegang kekuasaan kehakiman?",
  "Apa jaminan beragama menurut UUD 1945?",
];
const SARAN_EN = [
  "How long is the presidential term under the 1945 Constitution?",
  "Who holds judicial power?",
  "What religious guarantees exist in the 1945 Constitution?",
];

export default function ChatPanel({ turns, busy, error, dryRun, semuaSitasi, bahasa, totalToken, onKirim, onCiteClick }: {
  turns: ChatTurn[];
  busy: boolean;
  error: string;
  dryRun: boolean;
  semuaSitasi?: Citation[];
  bahasa: Lang;
  totalToken: { prompt_tokens: number; completion_tokens: number; total_tokens: number; cached_tokens: number };
  onKirim: (q: string) => void;
  onCiteClick: (c: Citation) => void;
}) {
  const L: Lang = bahasa || "id";
  const T = (k: string) => t(L, k);
  const LOC = L === "id" ? "id-ID" : "en-US";
  // global session lookup: old S numbers from earlier messages still resolve.
  const cariSitasi = (n: number, lokal: Citation[]): Citation | undefined =>
    lokal.find((c) => c.n === n)
    || (semuaSitasi || []).find((c) => c.n === n);
  const [teks, setTeks] = useState("");
  const kirim = () => {
    if (!teks.trim() || busy) return;
    onKirim(teks.trim());
    setTeks("");
  };

  // auto-scroll to bottom on new content
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, turns[turns.length - 1]?.text?.length, busy]);

  // auto-growing textarea
  const areaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(160, el.scrollHeight) + "px";
  }, [teks]);

  const saran = L === "id" ? SARAN_ID : SARAN_EN;

  return (
    <>
      <div className="token-pill">
        <span>input/output : {totalToken.prompt_tokens.toLocaleString(LOC)}/{totalToken.completion_tokens.toLocaleString(LOC)}</span>
        <span>total : {totalToken.total_tokens.toLocaleString(LOC)}</span>
      </div>
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-inner">
          {turns.length === 0 && (
            <div className="hero">
              <div className="hero-mark">⚖️</div>
              <h2>{T("chat.hero.judul")}</h2>
              <p>
                {T("chat.hero.a")} <b>{T("chat.hero.b")}</b> {T("chat.hero.c")}
                <i> {T("chat.hero.d")}</i> {T("chat.hero.e")}{" "}
                <b>{T("chat.hero.f")}</b>, <b>{T("chat.hero.g")}</b>
                {T("chat.hero.h")}
              </p>
              <div className="suggest-row">
                <button className="demo-btn" data-tip={T("chat.demo.tip")}
                        onClick={() => onKirim(L === "id" ? DEMO_Q : DEMO_Q_EN)}>
                  🏛️ {L === "id" ? DEMO_Q : DEMO_Q_EN}
                </button>
                {saran.map((s) => (
                  <button key={s} className="suggest-chip" onClick={() => onKirim(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((t, i) => (
            <Fragment key={i}>
              {t.role === "user" ? (
                <div className="msg user">
                  <div className="bubble user">{t.text}</div>
                </div>
              ) : (
                <div className="msg bot">
                  <div className="avatar">⚖️</div>
                  <div className="body">
                    {t.replayOfNodeId && (
                      <div style={{ fontSize: 11, color: "var(--accent-2)", marginBottom: 5 }}>
                        {T("chat.replay")} {t.replayOfNodeId}
                      </div>
                    )}
                    {t.mode === "dry-run" && <span className="dry">dry-run</span>}
                    {t.streaming && (
                      <div className="status-line">
                        <span className="spin" /> {t.status || T("chat.status")}
                      </div>
                    )}
                    {t.reasoning ? (
                      <PenalaranView reasoning={t.reasoning} mentah={t.reasoningRaw} lang={L} />
                    ) : null}
                    {t.trace != null && t.trace.length > 0 && (
                      <details className="proc" open={!!t.streaming}>
                        <summary>
                          🔍 {T("proses.judul")} ({t.trace.length} {T("proses.langkah")})
                          {t.streaming ? T("proses.jalan") : ""}
                        </summary>
                        <div className="proc-body">
                          {t.trace.map((n) => (
                            <div key={n.id} style={{ marginBottom: 2, fontSize: 12 }}>
                              <span style={{ color: "var(--accent-2)", fontWeight: 700 }}>{n.id}</span>
                              {" "}{n.label}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                    <BubbleDenganSitasi text={t.text} citations={t.citations || []} cariGlobal={cariSitasi} onCiteClick={onCiteClick} />
                    {t.streaming && !t.text && (
                      <div style={{ fontSize: 11, color: "var(--muted)" }}>
                        {T("chat.mengalir")}
                      </div>
                    )}

                    {(t.citations || []).length > 0 && (
                      <details className="proc">
                        <summary>
                          📚 {T("sumber.judul")} ({t.citations!.length})
                        </summary>
                        <div className="proc-body">
                          {(t.citations || []).map((c) => (
                            <details className="src-card" key={c.n}>
                              <summary style={{ cursor: "pointer", listStyle: "none" }}>
                                <div className="src-head">
                                  <span className="snum">[S{c.n}]</span>
                                  <span>{c.official_title} — {c.citation_str}</span>
                                </div>
                                <div className="src-head" style={{ marginBottom: 0 }}>
                                  {c.page != null &&
                                    <span className="tag">{T("chat.src.hal")} {c.page}
                                      {c.lines ? ` · ${T("chat.src.baris")} ${c.lines[0]}-${c.lines[1]}` : ""}</span>}
                                  {c.score > 0 && <span className="tag">cosine {c.score}</span>}
                                  {c.skor_bm25 != null && <span className="tag bm25">BM25 {c.skor_bm25}</span>}
                                  {c.skor_fusi != null && <span className="tag fusi">{T("page.fusi")} {c.skor_fusi}</span>}
                                  {c.score_rerank != null && <span className="tag rerank">rerank {c.score_rerank}</span>}
                                </div>
                              </summary>
                              <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>
                                {(c.text || "").slice(0, 600)}{(c.text || "").length > 600 && " …"}
                              </div>
                            </details>
                          ))}
                        </div>
                      </details>
                    )}

                    {t.usage?.tokens != null &&
                      (t.usage.tokens.total_tokens > 0 ||
                       t.usage.tokens.prompt_tokens > 0) && (
                      <div className="usage">
                        🎟️ {t.usage.tokens.prompt_tokens.toLocaleString(LOC)} in
                        {" + "}{t.usage.tokens.completion_tokens.toLocaleString(LOC)} out
                        {t.usage.tokens.cached_tokens > 0 &&
                          ` (${T("token.cache")} ${t.usage.tokens.cached_tokens.toLocaleString(LOC)})`}
                        {" = "}{t.usage.tokens.total_tokens.toLocaleString(LOC)} {T("page.token")}
                      </div>
                    )}

                    {t.historyId && i > 0 && (
                      <button className="mini-btn" style={{ marginTop: 9 }}
                              data-tip={T("chat.retry.tip")}
                              onClick={() => onKirim(turns[i - 1].text)}>
                        {T("chat.retry")}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </Fragment>
          ))}

          {busy && <div className="status-line"><span className="spin" /> {T("chat.status")}</div>}
          {error && <div className="warnline">⚠ {error}</div>}
        </div>
      </div>

      <div className="composer-zone">
        <div className="composer-inner">
          <div className="composer">
            <textarea
              ref={areaRef}
              rows={1}
              value={teks} placeholder={T("chat.ph")}
              onChange={(e) => setTeks(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); kirim(); } }}
            />
            <button className="send-btn" data-tip={T("chat.kirim.tip")} onClick={kirim}
                    disabled={busy || !teks.trim()} aria-label={T("chat.kirim")}>↑</button>
          </div>
          <div className="composer-hint">{T("chat.hint")}</div>
        </div>
      </div>
    </>
  );
}

function PenalaranView({ reasoning, mentah, lang }: {
  reasoning: string; mentah?: string | null; lang: Lang;
}) {
  const T = (k: string) => t(lang, k);
  // Default: RAW first (provider-pattern debug). Toggle to Neat when needed.
  const [lihatMentah, setLihatMentah] = useState(true);
  return (
    <details className="proc">
      <summary>
        {T("chat.nalar.judul")}
      </summary>
      <div className="proc-body">
        {mentah != null && (
          <div style={{ marginBottom: 6 }}>
            <button type="button" className="mini-btn"
                    data-tip={lihatMentah ? T("penalaran.rapi.tip") : T("penalaran.mentah.tip")}
                    onClick={() => setLihatMentah((v) => !v)}>
              {lihatMentah ? T("penalaran.rapi") : T("penalaran.mentah")}
            </button>
          </div>
        )}
        {lihatMentah && mentah != null ? (
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word",
                         fontSize: 11 }}>{JSON.stringify(mentah.slice(0, 2000))}</pre>
        ) : (
          <div style={{ whiteSpace: "pre-wrap" }}>{mentah ?? reasoning}</div>
        )}
      </div>
    </details>
  );
}

function BubbleDenganSitasi({ text, citations, cariGlobal, onCiteClick }: {
  text: string;
  citations: Citation[];
  cariGlobal: (n: number, lokal: Citation[]) => Citation | undefined;
  onCiteClick: (c: Citation) => void;
}) {
  // Light markdown: ## headings, -/* bullets, **bold**, [Sn] citations.
  // No extra deps — enough for our answer style.
  const labelSitasi = (n: number): string => {
    const c = cariGlobal(n, citations);
    if (!c?.citation_str && !c?.official_title) return `[S${n}]`;
    const dok = c.doc_type
      || (c.official_title || "").slice(0, 14) || "UU";
    return `[S${n} · ${dok}${c.citation_str ? " " + c.citation_str : ""}]`;
  };
  const klikSitasi = (b: string) => {
    const n = Number(b.slice(2, -1));
    onCiteClick(cariGlobal(n, citations) || { n } as Citation);
  };
  const inline = (bagian: string, kunci: string) =>
    bagian.split(/(\[S\d+\]|\*\*[^*]+\*\*)/).map((b, i) => {
      if (/^\[S\d+\]$/.test(b)) {
        const n = Number(b.slice(2, -1));
        return <span key={`${kunci}-${i}`} className="cite"
                     title={(() => { const c = cariGlobal(n, citations);
                       return c ? `${c.official_title} — ${c.citation_str}` : b; })()}
                     onClick={() => klikSitasi(b)}>{labelSitasi(n)}</span>;
      }
      if (/^\*\*[^*]+\*\*$/.test(b)) {
        return <strong key={`${kunci}-${i}`}>{b.slice(2, -2)}</strong>;
      }
      return <Fragment key={`${kunci}-${i}`}>{b}</Fragment>;
    });
  const baris = (text || "").split("\n");
  return (
    <span>
      {baris.map((ln, i) => {
        const m = ln.match(/^(#{1,3})\s+(.*)$/);
        if (m) {
          const Level = `h${Math.min(4, m[1].length + 3)}` as "h4" | "h5" | "h6";
          return <Level key={i} style={{ margin: "8px 0 4px" }}>
            {inline(m[2], `j${i}`)}</Level>;
        }
        if (/^\s*[-*]\s+/.test(ln)) {
          return <div key={i} style={{ paddingLeft: 14, marginBottom: 2 }}>
            <span style={{ marginRight: 6 }}>•</span>
            {inline(ln.replace(/^\s*[-*]\s+/, ""), `b${i}`)}
          </div>;
        }
        if (ln.trim() === "") return <div key={i} style={{ height: 6 }} />;
        return <div key={i} style={{ marginBottom: 2 }}>{inline(ln, `p${i}`)}</div>;
      })}
    </span>
  );
}
