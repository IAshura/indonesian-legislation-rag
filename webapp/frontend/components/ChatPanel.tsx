"use client";
import { Fragment, useEffect, useRef, useState } from "react";
import { api } from "../lib/store";
import { Citation, ChatTurn } from "../lib/types";
import { t, type Lang } from "../lib/i18n";
import { I } from "./icons";

export default function ChatPanel({ turns, busy, error, dryRun, semuaSitasi, bahasa,
                                    totalToken, onStop, onKirim }: {
  turns: ChatTurn[];
  busy: boolean;
  error: string;
  dryRun: boolean;
  semuaSitasi?: Citation[];
  bahasa: Lang;
  totalToken: { prompt_tokens: number; completion_tokens: number; total_tokens: number; cached_tokens: number };
  onStop: () => void;
  onKirim: (q: string) => void;
}) {
  const L: Lang = bahasa || "id";
  const T = (k: string) => t(L, k);
  const LOC = L === "id" ? "id-ID" : "en-US";
  // global session lookup: old S numbers from earlier messages still resolve.
  const cariSitasi = (n: number, lokal: Citation[]): Citation | undefined =>
    lokal.find((c) => c.n === n)
    || (semuaSitasi || []).find((c) => c.n === n);
  const [teks, setTeks] = useState("");
  const [copied, setCopied] = useState<string | null>(null);
  const [voted, setVoted] = useState<Record<number, "up" | "down">>({});
  const kirim = () => {
    if (!teks.trim() || busy) return;
    onKirim(teks.trim());
    setTeks("");
  };
  const salin = async (teksSalin: string, kunci: string) => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(teksSalin);
      else {
        const ta = document.createElement("textarea");
        ta.value = teksSalin; document.body.appendChild(ta);
        ta.select(); document.execCommand("copy"); document.body.removeChild(ta);
      }
      setCopied(kunci);
      setTimeout(() => setCopied(null), 1600);
    } catch { /* ignore */ }
  };

  // auto-scroll to bottom on new content
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, turns[turns.length - 1]?.text?.length, busy]);

  // auto-growing textarea (1–8 rows)
  const areaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(200, el.scrollHeight) + "px";
  }, [teks]);

  const saran = [
    { kat: T("chat.kat.0"), q: L === "id"
      ? "Berapa lama masa jabatan Presiden menurut UUD 1945?"
      : "How long is the presidential term under the 1945 Constitution?" },
    { kat: T("chat.kat.1"), q: L === "id"
      ? "Siapa pemegang kekuasaan kehakiman?"
      : "Who holds judicial power?" },
    { kat: T("chat.kat.2"), q: L === "id"
      ? "Apa jaminan beragama menurut UUD 1945?"
      : "What religious guarantees exist in the 1945 Constitution?" },
  ];

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
              <h2>{T("chat.hero.welcome")}</h2>
              <p className="sub">{T("chat.hero.sub")}</p>
              <div className="suggest-grid">
                {saran.map((s) => (
                  <button key={s.q} className="suggest-card" onClick={() => onKirim(s.q)}>
                    <span className="cat">{s.kat}</span>
                    <span className="txt">{s.q}</span>
                    <span className="arrow">→</span>
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
                  <div className="body">
                    {t.replayOfNodeId && (
                      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 5 }}>
                        {T("chat.replay")} {t.replayOfNodeId}
                      </div>
                    )}
                    {t.mode === "dry-run" && <span className="dry">dry-run</span>}
                    {t.streaming && (
                      <div className="status-line">
                        <span className="spin" /> {t.status || T("chat.mencari")}
                      </div>
                    )}
                    {t.streaming && !t.text && (
                      <div className="skel" aria-hidden>
                        <i style={{ width: "92%" }} /><i style={{ width: "78%" }} /><i style={{ width: "64%" }} />
                      </div>
                    )}
                    {t.reasoning ? (
                      <PenalaranView reasoning={t.reasoning} mentah={t.reasoningRaw} lang={L} />
                    ) : null}
                    {t.trace != null && t.trace.length > 0 && (
                      <details className="proc" open={!!t.streaming}>
                        <summary>
                          {T("proses.judul")} ({t.trace.length} {T("proses.langkah")})
                          {t.streaming ? T("proses.jalan") : ""}
                        </summary>
                        <div className="proc-body">
                          {t.trace.map((n) => (
                            <div key={n.id} style={{ marginBottom: 2, fontSize: 12 }}>
                              <span style={{ color: "var(--accent)", fontWeight: 700 }}>{n.id}</span>
                              {" "}{n.label}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                    <div className="prose">
                      <BubbleDenganSitasi text={t.text} citations={t.citations || []} cariGlobal={cariSitasi} />
                    </div>
                    {t.streaming && !t.text && (
                      <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                        {T("chat.mengalir")}
                      </div>
                    )}

                    {(t.citations || []).length > 0 && (
                      <details className="sources">
                        <summary>
                          {T("sumber.judul")} ({t.citations!.length})
                        </summary>
                        <div>
                          {(t.citations || []).map((c) => {
                            const skor = c.score_rerank ?? c.score ?? 0;
                            const pct = Math.max(6, Math.min(100, Math.round(skor * 100)));
                            return (
                              <div className="src-row" key={c.n}>
                                <span className="snum">[{c.n}]</span>
                                <div className="info">
                                  <div className="title">{c.official_title}</div>
                                  <div className="cite-line">
                                    {c.citation_str}
                                    {c.page != null &&
                                      ` · ${T("chat.src.hal")} ${c.page}` +
                                      (c.lines ? ` ${T("chat.src.baris")} ${c.lines[0]}-${c.lines[1]}` : "")}
                                  </div>
                                  <div className="snippet">{c.text || ""}</div>
                                  <div className="rel-bar" title={`${T("sumber.relevansi")}: ${skor}`}>
                                    <i style={{ width: `${pct}%` }} />
                                  </div>
                                </div>
                                <button className="act-btn copy-btn" data-tip={T("sumber.salin")}
                                        aria-label={T("sumber.salin")}
                                        onClick={() => salin(
                                          `[S${c.n}] ${c.official_title} — ${c.citation_str}\n${c.text || ""}`,
                                          `src-${i}-${c.n}`)}>
                                  {copied === `src-${i}-${c.n}` ? <I.check /> : <I.copy />}
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      </details>
                    )}

                    {t.usage?.tokens != null &&
                      (t.usage.tokens.total_tokens > 0 ||
                       t.usage.tokens.prompt_tokens > 0) && (
                      <div className="usage">
                        {t.usage.tokens.prompt_tokens.toLocaleString(LOC)} in
                        {" + "}{t.usage.tokens.completion_tokens.toLocaleString(LOC)} out
                        {t.usage.tokens.cached_tokens > 0 &&
                          ` (${T("token.cache")} ${t.usage.tokens.cached_tokens.toLocaleString(LOC)})`}
                        {" = "}{t.usage.tokens.total_tokens.toLocaleString(LOC)} {T("page.token")}
                      </div>
                    )}

                    {!t.streaming && t.text && (
                      <div className="msg-actions">
                        <button className="act-btn" data-tip={T("chat.salin")} aria-label={T("chat.salin")}
                                onClick={() => salin(t.text, `ans-${i}`)}>
                          {copied === `ans-${i}` ? <I.check /> : <I.copy />}
                        </button>
                        {t.historyId && i > 0 && (
                          <button className="act-btn" data-tip={T("chat.retry.tip")} aria-label={T("chat.retry")}
                                  onClick={() => onKirim(turns[i - 1].text)}>
                            <I.refresh />
                          </button>
                        )}
                        {t.historyId && (
                          <>
                            <a className="act-btn" data-tip={T("menu.md")} aria-label={T("menu.md")}
                               href={api.exportUrl(t.historyId, "md")} target="_blank" rel="noreferrer"
                               style={{ textDecoration: "none" }}>
                              <I.download />
                            </a>
                            <button className={`act-btn ${voted[i] === "up" ? "on" : ""}`}
                                    data-tip={T("chat.suka")} aria-label={T("chat.suka")}
                                    onClick={() => setVoted((v) => ({ ...v, [i]: v[i] === "up" ? undefined as never : "up" }))}>
                              <I.thumbUp />
                            </button>
                            <button className={`act-btn ${voted[i] === "down" ? "on" : ""}`}
                                    data-tip={T("chat.tidak")} aria-label={T("chat.tidak")}
                                    onClick={() => setVoted((v) => ({ ...v, [i]: v[i] === "down" ? undefined as never : "down" }))}>
                              <I.thumbDown />
                            </button>
                          </>
                        )}
                      </div>
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
              aria-label={T("chat.ph")}
              onChange={(e) => setTeks(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); kirim(); } }}
            />
            <div className="composer-foot">
              {busy ? (
                <button className="send-btn stop" data-tip={T("chat.stop.tip")} aria-label={T("chat.stop")}
                        onClick={onStop}><I.stop /></button>
              ) : (
                <button className="send-btn" data-tip={T("chat.kirim.tip")} aria-label={T("chat.kirim")}
                        onClick={kirim} disabled={!teks.trim()}><I.send /></button>
              )}
            </div>
          </div>
          {(totalToken.total_tokens > 0) && (
            <div className="token-line"
                 title={`input/output: ${totalToken.prompt_tokens.toLocaleString(LOC)}/${totalToken.completion_tokens.toLocaleString(LOC)}`}>
              {totalToken.total_tokens.toLocaleString(LOC)} {T("page.token")}
            </div>
          )}
          <div className="disclaimer">{T("chat.disclaimer")}</div>
        </div>
      </div>
    </>
  );
}

function PenalaranView({ reasoning, mentah, lang }: {
  reasoning: string; mentah?: string | null; lang: Lang;
}) {
  const T = (k: string) => t(lang, k);
  // Default: NEAT first (readable). Toggle to Raw when needed.
  const [lihatMentah, setLihatMentah] = useState(false);
  return (
    <details className="proc">
      <summary>{T("chat.nalar.judul")}</summary>
      <div className="proc-body">
        {mentah != null && (
          <div style={{ margin: "10px 0 8px" }}>
            <button type="button" className="mini-btn"
                    data-tip={lihatMentah ? T("penalaran.rapi.tip") : T("penalaran.mentah.tip")}
                    onClick={() => setLihatMentah((v) => !v)}>
              {lihatMentah ? T("penalaran.rapi") : T("penalaran.mentah")}
            </button>
          </div>
        )}
        {lihatMentah && mentah != null ? (
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 12 }}>{mentah}</pre>
        ) : (
          <div style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{mentah ?? reasoning}</div>
        )}
      </div>
    </details>
  );
}

function BubbleDenganSitasi({ text, citations, cariGlobal }: {
  text: string;
  citations: Citation[];
  cariGlobal: (n: number, lokal: Citation[]) => Citation | undefined;
}) {
  // Light markdown: ## headings, -/* bullets, **bold**, [Sn] citations with hover cards.
  const inline = (bagian: string, kunci: string) =>
    bagian.split(/(\[S\d+\])/).map((b, i) => {
      const m = b.match(/^\[S(\d+)\]$/);
      if (m) {
        const n = Number(m[1]);
        const c = cariGlobal(n, citations);
        return (
          <span key={`${kunci}-${i}`} className="cite-wrap">
            <span className="cite" tabIndex={0}>[{n}]</span>
            {c && (
              <span className="cite-pop" role="tooltip">
                <b>[{n}] {c.official_title}</b>
                <span>{c.citation_str}</span>
                <span>{(c.text || "").slice(0, 220)}{(c.text || "").length > 220 && "…"}</span>
              </span>
            )}
          </span>
        );
      }
      // **bold** inside the same chunk
      const parts = b.split(/(\*\*[^*]+\*\*)/).map((p, j) =>
        /^\*\*[^*]+\*\*$/.test(p)
          ? <strong key={j}>{p.slice(2, -2)}</strong>
          : <Fragment key={j}>{p}</Fragment>);
      return <Fragment key={`${kunci}-${i}`}>{parts}</Fragment>;
    });
  const baris = (text || "").split("\n");
  return (
    <span>
      {baris.map((ln, i) => {
        const h = ln.match(/^(#{1,3})\s+(.*)$/);
        if (h) {
          const Level = `h${Math.min(4, h[1].length + 3)}` as "h4" | "h5" | "h6";
          return <Level key={i} style={{ margin: "10px 0 6px" }}>{inline(h[2], `j${i}`)}</Level>;
        }
        if (/^\s*[-*]\s+/.test(ln)) {
          return <li key={i} style={{ marginBottom: 3 }}>{inline(ln.replace(/^\s*[-*]\s+/, ""), `b${i}`)}</li>;
        }
        if (/^\s*\d+[.)]\s+/.test(ln)) {
          return <li key={i} style={{ marginBottom: 3 }}>{inline(ln.replace(/^\s*\d+[.)]\s+/, ""), `n${i}`)}</li>;
        }
        if (ln.trim() === "") return <div key={i} style={{ height: 8 }} />;
        return <div key={i} style={{ marginBottom: 4 }}>{inline(ln, `p${i}`)}</div>;
      })}
    </span>
  );
}
