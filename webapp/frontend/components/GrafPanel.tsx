"use client";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { TraceNode } from "../lib/types";
import { t, type Lang } from "../lib/i18n";

import { I, nodeGlyph } from "./icons";

const AKSEN = "var(--accent)";
const NETRAL = "var(--text-muted)";
const TIPE: Record<string, { warna: string }> = {
  user_ask: { warna: AKSEN },
  llm_ask: { warna: NETRAL },
  embed_query: { warna: NETRAL },
  retrieve_dense: { warna: NETRAL },
  retrieve_bm25: { warna: NETRAL },
  fuse_rrf: { warna: NETRAL },
  rerank: { warna: NETRAL },
  join_parent: { warna: NETRAL },
  build_prompt: { warna: NETRAL },
  llm_answer: { warna: AKSEN },
  tool_call: { warna: NETRAL },
};

export default function GrafPanel({ trace, onReplay, bahasa }: {
trace: TraceNode[];
onReplay: (nodeIdx: number) => void;
bahasa: Lang;
}) {
  const L: Lang = bahasa || "id";
  const T = (k: string) => t(L, k);
  // --- pan + zoom (n8n-style: geser 4 arah · scroll-wheel zoom) ----------
  const [pan, setPan] = useState({ x: 40, y: 30 });
  const [zoom, setZoom] = useState(1);
  const dragRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const [render, setRender] = useState<"cleaner" | "raw">("cleaner");
  // giliran lama: ringkas 1 baris; hanya giliran TERAKHIR tampil penuh.
  const [turnBuka, setTurnBuka] = useState<Record<number, boolean>>({});
  // rantai proses yang sedang dibuka (popup daftar chain atas→bawah)
  const [selGrup, setSelGrup] = useState<number[] | null>(null);
  // kembali ke rantai: diisi saat lompat dari popup rantai ke detail node
  const [kembaliRantai, setKembaliRantai] = useState<number[] | null>(null);

  // Bagi trace datar menjadi giliran: tiap segmen = [user_ask … llm_answer].
  // Lane proses = node tengah (milik Q..A segmen itu). Segmen terakhir
  // boleh tanpa llm_answer (streaming berjalan).
  const giliran: { tanya: number; jawab: number | null; proses: number[] }[] = [];
  trace.forEach((n, i) => {
    if (n.type === "user_ask") giliran.push({ tanya: i, jawab: null, proses: [] });
    else if (n.type === "llm_answer") {
      const g = [...giliran].reverse().find((x) => x.jawab === null);
      if (g) g.jawab = i;
      else giliran.push({ tanya: -1, jawab: i, proses: [] });
    } else if (giliran.length > 0) {
      const g = giliran[giliran.length - 1];
      if (g.jawab === null) g.proses.push(i);
      else giliran.push({ tanya: -1, jawab: null, proses: [i] });
    } else {
      giliran.push({ tanya: -1, jawab: null, proses: [i] });
    }
  });
  const wrapRef = useRef<HTMLDivElement>(null);
  // mirror utk handler native (hindari stale closure)
  const panRef = useRef(pan); panRef.current = pan;
  const zoomRef = useRef(zoom); zoomRef.current = zoom;

  // ---- edge independen berkoordinat: diukur dari DOM, digambar di SVG ----
  // Tiap edge = {awal(x,y), akhir(x,y)} dalam koordinat canvas (ikut zoom
  // karena SVG-nya di dalam elemen yang sama). Diukur ulang tiap layout.
  const canvasRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef(new Map<string, HTMLElement>());
  const [edges, setEdges] = useState<
    { x1: number; y1: number; x2: number; y2: number; putus: boolean }[]>([]);
  const ukur = (kunci: string) => {
    const c = canvasRef.current;
    const el = nodeRefs.current.get(kunci);
    if (!c || !el || !el.isConnected) return null;
    // rect relatif kanvas dibagi zoom (SVG ikut transform yang sama)
    const z = zoomRef.current || 1;
    const cr = c.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return null;
    return { x: (r.left - cr.left) / z, y: (r.top - cr.top) / z,
             w: r.width / z, h: r.height / z };
  };
  // Dipanggil ulang tiap layout berubah; disimpan di ref agar observer
  // visibilitas selalu memakai hitungan TERBARU (bukan closure basi).
  const hitungRef = useRef<() => void>(() => {});

  // Hitung edge sinkron setelah DOM commit (layout effect) + saat
  // resize/font selesai — tak ada celah timing dengan streaming token.
  useLayoutEffect(() => {
    const tengahBawah = (r: { x: number; y: number; w: number; h: number }) =>
      ({ x: r.x + r.w / 2, y: r.y + r.h });
    const tengahAtas = (r: { x: number; y: number; w: number }) =>
      ({ x: r.x + r.w / 2, y: r.y });
    const jangkarBawah = (gi: number): { x: number; y: number } | null => {
      const g = giliran[gi];
      const terbuka = turnBuka[gi] ?? true;
      if (!terbuka) {
        const s = ukur(`s-${gi}`);
        return s ? { x: s.x + s.w / 2, y: s.y + s.h } : null;
      }
      const a = ukur(`a-${gi}`) || ukur(`w-${gi}`);
      const r = a || ukur(`q-${gi}`) || ukur(`turn-${gi}`);
      return r ? { x: r.x + r.w / 2, y: r.y + r.h } : null;
    };
    const jangkarAtas = (gi: number): { x: number; y: number } | null => {
      const g = giliran[gi];
      const terbuka = turnBuka[gi] ?? true;
      if (!terbuka) {
        const s = ukur(`s-${gi}`);
        return s ? { x: s.x + s.w / 2, y: s.y } : null;
      }
      const r = ukur(`q-${gi}`) || ukur(`w-${gi}`) || ukur(`a-${gi}`)
        || ukur(`turn-${gi}`);
      return r ? { x: r.x + r.w / 2, y: r.y } : null;
    };
    const hitung = () => {
      const garis: { x1: number; y1: number; x2: number; y2: number; putus: boolean }[] = [];
      giliran.forEach((g, gi) => {
        const terbuka = turnBuka[gi] ?? true;
        if (terbuka) {
          const q = ukur(`q-${gi}`), a = ukur(`a-${gi}`) || ukur(`w-${gi}`),
            j = ukur(`j-${gi}`);
          if (q && a) {
            const p1 = tengahBawah(q), p2 = tengahAtas(a);
            garis.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, putus: false });
          }
          const aj = a || q;
          if (aj && j) {
            garis.push({ x1: aj.x + aj.w, y1: aj.y + aj.h / 2,
                         x2: j.x, y2: j.y + j.h / 2, putus: false });
          }
        }
        if (gi < giliran.length - 1) {
          const p1 = jangkarBawah(gi), p2 = jangkarAtas(gi + 1);
          if (p1 && p2) {
            garis.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, putus: true });
          }
        }
      });
      setEdges(garis);
    };
    hitungRef.current = hitung;
    hitung();
    window.addEventListener("resize", hitung);
    let batal = false;
    if (document.fonts?.ready) {
      document.fonts.ready.then(() => { if (!batal) hitung(); }).catch(() => {});
    }
    return () => { batal = true; window.removeEventListener("resize", hitung); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, turnBuka]);

  // Panel Graf disembunyikan (display:none) saat tab Setting aktif — semua
  // ukuran = 0. Begitu tab Graf dibuka / panel terlihat, hitung ulang.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((masuk) => {
      if (masuk.some((e) => e.isIntersecting)) hitungRef.current();
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const clamp = (z: number) => Math.min(2.5, Math.max(0.4, z));
  const zoomKe = useCallback((f: number, cx: number, cy: number) => {
    const z0 = zoomRef.current, p0 = panRef.current;
    const z1 = clamp(z0 * f);
    if (z1 === z0) return;
    const wx = (cx - p0.x) / z0, wy = (cy - p0.y) / z0;
    setZoom(z1);
    setPan({ x: cx - wx * z1, y: cy - wy * z1 });   // titik di bawah kursor tetap
  }, []);

  // wheel zoom non-passive (perlu preventDefault) — KECUALI di popup
  // detail node, agar isinya bisa di-scroll mouse seperti biasa.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if ((e.target as HTMLElement).closest(".graf-popup")) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomKe(e.deltaY < 0 ? 1.12 : 1 / 1.12,
             e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomKe]);

  // ref elemen kartu berdasar kunci ("q-0", "a-0", "j-0", "turn-0", ...)
  const refKartu = (kunci: string) => (el: HTMLElement | null) => {
    if (el) nodeRefs.current.set(kunci, el);
    else nodeRefs.current.delete(kunci);
  };

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    // hanya drag pada backdrop, bukan pada node/popup
    if ((e.target as HTMLElement).closest(".nnode, .graf-popup")) return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, ox: pan.x, oy: pan.y };
    setDragging(true);
  }, [pan]);
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    setPan({ x: dragRef.current.ox + (e.clientX - dragRef.current.sx),
             y: dragRef.current.oy + (e.clientY - dragRef.current.sy) });
  }, []);
  const onMouseUp = useCallback(() => { dragRef.current = null; setDragging(false); }, []);

  // dot grid follows pan/zoom (it lives on the viewport, so it must be
  // repositioned manually — otherwise dots stay fixed while nodes move).
  const GRID = 22;
  return (
    <div ref={wrapRef}
         className={`graf-wrap ${dragging ? "dragging" : ""}`}
         data-testid="graf-panel"
         style={{
           backgroundImage: "radial-gradient(var(--border-2) 1px, transparent 0)",
           backgroundPosition: `${pan.x}px ${pan.y}px`,
           backgroundSize: `${GRID * zoom}px ${GRID * zoom}px`,
         }}
         onMouseDown={onMouseDown} onMouseMove={onMouseMove}
         onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>
      <div className="graf-hint">{T("graf.hint")}</div>

      <div className="graf-zoom">
        <button data-tip={T("graf.zoom.kecil")} onClick={(e) => { e.stopPropagation();
          const r = wrapRef.current!.getBoundingClientRect();
          zoomKe(1 / 1.25, r.width / 2, r.height / 2); }}>−</button>
        <span onClick={(e) => { e.stopPropagation(); setZoom(1); setPan({ x: 40, y: 30 }); }}
              title={T("graf.zoom.reset")}>
          {Math.round(zoom * 100)}%
        </span>
        <button data-tip={T("graf.zoom.besar")} onClick={(e) => { e.stopPropagation();
          const r = wrapRef.current!.getBoundingClientRect();
          zoomKe(1.25, r.width / 2, r.height / 2); }}>+</button>
      </div>

      <div className="graf-canvas" ref={canvasRef}
           style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
        {/* lapisan edge independen: garis berkoordinat awal→akhir */}
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%",
                      overflow: "visible", pointerEvents: "none" }}>
          {edges.map((e, k) => (
            <line key={k} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                  stroke={e.putus ? "#8b95a8" : "#a78bfa"} strokeWidth={2}
                  opacity={e.putus ? 0.5 : 0.9}
                  strokeDasharray={e.putus ? "5 4" : undefined} />
          ))}
        </svg>
        {trace.length === 0 ? (
          <div style={{ color: "var(--muted)", fontSize: 12.5, maxWidth: 240, padding: 8 }}>
            {T("graf.kosong")}
          </div>
        ) : (
          giliran.map((g, gi) => {
            const terakhir = gi === giliran.length - 1;
            // default TERBUKA semua — tidak ada auto-shrink giliran lama.
            const buka = turnBuka[gi] ?? true;
            const refTurn = (el: HTMLDivElement | HTMLButtonElement | null) => {
              if (el) nodeRefs.current.set(`turn-${gi}`, el as HTMLDivElement);
              else nodeRefs.current.delete(`turn-${gi}`);
            };
            if (!buka) {
              const q = g.tanya >= 0 ? trace[g.tanya] : null;
              const a = g.jawab != null ? trace[g.jawab] : null;
              const nLangkah = g.proses.length + (q ? 1 : 0) + (a ? 1 : 0);
              const statusA = !a
                ? (terakhir ? T("graf.menunggu")
                   : T("graf.gagal"))
                : /batas|kosong/i.test(a.label) ? "⚠ " + a.label
                : /dry-run/i.test(a.label) ? "🧪 " + a.label
                : "🤖 " + a.label;
              return (
                <div key={gi} ref={refTurn as React.Ref<HTMLDivElement>}>
                  <button type="button" className="lane-toggle" data-tip={T("graf.buka.tip")}
                          ref={refKartu(`s-${gi}`)}
                          style={{ marginBottom: 6, textAlign: "left", width: "100%" }}
                          onClick={(e) => {
                      e.stopPropagation();
                      setTurnBuka((s) => ({ ...s, [gi]: true }));
                    }}>
                    {q ? (
                      <div>🧑 {String(q.label).slice(0, 70)}</div>
                    ) : (
                      <div>⚙️ {T("graf.lanjutan")} ({g.proses.length} {T("proses.langkah")})</div>
                    )}
                    <div style={{ marginTop: 2, color: "var(--muted)" }}>
                      {String(statusA).slice(0, 70)} · {nLangkah} {T("proses.langkah")} ▼
                    </div>
                  </button>
                </div>
              );
            }
            return (
            <div key={gi} ref={refTurn as React.Ref<HTMLDivElement>}>
              {!terakhir && (
                <button type="button" className="lane-toggle" data-tip={T("graf.ringkas.tip")}
                        style={{ marginBottom: 6 }} onClick={(e) => {
                    e.stopPropagation();
                    setTurnBuka((s) => ({ ...s, [gi]: false }));
                  }}>
                  {T("graf.ringkas")}
                </button>
              )}
              <div className="tcol">
                {/* user_ask selalu di atas */}
                {g.tanya >= 0 && (
                  <KartuNode n={trace[g.tanya]} i={g.tanya} sel={sel} refCb={refKartu} lang={L}
                             refKunci={`q-${gi}`}
                             onPilih={(x) => { setSel(x); setRender("cleaner"); }} />
                )}
                {/* baris jawaban: llm_answer + jejak (garis via SVG) */}
                <div className="trow-jawab">
                  {g.jawab != null ? (
                    <KartuNode n={trace[g.jawab]} i={g.jawab} sel={sel} refCb={refKartu} lang={L}
                               refKunci={`a-${gi}`}
                               onPilih={(x) => { setSel(x); setRender("cleaner"); }} />
                  ) : (
                    <div className="nnode" ref={refKartu(`w-${gi}`)}
                         style={{ borderStyle: "dashed", opacity: 0.75 }}>
                      <div className="tipe" style={{ color: "#34d399" }}>
                        <span>🤖</span><span>llm_answer</span>
                      </div>
                      <div className="lbl">{T("graf.menunggu")}</div>
                    </div>
                  )}
                  {g.proses.length > 0 && (
                    <div className="tlane">
                      <div className="nnode" ref={refKartu(`j-${gi}`)}
                           style={{ borderStyle: "dashed", cursor: "pointer" }}
                           onClick={(e) => { e.stopPropagation(); setSelGrup(g.proses); }}>
                        <div className="tipe" style={{ color: "#a78bfa" }}>
                          <span>⚙️</span><span>{T("graf.jejak")}</span>
                        </div>
                        <div className="lbl">{g.proses.length} {T("graf.jejak.lbl")}</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            );
          })
        )}
      </div>

      {/* ---------- popup rantai proses (atas→bawah saling chain) ---------- */}
      {selGrup != null && (
        <div className="graf-popup-backdrop" onMouseDown={(e) => {
          if ((e.target as HTMLElement) === e.currentTarget) setSelGrup(null);
        }} onClick={(e) => e.stopPropagation()}>
          <div className="graf-popup">
            <div className="head">
              <span className="tipe">{T("graf.rantai")} ({selGrup.length} {T("proses.langkah")})</span>
              <button className="x" data-tip={T("graf.rantai.tutup")} onClick={() => setSelGrup(null)}>✕</button>
            </div>
            <div className="body" data-testid="popup-rantai">
              {selGrup.map((pi, k) => {
                const n = trace[pi];
                if (!n) return null;
                const info = TIPE[n.type] || { warna: "var(--text-faint)" };
                return (
                  <div key={n.id + pi}>
                    <div className="nnode" style={{ width: "100%", cursor: "pointer" }}
                         onClick={() => { setKembaliRantai(selGrup); setSelGrup(null); setSel(pi); setRender("cleaner"); }}>
                      <div className="tipe" style={{ color: info.warna }}>
                        <span>{nodeGlyph(n.type)}</span><span>{n.type}</span>
                      </div>
                      <div className="lbl">{n.label}</div>
                      <NodeSubTeks node={n} lang={L} />
                    </div>
                    {k < selGrup.length - 1 && <div className="nedge" style={{ margin: "2px auto" }} />}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ---------- popup detail node (bukan bagian bawah) ---------- */}
      {sel != null && trace[sel] && (
        <div className="graf-popup-backdrop" onMouseDown={(e) => {
          if ((e.target as HTMLElement) === e.currentTarget) {
            setSel(null); setKembaliRantai(null);
          }
        }} onClick={(e) => e.stopPropagation()}>
          <div className="graf-popup">
            <div className="head">
              {kembaliRantai != null && (
                <button className="icon-btn sq" data-tip={T("graf.kembali")}
                        aria-label={T("graf.kembali")}
                        onClick={() => { setSel(null); setSelGrup(kembaliRantai); setKembaliRantai(null); }}>
                  ←
                </button>
              )}
              <span style={{ fontSize: 17 }}>{nodeGlyph(trace[sel].type)}</span>
              <span className="tipe">{trace[sel].type}</span>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>{trace[sel].id}</span>
              <span className="mode-raw-cleaner" style={{ marginLeft: 14 }}>
                <button className={render === "cleaner" ? "aktif" : ""} data-tip={T("graf.node.bersih.tip")}
                        onClick={() => setRender("cleaner")}>{T("graf.node.bersih")}</button>
                <button className={render === "raw" ? "aktif" : ""} data-tip={T("graf.node.mentah.tip")}
                        onClick={() => setRender("raw")}>{T("graf.node.mentah")}</button>
              </span>
              <button className="icon-btn sq" data-tip={T("graf.node.tutup")}
                      aria-label={T("graf.node.tutup")}
                        onClick={() => { setSel(null); setKembaliRantai(null); }}><I.x /></button>
            </div>
            <div className="body" data-testid="popup-body">
              {render === "raw"
                ? <pre>{JSON.stringify(trace[sel].detail, null, 2)}</pre>
                : <CleanerNode node={trace[sel]} lang={L} />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function KartuNode({ n, i, sel, onPilih, refKunci, refCb, lang }: {
  n: TraceNode; i: number; sel: number | null; onPilih: (x: number) => void;
  refKunci?: string; refCb?: (kunci: string) => (el: HTMLElement | null) => void;
  lang: Lang;
}) {
  const L: Lang = lang || "id";
  const info = TIPE[n.type] || { warna: "var(--text-faint)" };
  const refEl = refKunci && refCb ? refCb(refKunci) : undefined;
  // Role identity: user question + final answer in accent, other steps neutral.
  const aksen = n.type === "user_ask" || n.type === "llm_answer" ? AKSEN : undefined;
  return (
    <div className={`nnode ${sel === i ? "klik" : ""}`}
         ref={refEl}
         style={aksen ? { borderLeft: `3px solid ${aksen}` } : undefined}
         onClick={(e) => { e.stopPropagation(); onPilih(i); }}>
      <div className="tipe" style={{ color: aksen || info.warna }}>
        <span>{nodeGlyph(n.type)}</span><span>{n.type}</span>
      </div>
      <div className="lbl">{n.label}</div>
      <NodeSubTeks node={n} lang={L} />
    </div>
  );
}

function NodeSubTeks({ node, lang }: { node: TraceNode; lang: Lang }) {
  const T = (k: string) => t(lang, k);
  const d = node.detail || {};
  if (d.hasil?.length) return <div className="sub">{d.hasil.length} {T("graf.kandidat")}</div>;
  if (d.retrieval_subtrace || d.daftar_hasil)
    return <div className="sub">{(d.daftar_hasil || []).length} {T("graf.hasil.tool")}</div>;
  if (d.prompt) return <div className="sub">{d.prompt.char_len} {T("graf.char")}</div>;
  if (d.detik != null) return <div className="sub">{d.detik} {T("graf.det")}</div>;
  return null;
}

function CleanerNode({ node, lang }: { node: TraceNode; lang: Lang }) {
  const T = (k: string) => t(lang, k);
  // Tampilan rapi: string ditampilkan apa adanya sehingga \n menjadi
  // baris baru beneran; struktur (prompt/tool/hasil) dipecah per bagian.
  const d = node.detail || {} as Record<string, any>;   // eslint-disable-line

  // settings / pertanyaan
  if (d.question || d.settings) {
    return (<>
      {d.question && <div className="kv">
        <span className="k">{T("graf.tanya")}</span><span>{d.question}</span>
      </div>}
      {d.settings && <SettingTable s={d.settings} />}
    </>);
  }

  // prompt: SYSTEM / KONTEKS / USER terpisah, baris baru asli
  if (d.prompt) {
    const p = d.prompt;
    const bagian: [string, string][] = [
      [T("graf.sistem"), p.system], [T("graf.konteks"), p.context], [T("graf.pengguna"), p.user],
    ];
    return (<>
      {bagian.filter(([, isi]) => isi).map(([nama, isi]) => (
        <div key={nama} style={{ marginBottom: 10 }}>
          <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 3 }}>
            {nama}{nama === T("graf.konteks") && p.char_len ? ` (${p.char_len} char)` : ""}
          </div>
          <div style={{ whiteSpace: "pre-wrap" }}>{String(isi)}</div>
        </div>
      ))}
    </>);
  }

  // tool_call: kartu sumber per hasil (teks asli, bukan JSON escape)
  if (Array.isArray(d.daftar_hasil)) {
    return (<>
      <div className="kv"><span className="k">{T("graf.tool")}</span><span>{d.tool}</span></div>
      {d.argumen && <div className="kv"><span className="k">{T("graf.argumen")}</span>
        <span style={{ whiteSpace: "pre-wrap" }}>{String(d.argumen)}</span></div>}
      {d.detik != null && <div className="kv"><span className="k">{T("graf.durasi")}</span>
        <span>{d.detik}s</span></div>}
      <div style={{ color: "var(--muted)", fontSize: 11, margin: "8px 0 4px" }}>
        {T("graf.hasil")} ({d.daftar_hasil.length})
      </div>
      {d.daftar_hasil.map((h: Record<string, any>, i: number) => (   // eslint-disable-line
        <div className="rank-row" key={i} style={{ display: "block", marginBottom: 8 }}>
          <div>
            <span className="no">#{h.n ?? i + 1}</span>{" "}
            <span className="judul">{h.official_title} — {h.citation_str}</span>
          </div>
          {h.text && <div style={{ whiteSpace: "pre-wrap", marginTop: 4 }}>{String(h.text)}</div>}
        </div>
      ))}
    </>);
  }

  // retrieval ranking (hasil WAJIB array; node tool_call mengirim string)
  if (Array.isArray(d.hasil) && d.hasil.length) {
    return (<>
      {d.query && <div className="kv"><span className="k">{T("graf.query")}</span><span>{d.query}</span></div>}
      {d.detik != null && <div className="kv"><span className="k">{T("graf.durasi")}</span><span>{d.detik}s</span></div>}
      {d.rrf_k != null && <div className="kv"><span className="k">{T("graf.rrfk")}</span><span>{d.rrf_k}</span></div>}
      {d.hasil.map((h: Record<string, any>, i: number) => (   // eslint-disable-line
        <div className="rank-row" key={i}>
          <span className="no">#{h.rank ?? i + 1}</span>
          <span className="judul">
            {h.official_title || "-"} — {h.citation_str || h.chunk_id}
          </span>
          <span className="skor">
            {h.cosine != null && `cos ${h.cosine} `}
            {h.skor_bm25 != null && `BM25 ${h.skor_bm25} `}
            {h.skor_fusi != null && `fusi ${h.skor_fusi} `}
            {h.score_rerank != null && `rer ${h.score_rerank}`}
          </span>
        </div>
      ))}
    </>);
  }

  // teks mentah (hasil string / payload / jawaban): baris baru asli
  for (const kunci of ["hasil", "teks", "payload"]) {
    if (typeof d[kunci] === "string" && d[kunci]) {
      return <div style={{ whiteSpace: "pre-wrap" }}>{d[kunci]}</div>;
    }
  }

  // join parent
  if (d.kandidat?.length) {
    return d.kandidat.map((k: Record<string, any>, i: number) => (  // eslint-disable-line
      <div className="rank-row" key={i}>
        <span className="no">#{i + 1}</span>
        <span className="judul">{k.official_title} — {k.citation_str}</span>
        <span className="skor">{k.char_len} char · {k.sumber}</span>
      </div>
    ));
  }

  // rerank
  if (d.model) return (<>
    <div className="kv"><span className="k">{T("graf.model")}</span><span>{d.model}</span></div>
    <div className="kv"><span className="k">{T("graf.pool")}</span><span>{d.pool} → {d.top_n}</span></div>
    {d.detik != null && <div className="kv"><span className="k">{T("graf.durasi")}</span><span>{d.detik}s</span></div>}
    {(Array.isArray(d.hasil) ? d.hasil : []).map((h: Record<string, any>, i: number) => (    // eslint-disable-line
      <div className="rank-row" key={i}>
        <span className="no">#{h.rank}</span>
        <span className="judul">{h.official_title} — {h.citation_str}</span>
        <span className="skor">rer {h.score_rerank}</span>
      </div>
    ))}
  </>);

  if (d.model && d.provider) return <SettingTable s={d} />;
  return <pre>{JSON.stringify(d, null, 2)}</pre>;
}

function SettingTable({ s }: { s: Record<string, any> }) {  // eslint-disable-line
  return <div className="kv">
    {Object.entries(s).filter(([, v]) => v != null && v !== "").map(([k, v]) => (
      <Row key={k} k={k} v={typeof v === "object" ? JSON.stringify(v) : String(v)} />
    ))}
  </div>;
}
function Row({ k, v }: { k: string; v: string }) {
  return (<>
    <span className="k">{k}</span><span>{v}</span>
  </>);
}
