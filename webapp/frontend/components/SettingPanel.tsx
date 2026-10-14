"use client";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api } from "../lib/store";
import { Settings, DOC_TYPES, ModelInfo } from "../lib/types";
import { t, type Lang } from "../lib/i18n";

const PROVIDER_PRESET: Record<string, { base_url: string; model: string }> = {
  openai:    { base_url: "https://api.openai.com/v1",  model: "gpt-4o-mini" },
  anthropic: { base_url: "https://api.anthropic.com/v1", model: "claude-sonnet-4-20250514" },
  google:    { base_url: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-2.0-flash" },
  zhipu:     { base_url: "https://open.bigmodel.cn/api/paas/v4", model: "glm-4-flash" },
  ollama:    { base_url: "http://localhost:11434/v1", model: "qwen3:4b" },
  custom:    { base_url: "", model: "" },
};

export default function SettingPanel({ setting, onChange, onUsage }: {
  setting: Settings;
  onChange: (s: Settings) => void;
  onUsage?: (tokens: { prompt_tokens?: number; completion_tokens?: number;
    total_tokens?: number; cached_tokens?: number }) => void;
}) {
  const set = (p: Partial<Settings>) => onChange({ ...setting, ...p });
  const L: Lang = setting.bahasa || "id";
  const T = (k: string) => t(L, k);
  const LOC = L === "id" ? "id-ID" : "en-US";

  // buka/tutup dropdown doc_type
  const [openDoc, setOpenDoc] = useState(false);

  // --- embedding list dari backend (dinamis: embedding baru auto-detect) ---
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [embMsg, setEmbMsg] = useState("");
  const [embErr, setEmbErr] = useState("");
  useEffect(() => {
    api.models().then((m) => setModels(m.daftar || [])).catch(() => {});
  }, []);

  const switchEmbedding = useCallback(async (tag: string) => {
    setEmbErr(""); setEmbMsg("mengganti model & memuat matrix…");
    try {
      const r = await api.switchEmbedding(tag);
      set({ embedding: tag });
      setEmbMsg(`✓ aktif: ${tag} (dim ${r.dim ?? "?"}, ${r.count ?? "?"} vektor)`);
      if (r.warning) setEmbMsg((m) => m + ` — ⚠ ${r.warning}`);
      const daftar = await api.models();
      setModels(daftar.daftar || []);
    } catch (e) { setEmbErr((e as Error).message); setEmbMsg(""); }
  }, [set]);

  // --- cek Milvus tiap pilih metode HNSW/IVF/FLAT ---------------------------
  type MilvusInfo = { ok: boolean; koleksi?: string; entitas?: number | null;
    dim_koleksi?: number | null; dim_aktif?: number | null; cocok_dim?: boolean;
    alasan?: string; peringatan?: string; instruksi?: string };
  const [milvus, setMilvus] = useState<MilvusInfo | null>(null);
  const [milvusCek, setMilvusCek] = useState(false);
  const cekMilvus = useCallback(async () => {
    setMilvusCek(true);
    try {
      setMilvus(await api.milvusStatus());
    } catch (e) {
      setMilvus({ ok: false, alasan: `backend tak terjangkau: ${(e as Error).message}` });
    } finally {
      setMilvusCek(false);
    }
  }, []);
  // auto-cek saat metode Milvus dipilih (dan saat panel dibuka bila sudah Milvus)
  useEffect(() => {
    if (setting.retrieval !== "bm25" && setting.denseMethod !== "exact") cekMilvus();
    else setMilvus(null);
  }, [setting.retrieval, setting.denseMethod, cekMilvus]);

  // --- doc_type checklist -------------------------------------------------
  const toggleDoc = (dt: string) => {
    const baru = setting.docTypes.includes(dt)
      ? setting.docTypes.filter((x) => x !== dt)
      : [...setting.docTypes, dt];
    set({ docTypes: baru });
  };

  // --- LLM test ------------------------------------------------------------
  const [testState, setTestState] = useState<"idle" | "busy" | "ok" | "fail">("idle");
  const [testMsg, setTestMsg] = useState("");
  const testKoneksi = async () => {
    setTestState("busy"); setTestMsg("");
    try {
      const r = await api.testKoneksi({ ...setting.llm });
      setTestState("ok");
      const teks = typeof r.answer === "string" ? r.answer
        : JSON.stringify(r.answer ?? r);
      const tk = (r as { usage?: { tokens?: {
        prompt_tokens?: number; completion_tokens?: number;
        total_tokens?: number; cached_tokens?: number } } }).usage?.tokens;
      if (tk && (tk.total_tokens || 0) > 0) {
        onUsage?.(tk);   // ping ikut bayar -> masuk total sesi
        setTestMsg(`OK — ${teks.slice(0, 90)}… • ` +
      `${(tk.prompt_tokens || 0).toLocaleString(LOC)} in + ` +
      `${(tk.completion_tokens || 0).toLocaleString(LOC)} out (${T("set.test.masuk")})`);
      } else {
        setTestMsg(`OK — ${teks.slice(0, 90)}…`);
      }
      if (setting.dryRun) set({ dryRun: false });   // koneksi OK -> user mau live
    } catch (e) { setTestState("fail"); setTestMsg((e as Error).message); }
  };

  const aktifModel = models.find((m) => m.tag === setting.embedding) || models.find((m) => m.aktif);

  // --- grup collapsible (posisi ingat-ingat di localStorage) ---
  const [grupBuka, setGrupBuka] = useState<Record<string, boolean>>({});
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("rag-hukum-grup");
      if (raw) setGrupBuka(JSON.parse(raw));
    } catch { /* abaikan */ }
  }, []);
  const bukaGrup = (id: string, bawaan: boolean) =>
    grupBuka[id] !== undefined ? grupBuka[id] : bawaan;
  const toggleGrup = (id: string, bawaan: boolean) => {
    const v = !bukaGrup(id, bawaan);
    setGrupBuka((g) => {
      const n = { ...g, [id]: v };
      try { window.localStorage.setItem("rag-hukum-grup", JSON.stringify(n)); } catch { /* abaikan */ }
      return n;
    });
  };

  return (
    <div>
      <Grup id="model" judul={T("grup.model")} bawaan={false}
            buka={bukaGrup} onToggle={toggleGrup} lang={L}>

      {/* --- embedding: dropdown dinamis --- */}
      <div className="field">
        <label>{T("set.emb")}
          <span style={{ textTransform: "none", letterSpacing: 0 }}>
            {models.length > 0 && ` — ${models.length} ${T("set.emb.terdeteksi")}`}</span>
        </label>
        <select value={setting.embedding}
                data-testid="dropdown-embedding"
                onChange={(e) => switchEmbedding(e.target.value)}>
          {models.map((m) => (
            <option key={m.tag} value={m.tag}>
              {m.tag} (dim: {m.dim ?? "?"}, {m.count ?? "?"} {T("set.emb.vektor")}{m.is_gpu ? " · GPU" : ""})
              {m.aktif ? ` — ${T("set.emb.aktif")}` : ""}
            </option>
          ))}
        </select>
        {aktifModel?.is_gpu && (
          <div className="warnbox">
            ⚠ {aktifModel.tag} (dim {aktifModel.dim}) {T("set.emb.warn")}
          </div>
        )}
        {embMsg && <div className="okbox">{embMsg}</div>}
        {embErr && <div className="errbox">⚠ {embErr}</div>}
      </div>

      {/* --- LLM --- */}
      <div className="field">
        <label>{T("set.llm")}</label>
        <select value={setting.llm.provider}
                onChange={(e) => {
                  const p = PROVIDER_PRESET[e.target.value];
                  set({ llm: { ...setting.llm,
                                provider: e.target.value as Settings["llm"]["provider"],
                                base_url: p.base_url, model: p.model } });
                }}>
          {Object.keys(PROVIDER_PRESET).map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>
      <div className="field">
        <label>{T("set.baseurl")}</label>
        <input type="text" value={setting.llm.base_url}
               onChange={(e) => set({ llm: { ...setting.llm, base_url: e.target.value } })} />
      </div>
      <div className="field">
        <label>{T("set.apikey")}</label>
        <input type="password" value={setting.llm.api_key}
               onChange={(e) => set({ llm: { ...setting.llm, api_key: e.target.value } })}
               placeholder={T("set.apikey.ph")} />
      </div>
      <div className="field">
        <label>{T("set.model")}</label>
        <input type="text" value={setting.llm.model}
               onChange={(e) => set({ llm: { ...setting.llm, model: e.target.value } })} />
      </div>
      <div className="field">
        <label>{T("set.temp")} ({(setting.llm.temperature ?? 0.2).toFixed(1)} — {T("set.temp.note")})</label>
        <input type="range" min={0} max={1} step={0.1}
               value={setting.llm.temperature ?? 0.2} style={{ width: "100%" }}
               onChange={(e) => set({ llm: { ...setting.llm, temperature: Number(e.target.value) } })} />
      </div>
      <div className="field">
        <label>{T("set.maxtok")}</label>
        <input type="number" min={16} max={8192} value={setting.llm.maxTokens ?? 1024}
               onChange={(e) => set({ llm: { ...setting.llm,
                 maxTokens: Math.max(16, Math.min(8192, Number(e.target.value) || 1024)) } })} />
      </div>
      <div style={{ marginBottom: 12 }}>
        <button className="btn secondary" data-tip={T("set.test.tip")} onClick={testKoneksi}
                disabled={testState === "busy"}>
          {testState === "busy" ? T("set.test.busy") : T("set.test")}
        </button>
        {testMsg && (
          <div className={testState === "ok" ? "okbox" : "errbox"}>
            {testState === "ok" ? "✓ " : "⚠ "}{testMsg}
          </div>)}
      </div>

      </Grup>

      <Grup id="strategi" judul={T("grup.strategi")} bawaan
            buka={bukaGrup} onToggle={toggleGrup} lang={L}>
      {/* --- mode --- */}
      <div className="field">
        <label>{T("set.mode")}</label>
        <div className="seg">
          {([
            ["fixed", T("set.mode.fixed"), T("set.mode.fixed.tip")],
            ["agent", T("set.mode.agent"), T("set.mode.agent.tip")],
          ] as const).map(([m, nama, tip]) => (
            <label key={m} className={setting.mode === m ? "aktif" : ""} data-tip={tip}>
              <input type="radio" name="mode" checked={setting.mode === m}
                     onChange={() => set({ mode: m as Settings["mode"] })} />
              {nama}
            </label>
          ))}
        </div>
      </div>

      {/* --- retrieval --- */}
      <div className="field">
        <label>{T("set.retrieval")}</label>
        <div className="seg">
          {([
            ["hybrid", T("set.ret.hybrid.tip")],
            ["dense", T("set.ret.dense.tip")],
            ["bm25", T("set.ret.bm25.tip")],
          ] as const).map(([r, tip]) => (
            <label key={r} className={setting.retrieval === r ? "aktif" : ""} data-tip={tip}>
              <input type="radio" name="retrieval" checked={setting.retrieval === r}
                     onChange={() => set({ retrieval: r as Settings["retrieval"] })} />
              {T(`set.ret.${r}`)}
            </label>
          ))}
        </div>
      </div>

      {/* --- metode index dense (berlaku bila retrieval dense/hybrid) --- */}
      <div className="field"
           style={setting.retrieval === "bm25"
             ? { opacity: 0.45, filter: "grayscale(1)", pointerEvents: "none" }
             : undefined}>
        <label>{T("set.idx")}</label>
        <div className="seg">
          {([
            ["exact", "exact", T("set.idx.exact.tip")],
            ["hnsw", "HNSW", T("set.idx.hnsw.tip")],
            ["ivf", "IVF", T("set.idx.ivf.tip")],
            ["flat", "FLAT", T("set.idx.flat.tip")],
          ] as const).map(([v, nama, tip]) => (
            <label key={v} className={setting.denseMethod === v ? "aktif" : ""} data-tip={tip}>
              <input type="radio" name="denseidx"
                     checked={setting.denseMethod === v}
                     disabled={setting.retrieval === "bm25"}
                     onChange={() => set({ denseMethod: v })} />
              {nama}
            </label>
          ))}
        </div>
        {setting.retrieval === "bm25" ? (
          <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
            {T("set.idx.bm25off")}
          </div>
        ) : setting.denseMethod === "exact" ? (
          <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
            {T("set.idx.exact.note")}
          </div>
        ) : (
          <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
            {T("set.idx.milvus.note")}
          </div>
        )}
        {setting.retrieval !== "bm25" && setting.denseMethod !== "exact" && (
          <div style={{ marginTop: 8 }}>
            {milvusCek && (
              <div style={{ color: "var(--muted)", fontSize: 11 }}>{T("set.idx.cek")}</div>
            )}
            {!milvusCek && milvus?.ok && milvus.cocok_dim !== false && (
              <div className="okbox">
                {T("set.idx.ok")}: koleksi {milvus.koleksi}
                {milvus.entitas != null ? ` (${Number(milvus.entitas).toLocaleString(LOC)} ${T("set.idx.entitas")}` : ""}
                {milvus.dim_koleksi != null ? `, ${T("set.idx.dim")} ${milvus.dim_koleksi}` : ""}{milvus.entitas != null ? ")" : ""}
              </div>
            )}
            {!milvusCek && milvus && (!milvus.ok || milvus.cocok_dim === false) && (
              <div className="errbox">
                <div>⚠ {(milvus as MilvusInfo).peringatan || (milvus as MilvusInfo).alasan || "Milvus"}</div>
                <div style={{ fontSize: 11, marginTop: 6, lineHeight: 1.6 }}>
                  {T("set.idx.docker0")}<br />
                  {T("set.idx.docker1a")}<br />
                  <code>cd C:\milvus<br />docker compose up -d</code>{" "}
                  <button type="button" className="mini-btn"
                          data-tip={T("set.idx.salin.tip")}
                          onClick={() => {
                            const cmd = "cd C:\\milvus\ndocker compose up -d";
                            if (navigator.clipboard?.writeText) {
                              navigator.clipboard.writeText(cmd).catch(() => {});
                            } else {
                              const ta = document.createElement("textarea");
                              ta.value = cmd; document.body.appendChild(ta);
                              ta.select(); document.execCommand("copy");
                              document.body.removeChild(ta);
                            }
                          }}>{T("set.idx.salin")}</button><br />
                  {T("set.idx.docker1b")} (<code>docker ps</code>).<br />
                  {(milvus as MilvusInfo).instruksi && (
                    <span>{(milvus as MilvusInfo).instruksi}</span>
                  )}
                </div>
                <button type="button" className="mini-btn" style={{ marginTop: 6 }} data-tip={T("set.idx.coba.tip")}
                        onClick={cekMilvus}>{T("set.idx.coba")}</button>
              </div>
            )}
          </div>
        )}
        {setting.retrieval !== "bm25" && setting.denseMethod === "hnsw" && (
          <div className="field" style={{ marginTop: 8 }}>
            <label>{T("set.ef")}</label>
            <input type="number" min={1} max={512} value={setting.denseEf}
                   onChange={(e) => set({ denseEf: Math.max(1, Math.min(512, Number(e.target.value) || 64)) })} />
          </div>
        )}
        {setting.retrieval !== "bm25" && setting.denseMethod === "ivf" && (
          <div className="field" style={{ marginTop: 8 }}>
            <label>{T("set.nprobe")}</label>
            <input type="number" min={1} max={1024} value={setting.denseNprobe}
                   onChange={(e) => set({ denseNprobe: Math.max(1, Math.min(1024, Number(e.target.value) || 16)) })} />
          </div>
        )}
      </div>
      </Grup>

      <Grup id="relevansi" judul={T("grup.relevansi")} bawaan={false}
            buka={bukaGrup} onToggle={toggleGrup} lang={L}>

      {/* --- rerank --- */}
      <div className="field">
        <label>{T("set.rerank")}</label>
        <div className="seg">
          <label className={setting.rerank ? "aktif" : ""} data-tip={T("set.rerank.on.tip")}>
            <input type="radio" name="rr" checked={setting.rerank}
                   onChange={() => set({ rerank: true })} /> {T("set.rerank.on")}
          </label>
          <label className={!setting.rerank ? "aktif" : ""} data-tip={T("set.rerank.off.tip")}>
            <input type="radio" name="rr" checked={!setting.rerank}
                   onChange={() => set({ rerank: false })} /> {T("set.rerank.off")}
          </label>
        </div>
      </div>

      {/* --- top-k + inject-k --- */}
      <div className="field">
        <div className="num3">
          <div className="field">
            <label>{T("set.topk")}</label>
            <input type="number" min={1} max={50} value={setting.topK}
                   onChange={(e) => {
                     const v = Math.max(1, Math.min(50, Number(e.target.value) || 1));
                     set({ topK: v, injectK: Math.min(v, setting.injectK) });
                   }} />
          </div>
          <div className="field">
            <label>{T("set.inject")}</label>
            <input type="number" min={1} max={setting.topK} value={setting.injectK}
                   onChange={(e) => {
                     const v = Math.max(1, Math.min(setting.topK, Number(e.target.value) || 1));
                     set({ injectK: v });
                   }} />
          </div>
        </div>
        <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
          {T("set.topk.note.a")} {setting.topK} {T("set.topk.note.b")} {Math.min(setting.injectK, setting.topK)} {T("set.topk.note.c")}{setting.rerank ? ` ${T("set.topk.note.rerank")}` : ""}.
        </div>
      </div>

      {/* --- doc_type: dropdown 2 kolom --- */}
      <div className="field">
        <label>{T("set.doctype")}</label>
        <div className="dd-multiselect" data-testid="check-doc-type">
          <button type="button" className="dd-toggle" data-tip={T("set.doctype.tip")}
                  onClick={() => setOpenDoc((v) => !v)}>
            <span>
              {setting.docTypes.length === 0 ? T("set.doctype.semua") :
                `${setting.docTypes.length} ${T("set.doctype.pilih")}: ${setting.docTypes.join(", ")}`}
            </span>
            <span className="chev">{openDoc ? "▲" : "▼"}</span>
          </button>
          {openDoc && (
            <div className="dd-panel">
              <div className="checklist kolom2">
                {DOC_TYPES.map((dt) => (
                  <label className="check-row" key={dt}>
                    <input type="checkbox" checked={setting.docTypes.includes(dt)}
                           onChange={() => toggleDoc(dt)} />
                    {dt}
                  </label>
                ))}
              </div>
              <div className="check-actions">
                <button type="button" className="mini-btn" data-tip={T("set.doctype.all.tip")}
                        onClick={() => set({ docTypes: [...DOC_TYPES] })}>{T("set.doctype.all")}</button>
                <button type="button" className="mini-btn" data-tip={T("set.doctype.none.tip")}
                        onClick={() => set({ docTypes: [] })}>{T("set.doctype.none")}</button>
                <span style={{ color: "var(--muted)", fontSize: 11, alignSelf: "center" }}>
                  {setting.docTypes.length === 0 ? T("set.doctype.aktif") :
                    `${setting.docTypes.length} ${T("set.doctype.terpilih")}`}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* --- year --- */}
      <div className="field">
        <label>{T("set.year")}</label>
        <input type="number" value={setting.year}
               onChange={(e) => set({ year: e.target.value })} placeholder={T("set.year.ph")} />
      </div>
      </Grup>

      <Grup id="jawaban" judul={T("grup.jawaban")} bawaan={false}
            buka={bukaGrup} onToggle={toggleGrup} lang={L}>

      <div className="field">
        <label>{T("set.gaya")}</label>
        <div className="seg">
          {([
            ["ringkas", T("set.gaya.ringkas"), T("set.gaya.ringkas.tip")],
            ["standar", T("set.gaya.standar"), T("set.gaya.standar.tip")],
            ["rinci", T("set.gaya.rinci"), T("set.gaya.rinci.tip")],
          ] as const).map(([v, nama, tip]) => (
            <label key={v} className={setting.gaya === v ? "aktif" : ""} data-tip={tip}>
              <input type="radio" name="gaya" checked={setting.gaya === v}
                     onChange={() => set({ gaya: v })} />
              {nama}
            </label>
          ))}
        </div>
        <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
          {setting.gaya === "ringkas" ? T("set.gaya.ringkas.note") :
           setting.gaya === "rinci" ? T("set.gaya.rinci.note") :
           T("set.gaya.standar.note")}
        </div>
      </div>

      {/* --- dry run --- */}
      <div className="field">
        <label>{T("set.dry")}</label>
        <div className="seg">
          <label className={setting.dryRun ? "aktif" : ""} data-tip={T("set.dry.on.tip")}>
            <input type="radio" name="dry" checked={setting.dryRun}
                   onChange={() => set({ dryRun: true })} /> {T("set.dry.on")}
          </label>
          <label className={!setting.dryRun ? "aktif" : ""} data-tip={T("set.dry.off.tip")}>
            <input type="radio" name="dry" checked={!setting.dryRun}
                   onChange={() => set({ dryRun: false })} /> {T("set.dry.off")}
          </label>
        </div>
      </div>
      </Grup>

    </div>
  );
}

function Grup({ id, judul, bawaan, buka, onToggle, children, lang }: {
  id: string; judul: string; bawaan: boolean;
  buka: (id: string, bawaan: boolean) => boolean;
  onToggle: (id: string, bawaan: boolean) => void;
  children: ReactNode; lang: Lang;
}) {
  const terbuka = buka(id, bawaan);
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--bg-2)",
                  marginBottom: 10, overflow: "hidden" }}>
      <button type="button" data-tip={t(lang, "grup.toggle.tip")} onClick={() => onToggle(id, bawaan)}
              style={{ width: "100%", display: "flex", justifyContent: "space-between",
                       alignItems: "center", padding: "9px 12px", background: "transparent",
                       border: "none", color: "inherit", cursor: "pointer",
                       fontWeight: 700, fontSize: 13 }}>
        <span>{judul}</span>
        <span style={{ color: "var(--muted)" }}>{terbuka ? "▲" : "▼"}</span>
      </button>
      {terbuka && <div style={{ padding: "2px 12px 12px" }}>{children}</div>}
    </div>
  );
}
