"use client";
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { api } from "../lib/store";
import { Settings, ModelInfo } from "../lib/types";
import { t, type Lang } from "../lib/i18n";
import { I } from "./icons";

const PROVIDER_PRESET: Record<string, { base_url: string; model: string }> = {
  openai:    { base_url: "https://api.openai.com/v1",  model: "gpt-4o-mini" },
  anthropic: { base_url: "https://api.anthropic.com/v1", model: "claude-sonnet-4-20250514" },
  google:    { base_url: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-2.0-flash" },
  zhipu:     { base_url: "https://open.bigmodel.cn/api/paas/v4", model: "glm-4-flash" },
  ollama:    { base_url: "http://localhost:11434/v1", model: "qwen3:4b" },
  custom:    { base_url: "", model: "" },
};

type Tab = "model" | "cari" | "gaya";

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
  const [tab, setTab] = useState<Tab>("model");

  // --- embedding list from backend (dynamic: new embeddings auto-detected) ---
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

  // --- Milvus check whenever an HNSW/IVF/FLAT method is picked ---------------
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
  // auto-check when a Milvus method is picked
  useEffect(() => {
    if (setting.retrieval !== "bm25" && setting.denseMethod !== "exact") cekMilvus();
    else setMilvus(null);
  }, [setting.retrieval, setting.denseMethod, cekMilvus]);

  // --- LLM connection test --------------------------------------------------
  const [testState, setTestState] = useState<"idle" | "busy" | "ok" | "fail">("idle");
  const [testMsg, setTestMsg] = useState("");
  const testKoneksi = async () => {
    setTestState("busy"); setTestMsg("");
    const t0 = performance.now();
    try {
      const r = await api.testKoneksi(setting.llm);
      const ms = Math.round(performance.now() - t0);
      onUsage?.((r.usage as { tokens?: {
        prompt_tokens?: number; completion_tokens?: number;
        total_tokens?: number; cached_tokens?: number } })?.tokens || {});
      setTestState("ok");
      setTestMsg(`${T("set.test.ok")} · ${ms} ms — “${String(r.answer).slice(0, 80)}”`);
      if (setting.dryRun) set({ dryRun: false });   // koneksi OK -> user mau live
    } catch (e) { setTestState("fail"); setTestMsg((e as Error).message); }
  };

  const aktifModel = models.find((m) => m.tag === setting.embedding) || models.find((m) => m.aktif);
  const [showKey, setShowKey] = useState(false);
  const temp = setting.llm.temperature ?? 0.2;

  return (
    <div>
      <div className="set-tabs" role="tablist">
        {(["model", "cari", "gaya"] as const).map((tb) => (
          <button key={tb} role="tab" aria-selected={tab === tb}
                  className={tab === tb ? "on" : ""}
                  onClick={() => setTab(tb)}>
            {T(`set.tab.${tb}`)}
          </button>
        ))}
      </div>

      {tab === "model" && (
        <div>
          <div className="set-sec">
            <div className="set-sec-t">{T("set.llm")}</div>
            <div className="field">
              <CSelect
                value={setting.llm.provider}
                options={Object.keys(PROVIDER_PRESET).map((p) => ({ value: p, label: p }))}
                onPick={(v) => {
                  const p = PROVIDER_PRESET[v];
                  set({ llm: { ...setting.llm, provider: v as Settings["llm"]["provider"],
                               base_url: p.base_url, model: p.model } });
                }} />
            </div>
            <div className="field">
              <label>{T("set.baseurl")}</label>
              <input type="text" value={setting.llm.base_url}
                     onChange={(e) => set({ llm: { ...setting.llm, base_url: e.target.value } })} />
            </div>
            <div className="field">
              <label>{T("set.apikey")}</label>
              <div style={{ position: "relative" }}>
                <input type={showKey ? "text" : "password"} value={setting.llm.api_key}
                       style={{ paddingRight: 44 }}
                       onChange={(e) => set({ llm: { ...setting.llm, api_key: e.target.value } })}
                       placeholder={T("set.apikey.ph")} />
                <button type="button" className="eye-btn"
                        data-tip={showKey ? T("set.apikey.sembunyi") : T("set.apikey.lihat")}
                        aria-label={showKey ? T("set.apikey.sembunyi") : T("set.apikey.lihat")}
                        onClick={() => setShowKey((v) => !v)}>
                  {showKey ? <I.eyeOff /> : <I.eye />}
                </button>
              </div>
              {setting.llm.api_key
                ? <div className="help"><span className="saved-tag">✓ {T("set.apikey.tersimpan")}</span></div>
                : <div className="help">{T("set.apikey.ph")}</div>}
            </div>
            <div className="field">
              <label>{T("set.model")}</label>
              <input type="text" value={setting.llm.model}
                     onChange={(e) => set({ llm: { ...setting.llm, model: e.target.value } })} />
            </div>
            <div className="field">
              <label>{T("set.temp")} <span className="val">{temp.toFixed(1)}</span></label>
              <div className="temp-row">
                <input type="range" className="temp" min={0} max={1} step={0.1} value={temp}
                       style={{ "--fill": `${Math.round(temp * 100)}%` } as CSSProperties}
                       onChange={(e) => set({ llm: { ...setting.llm, temperature: Number(e.target.value) } })}
                       aria-label={T("set.temp")} />
                <span className="val">{temp.toFixed(1)}</span>
              </div>
              <div className="temp-ends"><span>{T("set.temp.kaku")}</span><span>{T("set.temp.kreatif")}</span></div>
            </div>
            <div className="field">
              <label>{T("set.maxtok")}</label>
              <div className="stepper">
                <button type="button" aria-label="−"
                        onClick={() => set({ llm: { ...setting.llm,
                          maxTokens: Math.max(16, (setting.llm.maxTokens ?? 8192) - 512) } })}>−</button>
                <input type="number" min={16} max={8192} value={setting.llm.maxTokens ?? 8192}
                       onChange={(e) => set({ llm: { ...setting.llm,
                         maxTokens: Math.max(16, Math.min(8192, Number(e.target.value) || 8192)) } })} />
                <button type="button" aria-label="+"
                        onClick={() => set({ llm: { ...setting.llm,
                          maxTokens: Math.min(8192, (setting.llm.maxTokens ?? 8192) + 512) } })}>+</button>
              </div>
              <div className="help">{T("set.maxtok.help")}</div>
            </div>
            <div style={{ marginBottom: 4 }}>
              <button className="btn secondary" data-tip={T("set.test.tip")} onClick={testKoneksi}
                      disabled={testState === "busy"}>
                {testState === "busy" ? T("set.test.busy") : T("set.test")}
              </button>
              {testMsg && (
                <div className={testState === "ok" ? "okbox" : "errbox"}>
                  {testState === "ok" ? "✓ " : ""}{testMsg}
                </div>)}
            </div>
          </div>
        </div>
      )}

      {tab === "cari" && (
        <div>
          <div className="set-sec">
            <div className="set-sec-t">{T("set.mode")}</div>
            <div className="seg">
              {([
                ["fixed", T("set.mode.fixed"), T("set.mode.fixed.tip")],
                ["agent", T("set.mode.agent"), T("set.mode.agent.tip")],
              ] as const).map(([m, nama, tip]) => (
                <label key={m} className={setting.mode === m ? "on" : ""} data-tip={tip}>
                  <input type="radio" name="mode" checked={setting.mode === m}
                         onChange={() => set({ mode: m as Settings["mode"] })} />
                  {nama}
                </label>
              ))}
            </div>
          </div>

          <div className="set-sec">
            <div className="set-sec-t">{T("set.emb")}{models.length > 0 && ` — ${models.length} ${T("set.emb.terdeteksi")}`}</div>
            <CSelect
              testid="dropdown-embedding"
              value={setting.embedding}
              options={models.map((m) => ({
                value: m.tag,
                label: `${m.tag} (dim: ${m.dim ?? "?"}, ${m.count ?? "?"} ${T("set.emb.vektor")}${m.is_gpu ? " · GPU" : ""})${m.aktif ? ` — ${T("set.emb.aktif")}` : ""}`,
              }))}
              onPick={(v) => switchEmbedding(v)} />
            {aktifModel?.is_gpu && (
              <div className="warnbox">
                ⚠ {aktifModel.tag} (dim {aktifModel.dim}) {T("set.emb.warn")}
              </div>
            )}
            {embMsg && <div className="okbox">{embMsg}</div>}
            {embErr && <div className="errbox">⚠ {embErr}</div>}
          </div>

          <div className="set-sec">
            <div className="set-sec-t">{T("set.retrieval")}</div>
            <div className="seg">
              {([
                ["hybrid", T("set.ret.hybrid.tip")],
                ["dense", T("set.ret.dense.tip")],
                ["bm25", T("set.ret.bm25.tip")],
              ] as const).map(([r, tip]) => (
                <label key={r} className={setting.retrieval === r ? "on" : ""} data-tip={tip}>
                  <input type="radio" name="retrieval" checked={setting.retrieval === r}
                         onChange={() => set({ retrieval: r as Settings["retrieval"] })} />
                  {T(`set.ret.${r}`)}
                </label>
              ))}
            </div>
          </div>

          <div className="set-sec" style={setting.retrieval === "bm25"
              ? { opacity: 0.45, filter: "grayscale(1)", pointerEvents: "none" } : undefined}>
            <div className="set-sec-t">{T("set.idx")}</div>
            <div className="seg">
              {([
                ["exact", "exact", T("set.idx.exact.tip")],
                ["hnsw", "HNSW", T("set.idx.hnsw.tip")],
                ["ivf", "IVF", T("set.idx.ivf.tip")],
                ["flat", "FLAT", T("set.idx.flat.tip")],
              ] as const).map(([v, nama, tip]) => (
                <label key={v} className={setting.denseMethod === v ? "on" : ""} data-tip={tip}>
                  <input type="radio" name="denseidx"
                         checked={setting.denseMethod === v}
                         disabled={setting.retrieval === "bm25"}
                         onChange={() => set({ denseMethod: v })} />
                  {nama}
                </label>
              ))}
            </div>
            {setting.retrieval === "bm25" ? (
              <div className="help">{T("set.idx.bm25off")}</div>
            ) : setting.denseMethod === "exact" ? (
              <div className="help">{T("set.idx.exact.note")}</div>
            ) : (
              <div className="help">{T("set.idx.milvus.note")}</div>
            )}
            {setting.retrieval !== "bm25" && setting.denseMethod !== "exact" && (
              <div style={{ marginTop: 8 }}>
                {milvusCek && (
                  <div className="help">{T("set.idx.cek")}</div>
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
              <div className="field" style={{ marginTop: 10, marginBottom: 0 }}>
                <label>{T("set.ef")}</label>
                <input type="number" min={1} max={512} value={setting.denseEf}
                       onChange={(e) => set({ denseEf: Math.max(1, Math.min(512, Number(e.target.value) || 64)) })} />
              </div>
            )}
            {setting.retrieval !== "bm25" && setting.denseMethod === "ivf" && (
              <div className="field" style={{ marginTop: 10, marginBottom: 0 }}>
                <label>{T("set.nprobe")}</label>
                <input type="number" min={1} max={1024} value={setting.denseNprobe}
                       onChange={(e) => set({ denseNprobe: Math.max(1, Math.min(1024, Number(e.target.value) || 16)) })} />
              </div>
            )}
          </div>

          <div className="set-sec">
            <div className="set-sec-t">{T("set.rerank")}</div>
            <div className="seg">
              <label className={setting.rerank ? "on" : ""} data-tip={T("set.rerank.on.tip")}>
                <input type="radio" name="rr" checked={setting.rerank}
                       onChange={() => set({ rerank: true })} /> {T("set.rerank.on")}
              </label>
              <label className={!setting.rerank ? "on" : ""} data-tip={T("set.rerank.off.tip")}>
                <input type="radio" name="rr" checked={!setting.rerank}
                       onChange={() => set({ rerank: false })} /> {T("set.rerank.off")}
              </label>
            </div>
          </div>

          <div className="set-sec">
            <div className="set-sec-t">{T("set.topk")}</div>
            <div style={{ display: "flex", gap: 8 }}>
              <div className="field" style={{ flex: 1, marginBottom: 0 }}>
                <label>{T("set.topk")}</label>
                <input type="number" min={1} max={50} value={setting.topK}
                       onChange={(e) => {
                         const v = Math.max(1, Math.min(50, Number(e.target.value) || 1));
                         set({ topK: v, injectK: Math.min(v, setting.injectK) });
                       }} />
              </div>
              <div className="field" style={{ flex: 1, marginBottom: 0 }}>
                <label>{T("set.inject")}</label>
                <input type="number" min={1} max={setting.topK} value={setting.injectK}
                       onChange={(e) => {
                         const v = Math.max(1, Math.min(setting.topK, Number(e.target.value) || 1));
                         set({ injectK: v });
                       }} />
              </div>
            </div>
            <div className="help">
              {T("set.topk.note.a")} {setting.topK} {T("set.topk.note.b")} {Math.min(setting.injectK, setting.topK)} {T("set.topk.note.c")}{setting.rerank ? ` ${T("set.topk.note.rerank")}` : ""}.
            </div>
          </div>
        </div>
      )}

      {tab === "gaya" && (
        <div>
          <div className="set-sec">
            <div className="set-sec-t">{T("set.gaya")}</div>
            <div className="seg">
              {([
                ["ringkas", T("set.gaya.ringkas"), T("set.gaya.ringkas.tip")],
                ["standar", T("set.gaya.standar"), T("set.gaya.standar.tip")],
                ["rinci", T("set.gaya.rinci"), T("set.gaya.rinci.tip")],
              ] as const).map(([v, nama, tip]) => (
                <label key={v} className={setting.gaya === v ? "on" : ""} data-tip={tip}>
                  <input type="radio" name="gaya" checked={setting.gaya === v}
                         onChange={() => set({ gaya: v })} />
                  {nama}
                </label>
              ))}
            </div>
            <div className="help">
              {setting.gaya === "ringkas" ? T("set.gaya.ringkas.note") :
               setting.gaya === "rinci" ? T("set.gaya.rinci.note") :
               T("set.gaya.standar.note")}
            </div>
          </div>

          <div className="set-sec">
            <div className="set-sec-t">{T("set.dry")}</div>
            <div className="seg">
              <label className={setting.dryRun ? "on" : ""} data-tip={T("set.dry.on.tip")}>
                <input type="radio" name="dry" checked={setting.dryRun}
                       onChange={() => set({ dryRun: true })} /> {T("set.dry.on")}
              </label>
              <label className={!setting.dryRun ? "on" : ""} data-tip={T("set.dry.off.tip")}>
                <input type="radio" name="dry" checked={!setting.dryRun}
                       onChange={() => set({ dryRun: false })} /> {T("set.dry.off")}
              </label>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

function CSelect({ value, options, onPick, testid }: {
  value: string;
  options: { value: string; label: string }[];
  onPick: (v: string) => void;
  testid?: string;
}) {
  const [buka, setBuka] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!buka) return;
    const tutup = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setBuka(false);
    };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setBuka(false); };
    document.addEventListener("mousedown", tutup);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", tutup);
      document.removeEventListener("keydown", key);
    };
  }, [buka]);
  const aktif = options.find((o) => o.value === value);
  return (
    <div className="cselect" ref={ref}>
      <button type="button" className="cselect-btn" data-testid={testid}
              aria-haspopup="listbox" aria-expanded={buka}
              onClick={() => setBuka((v) => !v)}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {aktif ? aktif.label : value || "—"}
        </span>
        <span className="chev">{buka ? "▲" : "▼"}</span>
      </button>
      {buka && (
        <div className="cselect-pop" role="listbox">
          {options.map((o) => (
            <button key={o.value} role="option" aria-selected={o.value === value}
                    className={o.value === value ? "on" : ""}
                    onClick={() => { onPick(o.value); setBuka(false); }}>
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
