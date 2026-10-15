"use client";
import { useEffect, useRef, useState } from "react";
import type { Citation } from "../lib/types";
import { t, type Lang } from "../lib/i18n";
import { I } from "./icons";

function parseJudul(citationStr: string): { pasal: string | null; bab: string | null } {
  const pasal = citationStr.match(/pasal\s+([\w\d]+)/i);
  const bab = citationStr.match(/bab\s+([IVXLC\d]+)/i);
  return {
    pasal: pasal ? `Pasal ${pasal[1]}` : null,
    bab: bab ? `Bab ${bab[1].toUpperCase()}` : null,
  };
}

function pecahAyat(text: string): { no: string; isi: string }[] | null {
  // Split on "(1)", "(2)", … verse markers. Fall back to plain text.
  const parts = text.split(/(?=\(\d+\)\s)/);
  if (parts.length < 2) return null;
  const out: { no: string; isi: string }[] = [];
  for (const p of parts) {
    const m = p.match(/^\(?(\d+)\)?\s*([\s\S]*)$/);
    if (!m || !m[2].trim()) return null;
    out.push({ no: `(${m[1]})`, isi: m[2].trim() });
  }
  return out;
}

function potongKata(text: string, maks: number): { tampil: string; penuh: boolean } {
  if (text.length <= maks) return { tampil: text, penuh: true };
  const cut = text.lastIndexOf(" ", maks);
  const akhir = cut > maks * 0.5 ? cut : maks;
  return { tampil: text.slice(0, akhir).trimEnd() + "…", penuh: false };
}

export default function SourceModal({ list, idx, bahasa, onTutup, onPindah }: {
  list: Citation[];
  idx: number;
  bahasa: Lang;
  onTutup: () => void;
  onPindah: (idx: number) => void;
}) {
  const L: Lang = bahasa || "id";
  const T = (k: string) => t(L, k);
  const c = list[idx];
  const adaBanyak = list.length > 1;
  const boxRef = useRef<HTMLDivElement>(null);
  const [tersalin, setTersalin] = useState<"kutipan" | "sitasi" | null>(null);

  // initial focus on the dialog, body scroll lock
  useEffect(() => {
    boxRef.current?.focus();
    const lama = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = lama; };
  }, []);
  // reset copy feedback when switching sources
  useEffect(() => { setTersalin(null); }, [idx]);

  // Esc / arrows / simple focus trap
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onTutup(); }
      else if (e.key === "ArrowLeft" && adaBanyak) onPindah((idx - 1 + list.length) % list.length);
      else if (e.key === "ArrowRight" && adaBanyak) onPindah((idx + 1) % list.length);
      else if (e.key === "Tab" && boxRef.current) {
        const f = Array.from(
          boxRef.current.querySelectorAll<HTMLElement>(
            'button, [href], [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => !el.hasAttribute("disabled"));
        if (f.length === 0) return;
        const pertama = f[0];
        const terakhir = f[f.length - 1];
        if (e.shiftKey && document.activeElement === pertama) {
          e.preventDefault(); terakhir.focus();
        } else if (!e.shiftKey && document.activeElement === terakhir) {
          e.preventDefault(); pertama.focus();
        }
      }
    };
    document.addEventListener("keydown", key, true);
    return () => document.removeEventListener("keydown", key, true);
  }, [idx, list.length, adaBanyak, onTutup, onPindah]);

  if (!c) return null;
  const { pasal, bab } = parseJudul(c.citation_str || "");
  const judul = pasal || c.official_title || `S${c.n}`;
  const teks = c.text || "";
  const terpotongBackend = /(\.\.\.|…)\s*$/.test(teks);
  const ayat = pecahAyat(teks);
  const sitasiFormat = [c.official_title, bab, pasal].filter(Boolean).join(", ");

  const salin = async (teksSalin: string, mana: "kutipan" | "sitasi") => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(teksSalin);
      else {
        const ta = document.createElement("textarea");
        ta.value = teksSalin; document.body.appendChild(ta);
        ta.select(); document.execCommand("copy"); document.body.removeChild(ta);
      }
      setTersalin(mana);
      setTimeout(() => setTersalin(null), 1500);
    } catch { /* ignore */ }
  };

  return (
    <div className="src-scrim" onClick={onTutup}>
      <div ref={boxRef} className="src-modal" role="dialog" aria-modal="true"
           aria-labelledby="src-judul" tabIndex={-1}
           onClick={(e) => e.stopPropagation()}>
        <div className="src-head">
          <div className="src-ident">
            <span className="src-chipnum">{c.n}</span>
            <span className="src-dok">{c.official_title}</span>
          </div>
          <div className="src-aksi-head">
            {adaBanyak && (
              <>
                <button type="button" className="icon-btn sq" aria-label={T("src.sebelum")}
                        onClick={() => onPindah((idx - 1 + list.length) % list.length)}>‹</button>
                <button type="button" className="icon-btn sq" aria-label={T("src.sesudah")}
                        onClick={() => onPindah((idx + 1) % list.length)}>›</button>
              </>
            )}
            <button type="button" className="icon-btn sq" aria-label={T("about.tutup")}
                    onClick={onTutup}><I.x /></button>
          </div>
        </div>
        <h2 id="src-judul" className="src-judul">{judul}</h2>
        {bab && <div className="src-sub">{bab}</div>}
        <div className="src-body" key={idx}>
          {ayat ? (
            <div className="src-ayat">
              {ayat.map((a, i) => (
                <div className="src-ayat-baris" key={i}>
                  <span className="src-ayat-no">{a.no}</span>
                  <span className="src-ayat-isi">{a.isi}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="src-teks">{teks}</p>
          )}
          {terpotongBackend && <div className="src-potong">{T("src.potong")}</div>}
        </div>
        <div className="src-foot">
          <button type="button" className="src-tombol"
                  onClick={() => salin(teks, "kutipan")}>
            {tersalin === "kutipan" ? <I.check /> : <I.copy />}
            {tersalin === "kutipan" ? T("src.tersalin") : T("src.salin.kutipan")}
          </button>
          <button type="button" className="src-tombol"
                  onClick={() => salin(sitasiFormat, "sitasi")}>
            {tersalin === "sitasi" ? <I.check /> : <I.copy />}
            {tersalin === "sitasi" ? T("src.tersalin") : T("src.salin.sitasi")}
          </button>
        </div>
      </div>
    </div>
  );
}

export { potongKata };
