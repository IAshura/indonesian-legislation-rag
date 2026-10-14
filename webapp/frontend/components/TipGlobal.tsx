"use client";
import { useEffect, useState } from "react";

// Tooltip global: 1 elemen position:fixed untuk SEMUA [data-tip].
// Tahan kursor 1,5 dtk -> muncul; pergi/klik/scroll -> langsung hilang.
// Posisi dihitung dari rect target + dijepit ke viewport (tak terpotong).
export default function TipGlobal() {
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cur: Element | null = null;
    const W = 240;

    const sembunyi = (bersihTimer = true) => {
      if (bersihTimer && timer) { clearTimeout(timer); timer = null; }
      cur = null;
      setTip(null);
    };
    const ukurTampil = (t: Element, text: string) => {
      const r = t.getBoundingClientRect();
      // di bawah target, rata kanan target; jepit ke viewport
      let x = Math.min(Math.max(8, r.right - W), window.innerWidth - W - 8);
      let y = r.bottom + 8;
      if (y + 110 > window.innerHeight) y = Math.max(8, r.top - 118);
      setTip({ text, x, y });
    };

    const masuk = (e: MouseEvent) => {
      const t = (e.target as HTMLElement).closest?.("[data-tip]") as Element | null;
      if (t === cur) return;   // masih di elemen sama: biarkan timer jalan
      if (timer) { clearTimeout(timer); timer = null; }
      cur = t;
      if (!t) { setTip(null); return; }
      const text = t.getAttribute("data-tip") || "";
      if (!text) { setTip(null); return; }
      timer = setTimeout(() => ukurTampil(t, text), 1500);
    };

    const keluar = (e: MouseEvent) => {
      const t = (e.target as HTMLElement).closest?.("[data-tip]");
      if (!t) sembunyi();
    };
    const klik = () => sembunyi();
    const gulir = () => sembunyi(false);
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") sembunyi(); };
    document.addEventListener("mouseover", masuk);
    document.addEventListener("mouseout", keluar);
    document.addEventListener("mousedown", klik);
    document.addEventListener("scroll", gulir, true);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mouseover", masuk);
      document.removeEventListener("mouseout", keluar);
      document.removeEventListener("mousedown", klik);
      document.removeEventListener("scroll", gulir, true);
      document.removeEventListener("keydown", key);
      if (timer) clearTimeout(timer);
    };
  }, []);

  if (!tip) return null;
  return (
    <div style={{ position: "fixed", left: tip.x, top: tip.y, width: 240,
                  background: "var(--panel-3)", border: "1px solid var(--border-2)",
                  color: "var(--text)", fontSize: 11, padding: "6px 10px",
                  borderRadius: 8, zIndex: 9999, pointerEvents: "none",
                  boxShadow: "var(--shadow)" }}>
      {tip.text}
    </div>
  );
}
