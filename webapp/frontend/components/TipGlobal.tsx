"use client";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

// Tooltip global: 1 elemen position:fixed untuk SEMUA [data-tip].
// Tahan kursor 1,5 dtk -> muncul; pergi/klik/scroll -> langsung hilang.
// Posisi diukur dari ukuran tooltip ASLI (bukan tebakan): tengah-bawah
// target, balik ke atas bila sempit, selalu dijepit ke dalam viewport.
export default function TipGlobal() {
  const [tip, setTip] = useState<{ text: string } | null>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const target = useRef<Element | null>(null);

  // ukur setelah render (sebelum paint: tanpa kedip), lalu tempatkan
  useLayoutEffect(() => {
    if (!tip || !ref.current || !target.current) return;
    const r = target.current.getBoundingClientRect();
    const w = ref.current.offsetWidth;
    const h = ref.current.offsetHeight;
    const x = Math.min(
      Math.max(8, r.left + r.width / 2 - w / 2),
      Math.max(8, window.innerWidth - w - 8)
    );
    let y = r.bottom + 8;
    if (y + h > window.innerHeight - 8) y = Math.max(8, r.top - h - 8);
    setPos({ x, y });
  }, [tip]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cur: Element | null = null;

    const sembunyi = (bersihTimer = true) => {
      if (bersihTimer && timer) { clearTimeout(timer); timer = null; }
      cur = null;
      target.current = null;
      setTip(null);
      setPos(null);
    };
    const ukurTampil = (t: Element, text: string) => {
      target.current = t;
      setTip({ text });
    };

    const masuk = (e: MouseEvent) => {
      const t = (e.target as HTMLElement).closest?.("[data-tip]") as Element | null;
      if (t === cur) return;   // masih di elemen sama: biarkan timer jalan
      if (timer) { clearTimeout(timer); timer = null; }
      cur = t;
      if (!t) { setTip(null); setPos(null); return; }
      const text = t.getAttribute("data-tip") || "";
      if (!text) { setTip(null); setPos(null); return; }
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
    <div ref={ref} style={{ position: "fixed",
                  left: pos ? pos.x : -9999, top: pos ? pos.y : -9999,
                  width: "max-content", maxWidth: 260,
                  background: "var(--surface-2)",
                  border: "1px solid var(--border-strong)",
                  color: "var(--text)", fontSize: 12, lineHeight: 1.5,
                  padding: "8px 12px", borderRadius: 10, zIndex: 9999,
                  pointerEvents: "none", boxShadow: "var(--shadow)" }}>
      {tip.text}
    </div>
  );
}
