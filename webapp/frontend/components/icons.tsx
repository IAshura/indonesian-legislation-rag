"use client";
import type { SVGProps } from "react";

// Consistent 18px, 1.5-stroke icon set (lucide-style, zero emoji).
function base(props: SVGProps<SVGSVGElement>) {
  return {
    width: 18, height: 18, viewBox: "0 0 24 24", fill: "none",
    stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const, ...props,
  };
}

export const I = {
  scales: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M12 3v18" /><path d="M5 7h14" />
      <path d="M5 7l-2.5 6a3.5 3.5 0 0 0 7 0L7 7" /><path d="M19 7l-2.5 6a3.5 3.5 0 0 0 7 0L21 7" />
      <path d="M8 21h8" /></svg>
  ),
  menu: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M4 6h16" /><path d="M4 12h16" /><path d="M4 18h16" /></svg>
  ),
  x: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
  ),
  plus: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M12 5v14" /><path d="M5 12h14" /></svg>
  ),
  search: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>
  ),
  graph: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="8" r="2.5" />
      <circle cx="12" cy="18" r="2.5" /><path d="m8.2 7 7.3.7" /><path d="M7.3 8.2 10.8 16" />
      <path d="m16.7 10-3.2 5.7" /></svg>
  ),
  sun: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="12" cy="12" r="4" /><path d="M12 2v2" />
      <path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" /></svg>
  ),
  moon: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" /></svg>
  ),
  help: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><path d="M12 17h.01" /></svg>
  ),
  gear: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" /></svg>
  ),
  send: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>
  ),
  stop: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" /></svg>
  ),
  copy: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg>
  ),
  check: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M20 6 9 17l-5-5" /></svg>
  ),
  refresh: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M3 12a9 9 0 0 1 15-6.7L21 8" /><path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-15 6.7L3 16" /><path d="M3 21v-5h5" /></svg>
  ),
  download: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m7 10 5 5 5-5" /><path d="M12 15V3" /></svg>
  ),
  thumbUp: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" /></svg>
  ),
  thumbDown: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M17 14V2" />
      <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" /></svg>
  ),
  chevDown: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="m6 9 6 6 6-6" /></svg>
  ),
  eye: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" /></svg>
  ),
  eyeOff: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24" />
      <path d="M10.73 5.08A10.4 10.4 0 0 1 12 5c7 0 10 7 10 7a17.6 17.6 0 0 1-2.16 3.19" />
      <path d="M6.61 6.61A17.5 17.5 0 0 0 2 12s3 7 10 7a9.7 9.7 0 0 0 5.39-1.61" />
      <path d="m2 2 20 20" /></svg>
  ),
  trash: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M3 6h18" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
  ),
  dots: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><circle cx="5" cy="12" r="1" fill="currentColor" />
      <circle cx="12" cy="12" r="1" fill="currentColor" />
      <circle cx="19" cy="12" r="1" fill="currentColor" /></svg>
  ),
  arrowUpRight: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M7 17 17 7" /><path d="M7 7h10v10" /></svg>
  ),
  file: (p: SVGProps<SVGSVGElement> = {}) => (
    <svg {...base(p)}><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" /></svg>
  ),
};

// Small glyph per trace-node type (graph panel). Single accent/muted ink.
export function nodeGlyph(type: string) {
  const small = { width: 14, height: 14 };
  switch (type) {
    case "user_ask": return I.send(small);
    case "llm_ask": return I.help(small);
    case "embed_query": return I.graph(small);
    case "retrieve_dense": return I.search(small);
    case "retrieve_bm25": return I.file(small);
    case "fuse_rrf": return I.scales(small);
    case "rerank": return I.check(small);
    case "join_parent": return I.copy(small);
    case "build_prompt": return I.file(small);
    case "llm_answer": return I.check(small);
    case "tool_call": return I.gear(small);
    default: return null;
  }
}
