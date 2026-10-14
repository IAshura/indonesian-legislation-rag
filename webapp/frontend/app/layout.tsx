import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Indonesian Law RAG",
  description: "Ask Indonesian regulations — hybrid BM25 + dense retrieval, rerank, cited answers",
};

function themeInit() {
  return `(function(){try{var t=localStorage.getItem("rag-hukum-theme")||"dark";document.documentElement.setAttribute("data-theme",t)}catch(e){document.documentElement.setAttribute("data-theme","dark")}}())`;
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `(${themeInit.toString()})()` }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
