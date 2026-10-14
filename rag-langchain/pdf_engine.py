"""
pdf_engine.py — Mesin kecil untuk membuat PDF penjelasan kode (reportlab).
==========================================================================

DIPAKAI OLEH: build_docs_pdf.py (di folder yang sama).
Isi penjelasan ditulis di file itu; file ini hanya tahu CARA MENCETAK:

  - Platypus = "table of flowables": setiap paragraf/tabel/blok kode adalah
    flowable yang mengalir satu per satu dari atas ke bawah halaman.
  - DAFTAR ISI: reportlab butuh DUA KALI render (multiBuild) karena
    nomor halaman baru tahu setelah lewatan pertama. afterFlowable()
    memberi tahu TOC tiap kali judul bab lewat.
  - tabel_entitas(): membaca file .py dengan modul `ast` (Abstract Syntax
    Tree) untuk mendaftar class/function + signature + nomor baris +
    docstring — tabel di PDF SELALU sesuai kode, bukan salin-tempel.

Catatan font: font bawaan (Helvetica/Courier) memakai encoding WinAnsi;
karakter di luar itu (emoji, panah unicode) tidak dicetak — pakai "->".
"""

from __future__ import annotations

import ast
import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak,
                                PageTemplate, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

# ------------------------------------------------------------- PALET WARNA
TEAL = colors.HexColor("#0f766e")     # aksen utama (teal)
INK = colors.HexColor("#1f2937")      # teks utama
DIM = colors.HexColor("#6b7280")      # teks redup
GOLD = colors.HexColor("#b45309")     # aksen kedua
CODE_BG = colors.HexColor("#f4f4f5")  # latar blok kode
ROW_ALT = colors.HexColor("#f0fdfa")  # baris tabel genap

# ------------------------------------------------------------------ GAYA
_st = getSampleStyleSheet()
S = {
    "Judul": ParagraphStyle("Judul", parent=_st["Title"], fontSize=26,
                            leading=30, textColor=INK, spaceAfter=6),
    "Sub": ParagraphStyle("Sub", parent=_st["Normal"], fontSize=11,
                          leading=15, textColor=DIM),
    "H1": ParagraphStyle("H1", parent=_st["Heading1"], fontSize=16,
                         leading=20, textColor=TEAL, spaceBefore=6,
                         spaceAfter=8),
    "H2": ParagraphStyle("H2", parent=_st["Heading2"], fontSize=12.5,
                         leading=16, textColor=GOLD, spaceBefore=10,
                         spaceAfter=4),
    "Isi": ParagraphStyle("Isi", parent=_st["BodyText"], fontSize=10,
                          leading=14.5, textColor=INK, spaceAfter=6),
    "Bullet": ParagraphStyle("Bullet", parent=_st["BodyText"], fontSize=10,
                             leading=14, leftIndent=14, bulletIndent=4,
                             textColor=INK, spaceAfter=3),
    "Kode": ParagraphStyle("Kode", parent=_st["Code"], fontName="Courier",
                           fontSize=8.2, leading=11, backColor=CODE_BG,
                           borderPadding=6, leftIndent=6, rightIndent=6,
                           spaceBefore=4, spaceAfter=8),
    "Ket": ParagraphStyle("Ket", parent=_st["Normal"], fontSize=8,
                          leading=11, textColor=DIM),
}

UKURAN = A4
MARGIN = 2.0 * cm


# ---------------------------------------------------- KELAS DOKUMEN + TOC
class DocTOC(BaseDocTemplate):
    """DocTemplate + pengumuman ke TableOfContents tiap judul lewat."""

    def afterFlowable(self, flowable):
        # Paragraph menyimpan gaya; bila gaya H1/H2, catat utk TOC.
        if isinstance(flowable, Paragraph):
            st = flowable.style.name
            if st in ("H1", "H2"):
                teks = flowable.getPlainText()
                level = 0 if st == "H1" else 1
                self.notify("TOCEntry", (level, teks, self.page))


def _footer(canvas, doc):
    """Header/footer tiap halaman: nama dokumen kiri, nomor kanan."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DIM)
    canvas.drawString(MARGIN, 1.1 * cm, "Penjelasan Kode — "
                      + Path(doc.filename).parent.name)
    canvas.drawRightString(UKURAN[0] - MARGIN, 1.1 * cm, f"hal. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
    canvas.line(MARGIN, 1.4 * cm, UKURAN[0] - MARGIN, 1.4 * cm)
    canvas.restoreState()


# ---------------------------------------------------------- PEMBANGUN ISI
def cover(judul: str, subjudul: str, meta: list[str]) -> list:
    """Halaman sampul: judul besar + garis aksen + meta proyek."""
    isi = [Spacer(1, 5.5 * cm),
           Paragraph(judul, S["Judul"]),
           Paragraph(subjudul, S["Sub"]),
           Spacer(1, 1.0 * cm)]
    t = Table([[Paragraph(m, S["Isi"])] for m in meta], colWidths=[16 * cm])
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, TEAL),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafaf9")),
    ]))
    isi += [t, PageBreak()]
    return isi


def daftar_isi() -> list:
    """Flowable TOC + 'bab 0' pengantar cara membaca PDF ini."""
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc1", fontSize=10.5, leading=15, textColor=INK,
                       leftIndent=6),
        ParagraphStyle("toc2", fontSize=9.5, leading=13, textColor=DIM,
                       leftIndent=18),
    ]
    return [Paragraph("Daftar Isi", S["H1"]), toc,
            Paragraph("Setiap bab menjelaskan satu file kode: untuk apa "
                      "file itu ada, konsep kuncinya, tabel fungsi/class "
                      "(otomatis dari kode), catatan fungsi per fungsi, "
                      "contoh menjalankan, dan jebakan umum.", S["Isi"]),
            PageBreak()]


def bab(nomor: str, judul: str) -> list:
    return [Paragraph(f"{nomor} · {html.escape(judul)}", S["H1"])]


def sub(judul: str) -> list:
    return [Paragraph(html.escape(judul), S["H2"])]


def para(teks: str) -> list:
    """Paragraf; *bintang* -> miring, `backtick` -> font mono (pasangan)."""
    import re
    t = html.escape(teks)
    t = re.sub(r"`([^`]*)`", r"<font face='Courier' size='9'>\1</font>", t)
    t = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)
    return [Paragraph(t, S["Isi"])]


def bullets(daftar: list[str]) -> list:
    return [Paragraph("• " + html.escape(b), S["Bullet"]) for b in daftar]


def kode(teks: str) -> list:
    return [Preformatted(teks, S["Kode"])]


# ------------------------------------------------- TABEL ENTITAS (ast)
def tabel_entitas(path_py: str | Path, max_doc: int = 110) -> list:
    """
    Tabel fungsi/class satu file, dibaca OTOMATIS via `ast`:
    nama, jenis, baris awal, signature (disingkat), docstring terpenggal.
    Karena di-generate dari file, tabel tak mungkin basi bila kode berubah.
    """
    tree = ast.parse(Path(path_py).read_text(encoding="utf-8"))
    baris: list[list[str]] = [["Entitas", "Jenis", "Baris", "Signature / docstring"]]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            jenis = "class" if isinstance(node, ast.ClassDef) else "def"
            sig = _signature(node)
            doc = ast.get_docstring(node, clean=True) or ""
            doc = " ".join(doc.split())[:max_doc]
            isi = sig + ("<br/>" + html.escape(doc) if doc else "")
            baris.append([html.escape(node.name), jenis,
                          str(node.lineno), isi])
    t = Table(baris, colWidths=[3.4 * cm, 1.3 * cm, 1.2 * cm, 10.6 * cm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ]))
    return [t, Spacer(1, 6)]


def _signature(node) -> str:
    """Signature singkat: fungsi nama(arg) -> ret; class nama(Base)."""
    if isinstance(node, ast.ClassDef):            # class: tampilkan basis
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        sig = f"{node.name}({bases})"
    else:
        a = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        sig = f"{node.name}({a}){ret}"
    return html.escape(sig[:90] + ("..." if len(sig) > 90 else ""))


# ---------------------------------------------------------------- RAKIT
def bangun_pdf(path_output: str | Path, judul: str, subjudul: str,
               meta: list[str], isi: list) -> None:
    """Susun cover + TOC + isi; multiBuild agar TOC terisi nomor halaman."""
    doc = DocTOC(str(path_output), pagesize=UKURAN,
                 leftMargin=MARGIN, rightMargin=MARGIN,
                 topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                 title=judul, author="Codebuff")
    frame = Frame(MARGIN, 1.8 * cm, UKURAN[0] - 2 * MARGIN,
                  UKURAN[1] - 3.6 * cm, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_footer)])
    doc.multiBuild(cover(judul, subjudul, meta) + daftar_isi() + isi)
    print(f"PDF dibuat: {path_output}")
