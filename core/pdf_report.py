"""Дәрігерге арналған PDF есеп (4-қадам). reportlab + matplotlib, интернет керек емес."""
import io
import os
import re
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .curves import CURVE_KEYS, mean_sd_curves
from .data import CONDITIONS, PARAMS
from .i18n import cvlabel, plabel, tr
from .report import disclaimer

COL = {"Baseline": "#8A97A6", "Verbal": "#E69F00", "VR": "#0072B2"}
LOGO = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
_FONTS_DONE = False


def _register_fonts():
    global _FONTS_DONE
    if _FONTS_DONE:
        return
    base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")  # DejaVu: қазақ әріптерін қолдайды
    for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                    ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
        pdfmetrics.registerFont(TTFont(name, os.path.join(base, f)))
    addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B")
    addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")
    _FONTS_DONE = True


def _md(text: str) -> str:
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    return t


def _fig_png(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _forest(chg_ci, lang):
    d = chg_ci[chg_ci["ci_low"].notna()]
    if d.empty:
        return None
    params = list(dict.fromkeys(d["param"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(params) + 1.2))
    for off, cond in [(-0.15, "Verbal"), (0.15, "VR")]:
        s = d[d["condition"] == cond]
        ys = [params.index(p) + off for p in s["param"]]
        ax.errorbar(s["delta"], ys, xerr=[s["delta"] - s["ci_low"], s["ci_high"] - s["delta"]],
                    fmt="o", color=COL[cond], label=cond, capsize=3)
    ax.axvline(0, color="#8A97A6", ls=":")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([f"{p} ({PARAMS[p]['unit']})" for p in params], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(tr("pdf_forest_x", lang))
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="x", alpha=0.25)
    return _fig_png(fig)


def _curves(cs, lang):
    if cs is None or cs.empty:
        return None
    vars_ = CURVE_KEYS
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0))
    for ax, v in zip(axes.ravel(), vars_):
        x, res = mean_sd_curves(cs, v)
        for cond in CONDITIONS:
            if cond in res:
                m, sd = res[cond]["mean"], res[cond]["sd"]
                ax.plot(x, m, color=COL[cond], lw=1.8, label=f'{cond} (n={res[cond]["n"]})')
                ax.fill_between(x, m - sd, m + sd, color=COL[cond], alpha=0.15)
        ax.set_title(cvlabel(v, lang), fontsize=8)
        ax.set_xlabel(tr("gc", lang), fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center", fontsize=9, frameon=False)
    fig.tight_layout()
    return _fig_png(fig)


def build_pdf(label, side, d, curves_subj, chg_ci, summary_md, rules_md, lang="kk", note="") -> bytes:
    """summary_md — қосымша (ереже/ескертпесіз) қорытынды; ережелер мен ескертпе бөлек бөлім болып қосылады."""
    _register_fonts()
    from PIL import Image as PILImage
    S = lambda name, **kw: ParagraphStyle(name, **{"fontName": "DV", **kw})
    navy, blue = colors.HexColor("#0A1F44"), colors.HexColor("#0A6CE0")
    h1 = S("h1", fontName="DV-B", fontSize=16, leading=20, textColor=navy, spaceAfter=3)
    h2 = S("h2", fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4, keepWithNext=1,
           textColor=blue)
    body = S("body", fontSize=9, leading=12.5)
    small = S("small", fontSize=7.5, leading=10, textColor=colors.HexColor("#5B6B7B"))
    cell = S("cell", fontSize=7.5, leading=9.5)
    bul = S("bul", fontSize=8.5, leading=11.5, leftIndent=10, firstLineIndent=-6)
    warn = S("warn", fontSize=8.5, leading=11.5, backColor=colors.HexColor("#EAF2FF"),
             borderPadding=6, borderColor=blue, borderWidth=0.6)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"GaitLens — {label}", author="GaitLens")
    el = []
    if os.path.exists(LOGO):
        w, h = PILImage.open(LOGO).size
        head = Table([[Image(LOGO, width=30 * mm, height=30 * mm * h / w),
                       [Paragraph(tr("pdf_title", lang), h1),
                        Paragraph(_md(tr("pdf_line", lang, p=label, s=side, d=f"{date.today():%d.%m.%Y}")), body)]]],
                     colWidths=[34 * mm, 144 * mm])
        head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        el.append(head)
    else:
        el += [Paragraph(tr("pdf_title", lang), h1),
               Paragraph(_md(tr("pdf_line", lang, p=label, s=side, d=f"{date.today():%d.%m.%Y}")), body)]
    el += [Spacer(1, 4), Paragraph(_md(tr("pdf_attn", lang) + disclaimer(lang)), warn), Spacer(1, 6)]

    ns = " · ".join(f'{c}: {int((d["condition"] == c).sum())}' for c in CONDITIONS)
    el += [Paragraph(tr("pdf_cycles_h", lang), h2), Paragraph(ns, body)]

    el.append(Paragraph(tr("pdf_changes_h", lang), h2))
    rows = [[Paragraph(h, cell) for h in tr("pdf_cols", lang)]]
    vmap = {"detected": tr("v_detected", lang), "none": tr("v_none", lang), "few": tr("v_few", lang)}
    for _, r in chg_ci.iterrows():
        def f(v, spec="{:.2f}"):
            return spec.format(v) if v == v else "—"
        ci = f"[{r['ci_low']:+.2f}; {r['ci_high']:+.2f}]" if r["ci_low"] == r["ci_low"] else "—"
        rows.append([Paragraph(plabel(r["param"], lang), cell), Paragraph(r["condition"], cell),
                     f(r["mean_base"]), f(r["mean_cond"]), f(r["delta"], "{:+.2f}"), ci,
                     f(r["delta_pct"], "{:+.1f}"), f(r["effect_d"], "{:+.2f}"),
                     Paragraph(vmap.get(r["verdict"], r["verdict"]), cell)])
    t = Table(rows, colWidths=[36 * mm, 17 * mm, 16 * mm, 18 * mm, 13 * mm, 25 * mm, 12 * mm, 11 * mm, 30 * mm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EEFC")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D6DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFB")])]))
    el += [t, Paragraph(tr("pdf_tcap", lang), small)]

    fp = _forest(chg_ci, lang)
    if fp:
        w, h = PILImage.open(fp).size; fp.seek(0)
        el += [Spacer(1, 6), Image(fp, width=165 * mm, height=165 * mm * h / w)]

    cp = _curves(curves_subj, lang)
    if cp:
        el.append(Paragraph(tr("pdf_curves_h", lang), h2))
        w, h = PILImage.open(cp).size; cp.seek(0)
        el.append(Image(cp, width=150 * mm, height=150 * mm * h / w))

    el.append(Paragraph(tr("pdf_summary_h", lang), h2))
    for line in summary_md.splitlines():
        line = line.strip()
        if not line or (line.startswith("*") and not line.startswith("**")):
            continue
        if line.startswith("- "):
            el.append(Paragraph("• " + _md(line[2:]), bul))
        else:
            el.append(Paragraph(_md(line), body))
    if rules_md:
        el.append(Paragraph(tr("pdf_rules_h", lang), h2))
        for line in rules_md.splitlines():
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
            if line:
                el.append(Paragraph("• " + _md(line), bul))
    if note and note.strip():
        el.append(Paragraph(tr("pdf_note_h", lang), h2))
        for para in note.strip().splitlines():
            if para.strip():
                el.append(Paragraph(_md(para.strip()), body))
    el.append(Paragraph(tr("pdf_icf_h", lang), h2))
    el.append(Paragraph(_md(tr("icf_text", lang)), body))
    el += [Spacer(1, 8), Paragraph(_md("*" + disclaimer(lang) + "*"), small)]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor("#5B6B7B"))
        canvas.drawString(16 * mm, 9 * mm, tr("pdf_footer", lang))
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, tr("pdf_page", lang, n=doc_.page))
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
