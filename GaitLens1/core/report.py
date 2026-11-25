"""Шаблон негізіндегі қорытынды: сандар кодтан келеді, ойдан сан жоқ. Үш тілде."""
from .data import PARAMS
from .i18n import plabel, tr


def disclaimer(lang="kk"):
    return tr("disclaimer", lang)


def _fmt(p, r, lang):
    u = PARAMS[p]["unit"]
    word = tr("rp_up" if r["delta"] > 0 else "rp_down", lang)
    s = tr("rp_line", lang, p=plabel(p, lang), c=r["condition"], w=word, d=r["delta"], u=u)
    if r["delta_pct"] == r["delta_pct"]:
        s += f' ({r["delta_pct"]:+.1f}%)'
    s += tr("rp_tail", lang, lo=r["ci_low"], hi=r["ci_high"], n=int(r["n_cond"]),
            nb=int(r["n_base"]), ef=r["effect_d"])
    if PARAMS[p]["direction"] == "lower":
        s += tr("rp_lower", lang)
    return s


def build_summary(label, side, chg_ci, lang="kk", top_k=5, extra_md="", note=""):
    """chg_ci: bootstrap_change нәтижесі. Қайтарады: мәтін (markdown)."""
    lines = ["**" + tr("rp_subject", lang, l=label, s=side) + "**", ""]
    ok = chg_ci[chg_ci["verdict"] == "detected"].copy()
    for cond in ["Verbal", "VR"]:
        c = ok[ok["condition"] == cond]
        c = c.reindex(c["effect_d"].abs().sort_values(ascending=False).index).head(top_k)
        lines.append("**" + tr("rp_vs", lang, c=cond) + "**")
        if c.empty:
            lines.append(tr("rp_nochange", lang))
        for _, r in c.iterrows():
            lines.append("- " + _fmt(r["param"], r, lang))
        lines.append("")
    low = chg_ci[chg_ci["verdict"] == "few"]
    if not low.empty:
        lines.append(tr("rp_insuff", lang, n=", ".join(sorted(set(low["param"])))))
    if extra_md:
        lines += ["", "**" + tr("rp_rules", lang) + "**", extra_md]
    if note and note.strip():
        lines += ["", "**" + tr("rp_note", lang) + "**", note.strip()]
    lines += ["", "*" + tr("rp_method", lang) + "*", "", "*" + disclaimer(lang) + "*"]
    return "\n".join(lines)
