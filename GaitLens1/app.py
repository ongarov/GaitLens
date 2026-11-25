"""GaitLens — дәрігерге арналған жүріс талдау көмекшісі (kk / ru / en)."""
import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.cohort import cohort_effects, cohort_gps
from core.curves import CURVE_KEYS, build_curves, curve_columns, mean_sd_curves, unstable_cycles
from core.data import CONDITIONS, PARAMS, PARAM_COLS, build_dataset, quality_table, subject_labels
from core.i18n import LANGS, cdesc, cvlabel, phint, plabel, tr
from core.model import interpret, loso_eval, pairwise_eval
from core.pdf_report import build_pdf
from core.report import build_summary
from core.rules import (CMP_KEY, COMPARES, DEFAULT_RULES, MSG_FIELDS, OP_KEY, OPERATORS, SEV_KEY,
                        SEVERITIES, author_label, evaluate_rules, new_rule, rule_name,
                        rules_from_json, rules_summary_md, rules_to_json, validate_rule)
from core.stats import arrow, bootstrap_change, change_table, summary_table

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE, "sample_data")
RULES_PATH = os.path.join(BASE, "rules.json")
LOGO = os.path.join(BASE, "assets", "logo.png")
ICON = os.path.join(BASE, "assets", "icon.png")
NAVY, BLUE = "#0A1F44", "#0A6CE0"
# Түс соқырлығына лайық палитра (Okabe–Ito) + таңбалар
COLORS = {"Baseline": "#8A97A6", "Verbal": "#E69F00", "VR": "#0072B2"}
VCOL = {"down": "#009E73", "none": "#8A97A6", "up": "#D55E00", "na": "#C9D1D9"}

try:
    _icon = Image.open(ICON)
except Exception:
    _icon = "🦿"
st.set_page_config(page_title="GaitLens", page_icon=_icon, layout="wide")

st.markdown(
    f"""
    <style>
    div[data-testid="stMetric"] {{background:#F2F6FC; border-radius:14px; padding:12px 16px;
        border:1px solid #DCE6F2;}}
    .gl-note {{background:#EAF2FF; border-left:4px solid {BLUE}; padding:10px 14px;
        border-radius:8px; font-size:0.9rem;}}
    h1, h2, h3 {{color:{NAVY};}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Тіл ----------------
with st.sidebar:
    if os.path.exists(LOGO):
        st.image(LOGO, width=210)
    lang = st.radio("Тіл / Язык / Language", list(LANGS), format_func=lambda k: LANGS[k],
                    horizontal=True, key="lang")


def T(key, **kw):
    return tr(key, lang, **kw)


# ---------------- Ережелер (session) ----------------
def save_rules():
    try:
        with open(RULES_PATH, "w", encoding="utf-8") as f:
            f.write(rules_to_json(st.session_state["rules"]))
    except OSError:
        pass


if "rules" not in st.session_state:
    _loaded = []
    if os.path.exists(RULES_PATH):
        try:
            with open(RULES_PATH, encoding="utf-8") as f:
                _loaded, _ = rules_from_json(f.read(), lang)
        except OSError:
            _loaded = []
    st.session_state["rules"] = _loaded or [dict(r) for r in DEFAULT_RULES]


# ---------------- Деректер ----------------
@st.cache_data(show_spinner=False)
def load_uploaded(payload, lang_):
    class _F:  # UploadedFile-ға ұқсас орама
        def __init__(self, b):
            self._b = b

        def getvalue(self):
            return self._b

    files = [(n, _F(b)) for n, b in payload]
    data_, w1 = build_dataset(files, lang_)
    curves_, w2 = build_curves(files, lang_)
    return data_, curves_, w1 + w2


@st.cache_data(show_spinner=False)
def load_sample(lang_):
    files = [(os.path.basename(f), f) for f in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.csv")))]
    data_, w1 = build_dataset(files, lang_)
    curves_, w2 = build_curves(files, lang_)
    return data_, curves_, w1 + w2


@st.cache_data(show_spinner=False)
def get_cohort(df):
    return cohort_gps(df), cohort_effects(df)


@st.cache_data(show_spinner=False)
def run_model(df):
    return loso_eval(df), pairwise_eval(df)


with st.sidebar:
    st.markdown("**" + T("sb_data") + "**")
    uploads = st.file_uploader(T("uploader"), type=["csv"], accept_multiple_files=True,
                               help=T("uploader_help"))
    if uploads:
        data, curves, warns = load_uploaded(tuple((u.name, u.getvalue()) for u in uploads), lang)
        source = T("src_upload")
    else:
        data, curves, warns = load_sample(lang)
        source = T("src_sample")
    st.caption(T("source", s=source))

if data.empty:
    st.error(T("data_error"))
    for w in warns:
        st.warning(w)
    st.stop()

with st.sidebar:
    st.markdown("**" + T("sb_subject") + "**")
    show_id = st.checkbox(T("show_id"), value=False, help=T("show_id_help"))
    labels = subject_labels(data, show_real_id=show_id)
    subject = st.selectbox(T("pick_subject"), options=list(labels.keys()), format_func=lambda s: labels[s])
    for w in warns:
        st.warning(w)

d = data[data["subject"] == subject]
params = [p for p in PARAM_COLS if p in d.columns]
pid = labels[subject].split(" · ")[0]
sub_info = d.iloc[0]
chg = change_table(d, params)
chg_ci = bootstrap_change(d, params)
VMAP = {"detected": T("v_detected"), "none": T("v_none"), "few": T("v_few")}
NMAP = {"few": T("note_few"), "lown": T("note_lown"), "": ""}

# ---------------- Бас бет ----------------
hc1, hc2 = st.columns([1, 9])
if os.path.exists(ICON):
    hc1.image(ICON, width=70)
hc2.title(T("title"))
hc2.caption(T("tagline"))
st.markdown(f'<div class="gl-note">{T("banner")}</div>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric(T("m_subject"), pid)
c2.metric(T("m_side"), sub_info["side"])
c3.metric(T("m_hw"), f'{sub_info["height"]:.2f} {T("m")} / {sub_info["weight"]:.1f} {T("kg")}'
          if {"height", "weight"} <= set(d.columns) else "—")
c4.metric(T("m_cycles"), T("total", n=int(len(d))))
n_text = " · ".join(f'{c}: {int((d["condition"] == c).sum())}' for c in CONDITIONS)
st.caption(T("cycles_line", s=n_text))

(tab_card, tab_cmp, tab_curve, tab_det, tab_cohort, tab_rules, tab_sum, tab_model, tab_q) = st.tabs(
    [T("t_card"), T("t_cmp"), T("t_curve"), T("t_det"), T("t_cohort"), T("t_rules"), T("t_sum"),
     T("t_model"), T("t_quality")])


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ================= Пациент картасы =================
with tab_card:
    st.subheader(T("card_h"))
    key = [p for p in ["GPS_a", "GVS_aSagP", "GVS_aSagH", "GVS_aSagK", "GVS_aSagA"] if p in params]
    for cond in ["Verbal", "VR"]:
        st.markdown(f"**{T('card_vs', c=cond)}** · {cdesc(cond, lang)}")
        cols = st.columns(len(key))
        for col, p in zip(cols, key):
            r = chg[(chg["param"] == p) & (chg["condition"] == cond)].iloc[0]
            short = "GPS" if p == "GPS_a" else "GVS " + p.replace("GVS_aSag", "")
            if np.isnan(r["delta"]):
                col.metric(short, "—", T("card_few"))
                continue
            col.metric(short, f'{r["mean_cond"]:.2f}°', f'{r["delta"]:+.2f}°', delta_color="inverse")
            col.caption(T("card_n", n=int(r["n_cond"]), nb=int(r["n_base"]))
                        + (f' · ⚠ {NMAP[r["note"]]}' if r["note"] else ""))

    st.subheader(T("table_h"))
    out = pd.DataFrame({
        T("c_param"): chg["param"].map(lambda p: plabel(p, lang)),
        T("c_cond"): chg["condition"],
        "Baseline": chg["mean_base"].round(2),
        T("c_condval"): chg["mean_cond"].round(2),
        T("c_delta"): chg["delta"].round(2),
        "Δ %": chg["delta_pct"].round(1),
        T("c_dir"): chg["delta"].map(arrow),
        T("c_unit"): chg["param"].map(lambda p: PARAMS[p]["unit"]),
        T("c_nb"): chg["n_base"],
        T("c_nc"): chg["n_cond"],
        T("c_d"): chg["effect_d"].round(2),
        T("c_note"): chg["note"].map(NMAP),
    })
    st.dataframe(out, hide_index=True)
    st.caption(T("table_cap"))

# ================= Салыстыру =================
with tab_cmp:
    p = st.selectbox(T("pick_param"), params, format_func=lambda x: f"{plabel(x, lang)}  ({x})")
    st.caption(phint(p, lang))
    left, right = st.columns(2)
    fig = go.Figure()
    for cond in CONDITIONS:
        v = d.loc[d["condition"] == cond, p].dropna()
        if v.empty:
            continue
        fig.add_trace(go.Box(y=v, name=cond, boxpoints="all", jitter=0.4, pointpos=0,
                             marker_color=COLORS[cond], line_color=COLORS[cond], boxmean=True))
    fig.update_layout(title=T("box_title"), yaxis_title=f'{p}, {PARAMS[p]["unit"]}', showlegend=False,
                      template="plotly_white", height=420, margin=dict(t=60, b=20))
    left.plotly_chart(fig)
    fig2 = go.Figure()
    for cond in CONDITIONS:
        s = d[d["condition"] == cond].sort_values("cycle")
        if s.empty:
            continue
        fig2.add_trace(go.Scatter(x=s["cycle"], y=s[p], mode="lines+markers", name=cond,
                                  line=dict(color=COLORS[cond], width=2), marker=dict(size=5)))
    fig2.update_layout(title=T("line_title"), xaxis_title=T("cycle_no"),
                       yaxis_title=f'{p}, {PARAMS[p]["unit"]}', template="plotly_white", height=420,
                       margin=dict(t=60, b=20), legend=dict(orientation="h", y=-0.2))
    right.plotly_chart(fig2)
    st.markdown("**" + T("stats_h") + "**")
    sm = summary_table(d, [p]).drop(columns="param")
    sm = sm.rename(columns={"condition": T("c_cond"), "mean": T("c_mean"), "sd": "SD",
                            "median": T("c_median")}).round(2)
    st.dataframe(sm, hide_index=True)

# ================= Қисықтар =================
with tab_curve:
    cs = curves[curves["subject"] == subject] if not curves.empty else curves
    if cs.empty:
        st.info(T("curve_none"))
    else:
        v = st.selectbox(T("curve_var"), CURVE_KEYS, format_func=lambda x: f"{cvlabel(x, lang)}  ({x})")
        o1, o2 = st.columns(2)
        show_cycles = o1.checkbox(T("show_cycles"), value=False)
        k = o2.slider(T("flag_slider"), 2.0, 6.0, 3.0, 0.5, help=T("flag_help"))
        x, res = mean_sd_curves(cs, v)
        figc = go.Figure()
        for cond in CONDITIONS:
            if cond not in res:
                continue
            m, sd = res[cond]["mean"], res[cond]["sd"]
            figc.add_trace(go.Scatter(x=x, y=m + sd, mode="lines", line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"))
            figc.add_trace(go.Scatter(x=x, y=m - sd, mode="lines", line=dict(width=0), fill="tonexty",
                                      fillcolor=_rgba(COLORS[cond], 0.18), showlegend=False,
                                      hoverinfo="skip"))
            figc.add_trace(go.Scatter(x=x, y=m, mode="lines", name=f'{cond} (n={res[cond]["n"]})',
                                      line=dict(color=COLORS[cond], width=3)))
        flag = unstable_cycles(cs, v, k)
        cols_, _ = curve_columns(cs, v)
        if show_cycles:
            for _, r in cs.iterrows():
                figc.add_trace(go.Scatter(x=x, y=r[cols_].astype(float), mode="lines",
                                          line=dict(color=_rgba(COLORS[r["condition"]], 0.25), width=1),
                                          showlegend=False, hoverinfo="skip"))
        if not flag.empty:
            for _, r in flag[flag["flagged"]].iterrows():
                row = cs[(cs["condition"] == r["condition"]) & (cs["cycle"] == r["cycle"])].iloc[0]
                figc.add_trace(go.Scatter(x=x, y=row[cols_].astype(float), mode="lines",
                                          line=dict(color="#D55E00", width=2, dash="dash"),
                                          name=T("flag_name", c=r["condition"], n=r["cycle"])))
        figc.update_layout(title=T("curve_title"), xaxis_title=T("gc"), yaxis_title=cvlabel(v, lang),
                           template="plotly_white", height=480, margin=dict(t=60, b=20),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figc)
        if not flag.empty:
            nfl = flag[flag["flagged"]]
            if nfl.empty:
                st.success(T("flag_ok"))
            else:
                st.warning(T("flag_warn", k=len(nfl), n=len(flag)))
                st.dataframe(nfl.rename(columns={"condition": T("c_cond"), "cycle": T("cycle_no"),
                                                 "distance": T("c_dist"), "z": T("c_z")})
                             .drop(columns="flagged").round(2), hide_index=True)
        st.caption(T("curve_cap"))

# ================= Өзгеріс детекторы =================
with tab_det:
    st.subheader(T("det_h"))
    st.caption(T("det_cap"))
    rank = chg_ci.copy()
    rank["_abs"] = rank["effect_d"].abs()
    rank = rank.sort_values(["condition", "_abs"], ascending=[True, False])
    tbl = pd.DataFrame({
        T("c_cond"): rank["condition"],
        T("c_param"): rank["param"].map(lambda q: plabel(q, lang)),
        "Δ": rank["delta"].round(2),
        T("c_ci"): [f"[{lo:+.2f}; {hi:+.2f}]" if lo == lo else "—"
                    for lo, hi in zip(rank["ci_low"], rank["ci_high"])],
        T("c_d"): rank["effect_d"].round(2),
        T("c_result"): rank["verdict"].map(VMAP),
        T("c_nbc"): [f"{int(a)}/{int(b)}" for a, b in zip(rank["n_base"], rank["n_cond"])],
    })
    st.dataframe(tbl, hide_index=True)
    figf = go.Figure()
    for cond in ["Verbal", "VR"]:
        s_ = chg_ci[(chg_ci["condition"] == cond) & chg_ci["ci_low"].notna()]
        figf.add_trace(go.Scatter(
            x=s_["delta"], y=[f"{q} ({PARAMS[q]['unit']})" for q in s_["param"]], mode="markers", name=cond,
            marker=dict(color=COLORS[cond], size=10, symbol="circle" if cond == "Verbal" else "diamond"),
            error_x=dict(type="data", symmetric=False, array=s_["ci_high"] - s_["delta"],
                         arrayminus=s_["delta"] - s_["ci_low"], color=COLORS[cond])))
    figf.add_vline(x=0, line_dash="dot", line_color="#8A97A6")
    figf.update_layout(title=T("forest_title"), template="plotly_white", height=460,
                       margin=dict(t=60, b=20), xaxis_title=T("forest_x"),
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(figf)
    st.caption(T("forest_cap"))

# ================= Когорта =================
with tab_cohort:
    st.subheader(T("co_h"))
    st.caption(T("co_cap"))
    cg, ce = get_cohort(data)
    lab_of = lambda s: labels[s].split(" · ")[0]
    vlab = {"down": T("co_down"), "none": T("co_none"), "up": T("co_up"), "na": T("co_na")}
    figb = go.Figure()
    for cat in ["down", "none", "up", "na"]:
        figb.add_trace(go.Bar(name=vlab[cat], x=["Verbal", "VR"],
                              y=[int((cg["verdict_verbal"] == cat).sum()), int((cg["verdict_vr"] == cat).sum())],
                              marker_color=VCOL[cat]))
    figb.update_layout(barmode="stack", title=T("co_bar"), template="plotly_white", height=360,
                       margin=dict(t=60, b=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(figb)

    cond_sel = st.radio(T("co_heat_cond"), ["Verbal", "VR"], horizontal=True, key="co_cond")
    kk_ = cond_sel.lower()
    cs_ = cg.dropna(subset=[f"delta_{kk_}"]).sort_values(f"delta_{kk_}")
    figd = go.Figure()
    figd.add_trace(go.Scatter(
        x=[lab_of(s) for s in cs_["subject"]], y=cs_[f"delta_{kk_}"], mode="markers", name=cond_sel,
        marker=dict(color=[VCOL[v_] for v_ in cs_[f"verdict_{kk_}"]], size=9),
        error_y=dict(type="data", symmetric=False, array=cs_[f"hi_{kk_}"] - cs_[f"delta_{kk_}"],
                     arrayminus=cs_[f"delta_{kk_}"] - cs_[f"lo_{kk_}"], color="#B8C4D0")))
    me = cs_[cs_["subject"] == subject]
    if not me.empty:
        figd.add_trace(go.Scatter(x=[pid], y=me[f"delta_{kk_}"], mode="markers", name=T("co_you"),
                                  marker=dict(symbol="star", size=18, color=NAVY)))
    figd.add_hline(y=0, line_dash="dot", line_color="#8A97A6")
    figd.update_layout(title=T("co_dot"), template="plotly_white", height=420, yaxis_title="Δ GPS (°)",
                       margin=dict(t=60, b=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(figd)

    ef = ce[ce["condition"] == cond_sel].pivot(index="subject", columns="param", values="effect_d")
    ef = ef.reindex(columns=PARAM_COLS)
    figh = go.Figure(go.Heatmap(z=ef.values, x=[q for q in ef.columns], y=[lab_of(s) for s in ef.index],
                                colorscale="RdBu_r", zmid=0, colorbar=dict(title="d")))
    figh.update_layout(title=f'{T("co_heat")} — {cond_sel}', template="plotly_white",
                       height=max(420, 16 * len(ef) + 120), margin=dict(t=60, b=20),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(figh)

    base_all = cg["base"].dropna()
    mine = cg.loc[cg["subject"] == subject, "base"]
    if len(base_all) and len(mine) and mine.iloc[0] == mine.iloc[0]:
        st.info(T("co_pct", p=float((base_all < mine.iloc[0]).mean() * 100)))
    st.caption(T("co_note"))
    st.markdown("**" + T("co_table") + "**")
    tab_c = pd.DataFrame({
        T("m_subject"): [lab_of(s) for s in cg["subject"]],
        T("m_side"): cg["side"],
        T("c_base"): cg["base"].round(2),
        T("c_dv"): cg["delta_verbal"].round(2), T("c_rv"): cg["verdict_verbal"].map(vlab),
        T("c_dr"): cg["delta_vr"].round(2), T("c_rr"): cg["verdict_vr"].map(vlab),
    })
    st.dataframe(tab_c, hide_index=True)

# ================= Ережелер =================
with tab_rules:
    st.subheader(T("r_h"))
    st.caption(T("r_cap"))
    with st.expander(T("r_add"), expanded=False):
        with st.form("rule_form", clear_on_submit=False):
            r_name = st.text_input(T("r_name"), placeholder=T("r_name_ph"))
            f1, f2, f3 = st.columns(3)
            r_col = f1.selectbox(T("r_col"), PARAM_COLS, format_func=lambda x: f"{plabel(x, lang)} ({x})")
            r_cmp = f2.selectbox(T("r_cmp"), COMPARES, format_func=lambda x: T(CMP_KEY[x]))
            r_op = f3.selectbox(T("r_op"), OPERATORS, format_func=lambda x: T(OP_KEY[x]))
            g1, g2, g3 = st.columns(3)
            r_val = g1.number_input(T("r_val"), value=0.0, step=0.1, format="%.2f")
            r_conds = g2.multiselect(T("r_conds"), CONDITIONS, default=["Verbal", "VR"])
            r_sev = g3.selectbox(T("r_sev"), SEVERITIES, format_func=lambda x: T(SEV_KEY[x]))
            r_msg = st.text_area(T("r_msg"), value=T("r_msg_default"),
                                 help=T("r_msg_help", f=", ".join("{" + f + "}" for f in MSG_FIELDS)))
            submitted = st.form_submit_button(T("r_save"))
        if submitted:
            rule = new_rule(name=r_name.strip(), column=r_col, compare=r_cmp, operator=r_op,
                            value=r_val, conditions=r_conds, severity=r_sev, message=r_msg)
            errs = validate_rule(rule, lang)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.session_state["rules"].append(rule)
                save_rules()
                st.success(T("r_added", n=rule["name"]))

    rules = st.session_state["rules"]
    if rules:
        st.markdown("**" + T("r_saved_h", n=len(rules)) + "**")
        st.dataframe(pd.DataFrame({
            T("r_c_name"): [rule_name(x, lang) for x in rules],
            T("r_c_col"): [x["column"] for x in rules],
            T("r_c_type"): [T(CMP_KEY[x["compare"]]) for x in rules],
            T("r_c_cond"): [f'{x["operator"]} {x["value"]:g}' for x in rules],
            T("r_c_conds"): [", ".join(x["conditions"]) for x in rules],
            T("r_c_author"): [author_label(x["author"], lang) for x in rules],
        }), hide_index=True)
        dc1, dc2 = st.columns([3, 1])
        to_del = dc1.multiselect(T("r_del_pick"), [x["id"] for x in rules],
                                 format_func=lambda i: next(rule_name(x, lang) for x in rules if x["id"] == i))
        if dc2.button(T("r_del"), disabled=not to_del):
            st.session_state["rules"] = [x for x in rules if x["id"] not in to_del]
            save_rules()
            st.rerun()
    else:
        st.info(T("r_none"))

    ex1, ex2 = st.columns(2)
    ex1.download_button(T("r_dl"), rules_to_json(st.session_state["rules"]),
                        file_name="gaitlens_rules.json", mime="application/json")
    up = ex2.file_uploader(T("r_up"), type=["json"], key="rules_up")
    if up is not None and ex2.button(T("r_up_btn")):
        new, errs = rules_from_json(up.getvalue().decode("utf-8-sig", errors="replace"), lang)
        have = {x["id"] for x in st.session_state["rules"]}
        added = [x for x in new if x["id"] not in have]
        st.session_state["rules"].extend(added)
        save_rules()
        for e in errs:
            st.error(e)
        st.success(T("r_up_ok", n=len(added)))

    st.divider()
    st.subheader(T("r_res_h", p=pid))
    rule_results = evaluate_rules(d, st.session_state["rules"], lang)
    fired = [x for x in rule_results if x["fired"]]
    if not fired:
        st.success(T("r_none_fired"))
    for x in fired:
        box = st.warning if x["rule"]["severity"] == "attention" else st.info
        box(f'**{rule_name(x["rule"], lang)}** ({T("r_by", a=author_label(x["rule"]["author"], lang))}): {x["text"]}')
    notev = [x for x in rule_results if x["status"] != "ok"]
    if notev:
        with st.expander(T("r_unev", n=len(notev))):
            for x in notev:
                st.write(f'{rule_name(x["rule"], lang)} — {x["condition"]}: {T("r_unev_status")}')

# ================= Қорытынды =================
with tab_sum:
    st.subheader(T("s_h"))
    note = st.text_area(T("s_note_h"), key=f"note_{subject}", placeholder=T("s_note_ph"))
    rules_md = rules_summary_md(rule_results, lang)
    core_txt = build_summary(pid, sub_info["side"], chg_ci, lang)
    txt = build_summary(pid, sub_info["side"], chg_ci, lang, extra_md=rules_md, note=note)
    st.markdown(txt)

    csv_df = chg_ci.drop(columns=["note"]).copy()
    csv_df.insert(0, "participant", pid)
    dl1, dl2 = st.columns(2)
    dl1.download_button(T("s_dl_md"), txt, file_name=f"gaitlens_{pid}.md")
    dl2.download_button(T("s_dl_csv"), csv_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                        file_name=f"gaitlens_{pid}.csv", mime="text/csv")
    st.divider()
    if st.button(T("s_pdf_btn")):
        with st.spinner(T("s_pdf_spin")):
            cs_ = curves[curves["subject"] == subject] if not curves.empty else None
            st.session_state["pdf"] = ((subject, lang, note), build_pdf(
                pid, sub_info["side"], d, cs_, chg_ci, core_txt, rules_md, lang, note))
    pdf_state = st.session_state.get("pdf")
    if pdf_state and pdf_state[0] == (subject, lang, note):
        st.download_button(T("s_pdf_dl"), pdf_state[1], file_name=f"gaitlens_{pid}_{lang}.pdf",
                           mime="application/pdf")
    with st.expander(T("s_icf_h")):
        st.markdown(T("icf_text"))

# ================= Модель сапасы =================
with tab_model:
    st.subheader(T("mo_h"))
    st.markdown(T("mo_text"))
    if len(data["subject"].unique()) < 5:
        st.info(T("mo_few"))
    elif st.button(T("mo_run")):
        with st.spinner(T("mo_spin")):
            st.session_state["model_res"] = run_model(data)
    if "model_res" in st.session_state:
        res, pair = st.session_state["model_res"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f'{res["accuracy"]:.3f}')
        m2.metric("Balanced accuracy", f'{res["bal_acc"]:.3f}')
        m3.metric("Macro-F1", f'{res["macro_f1"]:.3f}')
        m4.metric(T("mo_chance"), f'{res["chance_bal"]:.3f}')
        st.info(interpret(res, lang))
        cm = res["cm"]
        figm = go.Figure(go.Heatmap(z=cm, x=[T("mo_pred", l=l) for l in res["labels"]],
                                    y=[T("mo_true", l=l) for l in res["labels"]], text=cm,
                                    texttemplate="%{text}", colorscale="Blues", showscale=False))
        figm.update_layout(title=T("mo_cm"), height=380, template="plotly_white",
                           yaxis=dict(autorange="reversed"), margin=dict(t=60, b=20))
        cA, cB = st.columns(2)
        cA.plotly_chart(figm)
        imp = res["importance"]
        figi = go.Figure(go.Bar(x=imp.values[::-1], y=[plabel(i, lang) for i in imp.index][::-1],
                                orientation="h", marker_color=BLUE))
        figi.update_layout(title=T("mo_imp"), height=380, template="plotly_white", margin=dict(t=60, b=20))
        cB.plotly_chart(figi)
        st.markdown("**" + T("mo_pair") + "**")
        pdsp = pair.rename(columns={"pair": T("mo_pair_c"), "bal_acc": "Balanced accuracy",
                                    "macro_f1": "Macro-F1", "chance": T("mo_chance")}).round(3)
        st.dataframe(pdsp, hide_index=True)
        figs = go.Figure(go.Histogram(x=res["per_subject"].values, nbinsx=12, marker_color=COLORS["VR"]))
        figs.update_layout(title=T("mo_ps"), xaxis_title=T("mo_acc"), yaxis_title=T("mo_cnt"), height=300,
                           template="plotly_white", margin=dict(t=60, b=20))
        st.plotly_chart(figs)
        st.caption(T("mo_cap", s=res["n_subjects"], c=res["n_cycles"]))

# ================= Деректер сапасы =================
with tab_q:
    st.subheader(T("q_all"))
    st.dataframe(quality_table(data, lang), hide_index=True)
    st.subheader(T("q_subj"))
    nan = d.groupby("condition")[params].apply(lambda x: x.isna().sum()).T
    nan.index = [plabel(i, lang) for i in nan.index]
    st.dataframe(nan.reset_index().rename(columns={"index": T("c_param")}), hide_index=True)
    st.caption(T("q_nan_cap"))
