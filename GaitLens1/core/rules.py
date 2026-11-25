"""Дәрігер ережелері (3-қадам).

Ереже — код емес, ДЕРЕК (JSON). Орындаушы тіркелген, тексерілген; ешқандай exec/eval жоқ,
интернет те, API де керек емес.
"""
import json
import math
import re
import uuid

import numpy as np

from .data import PARAMS, PARAM_COLS, CONDITIONS
from .i18n import tr, plabel

OP_KEY = {">": "op_gt", ">=": "op_ge", "<": "op_lt", "<=": "op_le", "abs>": "op_abs"}
CMP_KEY = {"delta": "cmp_delta", "delta_pct": "cmp_pct", "value": "cmp_value", "n_cycles": "cmp_n"}
SEV_KEY = {"info": "sev_info", "attention": "sev_attention"}
OPERATORS = list(OP_KEY)
COMPARES = list(CMP_KEY)
SEVERITIES = list(SEV_KEY)
MSG_FIELDS = ["param", "condition", "delta", "delta_pct", "value", "baseline", "n", "n_base",
              "threshold", "operator"]
_field_re = re.compile(r"\{([^{}]*)\}")
_field_ok = re.compile(r"^(" + "|".join(MSG_FIELDS) + r")(:[+\-.0-9a-z%]*)?$")

DEFAULT_RULES = [{
    "id": "default_n_low", "builtin": True, "name": "", "column": PARAM_COLS[0],
    "compare": "n_cycles", "operator": "<", "value": 10.0,
    "conditions": ["Baseline", "Verbal", "VR"], "severity": "attention",
    "message": "", "author": "sample",
}]


def rule_name(r, lang):
    return tr("default_name", lang) if r.get("builtin") else r["name"]


def rule_message(r, lang):
    return tr("default_msg", lang) if r.get("builtin") else r["message"]


def author_label(a, lang):
    return {"doctor": tr("author_doc", lang), "sample": tr("author_sample", lang)}.get(a, a)


def validate_rule(r: dict, lang="kk") -> list:
    errs = []
    if not r.get("builtin") and not str(r.get("name", "")).strip():
        errs.append(tr("e_name", lang))
    if r.get("column") not in PARAMS:
        errs.append(tr("e_col", lang))
    if r.get("compare") not in CMP_KEY:
        errs.append(tr("e_cmp", lang))
    if r.get("operator") not in OP_KEY:
        errs.append(tr("e_op", lang))
    try:
        if not math.isfinite(float(r.get("value"))):
            errs.append(tr("e_val", lang))
    except (TypeError, ValueError):
        errs.append(tr("e_val", lang))
    conds = r.get("conditions") or []
    if not conds or any(c not in CONDITIONS for c in conds):
        errs.append(tr("e_conds", lang))
    if r.get("compare") in ("delta", "delta_pct") and conds == ["Baseline"]:
        errs.append(tr("e_base", lang))
    if r.get("severity") not in SEV_KEY:
        errs.append(tr("e_sev", lang))
    msg = str(r.get("message", ""))
    if not r.get("builtin") and not msg.strip():
        errs.append(tr("e_msg", lang))
    for f in _field_re.findall(msg):
        if not _field_ok.match(f):
            errs.append(tr("e_field", lang, f=f, ok=", ".join(MSG_FIELDS)))
    return errs


def new_rule(**kw) -> dict:
    r = {"id": uuid.uuid4().hex[:8], "author": "doctor"}
    r.update(kw)
    r["value"] = float(r["value"])
    return r


def rules_to_json(rules) -> str:
    return json.dumps(rules, ensure_ascii=False, indent=2)


def rules_from_json(text: str, lang="kk"):
    """-> (дұрыс ережелер, қателер)."""
    try:
        data = json.loads(text)
    except Exception as e:
        return [], [tr("e_json", lang, e=e)]
    if not isinstance(data, list):
        return [], [tr("e_list", lang)]
    good, errs = [], []
    for i, r in enumerate(data, 1):
        if not isinstance(r, dict):
            errs.append(tr("e_obj", lang, i=i))
            continue
        e = validate_rule(r, lang)
        if e:
            errs.append(tr("e_rule", lang, i=i, n=r.get("name", ""), e=" ".join(e)))
        else:
            r = dict(r)
            r.setdefault("id", uuid.uuid4().hex[:8])
            r.setdefault("author", "doctor")
            r["value"] = float(r["value"])
            good.append(r)
    return good, errs


def _cmp(op, x, thr):
    if op == ">":
        return x > thr
    if op == ">=":
        return x >= thr
    if op == "<":
        return x < thr
    if op == "<=":
        return x <= thr
    if op == "abs>":
        return abs(x) > thr
    return False


def _context(d, cond, p):
    b = d.loc[d["condition"] == "Baseline", p].dropna().to_numpy(float)
    c = d.loc[d["condition"] == cond, p].dropna().to_numpy(float)
    ctx = dict(n=len(c), n_base=len(b), baseline=np.nan, value=np.nan, delta=np.nan, delta_pct=np.nan)
    if len(c):
        ctx["value"] = c.mean()
    if len(b):
        ctx["baseline"] = b.mean()
    if len(b) and len(c):
        ctx["delta"] = c.mean() - b.mean()
        if PARAMS[p]["positive"] and b.mean() > 1e-6:
            ctx["delta_pct"] = ctx["delta"] / b.mean() * 100
    return ctx


def evaluate_rules(d, rules, lang="kk"):
    """d — бір қатысушының (барлық жағдай) кестесі. Әр ереже × жағдай үшін нәтиже тізімі."""
    out = []
    for r in rules:
        p = r["column"]
        if p not in d.columns:
            continue
        for cond in r["conditions"]:
            if r["compare"] in ("delta", "delta_pct") and cond == "Baseline":
                continue
            ctx = _context(d, cond, p)
            if r["compare"] == "n_cycles":
                x = float(len(d[d["condition"] == cond]))
                ctx["n"] = int(x)
            else:
                x = ctx[{"delta": "delta", "delta_pct": "delta_pct", "value": "value"}[r["compare"]]]
            res = dict(rule=r, condition=cond, observed=x, fired=False, text="", status="ok")
            if x is None or (isinstance(x, float) and np.isnan(x)):
                res["status"] = "insufficient"
                out.append(res)
                continue
            fired = _cmp(r["operator"], x, float(r["value"]))
            res["fired"] = bool(fired)
            if fired:
                fmt = dict(ctx, param=plabel(p, lang), condition=cond,
                           threshold=float(r["value"]), operator=r["operator"])
                msg = rule_message(r, lang)
                try:
                    res["text"] = msg.format(**fmt)
                except Exception:
                    res["text"] = msg + tr("r_bad_text", lang)
            out.append(res)
    return out


def rules_summary_md(results, lang="kk"):
    fired = [x for x in results if x["fired"]]
    if not fired:
        return tr("r_none_summary", lang)
    lines = []
    for x in fired:
        icon = tr(SEV_KEY[x["rule"]["severity"]], lang).split(" ")[0]
        lines.append(f'- {icon} **{rule_name(x["rule"], lang)}** ({tr("r_by", lang, a=author_label(x["rule"]["author"], lang))}): {x["text"]}')
    return "\n".join(lines)
