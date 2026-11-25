"""Салыстыру статистикасы (1-қадам): n, орташа, SD, Baseline-ға қатысты өзгеріс."""
import numpy as np
import pandas as pd
from .data import PARAMS, CONDITIONS, MIN_CYCLES


def _vals(d, cond, p):
    return d.loc[d["condition"] == cond, p].dropna().to_numpy(dtype=float)


def summary_table(d: pd.DataFrame, params) -> pd.DataFrame:
    rows = []
    for p in params:
        for cond in CONDITIONS:
            v = _vals(d, cond, p)
            rows.append({
                "param": p, "condition": cond, "n": len(v),
                "mean": v.mean() if len(v) else np.nan,
                "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                "median": np.median(v) if len(v) else np.nan,
            })
    return pd.DataFrame(rows)


def change_table(d: pd.DataFrame, params) -> pd.DataFrame:
    """Verbal және VR-дың Baseline-мен айырмасы (бір қатысушы ішінде)."""
    rows = []
    for p in params:
        info = PARAMS[p]
        b = _vals(d, "Baseline", p)
        for cond in ["Verbal", "VR"]:
            c = _vals(d, cond, p)
            row = {"param": p, "condition": cond, "n_base": len(b), "n_cond": len(c),
                   "mean_base": np.nan, "mean_cond": np.nan, "delta": np.nan,
                   "delta_pct": np.nan, "effect_d": np.nan, "note": ""}
            if len(b) < 2 or len(c) < 2:
                row["note"] = "few"
                rows.append(row)
                continue
            mb, mc = b.mean(), c.mean()
            row.update(mean_base=mb, mean_cond=mc, delta=mc - mb)
            # % тек әрқашан оң көрсеткіштерге және Baseline нөлге жақын емес болса
            if info["positive"] and mb > 1e-6:
                row["delta_pct"] = (mc - mb) / mb * 100
            sp = np.sqrt(((len(b) - 1) * b.var(ddof=1) + (len(c) - 1) * c.var(ddof=1))
                         / (len(b) + len(c) - 2))
            if sp > 1e-9:
                row["effect_d"] = (mc - mb) / sp
            if min(len(b), len(c)) < MIN_CYCLES:
                row["note"] = "lown"
            rows.append(row)
    return pd.DataFrame(rows)


def arrow(delta, tol=1e-9):
    if delta is None or np.isnan(delta):
        return "—"
    if abs(delta) < tol:
        return "→"
    return "↑" if delta > 0 else "↓"


def bootstrap_change(d: pd.DataFrame, params, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Орташа айырманың (шарт − Baseline) бутстреп сенімділік интервалы:
    циклдер әр жағдай ішінде қайта таңдалады. Интервал нөлді қамтымаса, өзгеріс «анықталды»."""
    rng = np.random.default_rng(seed)
    base = change_table(d, params)
    lo, hi, verdict = [], [], []
    for _, r in base.iterrows():
        b = _vals(d, "Baseline", r["param"])
        c = _vals(d, r["condition"], r["param"])
        if len(b) < MIN_CYCLES or len(c) < MIN_CYCLES:
            lo.append(np.nan); hi.append(np.nan); verdict.append("few")
            continue
        bs = (rng.choice(c, (n_boot, len(c))).mean(axis=1) - rng.choice(b, (n_boot, len(b))).mean(axis=1))
        l, h = np.percentile(bs, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        lo.append(l); hi.append(h)
        verdict.append("detected" if (l > 0 or h < 0) else "none")
    base["ci_low"], base["ci_high"], base["verdict"] = lo, hi, verdict
    return base
