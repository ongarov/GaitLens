"""Когорта шолуы (5-қадам): барлық қатысушылар бойынша GPS өзгерісі мен эффект өлшемдері."""
import numpy as np
import pandas as pd

from .data import CONDITIONS, MIN_CYCLES, PARAM_COLS
from .stats import change_table


def cohort_gps(data: pd.DataFrame, param: str = "GPS_a", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Әр қатысушы үшін Baseline-ға қатысты GPS өзгерісі (бутстреп 95% ИН).
    verdict_*: 'down' | 'up' | 'none' | 'na'."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, d in data.groupby("subject"):
        b = d.loc[d["condition"] == "Baseline", param].dropna().to_numpy(float)
        row = dict(subject=s, side=d["side"].iloc[0], base=b.mean() if len(b) else np.nan)
        for cond in ["Verbal", "VR"]:
            c = d.loc[d["condition"] == cond, param].dropna().to_numpy(float)
            k = cond.lower()
            row[f"delta_{k}"] = np.nan
            row[f"lo_{k}"] = row[f"hi_{k}"] = np.nan
            row[f"verdict_{k}"] = "na"
            if len(b) >= MIN_CYCLES and len(c) >= MIN_CYCLES:
                bs = rng.choice(c, (n_boot, len(c))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                row[f"delta_{k}"] = c.mean() - b.mean()
                row[f"lo_{k}"], row[f"hi_{k}"] = lo, hi
                row[f"verdict_{k}"] = "down" if hi < 0 else ("up" if lo > 0 else "none")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def cohort_effects(data: pd.DataFrame) -> pd.DataFrame:
    """Қатысушы × көрсеткіш × жағдай бойынша эффект өлшемі d."""
    out = []
    for s, d in data.groupby("subject"):
        ch = change_table(d, PARAM_COLS)
        ch = ch.assign(subject=s)[["subject", "param", "condition", "effect_d", "delta"]]
        out.append(ch)
    return pd.concat(out, ignore_index=True)
