"""Қадам циклінің қисықтары (2-қадам)."""
import re
import numpy as np
import pandas as pd
from .data import read_gait_csv, guess_condition, CONDITIONS
from .i18n import tr

CURVE_KEYS = ["aSagP", "aSagH", "aSagK", "aSagA", "mSagH"]  # атаулары core/i18n.py ішінде
_pat = re.compile(r"^(\w+?)_pct_GC_(\d+)$")


def curve_columns(df, var):
    cols = []
    for c in df.columns:
        m = _pat.match(c)
        if m and m.group(1) == var:
            cols.append((int(m.group(2)), c))
    cols.sort()
    return [c for _, c in cols], np.array([x for x, _ in cols]) / 10.0  # 0..100 % GC


def build_curves(files, lang="kk"):
    """files: [(аты, src)] -> (кесте, ескертулер). Тек Kinematics_Kinetics файлдары."""
    frames, warns = [], []
    for name, src in files:
        if "kinematics" not in name.lower():
            continue
        cond = guess_condition(name)
        if cond is None:
            warns.append(tr("w_cond", lang, f=name))
            continue
        try:
            df = read_gait_csv(src, lang)
        except Exception as e:
            warns.append(tr("w_read", lang, f=name, e=e))
            continue
        df = df.assign(condition=cond)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), warns
    out = pd.concat(frames, ignore_index=True).copy()
    out = out.assign(cycle=out.groupby(["subject", "condition"]).cumcount() + 1)
    return out, warns


def mean_sd_curves(d, var):
    cols, x = curve_columns(d, var)
    res = {}
    for cond in CONDITIONS:
        m = d.loc[d["condition"] == cond, cols].to_numpy(dtype=float)
        if len(m) == 0:
            continue
        res[cond] = dict(mean=np.nanmean(m, axis=0), sd=np.nanstd(m, axis=0, ddof=1) if len(m) > 1 else np.zeros(len(x)),
                         n=len(m))
    return x, res


def unstable_cycles(d, var, k=3.0):
    """Әр жағдайда циклдің жағдайдың медиана қисығынан RMS қашықтығы.
    Робасты z = (қашықтық - медиана)/(1.4826*MAD) > k болса, цикл «ерекшеленеді».
    Бұл статистикалық белгі, клиникалық шек емес."""
    cols, _ = curve_columns(d, var)
    rows = []
    for cond in CONDITIONS:
        s = d[d["condition"] == cond]
        if len(s) < 5:
            continue
        m = s[cols].to_numpy(dtype=float)
        med = np.nanmedian(m, axis=0)
        dist = np.sqrt(np.nanmean((m - med) ** 2, axis=1))
        mad = np.median(np.abs(dist - np.median(dist)))
        scale = 1.4826 * mad if mad > 1e-9 else np.nan
        z = (dist - np.median(dist)) / scale if scale == scale else np.zeros_like(dist)
        for cyc, di, zi in zip(s["cycle"], dist, z):
            rows.append(dict(condition=cond, cycle=int(cyc), distance=float(di),
                             z=float(zi), flagged=bool(zi > k)))
    return pd.DataFrame(rows)
