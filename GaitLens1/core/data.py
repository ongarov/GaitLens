"""Деректі оқу және тазалау (1-қадам).

CSV: бөлгіш ';', ондық бөлгіш ',', бос мәндер NaN.
"""
import io
import numpy as np
import pandas as pd

from .i18n import tr

CONDITIONS = ["Baseline", "Verbal", "VR"]
REQUIRED_COLS = ["subject", "side"]

# Көрсеткіш сипаттамалары. direction: "lower" = төмен мән қалыпты жүріске жақын,
# None = бағыты көрсетілмеген (клиникалық мағынаны дәрігер белгілейді).
# Бағандардың мағынасы баған атауларына қарай; ұйымдастырушылардың сипаттамасымен растаңыз.
PARAMS = {
    # direction: "lower" = төмен мән қалыпты жүріске жақын; None = бағыты көрсетілмеген (дәрігер белгілейді).
    # Атаулар мен түсіндірмелер core/i18n.py ішінде.
    "GPS_a": dict(unit="°", direction="lower", positive=True),
    "GVS_aSagP": dict(unit="°", direction="lower", positive=True),
    "GVS_aSagH": dict(unit="°", direction="lower", positive=True),
    "GVS_aSagK": dict(unit="°", direction="lower", positive=True),
    "GVS_aSagA": dict(unit="°", direction="lower", positive=True),
    "aMinSagH": dict(unit="°", direction=None, positive=False),
    "pctmZeroStSagH": dict(unit="%", direction=None, positive=True),
}
PARAM_COLS = list(PARAMS.keys())

MIN_CYCLES = 10  # осыдан аз цикл болса «n аз» ескертуі


def read_gait_csv(src, lang="kk") -> pd.DataFrame:
    """src: файл жолы немесе Streamlit UploadedFile."""
    if hasattr(src, "getvalue"):
        raw = src.getvalue()
    else:
        with open(src, "rb") as f:
            raw = f.read()
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") >= first.count(",") else ","
    decimal = "," if sep == ";" else "."
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal,
                     na_values=["NaN", "nan", "NA", ""])
    df = df.copy()  # үлкен кестені дефрагменттеу
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(tr("w_cols", lang, c=", ".join(missing)))
    df["subject"] = df["subject"].astype(str).str.strip()
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    for c in PARAM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def guess_condition(filename: str):
    n = filename.lower()
    if "baseline" in n:
        return "Baseline"
    if "verbal" in n:
        return "Verbal"
    if "vr" in n:
        return "VR"
    return None


def build_dataset(files, lang="kk"):
    """files: [(аты, src), ...] -> (ұзын кесте, ескертулер тізімі)."""
    frames, warnings = [], []
    for name, src in files:
        if "kinematics" in name.lower():
            continue  # қисық файлдарын core/curves.py оқиды
        cond = guess_condition(name)
        if cond is None:
            warnings.append(tr("w_cond", lang, f=name))
            continue
        try:
            df = read_gait_csv(src, lang)
        except Exception as e:
            warnings.append(tr("w_read", lang, f=name, e=e))
            continue
        df = df.assign(condition=cond)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), warnings
    data = pd.concat(frames, ignore_index=True)
    data["cycle"] = data.groupby(["subject", "condition"]).cumcount() + 1
    present = set(data["condition"].unique())
    for c in CONDITIONS:
        if c not in present:
            warnings.append(tr("w_missing_file", lang, c=c))
    return data, warnings


def subject_labels(data: pd.DataFrame, show_real_id: bool = False) -> dict:
    """{subject_id: 'P01 · RIGHT'} — әдепкіде ID анонимді."""
    ids = sorted(data["subject"].unique())
    side = data.groupby("subject")["side"].first()
    labels = {}
    for i, s in enumerate(ids, 1):
        base = s if show_real_id else f"P{i:02d}"
        labels[s] = f"{base} · {side[s]}"
    return labels


def quality_table(data: pd.DataFrame, lang="kk") -> pd.DataFrame:
    """Әр жағдай бойынша циклдер саны мен бос мәндер үлесі."""
    rows = []
    for cond in CONDITIONS:
        d = data[data["condition"] == cond]
        if d.empty:
            continue
        per_subj = d.groupby("subject").size()
        cols = [c for c in PARAM_COLS if c in d.columns]
        rows.append({
            tr("q_cond", lang): cond,
            tr("q_subjects", lang): d["subject"].nunique(),
            tr("q_cycles", lang): len(d),
            tr("q_range", lang): f"{per_subj.min()}–{per_subj.max()}",
            tr("q_nan", lang): round(d[cols].isna().mean().mean() * 100, 1),
        })
    return pd.DataFrame(rows)
