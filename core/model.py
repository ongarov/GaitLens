"""ML-компонент: жағдайды (Baseline/Verbal/VR) цикл деңгейінде жіктеу (4-қадам).

Қатаң Leave-One-Subject-Out: бір қатысушының барлық циклдері не train-де, не test-те.
Бос мәндер импутациясы да тек train бөлігінде үйретіледі (Pipeline), сондықтан ағып кету жоқ.
Бұл диагностика емес: «шартты деректен қаншалықты ажыратуға болады» деген өлшем.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline

from .data import PARAM_COLS

FEATURES = PARAM_COLS  # тек жүріс көрсеткіштері (бой/салмақ сияқты қатысушыны танытатын бағандарсыз)
ALL = ("Baseline", "Verbal", "VR")


def loso_eval(data: pd.DataFrame, conds=ALL, n_estimators: int = 150, seed: int = 0) -> dict:
    d = data[data["condition"].isin(conds)].reset_index(drop=True)
    X = d[FEATURES].to_numpy(float)
    y = d["condition"].to_numpy()
    g = d["subject"].to_numpy()
    pred = np.empty(len(d), dtype=object)
    imps = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        assert set(g[tr]).isdisjoint(set(g[te])), "Subject ағып кетуі!"
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        imps.append(m[-1].feature_importances_)
    labels = list(conds)
    shares = pd.Series(y).value_counts(normalize=True)
    per_subj = pd.Series(pred == y).groupby(g).mean()
    return dict(
        labels=labels, n_cycles=len(d), n_subjects=int(pd.Series(g).nunique()),
        accuracy=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", labels=labels),
        chance_bal=1.0 / len(labels), majority_acc=float(shares.max()),
        cm=confusion_matrix(y, pred, labels=labels),
        importance=pd.Series(np.mean(imps, axis=0), index=FEATURES).sort_values(ascending=False),
        per_subject=per_subj,
    )


def pairwise_eval(data: pd.DataFrame, n_estimators: int = 150) -> pd.DataFrame:
    rows = []
    for a, b in [("Baseline", "Verbal"), ("Baseline", "VR"), ("Verbal", "VR")]:
        r = loso_eval(data, (a, b), n_estimators)
        rows.append({"pair": f"{a} vs {b}", "bal_acc": r["bal_acc"], "macro_f1": r["macro_f1"],
                     "chance": r["chance_bal"]})
    return pd.DataFrame(rows)


def interpret(r: dict, lang: str = "kk", margin: float = 0.05) -> str:
    from .i18n import tr
    gap = r["bal_acc"] - r["chance_bal"]
    if gap <= margin:
        return tr("mo_i_chance", lang)
    if r["bal_acc"] < 0.7:
        return tr("mo_i_weak", lang)
    return tr("mo_i_good", lang)
