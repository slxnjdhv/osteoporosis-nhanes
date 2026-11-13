"""Step 3: full modelling and evaluation pipeline (revision 3).

  A. Descriptive tables: by outcome (unweighted + survey-weighted), by data set,
     included vs excluded participants, cohort summary
  B. Feature selection (Boruta, Lasso, XGBoost) - only for the re-implementation
  C. Re-implementation of the Zhang et al. approach: 3 selectors x 4 models
  D. Model comparison on all predictors: 8 models (incl. spline LR) x 4 imbalance strategies,
     raw and cross-fitted recalibrated Brier scores
  E. Optuna tuning of the best configuration
  F. Temporal validation vs benchmarks (OST, 3-variable LR, re-implemented Boruta-XGBoost),
     bootstrap CIs incl. calibration slope/intercept, DeLong + Holm, survey-weighted metrics,
     OST at published cut-offs vs the model at matched referral rates, decision curves
  G. Incremental value of predictor blocks (nested models)
  H. Fairness: subgroup metrics with CIs and event counts, sex-specific thresholds, race-free model
  I. SHAP (importance on development data) and race-free short-form model
  J. Final model coefficients (original scale, after recalibration)
  K. Sensitivity analyses: spine-based outcomes, development set without 2005-2006

All tables -> outputs/tables, all figures -> outputs/figures.
"""
from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")   # also silences parallel worker processes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import statsmodels.api as sm
from boruta import BorutaPy
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from config import CYCLES as CYCLES_YEARS
from config import (BINARY_FEATURES, CV_FOLDS, CV_FOLDS_BASELINE, DATASET_CSV, FEATURES,
                    FIG_DIR, INCREMENTAL_BLOCKS, LAB_FEATURES, N_BOOT, N_JOBS, NUMERIC_FEATURES,
                    OPTUNA_TRIALS, PRIMARY_OUTCOME, PROC_DIR, PRETTY, RACE_FEATURES, SEED,
                    SENSITIVITY_OUTCOMES, SHORT_FORM_K, TABLE_DIR, TEST_CYCLES, TRAIN_CYCLES)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RNG = np.random.default_rng(SEED)

BASE_MODELS = ["LR", "SVM", "RF", "XGBoost"]                 # used in the published study
ALL_MODELS = ["LR", "LR-spline", "SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
STRATEGIES = ["none", "class_weight", "smote"]   # SMOTE-Tomek removed no samples (see log)
TREE_MODELS = {"RF", "XGBoost", "LightGBM", "CatBoost"}
OST_NAME = "OST: 0.2 x (weight - age)"
ZHANG_NAME = "Zhang et al. approach (Boruta + XGBoost, re-implemented)"
LR3_NAME = "LR: age, weight, sex"
ORAI_NAME = "ORAI"
SCORE_NAME = "SCORE"
EQ_MARGIN = 0.02     # equivalence margin for ML vs LR (AUROC)
DISPLAY = {"Proposed": "Full model (LR)", ZHANG_NAME: "Boruta + XGBoost pipeline",
           LR3_NAME: "Age, weight, sex (LR)", OST_NAME: "OST", ORAI_NAME: "ORAI", SCORE_NAME: "SCORE"}


# =========================================================================== models
def make_estimator(name: str, balanced: bool, pos_weight: float, params: dict | None = None):
    p = params or {}
    cw = "balanced" if balanced else None
    if name in ("LR", "LR-spline"):
        return LogisticRegression(max_iter=5000, class_weight=cw, C=p.get("C", 1.0))
    if name == "SVM":
        return SVC(probability=True, kernel="rbf", class_weight=cw, random_state=SEED,
                   C=p.get("C", 1.0), gamma=p.get("gamma", "scale"))
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 500), max_depth=p.get("max_depth"),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            max_features=p.get("max_features", "sqrt"),
            class_weight=cw, n_jobs=N_JOBS, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 4), subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            min_child_weight=p.get("min_child_weight", 1), reg_lambda=p.get("reg_lambda", 1.0),
            scale_pos_weight=pos_weight if balanced else 1.0, eval_metric="logloss",
            n_jobs=N_JOBS, random_state=SEED, verbosity=0)
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 15), min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.8), subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8), reg_lambda=p.get("reg_lambda", 0.0),
            class_weight=cw, n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=p.get("iterations", 500), depth=p.get("depth", 4),
            learning_rate=p.get("learning_rate", 0.05), l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            auto_class_weights="Balanced" if balanced else None,
            random_seed=SEED, verbose=0, thread_count=N_JOBS)
    if name == "Stacking":
        base = [(n.lower(), make_estimator(n, balanced, pos_weight, p.get(n)))
                for n in ["RF", "XGBoost", "LightGBM", "CatBoost"]]
        return StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=2000),
                                  cv=5, stack_method="predict_proba", n_jobs=1)
    raise ValueError(name)


def make_pipeline(name: str, strategy: str, pos_weight: float, params: dict | None = None,
                  cols: list | None = None):
    steps = [("impute", IterativeImputer(max_iter=10, random_state=SEED)),
             ("scale", StandardScaler())]
    if name == "LR-spline":
        num_idx = [i for i, c in enumerate(cols) if c in NUMERIC_FEATURES]
        steps.append(("spline", ColumnTransformer(
            [("spl", SplineTransformer(n_knots=4, degree=3, knots="quantile", extrapolation="linear"), num_idx)], remainder="passthrough")))
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif strategy == "smote_tomek":
        steps.append(("sampler", SMOTETomek(random_state=SEED)))
    steps.append(("model", make_estimator(name, strategy == "class_weight", pos_weight, params)))
    return Pipeline(steps)


# =========================================================================== metrics
def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_recalibrator(y, p_oof):
    """Platt (logistic) recalibration on out-of-fold development predictions.
    Returns a function; .a and .b hold intercept and slope on the logit scale."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_oof).reshape(-1, 1), y)

    def apply(p):
        return lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    apply.a, apply.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    return apply


def crossfit_recal(y, p_raw):
    """Honest internal recalibration: each fold recalibrated using only the other folds."""
    out = np.empty_like(p_raw, dtype=float)
    for a, b in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + 1).split(p_raw, y):
        out[b] = fit_recalibrator(y[a], p_raw[a])(p_raw[b])
    return out


def calibration_slope_intercept(y, p):
    lp = logit(p)
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        return float(slope), float(intercept)
    except Exception:
        return np.nan, np.nan


def point_metrics(y, p, thr) -> dict:
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Sensitivity": sens, "Specificity": spec,
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "F1": f1_score(y, pred, zero_division=0), "Balanced_acc": np.nanmean([sens, spec]),
            "Accuracy": accuracy_score(y, pred), "Referred_pct": pred.mean() * 100,
            "Events": int(y.sum()), "Detected": int(tp)}


def bootstrap_metrics(y, p, thr, n_boot=N_BOOT, calib=True) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    est = point_metrics(y, p, thr)
    if calib:
        est["Calibration_slope"], est["Calibration_intercept"] = calibration_slope_intercept(y, p)
    boots = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        b = point_metrics(y[idx], p[idx], thr)
        if calib:
            b["Calibration_slope"], b["Calibration_intercept"] = calibration_slope_intercept(y[idx], p[idx])
        boots.append(b)
    boots = pd.DataFrame(boots)
    rows = [{"metric": k, "estimate": v, "ci_low": boots[k].quantile(0.025),
             "ci_high": boots[k].quantile(0.975)} for k, v in est.items()]
    rows.append({"metric": "Threshold", "estimate": thr})
    return pd.DataFrame(rows)


def fmt_ci(df: pd.DataFrame, metric: str) -> str:
    r = df.set_index("metric").loc[metric]
    return f"{r.estimate:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})"


def delong_test(y, p1, p2):
    """DeLong test for two correlated AUROCs (Sun & Xu 2014 fast algorithm)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")       # positives first
    preds = np.vstack([p1, p2])[:, order]
    m = int(y.sum()); n = len(y) - m

    def midrank(x):
        j = np.argsort(x); z = x[j]; t = np.zeros(len(x)); i = 0
        while i < len(x):
            k = i
            while k < len(x) and z[k] == z[i]:
                k += 1
            t[i:k] = 0.5 * (i + k - 1) + 1
            i = k
        out = np.empty(len(x)); out[j] = t
        return out

    tx = np.array([midrank(r[:m]) for r in preds])
    ty = np.array([midrank(r[m:]) for r in preds])
    tz = np.array([midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def net_benefit(y, p, thresholds):
    y = np.asarray(y); n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def boot_auc_diff(y, p1, p2, n_boot=N_BOOT, return_p=False):
    """Paired bootstrap 95% CI (and two-sided percentile P) for AUROC(p1) - AUROC(p2)."""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2); d = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    out = (roc_auc_score(y, p1) - roc_auc_score(y, p2), np.quantile(d, 0.025), np.quantile(d, 0.975))
    if return_p:
        pv = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))
        return out + (min(pv, 1.0),)
    return out


def boot_auc_diff_ci90(y, p1, p2, n_boot=N_BOOT):
    """Paired bootstrap 90% CI for AUROC(p1) - AUROC(p2) (equivalence testing, TOST)."""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2); d = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95)), float(np.std(d, ddof=1))


def random_effects(est, se):
    """DerSimonian-Laird random-effects pooling; returns pooled, 95% CI, tau2, I2, 95% PI."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2; k = len(est)
    fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - fe) ** 2)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    wr = 1 / (se ** 2 + tau2)
    pooled = np.sum(wr * est) / np.sum(wr); sep = np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    tcrit = stats.t.ppf(0.975, k - 2) if k > 2 else np.nan
    pi = np.sqrt(tau2 + sep ** 2) * tcrit
    return {"pooled": pooled, "low": pooled - 1.96 * sep, "high": pooled + 1.96 * sep,
            "tau2": tau2, "I2": i2, "pi_low": pooled - pi, "pi_high": pooled + pi}


def design_boot(stat, w, strata, psu, n_boot=500):
    """Rao-Wu rescaled bootstrap over PSUs within strata; stat(weights) -> float."""
    w, strata, psu = np.asarray(w, float), np.asarray(strata), np.asarray(psu)
    groups = {h: np.unique(psu[strata == h]) for h in np.unique(strata)}
    out = []
    for _ in range(n_boot):
        mult = np.zeros(len(w))
        for h, ps in groups.items():
            nh = len(ps)
            if nh < 2:
                mult[strata == h] = 1.0
                continue
            for c in RNG.choice(ps, nh - 1, replace=True):
                mult[(strata == h) & (psu == c)] += nh / (nh - 1)
        try:
            out.append(stat(w * mult))
        except ValueError:
            continue
    return np.quantile(out, 0.025), np.quantile(out, 0.975)


def design_boot_weighted_auc(y, p, w, strata, psu, n_boot=500):
    """Rao-Wu rescaled bootstrap (one PSU drawn per stratum, weight x2 for 2-PSU strata)."""
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    strata, psu = np.asarray(strata), np.asarray(psu)
    groups = {h: np.unique(psu[strata == h]) for h in np.unique(strata)}
    out = []
    for _ in range(n_boot):
        mult = np.zeros(len(y))
        for h, ps in groups.items():
            nh = len(ps)
            if nh < 2:
                mult[strata == h] = 1.0
                continue
            draw = RNG.choice(ps, nh - 1, replace=True)
            for c in draw:
                mult[(strata == h) & (psu == c)] += nh / (nh - 1)
        ww = w * mult
        if ww[y == 1].sum() == 0:
            continue
        out.append(roc_auc_score(y, p, sample_weight=ww))
    return np.quantile(out, 0.025), np.quantile(out, 0.975)


def net_benefit_ci(y, p, thresholds, n_boot=N_BOOT):
    y = np.asarray(y); n = len(y); rows = []
    for t in thresholds:
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        boots = []
        for _ in range(n_boot):
            i = RNG.integers(0, n, n)
            boots.append(net_benefit(y[i], p[i], [t])[0])
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * t / (1 - t),
                     "nb_low": np.quantile(boots, 0.025), "nb_high": np.quantile(boots, 0.975),
                     "tp_per_100": tp / n * 100, "fp_per_100": fp / n * 100})
    return pd.DataFrame(rows)


def weighted_metrics(y, p, w) -> dict:
    """Survey-weighted point estimates (MEC weights; design-based SEs not computed)."""
    w = np.asarray(w, dtype=float)
    ok = ~np.isnan(w)
    y, p, w = np.asarray(y)[ok], np.asarray(p)[ok], w[ok]
    return {"AUROC_weighted": roc_auc_score(y, p, sample_weight=w),
            "Brier_weighted": np.average((p - y) ** 2, weights=w),
            "Observed_weighted_pct": np.average(y, weights=w) * 100,
            "Predicted_weighted_pct": np.average(p, weights=w) * 100}


# =========================================================================== A. descriptive tables
def wmean(g, v):
    m = g[v].notna() & g["wt_pooled"].notna()
    return np.average(g.loc[m, v], weights=g.loc[m, "wt_pooled"]) if m.any() else np.nan


def describe(groups: dict, df: pd.DataFrame, weighted=True, test=True) -> pd.DataFrame:
    """Mean (SD) / n (%) per group; optional survey-weighted columns and P values (2 groups)."""
    rows = [{"variable": "N", **{k: len(g) for k, g in groups.items()}}]
    gs = list(groups.values())
    for v in NUMERIC_FEATURES + BINARY_FEATURES:
        r = {"variable": v + (", mean (SD)" if v in NUMERIC_FEATURES else ", n (%)")}
        for k, g in groups.items():
            if v in NUMERIC_FEATURES:
                r[k] = f"{g[v].mean():.2f} ({g[v].std():.2f})"
                if weighted:
                    r[k + " weighted"] = f"{wmean(g, v):.2f}"
            else:
                r[k] = f"{int(g[v].sum())} ({g[v].mean() * 100:.1f})"
                if weighted:
                    r[k + " weighted"] = f"{wmean(g, v) * 100:.1f}%"
        if test and len(gs) == 2:
            if v in NUMERIC_FEATURES:
                a, b = gs[0][v].dropna(), gs[1][v].dropna()
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
            else:
                tab = pd.crosstab(pd.concat([gs[0][v], gs[1][v]]),
                                  np.r_[np.zeros(len(gs[0])), np.ones(len(gs[1]))])
                p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
            r["p"] = f"{p:.3g}"
        r["missing_%"] = f"{df[v].isna().mean() * 100:.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


# =========================================================================== B. feature selection
def select_features(X: pd.DataFrame, y: np.ndarray) -> dict:
    imp = IterativeImputer(max_iter=10, random_state=SEED).fit(X)
    Xi = pd.DataFrame(imp.transform(X), columns=X.columns)
    Xs = StandardScaler().fit_transform(Xi)

    rf = RandomForestClassifier(n_jobs=N_JOBS, class_weight="balanced", max_depth=5,
                                random_state=SEED)
    bor = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=SEED, verbose=0)
    bor.fit(Xi.values, y)
    boruta = list(X.columns[bor.support_])
    if len(boruta) < 5:
        boruta = list(X.columns[bor.support_ | bor.support_weak_])

    lasso = LogisticRegressionCV(penalty="l1", solver="saga", Cs=20, cv=5, scoring="roc_auc",
                                 max_iter=5000, random_state=SEED, n_jobs=N_JOBS).fit(Xs, y)
    lasso_feats = list(X.columns[np.abs(lasso.coef_[0]) > 1e-6])

    k = max(len(boruta), 10)
    xgb = make_estimator("XGBoost", False, 1.0).fit(Xi, y)
    gain = pd.Series(xgb.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0)
    xgb_feats = list(gain.sort_values(ascending=False).index[:k])
    return {"boruta": boruta, "lasso": lasso_feats, "xgboost": xgb_feats}


# =========================================================================== main
def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    train = df[df.cycle.isin(TRAIN_CYCLES)].reset_index(drop=True)
    test = df[df.cycle.isin(TEST_CYCLES)].reset_index(drop=True)
    y_tr, y_te = train[PRIMARY_OUTCOME].values, test[PRIMARY_OUTCOME].values
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"Train n={len(train)} events={y_tr.sum()} (prev {y_tr.mean():.3f}) | Temporal test "
          f"n={len(test)} events={y_te.sum()} (prev {y_te.mean():.3f})")

    # ---- A. descriptive tables and cohort summary
    describe({"No osteoporosis": df[df[PRIMARY_OUTCOME] == 0], "Osteoporosis": df[df[PRIMARY_OUTCOME] == 1]},
             df).to_csv(TABLE_DIR / "table1_baseline_characteristics.csv", index=False)
    describe({"Development 2005-2014": train, "Validation 2017-2018": test}, df,
             weighted=False).to_csv(TABLE_DIR / "tableS_development_vs_validation.csv", index=False)
    exc = pd.read_csv(PROC_DIR / "excluded_no_dxa.csv")
    cmp_rows = []
    for v in ["age", "female", "bmi", "weight", "prior_fracture", "steroid_use", "race_black"]:
        a, b = df[v].dropna(), exc[v].dropna()
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        cmp_rows.append({"variable": v, "included_mean": a.mean(), "excluded_mean": b.mean(),
                         "excluded_missing_%": exc[v].isna().mean() * 100, "p": p})
    pd.DataFrame(cmp_rows).to_csv(TABLE_DIR / "tableS_included_vs_excluded.csv", index=False)
    summary = {
        "n": len(df), "events": int(df[PRIMARY_OUTCOME].sum()), "prev_pct": df[PRIMARY_OUTCOME].mean() * 100,
        "prev_weighted_pct": np.average(df[PRIMARY_OUTCOME], weights=df.wt_pooled) * 100,
        "prev_by_cycle_pct": (df.groupby("cycle_years")[PRIMARY_OUTCOME].mean() * 100).round(2).to_dict(),
        "train_n": len(train), "train_events": int(y_tr.sum()), "test_n": len(test),
        "test_events": int(y_te.sum()), "events_per_predictor": y_tr.sum() / len(FEATURES),
        "mean_age": df.age.mean(), "female_pct": df.female.mean() * 100, "n_excluded_no_dxa": len(exc),
        "n_excluded_diagnosed": int(pd.read_csv(PROC_DIR / "nhanes_osteoporosis_incl_diagnosed.csv").known_osteo.sum()),
        "fn_only_prev_pct": (df.fn_t <= -2.5).mean() * 100,
        "sensitivity_outcome_prev_pct": {k: df[k].mean() * 100 for k in SENSITIVITY_OUTCOMES},
        "sensitivity_outcome_prev_by_cycle_pct": {k: (df.groupby("cycle_years")[k].mean() * 100).round(2).to_dict()
                                                  for k in SENSITIVITY_OUTCOMES},
        "spine_available_pct": df.ls_bmd.notna().mean() * 100,
        "partial_spine_pct_by_cycle": (df.groupby("cycle_years").ls_partial.mean() * 100).round(1).to_dict(),
    }
    json.dump(summary, open(TABLE_DIR / "cohort_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float))

    # ---- B. feature selection (for the re-implementation only)
    print("\n[B] Feature selection ...")
    sel = select_features(train[FEATURES], y_tr)
    json.dump(sel, open(TABLE_DIR / "selected_features.json", "w"), indent=2)
    for k, v in sel.items():
        print(f"  {k:8s} ({len(v)}): {v}")

    # ---- C. re-implementation of the published approach
    print("\n[C] Re-implementation of Zhang et al. approach (10-fold CV) ...")
    cv10 = StratifiedKFold(CV_FOLDS_BASELINE, shuffle=True, random_state=SEED)
    rows = []
    for s_name, feats_ in sel.items():
        for m in BASE_MODELS:
            pipe = make_pipeline(m, "none", pos_weight, cols=feats_)
            auc = cross_val_score(pipe, train[feats_], y_tr, cv=cv10, scoring="roc_auc", n_jobs=N_JOBS)
            acc = cross_val_score(pipe, train[feats_], y_tr, cv=cv10, scoring="accuracy", n_jobs=N_JOBS)
            rows.append({"selector": s_name, "model": m, "AUROC_mean": auc.mean(), "AUROC_sd": auc.std(),
                         "Accuracy_mean": acc.mean(), "Accuracy_sd": acc.std()})
            print(f"  {s_name:8s} {m:8s} AUROC {auc.mean():.3f}±{auc.std():.3f}  ACC {acc.mean():.3f}")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table3a_baseline_reproduction_cv10.csv", index=False)

    # ---- D. model comparison on all predictors
    feats = list(FEATURES)
    print(f"\n[D] Model comparison on all {len(feats)} predictors ({CV_FOLDS}-fold CV) ...")
    Xtr, Xte = train[feats], test[feats]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for m in ALL_MODELS:
        for s in STRATEGIES:
            pipe = make_pipeline(m, s, pos_weight, cols=feats)
            oof = cross_val_predict(pipe, Xtr, y_tr, cv=cv, method="predict_proba",
                                    n_jobs=1 if m == "Stacking" else N_JOBS)[:, 1]
            r = {"model": m, "strategy": s, **point_metrics(y_tr, oof, youden_threshold(y_tr, oof))}
            r["Brier_recalibrated"] = brier_score_loss(y_tr, crossfit_recal(y_tr, oof))
            rows.append(r)
            print(f"  {m:9s} {s:12s} AUROC {r['AUROC']:.3f}  AUPRC {r['AUPRC']:.3f}  "
                  f"Brier raw {r['Brier']:.3f} recal {r['Brier_recalibrated']:.3f}")
    comp = pd.DataFrame(rows).assign(AUROC_r=lambda d: d.AUROC.round(3))
    comp = comp.sort_values(["AUROC_r", "Brier_recalibrated"], ascending=[False, True]).drop(columns="AUROC_r")
    comp.to_csv(TABLE_DIR / "table4_model_imbalance_comparison_cv.csv", index=False)
    best_model, best_strategy = comp.iloc[0][["model", "strategy"]]
    Zi = StandardScaler().fit_transform(IterativeImputer(max_iter=10, random_state=SEED).fit_transform(Xtr))
    n_sm = len(SMOTE(random_state=SEED).fit_resample(Zi, y_tr)[1])
    n_st = len(SMOTETomek(random_state=SEED).fit_resample(Zi, y_tr)[1])
    json.dump({"after_smote": n_sm, "after_smote_tomek": n_st, "tomek_removed": n_sm - n_st},
              open(TABLE_DIR / "smote_tomek_check.json", "w"))
    print(f"  SMOTE n={n_sm}, SMOTE-Tomek n={n_st} (Tomek links removed {n_sm - n_st})")
    print(f"  -> best: {best_model} + {best_strategy}")

    # ---- E. tuning: every model family receives the same Optuna budget
    def space(t: optuna.Trial, mname: str) -> dict:
        if mname in ("LR", "LR-spline"):
            return {"C": t.suggest_float("C", 1e-3, 100, log=True)}
        if mname == "SVM":
            return {"C": t.suggest_float("C", 1e-2, 100, log=True),
                    "gamma": t.suggest_float("gamma", 1e-5, 1, log=True)}
        if mname == "RF":
            return {"n_estimators": t.suggest_int("n_estimators", 200, 1000, step=100),
                    "max_depth": t.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 60),
                    "max_features": t.suggest_float("max_features", 0.05, 1.0)}
        if mname == "XGBoost":
            return {"n_estimators": t.suggest_int("n_estimators", 100, 2000, step=50),
                    "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                    "max_depth": t.suggest_int("max_depth", 1, 8),
                    "subsample": t.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
                    "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
        if mname == "LightGBM":
            return {"n_estimators": t.suggest_int("n_estimators", 100, 2000, step=50),
                    "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                    "num_leaves": t.suggest_int("num_leaves", 2, 64),
                    "min_child_samples": t.suggest_int("min_child_samples", 1, 200),
                    "subsample": t.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
        if mname == "CatBoost":
            return {"iterations": t.suggest_int("iterations", 200, 2000, step=100),
                    "depth": t.suggest_int("depth", 1, 8),
                    "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                    "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}
        return {}


    def tune(mname, strategy, cols, n_trials):
        if not n_trials:
            return {}, np.nan
        def objective(t):
            pipe = make_pipeline(mname, strategy, pos_weight, space(t, mname), cols=cols)
            return cross_val_score(pipe, train[cols], y_tr, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(objective, n_trials=n_trials)
        return st.best_params, st.best_value

    print(f"\n[E] Optuna tuning of every model family ({OPTUNA_TRIALS} trials each) ...")
    fam_strategy = {m: comp[comp.model == m].iloc[0].strategy for m in ALL_MODELS}
    tuned, trows = {}, []
    for m in ALL_MODELS:
        if m == "Stacking":
            continue
        default_auc = comp[(comp.model == m) & (comp.strategy == fam_strategy[m])].AUROC.iloc[0]
        prm, val = tune(m, fam_strategy[m], feats, OPTUNA_TRIALS)
        tuned[m] = prm if (not np.isnan(val) and val > default_auc) else {}
        trows.append({"model": m, "strategy": fam_strategy[m], "default_cv_AUROC": default_auc,
                      "tuned_cv_AUROC": max(val, default_auc) if not np.isnan(val) else default_auc,
                      "params": json.dumps(tuned[m])})
        print(f"  {m:9s} {fam_strategy[m]:12s} default {default_auc:.3f} tuned {trows[-1]['tuned_cv_AUROC']:.3f} {tuned[m]}")
    tuned["Stacking"] = {b: tuned[b] for b in ["RF", "XGBoost", "LightGBM", "CatBoost"]}
    st_auc = cross_val_score(make_pipeline("Stacking", fam_strategy["Stacking"], pos_weight, tuned["Stacking"],
                                           cols=feats), Xtr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    trows.append({"model": "Stacking", "strategy": fam_strategy["Stacking"],
                  "default_cv_AUROC": comp[(comp.model == "Stacking") & (comp.strategy == fam_strategy["Stacking"])].AUROC.iloc[0],
                  "tuned_cv_AUROC": st_auc, "params": "tuned base learners"})
    print(f"  Stacking (tuned base learners) {st_auc:.3f}")
    spaces = {}
    for m in ALL_MODELS:
        if m == "Stacking":
            continue
        rec = {}
        class _Rec:
            def suggest_float(self, n, lo, hi, log=False): rec[n] = (lo, hi, "log" if log else "linear"); return lo
            def suggest_int(self, n, lo, hi, step=1): rec[n] = (lo, hi, f"int step {step}"); return lo
        space(_Rec(), m)
        spaces[m] = dict(rec)
    hp_rows = []
    for m, sp in spaces.items():
        for n, (lo, hi, kind) in sp.items():
            v = tuned[m].get(n)
            at_bound = v is not None and (np.isclose(v, lo, rtol=0.05) or np.isclose(v, hi, rtol=0.05))
            hp_rows.append({"model": m, "hyperparameter": n, "lower": lo, "upper": hi, "scale": kind,
                            "selected": v, "at_bound": bool(at_bound),
                            "trials": OPTUNA_TRIALS})
    pd.DataFrame(hp_rows).to_csv(TABLE_DIR / "tableS_hyperparameters.csv", index=False)
    tdf = pd.DataFrame(trows)
    tdf["order"] = tdf.model.map({m: i for i, m in enumerate(ALL_MODELS)})
    tdf = tdf.assign(r=tdf.tuned_cv_AUROC.round(3)).sort_values(["r", "order"], ascending=[False, True])
    tdf.drop(columns=["r", "order"]).to_csv(TABLE_DIR / "table4b_tuned_families_cv.csv", index=False)
    best_model = tdf.iloc[0].model; best_strategy = fam_strategy[best_model]; best_params = tuned[best_model]
    print(f"  -> selected: {best_model} + {best_strategy}")
    zhang_params, _ = tune("XGBoost", "none", sel["boruta"], OPTUNA_TRIALS)
    json.dump({"model": best_model, "strategy": best_strategy, "params": best_params,
               "features": feats}, open(TABLE_DIR / "final_model_config.json", "w"), indent=2)

    # ---- F. temporal validation and benchmarks
    print("\n[F] Temporal validation and benchmarks ...")

    def develop(cols, model, strategy, params, y_dev=None, y_val=None, tr=None, te=None):
        tr = train if tr is None else tr
        te = test if te is None else te
        y_dev = y_tr if y_dev is None else y_dev
        pw = (y_dev == 0).sum() / max((y_dev == 1).sum(), 1)
        pipe = make_pipeline(model, strategy, pw, params, cols=cols)
        raw = cross_val_predict(pipe, tr[cols], y_dev, cv=cv, method="predict_proba",
                                n_jobs=1 if model == "Stacking" else N_JOBS)[:, 1]
        recal_ = fit_recalibrator(y_dev, raw)
        thr_ = youden_threshold(y_dev, recal_(raw))
        pipe.fit(tr[cols], y_dev)
        return {"pipe": pipe, "oof": crossfit_recal(y_dev, raw), "oof_full": recal_(raw), "thr": thr_,
                "recal": recal_, "p": recal_(pipe.predict_proba(te[cols])[:, 1]), "cols": cols}

    med = train[["weight", "age"]].median()

    def ost_score(d):
        return 0.2 * (d["weight"].fillna(med["weight"]) - d["age"].fillna(med["age"])).values

    def score_model(score_tr, score_te, y_dev=None):
        """Fixed clinical score -> probability via logistic recalibration on development data."""
        y_dev = y_tr if y_dev is None else y_dev
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(score_tr.reshape(-1, 1), y_dev)
        p_tr = lr.predict_proba(score_tr.reshape(-1, 1))[:, 1]
        return {"oof": p_tr, "oof_full": p_tr, "thr": youden_threshold(y_dev, p_tr),
                "p": lr.predict_proba(score_te.reshape(-1, 1))[:, 1]}

    def orai_score(d):
        """ORAI (Cadarette 2000): age, weight, estrogen use (ever use as proxy for current)."""
        a = d["age"].fillna(med["age"]).values; w = d["weight"].fillna(med["weight"]).values
        e = d["hormone_use"].fillna(0).values
        return (np.select([a >= 75, a >= 65, a >= 55], [15, 9, 5], 0) +
                np.select([w < 60, w < 70], [9, 3], 0) + 2 * (e == 0))

    def score_score(d):
        """SCORE (Lydick 1998): race, rheumatoid arthritis, fractures, age, estrogen, weight."""
        a = d["age"].fillna(med["age"]).values; w = d["weight"].fillna(med["weight"]).values
        return (5 * (d["race_black"].fillna(0).values == 0) + 4 * d["rheumatoid"].fillna(0).values +
                4 * np.minimum(d["n_fracture_sites"].fillna(0).values, 3) + 3 * np.floor(a / 10) +
                1 * (d["hormone_use"].fillna(0).values == 0) - np.floor(w * 2.2046 / 10))

    no_race = [f for f in feats if f not in RACE_FEATURES]
    M = {
        "Proposed": develop(feats, best_model, best_strategy, best_params),
        "Proposed without race": develop(no_race, best_model, best_strategy, best_params),
        ZHANG_NAME: develop(sel["boruta"], "XGBoost", "none", zhang_params),
        LR3_NAME: develop(["age", "weight", "female"], "LR", "none", {}),
        OST_NAME: score_model(-ost_score(train), -ost_score(test)),
        ORAI_NAME: score_model(orai_score(train).astype(float), orai_score(test).astype(float)),
        SCORE_NAME: score_model(score_score(train).astype(float), score_score(test).astype(float)),
    }
    for m in ALL_MODELS:
        if m != best_model:
            M[f"{m} (tuned)"] = develop(feats, m, fam_strategy[m], tuned[m])

    final, p_te, thr = M["Proposed"]["pipe"], M["Proposed"]["p"], M["Proposed"]["thr"]
    res_int = bootstrap_metrics(y_tr, M["Proposed"]["oof"], thr).assign(model="Proposed", data="internal_cv")
    perf = [res_int]
    for name, r in M.items():
        perf.append(bootstrap_metrics(y_te, r["p"], r["thr"]).assign(model=name, data="temporal_test"))
    perf = pd.concat(perf)
    perf.to_csv(TABLE_DIR / "table5_final_performance_bootstrap.csv", index=False)
    res_te = perf[(perf.model == "Proposed") & (perf.data == "temporal_test")]

    # Pre-specified hierarchy: primary = full model vs OST (single test); secondary = other
    # simple tools and pipelines (Holm within family); ML families vs LR = equivalence (90% CI
    # within +/- EQ_MARGIN), with Holm-adjusted superiority P within that family.
    def family(name):
        if name == OST_NAME:
            return "primary"
        return "ml_equivalence" if name.endswith("(tuned)") else "secondary"
    dl = []
    for name, r in M.items():
        if name == "Proposed":
            continue
        a1, a2, pv = delong_test(y_te, p_te, r["p"])
        _, dlo, dhi, bp = boot_auc_diff(y_te, p_te, r["p"], return_p=True)
        l90, h90, sd_ = boot_auc_diff_ci90(y_te, p_te, r["p"])
        dl.append({"comparator": name, "family": family(name), "AUROC_proposed": a1, "AUROC_comparator": a2,
                   "difference": a1 - a2, "diff_low": dlo, "diff_high": dhi, "diff_low90": l90,
                   "diff_high90": h90, "boot_p": bp, "DeLong_p": pv,
                   "equivalent": bool(l90 > -EQ_MARGIN and h90 < EQ_MARGIN)})
    dl = pd.DataFrame(dl)
    dl["p_holm"] = dl.boot_p
    for fam_ in ["secondary", "ml_equivalence"]:
        k_ = dl.family == fam_
        dl.loc[k_, "p_holm"] = multipletests(dl.loc[k_, "boot_p"], method="holm")[1]
    dl.to_csv(TABLE_DIR / "table5b_delong_test.csv", index=False)

    # DXA non-response adjustment: weight / P(valid DXA | covariates) among aged >= 50
    inc_all = pd.read_csv(PROC_DIR / "nhanes_osteoporosis_incl_diagnosed.csv")
    exc_all = pd.read_csv(PROC_DIR / "excluded_no_dxa.csv")
    resp = pd.concat([inc_all.assign(R=1), exc_all.assign(R=0)], ignore_index=True)
    def rx(d):
        X = pd.DataFrame({"age": d.age, "female": d.female, "education": d.education.fillna(3),
                          "edu_miss": d.education.isna().astype(int), "pir": d.pir.fillna(2.5),
                          "pir_miss": d.pir.isna().astype(int), "bmi": d.bmi.fillna(28),
                          "bmi_miss": d.bmi.isna().astype(int)})
        for c in ["race_mexican", "race_other_hisp", "race_black", "race_other"]:
            X[c] = d[c]
        for c in ["E", "F", "H", "J"]:
            X["cyc_" + c] = (d.cycle == c).astype(int)
        X["bmi2"] = X.bmi ** 2
        return X
    sc_r = StandardScaler().fit(rx(resp))
    rmod = LogisticRegression(max_iter=5000).fit(sc_r.transform(rx(resp)), resp.R)
    p_resp = rmod.predict_proba(sc_r.transform(rx(test)))[:, 1]
    w_adj = test.wt_mec.values / p_resp
    json.dump({"response_rate": float(resp.R.mean()), "p_resp_min": float(p_resp.min()),
               "p_resp_max": float(p_resp.max())}, open(TABLE_DIR / "nonresponse_model.json", "w"))
    wrows = []
    for n_, r in M.items():
        pp = r["p"]
        lo, hi = design_boot(lambda ww: roc_auc_score(y_te, pp, sample_weight=ww), test.wt_mec, test.strata, test.psu)
        alo, ahi = design_boot(lambda ww: roc_auc_score(y_te, pp, sample_weight=ww), w_adj, test.strata, test.psu)
        row = {"model": n_, **weighted_metrics(y_te, pp, test.wt_mec),
               "AUROC_weighted_low": lo, "AUROC_weighted_high": hi,
               "AUROC_nr_adjusted": roc_auc_score(y_te, pp, sample_weight=w_adj),
               "AUROC_nr_adjusted_low": alo, "AUROC_nr_adjusted_high": ahi,
               "Observed_nr_adjusted_pct": np.average(y_te, weights=w_adj) * 100,
               "Predicted_nr_adjusted_pct": np.average(pp, weights=w_adj) * 100}
        if n_ != "Proposed":
            dlo_, dhi_ = design_boot(lambda ww: roc_auc_score(y_te, p_te, sample_weight=ww) -
                                     roc_auc_score(y_te, pp, sample_weight=ww), w_adj, test.strata, test.psu)
            row.update({"diff_nr_adjusted": roc_auc_score(y_te, p_te, sample_weight=w_adj) - row["AUROC_nr_adjusted"],
                        "diff_low": dlo_, "diff_high": dhi_})
        wrows.append(row)
    pd.DataFrame(wrows).to_csv(TABLE_DIR / "table5c_survey_weighted_validation.csv", index=False)
    print(f"  internal AUROC {fmt_ci(res_int, 'AUROC')}")
    for name in M:
        b = perf[(perf.model == name) & (perf.data == "temporal_test")]
        print(f"  {name:58s} temporal AUROC {fmt_ci(b, 'AUROC')}")
    print(dl.round(4).to_string(index=False))
    print(pd.DataFrame(wrows).round(3).to_string(index=False))

    # OST at published cut-offs (no fitting) vs the proposed model at the same referral rate
    ost_te = ost_score(test)
    rows = []
    for cut in [2.0, -1.0]:
        refer = ost_te < cut
        rate = refer.mean()
        thr_match = np.quantile(p_te, 1 - rate)
        prop = p_te >= thr_match
        for name, ref in [(f"OST < {cut:g}", refer), (f"Proposed at matched referral rate", prop)]:
            tp = (ref & (y_te == 1)).sum(); fp = (ref & (y_te == 0)).sum()
            rows.append({"rule": name, "ost_cutoff": cut, "referred_pct": ref.mean() * 100,
                         "sensitivity": tp / y_te.sum(), "specificity": 1 - fp / (y_te == 0).sum(),
                         "detected": int(tp), "events": int(y_te.sum())})
        dets = []
        for _ in range(N_BOOT):
            ii = RNG.integers(0, len(y_te), len(y_te))
            dets.append(((prop[ii]) & (y_te[ii] == 1)).sum() - ((refer[ii]) & (y_te[ii] == 1)).sum())
        rows[-1].update({"extra_detected_low": np.quantile(dets, 0.025), "extra_detected_high": np.quantile(dets, 0.975)})
    ostc = pd.DataFrame(rows)
    ostc.to_csv(TABLE_DIR / "table5d_ost_cutoffs.csv", index=False)

    # OST by sex (derived in women; no sex term)
    srows = []
    for sx, mask in [("Female", test.female.values == 1), ("Male", test.female.values == 0)]:
        yy = y_te[mask]
        r_ = {"sex": sx, "n": int(mask.sum()), "events": int(yy.sum())}
        for nm, key in [("Proposed", "Proposed"), ("LR3", LR3_NAME), ("OST", OST_NAME)]:
            r_[f"AUROC_{nm}"] = roc_auc_score(yy, M[key]["p"][mask])
        d_, lo_, hi_ = boot_auc_diff(yy, M["Proposed"]["p"][mask], -ost_te[mask])
        r_.update({"diff_vs_OST": d_, "diff_low": lo_, "diff_high": hi_})
        for cut in [2.0, -1.0]:
            ref = ost_te[mask] < cut
            r_[f"OST<{cut:g}_sens"] = (ref & (yy == 1)).sum() / max(yy.sum(), 1)
            r_[f"OST<{cut:g}_spec"] = 1 - (ref & (yy == 0)).sum() / (yy == 0).sum()
            r_[f"OST<{cut:g}_referred_pct"] = ref.mean() * 100
        srows.append(r_)
    pd.DataFrame(srows).to_csv(TABLE_DIR / "table5g_ost_by_sex.csv", index=False)
    print(pd.DataFrame(srows).round(3).to_string(index=False))

    # Operating points of the proposed model (thresholds fixed in development data)
    oof_p = M["Proposed"]["oof_full"]
    fpr_, tpr_, thr_all = roc_curve(y_tr, oof_p)
    thr90 = float(thr_all[np.argmax(tpr_ >= 0.90)])
    orows = []
    for lab_, t_ in [("Youden (development)", thr), ("Sensitivity 0.90 in development", thr90),
                     ("Fixed 3%", 0.03), ("Fixed 5%", 0.05), ("Fixed 10%", 0.10)]:
        pm = point_metrics(y_te, p_te, t_)
        orows.append({"operating_point": lab_, "threshold": t_, **{k: pm[k] for k in
                      ["Sensitivity", "Specificity", "PPV", "NPV", "Referred_pct", "Detected", "Events"]}})
    pd.DataFrame(orows).to_csv(TABLE_DIR / "table5h_operating_points.csv", index=False)
    print(pd.DataFrame(orows).round(3).to_string(index=False))
    print(ostc.round(3).to_string(index=False))

    # Figures: ROC, calibration, decision curves (+ net benefit table)
    show = ["Proposed", ZHANG_NAME, LR3_NAME, OST_NAME, ORAI_NAME, SCORE_NAME]
    short_lab = lambda n: DISPLAY.get(n, n)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in show:
        fpr, tpr, _ = roc_curve(y_te, M[name]["p"])
        if name in (ORAI_NAME, SCORE_NAME):
            ax.plot(fpr, tpr, ls=":", lw=1.2, label=f"{short_lab(name)} (AUC {roc_auc_score(y_te, M[name]['p']):.3f})")
            continue
        ax.plot(fpr, tpr, label=f"{short_lab(name)} (AUC {roc_auc_score(y_te, M[name]['p']):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC curves, temporal validation 2017-2018")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_roc.png", dpi=300); plt.close(fig)

    from statsmodels.nonparametric.smoothers_lowess import lowess
    lim = float(np.ceil(np.quantile(p_te, 0.98) * 20) / 20)
    grid = np.linspace(float(np.quantile(p_te, 0.01)), lim, 60)
    def _loess(yy, pp):
        sm_ = lowess(yy, pp, frac=0.6, it=0, return_sorted=True)   # it=0: no robustness weights for 0/1 data
        return np.interp(grid, sm_[:, 0], sm_[:, 1])
    rng_c = np.random.default_rng(SEED); bands = []
    for _ in range(200):
        ii = rng_c.integers(0, len(y_te), len(y_te))
        bands.append(_loess(y_te[ii], p_te[ii]))
    bands = np.array(bands)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.fill_between(grid, np.quantile(bands, 0.025, 0), np.quantile(bands, 0.975, 0), color="C1", alpha=0.2,
                    label="95% CI (loess, temporal)")
    ax.plot(grid, _loess(y_te, p_te), "-", color="C1", lw=1.8, label="Temporal validation (loess)")
    for y_, p_, lab, mk in [(y_tr, M["Proposed"]["oof"], "Internal CV, deciles", "s"),
                            (y_te, p_te, "Temporal validation, deciles", "o")]:
        fp_, mp_ = calibration_curve(y_, p_, n_bins=10, strategy="quantile")
        ax.plot(mp_, fp_, mk, ms=5, label=lab)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Ideal")
    ax.set(ylabel="Observed proportion", title="Calibration (full model)", xlim=(0, lim), ylim=(0, lim))
    ax.legend(fontsize=8, loc="upper left")
    bins_ = np.linspace(0, lim, 36)
    h0, _ = np.histogram(p_te[y_te == 0], bins_); h1, _ = np.histogram(p_te[y_te == 1], bins_)
    axh.bar(bins_[:-1], h0, width=np.diff(bins_), align="edge", color="grey", alpha=0.7, label="No osteoporosis (up)")
    axh.bar(bins_[:-1], -h1 * 10, width=np.diff(bins_), align="edge", color="C3", alpha=0.9,
            label="Osteoporosis (down, x10)")
    axh.axhline(0, color="k", lw=0.5)
    axh.set(xlabel="Predicted probability", ylabel="Count"); axh.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_calibration.png", dpi=300); plt.close(fig)
    json.dump({"axis_limit": lim, "share_above_limit": float((p_te > lim).mean())},
              open(TABLE_DIR / "calibration_plot_info.json", "w"))

    ts = np.linspace(0.01, 0.40, 40)
    prev = y_te.mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name in show:
        ax.plot(ts, net_benefit(y_te, M[name]["p"], ts), label=short_lab(name),
                ls=":" if name in (ORAI_NAME, SCORE_NAME) else "-")
    ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), "k:", label="Treat all")
    ax.axhline(0, color="grey", lw=0.8, label="Treat none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curves, temporal validation",
           ylim=(-0.02, max(prev, 0.05) * 1.1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_decision_curve.png", dpi=300); plt.close(fig)
    nb_t = [0.05, 0.10, 0.15, 0.20]
    pd.concat([net_benefit_ci(y_te, M[n]["p"], nb_t).assign(model=short_lab(n)) for n in show]).to_csv(
        TABLE_DIR / "table5f_net_benefit_ci.csv", index=False)
    chk = {short_lab(n): {"min_margin_vs_default": float(np.min(net_benefit(y_te, M[n]["p"], ts) -
                                                              np.maximum(prev - (1 - prev) * ts / (1 - ts), 0))),
                          "thresholds_not_above_default": [round(float(t), 2) for t, v in zip(ts,
                              net_benefit(y_te, M[n]["p"], ts) - np.maximum(prev - (1 - prev) * ts / (1 - ts), 0))
                              if v <= 0]} for n in show}
    json.dump(chk, open(TABLE_DIR / "nb_range_check.json", "w"), indent=1)
    pd.DataFrame({"threshold": nb_t, **{short_lab(n): net_benefit(y_te, M[n]["p"], nb_t) for n in show},
                  "treat_all": [prev - (1 - prev) * t / (1 - t) for t in nb_t]}).to_csv(
        TABLE_DIR / "table5e_net_benefit.csv", index=False)

    # ---- G. incremental value of predictor blocks (nested LR models)
    print("\n[G] Incremental value of predictor blocks ...")
    rows, cols, prev_p = [], [], None
    for label, block in INCREMENTAL_BLOCKS:
        cols = cols + block
        # first block = the age/weight/sex benchmark; later blocks use the selected LR settings,
        # so the last block is the full model (same predictions as in Table 4)
        r = M[LR3_NAME] if prev_p is None else (
            develop(list(cols), best_model, best_strategy, best_params) if best_model == "LR"
            else develop(list(cols), "LR", "none", {}))
        b = bootstrap_metrics(y_te, r["p"], r["thr"], n_boot=N_BOOT, calib=False).set_index("metric")
        row = {"block": label, "n_predictors": len(cols), "cv_AUROC": roc_auc_score(y_tr, r["oof"]),
               "AUROC": b.loc["AUROC", "estimate"], "AUROC_low": b.loc["AUROC", "ci_low"],
               "AUROC_high": b.loc["AUROC", "ci_high"], "AUPRC": b.loc["AUPRC", "estimate"]}
        if prev_p is not None:
            row["delta"], row["delta_low"], row["delta_high"], row["boot_p_vs_previous"] = \
                boot_auc_diff(y_te, r["p"], prev_p, return_p=True)
        rows.append(row)
        prev_p = r["p"]
        print(f"  {label:45s} k={len(cols):2d} AUROC {row['AUROC']:.3f}")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table6b_incremental_value.csv", index=False)

    # ---- H. fairness
    print("\n[H] Subgroup / fairness analysis ...")
    fem_tr = train.female.values == 1
    thr_sex = {"Female": youden_threshold(y_tr[fem_tr], M["Proposed"]["oof_full"][fem_tr]),
               "Male": youden_threshold(y_tr[~fem_tr], M["Proposed"]["oof_full"][~fem_tr])}
    test_s = test.assign(sex=np.where(test.female == 1, "Female", "Male"),
                         age_group=pd.cut(test.age, [49, 64, 79, 200], labels=["50-64", "65-79", "80+"]),
                         race2=np.where(test.asian == 1, "NH Asian",
                                        np.where(test.race == "Other/Multi", "Other/Multi (non-Asian)", test.race)))
    rows = []
    for model_name in ["Proposed", "Proposed without race"]:
        prob, thr_m = M[model_name]["p"], M[model_name]["thr"]
        for var in ["sex", "age_group", "race2"]:
            for level, g in test_s.assign(prob=prob).groupby(var, observed=True):
                if len(g) < 30 or g[PRIMARY_OUTCOME].nunique() < 2:
                    continue
                yy, pp = g[PRIMARY_OUTCOME].values, g.prob.values
                b = bootstrap_metrics(yy, pp, thr_m, n_boot=N_BOOT, calib=False).set_index("metric")
                row = {"model": model_name, "threshold": thr_m, "group": var, "level": level, "n": len(g),
                       "events": int(yy.sum()), "prevalence": yy.mean(), "mean_predicted": pp.mean(),
                       "AUROC": b.loc["AUROC", "estimate"], "AUROC_low": b.loc["AUROC", "ci_low"],
                       "AUROC_high": b.loc["AUROC", "ci_high"],
                       "Sensitivity": b.loc["Sensitivity", "estimate"],
                       "Sens_low": b.loc["Sensitivity", "ci_low"], "Sens_high": b.loc["Sensitivity", "ci_high"],
                       "Specificity": b.loc["Specificity", "estimate"],
                       "Detected": int(b.loc["Detected", "estimate"]),
                       "Referred_pct": b.loc["Referred_pct", "estimate"]}
                if model_name == "Proposed" and var == "sex":
                    pm = point_metrics(yy, pp, thr_sex[level])
                    row.update({"sex_threshold": thr_sex[level], "Sensitivity_sex_thr": pm["Sensitivity"],
                                "Specificity_sex_thr": pm["Specificity"], "Detected_sex_thr": pm["Detected"],
                                "Referred_pct_sex_thr": pm["Referred_pct"]})
                rows.append(row)
    fair = pd.DataFrame(rows)
    fair.to_csv(TABLE_DIR / "table6_fairness_subgroups.csv", index=False)
    print(fair.round(3).to_string(index=False))

    # ---- H2. decision-relevant clinical strata (where a risk tool decides who gets DXA)
    print("\n[H2] Clinical strata ...")
    a_te, f_te = test.age.values, test.female.values == 1
    strata_def = [("Women 50-64 years", f_te & (a_te < 65)), ("Women 65+ years", f_te & (a_te >= 65)),
                  ("Men 50-69 years", ~f_te & (a_te < 70)), ("Men 70+ years", ~f_te & (a_te >= 70)),
                  ("All women", f_te), ("All men", ~f_te)]
    rows = []
    for lab_, mk in strata_def:
        yy = y_te[mk]
        r_ = {"stratum": lab_, "n": int(mk.sum()), "events": int(yy.sum())}
        if yy.sum() < 2:
            rows.append(r_); continue
        for nm, key in [("Full", "Proposed"), ("LR3", LR3_NAME), ("OST", OST_NAME), ("ORAI", ORAI_NAME),
                        ("SCORE", SCORE_NAME)]:
            r_[f"AUROC_{nm}"] = roc_auc_score(yy, M[key]["p"][mk])
        d_, lo_, hi_ = boot_auc_diff(yy, M["Proposed"]["p"][mk], M[OST_NAME]["p"][mk])
        r_.update({"diff_vs_OST": d_, "diff_vs_OST_low": lo_, "diff_vs_OST_high": hi_})
        d_, lo_, hi_ = boot_auc_diff(yy, M["Proposed"]["p"][mk], M[LR3_NAME]["p"][mk])
        r_.update({"diff_vs_LR3": d_, "diff_vs_LR3_low": lo_, "diff_vs_LR3_high": hi_})
        pm = point_metrics(yy, M["Proposed"]["p"][mk], thr)
        r_.update({"Full_sens": pm["Sensitivity"], "Full_referred_pct": pm["Referred_pct"], "Full_detected": pm["Detected"]})
        ref = ost_te[mk] < 2
        r_.update({"OST<2_sens": (ref & (yy == 1)).sum() / yy.sum(), "OST<2_referred_pct": ref.mean() * 100,
                   "OST<2_detected": int((ref & (yy == 1)).sum())})
        rows.append(r_)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table6c_clinical_strata.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- H3. internal-external cross-validation by cycle (all five cycles, all events)
    print("\n[H3] Internal-external cross-validation by cycle ...")
    full_df = pd.concat([train, test], ignore_index=True)
    rows = []
    for c_ in ["D", "E", "F", "H", "J"]:
        tr_c = full_df[full_df.cycle != c_].reset_index(drop=True)
        te_c = full_df[full_df.cycle == c_].reset_index(drop=True)
        yd_c, yv_c = tr_c[PRIMARY_OUTCOME].values, te_c[PRIMARY_OUTCOME].values
        med_c = tr_c[["weight", "age"]].median()
        ost_c = lambda d: 0.2 * (d["weight"].fillna(med_c["weight"]) - d["age"].fillna(med_c["age"])).values
        rf_ = develop(feats, best_model, best_strategy, best_params, y_dev=yd_c, tr=tr_c, te=te_c)
        r3_ = develop(["age", "weight", "female"], "LR", "none", {}, y_dev=yd_c, tr=tr_c, te=te_c)
        ro_ = score_model(-ost_c(tr_c), -ost_c(te_c), y_dev=yd_c)
        row = {"held_out_cycle": CYCLES_YEARS.get(c_, c_), "n": len(te_c), "events": int(yv_c.sum()),
               "AUROC_full": roc_auc_score(yv_c, rf_["p"]), "AUROC_LR3": roc_auc_score(yv_c, r3_["p"]),
               "AUROC_OST": roc_auc_score(yv_c, ro_["p"]),
               "calib_slope_full": calibration_slope_intercept(yv_c, rf_["p"])[0],
               "O_E_full": yv_c.mean() / rf_["p"].mean()}
        for nm, other in [("OST", ro_), ("LR3", r3_)]:
            d_, lo_, hi_ = boot_auc_diff(yv_c, rf_["p"], other["p"])
            row.update({f"diff_vs_{nm}": d_, f"diff_vs_{nm}_low": lo_, f"diff_vs_{nm}_high": hi_,
                        f"diff_vs_{nm}_se": (hi_ - lo_) / 3.92})
        rows.append(row)
        print(f"  {c_}: n={len(te_c)} ev={int(yv_c.sum())} AUROC full {row['AUROC_full']:.3f} "
              f"OST {row['AUROC_OST']:.3f} LR3 {row['AUROC_LR3']:.3f}")
    iecv = pd.DataFrame(rows)
    iecv.to_csv(TABLE_DIR / "table9b_iecv_by_cycle.csv", index=False)
    pooled = []
    for nm in ["OST", "LR3"]:
        pooled.append({"comparison": f"Full model - {nm}",
                       **random_effects(iecv[f"diff_vs_{nm}"], iecv[f"diff_vs_{nm}_se"])})
    for col_ in ["AUROC_full", "AUROC_LR3", "AUROC_OST"]:
        # SE of a single AUROC (Hanley-McNeil)
        a_ = iecv[col_].values; n1 = iecv.events.values; n0 = (iecv.n - iecv.events).values
        q1, q2 = a_ / (2 - a_), 2 * a_ ** 2 / (1 + a_)
        se_ = np.sqrt((a_ * (1 - a_) + (n1 - 1) * (q1 - a_ ** 2) + (n0 - 1) * (q2 - a_ ** 2)) / (n1 * n0))
        pooled.append({"comparison": col_, **random_effects(a_, se_)})
    pd.DataFrame(pooled).to_csv(TABLE_DIR / "table9c_iecv_pooled.csv", index=False)
    print(pd.DataFrame(pooled).round(4).to_string(index=False))

    # ---- H4. uncertainty from model refitting: bootstrap development AND validation sets
    print("\n[H4] Refit bootstrap (development + validation resampling) ...")
    from joblib import Parallel, delayed
    ost_tr_s, ost_te_s = -ost_score(train), -ost_score(test)
    def _refit(seed_):
        rg = np.random.default_rng(seed_)
        idd = rg.integers(0, len(train), len(train)); idv = rg.integers(0, len(test), len(test))
        trb, yb = train.iloc[idd].reset_index(drop=True), y_tr[idd]
        pw_ = (yb == 0).sum() / max(yb.sum(), 1)
        pf = make_pipeline(best_model, best_strategy, pw_, best_params, cols=feats).fit(trb[feats], yb)
        p3 = make_pipeline("LR", "none", pw_, {}, cols=["age", "weight", "female"]).fit(
            trb[["age", "weight", "female"]], yb)
        yv = y_te[idv]
        if yv.min() == yv.max():
            return None
        a_f = roc_auc_score(yv, pf.predict_proba(test.iloc[idv][feats])[:, 1])
        a_3 = roc_auc_score(yv, p3.predict_proba(test.iloc[idv][["age", "weight", "female"]])[:, 1])
        a_o = roc_auc_score(yv, ost_te_s[idv])
        return a_f, a_3, a_o
    nrb = 200 if N_BOOT >= 200 else 20
    res_rb = [r for r in Parallel(n_jobs=N_JOBS)(delayed(_refit)(SEED + 1 + i) for i in range(nrb)) if r is not None]
    rb = np.array(res_rb)
    rb_out = {"n_boot": len(rb),
              "AUROC_full": [float(np.quantile(rb[:, 0], q)) for q in (0.025, 0.975)],
              "diff_vs_OST": [float(np.quantile(rb[:, 0] - rb[:, 2], q)) for q in (0.025, 0.975)],
              "diff_vs_LR3": [float(np.quantile(rb[:, 0] - rb[:, 1], q)) for q in (0.025, 0.975)],
              "p_diff_vs_OST_le0": float((rb[:, 0] - rb[:, 2] <= 0).mean())}
    json.dump(rb_out, open(TABLE_DIR / "refit_bootstrap.json", "w"), indent=1)
    print(rb_out)

    # ---- I. SHAP (importance on development data) and race-free short form
    print("\n[I] SHAP and short form ...")
    steps = final.named_steps
    mdl = steps["model"]

    def transform(X):
        Xi = pd.DataFrame(steps["impute"].transform(X), columns=feats)
        Z = steps["scale"].transform(Xi)
        return Xi, Z

    def explain(X):
        Xi, Z = transform(X)
        Zdf = pd.DataFrame(Z, columns=feats)
        if best_model in TREE_MODELS:
            e = shap.TreeExplainer(mdl)(Zdf)
        else:
            head = (lambda a: mdl.predict_proba(steps["spline"].transform(a))[:, 1]) if "spline" in steps \
                else (lambda a: mdl.predict_proba(a)[:, 1])
            bg = shap.sample(pd.DataFrame(transform(Xtr)[1], columns=feats), 100, random_state=SEED)
            e = shap.Explainer(lambda a: head(np.asarray(a)), bg)(Zdf)
        if e.values.ndim == 3:
            e = e[..., 1]
        e.data = Xi.values
        e.feature_names = feats
        return e

    dev_idx = RNG.choice(len(Xtr), min(500, len(Xtr)), replace=False)
    exp_dev = explain(Xtr.iloc[dev_idx])
    mean_abs = pd.Series(np.abs(exp_dev.values).mean(0), index=feats).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap_development").to_csv(TABLE_DIR / "table7_shap_importance.csv")

    exp = exp_dev                       # all SHAP figures use the same development sample
    exp.feature_names = [PRETTY.get(f, f) for f in feats]
    plt.figure(); shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close("all")
    for f in FIG_DIR.glob("fig7_shap_dependence_*.png"):
        f.unlink()
    for i, f in enumerate(mean_abs.index[:4], 1):
        plt.figure(); shap.plots.scatter(exp[:, PRETTY.get(f, f)], color=exp, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / f"fig7_shap_dependence_{i}_{f}.png", dpi=300, bbox_inches="tight"); plt.close("all")
    risk = exp.values.sum(1)
    for tag, i in [("high_risk", int(np.argmax(risk))), ("low_risk", int(np.argmin(risk)))]:
        plt.figure(); shap.plots.waterfall(exp[i], max_display=12, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / f"fig8_shap_waterfall_{tag}.png", dpi=300, bbox_inches="tight"); plt.close("all")

    # race-free, clinically obtainable: no labs, income, education, or derived BMI
    not_clin = set(RACE_FEATURES) | set(LAB_FEATURES) | {"pir", "education", "bmi"}
    short = [f for f in mean_abs.index if f not in not_clin][:SHORT_FORM_K]
    sf = develop(short, best_model, best_strategy, best_params)
    res_sf = bootstrap_metrics(y_te, sf["p"], sf["thr"]).assign(
        model=f"Short-form ({len(short)} features, race-free)", data="temporal_test")
    a_full, a_sf = roc_auc_score(y_te, p_te), roc_auc_score(y_te, sf["p"])
    a_ost, a_lr3 = roc_auc_score(y_te, M[OST_NAME]["p"]), roc_auc_score(y_te, M[LR3_NAME]["p"])
    *_, p_sf_full = boot_auc_diff(y_te, p_te, sf["p"], return_p=True)
    *_, p_sf_ost = boot_auc_diff(y_te, sf["p"], M[OST_NAME]["p"], return_p=True)
    *_, p_sf_lr3 = boot_auc_diff(y_te, sf["p"], M[LR3_NAME]["p"], return_p=True)
    sf_d = {k: boot_auc_diff(y_te, sf["p"], M[k]["p"]) for k in [OST_NAME, LR3_NAME]}
    pd.concat([res_te, res_sf]).to_csv(TABLE_DIR / "table8_full_vs_short_form.csv", index=False)
    sx_rows = []
    for sx, mk in [("Female", test.female.values == 1), ("Male", test.female.values == 0)]:
        yy = y_te[mk]
        sx_rows.append({"sex": sx, "n": int(mk.sum()), "events": int(yy.sum()),
                        "AUROC_short": roc_auc_score(yy, sf["p"][mk]), "AUROC_full": roc_auc_score(yy, p_te[mk]),
                        "AUROC_LR3": roc_auc_score(yy, M[LR3_NAME]["p"][mk]),
                        "AUROC_OST": roc_auc_score(yy, M[OST_NAME]["p"][mk]),
                        "short_sens": point_metrics(yy, sf["p"][mk], sf["thr"])["Sensitivity"],
                        "short_referred_pct": point_metrics(yy, sf["p"][mk], sf["thr"])["Referred_pct"]})
    pd.DataFrame(sx_rows).to_csv(TABLE_DIR / "table8c_short_form_by_sex.csv", index=False)
    print(pd.DataFrame(sx_rows).round(3).to_string(index=False))
    pd.DataFrame([{"short_form_features": ";".join(short), "AUROC_full": a_full, "AUROC_short": a_sf,
                   "boot_p_vs_full": p_sf_full, "AUROC_OST": a_ost, "boot_p_vs_OST": p_sf_ost,
                   "AUROC_LR3": a_lr3, "boot_p_vs_LR3": p_sf_lr3,
                   "diff_vs_OST_low": sf_d[OST_NAME][1], "diff_vs_OST_high": sf_d[OST_NAME][2],
                   "diff_vs_LR3_low": sf_d[LR3_NAME][1], "diff_vs_LR3_high": sf_d[LR3_NAME][2]}]).to_csv(
        TABLE_DIR / "table8b_short_form_tests.csv", index=False)
    print(f"  short-form: {short}\n  AUROC {fmt_ci(res_sf, 'AUROC')} (vs full p={p_sf_full:.4f}; "
          f"vs OST p={p_sf_ost:.4f}; vs LR3 p={p_sf_lr3:.4f})")

    # ---- J. coefficients of the final linear models on the original scale (after recalibration)
    def coefficients(r, label):
        pipe = r["pipe"]
        if "spline" in pipe.named_steps or not isinstance(pipe.named_steps["model"], LogisticRegression):
            return None
        sc, lrm = pipe.named_steps["scale"], pipe.named_steps["model"]
        beta = lrm.coef_[0] / sc.scale_
        b0 = lrm.intercept_[0] - np.sum(lrm.coef_[0] * sc.mean_ / sc.scale_)
        a, b = r["recal"].a, r["recal"].b
        out = pd.DataFrame({"term": ["(intercept)"] + list(r["cols"]),
                            "coefficient": np.r_[a + b * b0, b * beta]})
        out["odds_ratio_per_unit"] = np.where(out.term == "(intercept)", np.nan, np.exp(out.coefficient))
        out.to_csv(TABLE_DIR / f"tableS_coefficients_{label}.csv", index=False)
        return out

    coefficients(M["Proposed"], "full_model")
    coefficients(sf, "short_form")
    lin = M["Proposed"] if best_model == "LR" else M.get("LR (tuned)")
    if lin is not None and lin is not M["Proposed"]:
        coefficients(lin, "linear_LR_all_predictors")

    # ---- K. sensitivity analyses
    print("\n[K] Sensitivity analyses ...")
    rows = []
    for out_col, label in SENSITIVITY_OUTCOMES.items():
        yd, yv = train[out_col].values, test[out_col].values
        rp = develop(feats, best_model, best_strategy, best_params, y_dev=yd)
        ro = score_model(-ost_score(train), -ost_score(test), y_dev=yd)
        r3 = develop(["age", "weight", "female"], "LR", "none", {}, y_dev=yd)
        for nm, rr in [("Proposed", rp), ("OST", ro), ("LR: age, weight, sex", r3)]:
            rows.append({"analysis": label, "model": nm, "dev_prev_pct": yd.mean() * 100,
                         "val_prev_pct": yv.mean() * 100, "AUROC": roc_auc_score(yv, rr["p"]),
                         "calib_slope": calibration_slope_intercept(yv, rr["p"])[0]})
    # excluding non-Hispanic Asian participants (identifiable from 2011 only)
    tr_a = train[train.asian == 0].reset_index(drop=True); te_a = test[test.asian == 0].reset_index(drop=True)
    ra = develop(feats, best_model, best_strategy, best_params, y_dev=tr_a[PRIMARY_OUTCOME].values, tr=tr_a, te=te_a)
    rows.append({"analysis": "Excluding non-Hispanic Asian participants", "model": "Proposed",
                 "dev_prev_pct": tr_a[PRIMARY_OUTCOME].mean() * 100, "val_prev_pct": te_a[PRIMARY_OUTCOME].mean() * 100,
                 "AUROC": roc_auc_score(te_a[PRIMARY_OUTCOME], ra["p"]),
                 "calib_slope": calibration_slope_intercept(te_a[PRIMARY_OUTCOME].values, ra["p"])[0]})
    # development on 2013-2014 only: same DXA scanner type as the validation cycle
    tr_h = train[train.cycle == "H"].reset_index(drop=True)
    rh = develop(feats, best_model, best_strategy, best_params, y_dev=tr_h[PRIMARY_OUTCOME].values, tr=tr_h)
    rows.append({"analysis": "Development on 2013-2014 only (same scanner type)", "model": "Proposed",
                 "dev_prev_pct": tr_h[PRIMARY_OUTCOME].mean() * 100, "val_prev_pct": y_te.mean() * 100,
                 "AUROC": roc_auc_score(y_te, rh["p"]), "calib_slope": calibration_slope_intercept(y_te, rh["p"])[0]})
    keep = train.cycle != "D"
    tr2 = train[keep].reset_index(drop=True)
    r2 = develop(feats, best_model, best_strategy, best_params, y_dev=tr2[PRIMARY_OUTCOME].values, tr=tr2)
    rows.append({"analysis": "Development set without 2005-2006", "model": "Proposed",
                 "dev_prev_pct": tr2[PRIMARY_OUTCOME].mean() * 100, "val_prev_pct": y_te.mean() * 100,
                 "AUROC": roc_auc_score(y_te, r2["p"]), "calib_slope": calibration_slope_intercept(y_te, r2["p"])[0]})
    # without laboratory predictors / without alcohol (definition changed in 2017-2018)
    for lab_, cols_ in [("Without laboratory predictors", [f for f in feats if f not in LAB_FEATURES]),
                        ("Without alcohol predictor", [f for f in feats if f != "alcohol_3plus"])]:
        rr = develop(cols_, best_model, best_strategy, best_params)
        rows.append({"analysis": lab_, "model": "Proposed", "dev_prev_pct": y_tr.mean() * 100,
                     "val_prev_pct": y_te.mean() * 100, "AUROC": roc_auc_score(y_te, rr["p"]),
                     "calib_slope": calibration_slope_intercept(y_te, rr["p"])[0]})
    # including participants previously told they have osteoporosis
    inc = pd.read_csv(PROC_DIR / "nhanes_osteoporosis_incl_diagnosed.csv")
    tr_i = inc[inc.cycle.isin(TRAIN_CYCLES)].reset_index(drop=True)
    te_i = inc[inc.cycle.isin(TEST_CYCLES)].reset_index(drop=True)
    yd_i, yv_i = tr_i[PRIMARY_OUTCOME].values, te_i[PRIMARY_OUTCOME].values
    for nm, rr in [("Proposed", develop(feats, best_model, best_strategy, best_params, y_dev=yd_i, tr=tr_i, te=te_i)),
                   ("OST", score_model(-ost_score(tr_i), -ost_score(te_i), y_dev=yd_i)),
                   ("LR: age, weight, sex", develop(["age", "weight", "female"], "LR", "none", {},
                                                    y_dev=yd_i, tr=tr_i, te=te_i))]:
        rows.append({"analysis": "Including previously diagnosed or treated participants", "model": nm,
                     "dev_prev_pct": yd_i.mean() * 100, "val_prev_pct": yv_i.mean() * 100,
                     "AUROC": roc_auc_score(yv_i, rr["p"]),
                     "calib_slope": calibration_slope_intercept(yv_i, rr["p"])[0]})
    sens = pd.DataFrame(rows)
    sens.to_csv(TABLE_DIR / "table9_sensitivity_analyses.csv", index=False)
    print(sens.round(3).to_string(index=False))
    print(f"\nDone. Tables -> {TABLE_DIR}\n      Figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
