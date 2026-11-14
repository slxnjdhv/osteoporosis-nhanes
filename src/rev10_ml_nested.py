"""Revision-10: fully nested IECV for the ML families (run after rev8_analyses.py).

For each held-out cycle, every single-learner family (LR-spline, SVM, RF, XGBoost, LightGBM, CatBoost)
re-selects its class-imbalance strategy (none / class weighting / SMOTE, default hyperparameters) and is
re-tuned with Optuna (40 TPE trials, same search spaces as the main analysis) on the four development
cycles only. As in the main analysis, default hyperparameters are kept if tuning does not improve the
cross-validated AUROC. Models are Platt-recalibrated on out-of-fold development predictions and compared
with the full model (penalty re-tuned within the same development cycles, from rev8).

Usage:  python rev10_ml_nested.py D      (one held-out cycle; run D E F H J, possibly in parallel)
        python rev10_ml_nested.py pool   (combine, pool with DL + HKSJ, equivalence at +/-0.02)
Outputs: outputs/tables/rev10_ml_nested_{cycle}.json, rev10_ml_nested_by_cycle.csv,
         rev10_ml_nested_pooled.csv, rev10_ml_nested_predictions.csv
"""
from __future__ import annotations

import json
import sys

import config
config.N_JOBS = 1          # one thread per model; parallelism comes from the CV folds below

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from analysis import fit_recalibrator, make_pipeline
from config import CV_FOLDS, DATASET_CSV, OPTUNA_TRIALS, PRIMARY_OUTCOME, SEED, TABLE_DIR, TEST_CYCLES, TRAIN_CYCLES

optuna.logging.set_verbosity(optuna.logging.WARNING)
CV_JOBS = 2
N_BOOT, MARGIN = 1000, 0.02
CYC = ["D", "E", "F", "H", "J"]
FAMILIES = ["LR-spline", "SVM", "RF", "XGBoost", "LightGBM", "CatBoost"]
STRATEGIES = ["none", "class_weight", "smote"]
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
FEATS = json.load(open(TABLE_DIR / "final_model_config.json"))["features"]


def space(t, m):
    if m == "LR-spline":
        return {"C": t.suggest_float("C", 1e-3, 100, log=True)}
    if m == "SVM":
        return {"C": t.suggest_float("C", 1e-2, 100, log=True), "gamma": t.suggest_float("gamma", 1e-5, 1, log=True)}
    if m == "RF":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 1000, step=100),
                "max_depth": t.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 60),
                "max_features": t.suggest_float("max_features", 0.05, 1.0)}
    if m == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 2000, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                "max_depth": t.suggest_int("max_depth", 1, 8),
                "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if m == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 2000, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                "num_leaves": t.suggest_int("num_leaves", 2, 64),
                "min_child_samples": t.suggest_int("min_child_samples", 1, 200),
                "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if m == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 200, 2000, step=100),
                "depth": t.suggest_int("depth", 1, 8),
                "learning_rate": t.suggest_float("learning_rate", 0.002, 0.3, log=True),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}
    raise ValueError(m)


def run_cycle(c):
    df = pd.read_csv(DATASET_CSV)
    df = df[df.cycle.isin(TRAIN_CYCLES + TEST_CYCLES)].reset_index(drop=True)
    tr = df[df.cycle != c].reset_index(drop=True); te = df[df.cycle == c].reset_index(drop=True)
    y = tr[PRIMARY_OUTCOME].values; pw = (y == 0).sum() / y.sum()
    X = tr[FEATS]
    out = {"cycle": c, "families": {}}
    for m in FAMILIES:
        dflt = {s: cross_val_score(make_pipeline(m, s, pw, {}, cols=FEATS), X, y, cv=cv, scoring="roc_auc",
                                   n_jobs=CV_JOBS).mean() for s in STRATEGIES}
        s = max(dflt, key=dflt.get)

        def obj(t):
            return cross_val_score(make_pipeline(m, s, pw, space(t, m), cols=FEATS), X, y, cv=cv,
                                   scoring="roc_auc", n_jobs=CV_JOBS).mean()
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(obj, n_trials=OPTUNA_TRIALS)
        prm = st.best_params if st.best_value > dflt[s] else {}
        pipe = make_pipeline(m, s, pw, prm, cols=FEATS)
        raw = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=CV_JOBS)[:, 1]
        rc = fit_recalibrator(y, raw)
        pipe.fit(X, y)
        p = rc(pipe.predict_proba(te[FEATS])[:, 1])
        out["families"][m] = {"strategy": s, "default_cv": dflt, "tuned_cv": st.best_value, "params": prm,
                              "auroc": roc_auc_score(te[PRIMARY_OUTCOME].values, p), "pred": p.tolist()}
        print(c, m, s, round(max(st.best_value, dflt[s]), 4), "->", round(out["families"][m]["auroc"], 4), prm,
              flush=True)
        json.dump(out, open(TABLE_DIR / f"rev10_ml_nested_{c}.json", "w"))
    return out


def boot_diff(y, p1, p2, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), float(np.std(d, ddof=1))


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg != "pool":
        run_cycle(arg); sys.exit()
    from scipy import stats

    def pool(est, se):
        est, se = np.asarray(est, float), np.asarray(se, float); k = len(est)
        w = 1 / se ** 2; fe = np.sum(w * est) / np.sum(w); q = np.sum(w * (est - fe) ** 2)
        tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        wr = 1 / (se ** 2 + tau2); mu = np.sum(wr * est) / np.sum(wr); se_dl = np.sqrt(1 / np.sum(wr))
        se_h = np.sqrt(max(np.sum(wr * (est - mu) ** 2) / (k - 1), 1.0) * se_dl ** 2)
        t95, t90 = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.95, k - 1)
        return {"pooled": mu, "hksj_low": mu - t95 * se_h, "hksj_high": mu + t95 * se_h,
                "hksj90_low": mu - t90 * se_h, "hksj90_high": mu + t90 * se_h,
                "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}

    pr = pd.read_csv(TABLE_DIR / "rev8_iecv_predictions.csv")
    rows, preds = [], []
    for c in CYC:
        r_ = json.load(open(TABLE_DIR / f"rev10_ml_nested_{c}.json"))
        P = pr[pr.cycle == c].reset_index(drop=True); yv = P.y.values
        row = {"cycle": c, "AUROC_Full": roc_auc_score(yv, P.Full.values)}
        pc = pd.DataFrame({"cycle": c, "y": yv, "Full": P.Full.values})
        for m in FAMILIES:
            f = r_["families"][m]; p = np.array(f["pred"]); pc[m] = p
            d, s = boot_diff(yv, p, P.Full.values)
            row.update({f"strategy_{m}": f["strategy"], f"params_{m}": json.dumps(f["params"]),
                        f"AUROC_{m}": f["auroc"], f"AUROC_fixed_{m}": roc_auc_score(yv, P[m].values),
                        f"d[{m}]": d, f"se[{m}]": s})
        rows.append(row); preds.append(pc)
    by = pd.DataFrame(rows); by.to_csv(TABLE_DIR / "rev10_ml_nested_by_cycle.csv", index=False)
    pd.concat(preds).to_csv(TABLE_DIR / "rev10_ml_nested_predictions.csv", index=False)
    pooled = []
    for m in FAMILIES:
        p_ = pool(by[f"d[{m}]"], by[f"se[{m}]"])
        p_.update({"comparison": f"{m} minus Full",
                   "equivalent_pm0.02": bool(p_["hksj90_low"] > -MARGIN and p_["hksj90_high"] < MARGIN)})
        pooled.append(p_)
    pd.DataFrame(pooled).to_csv(TABLE_DIR / "rev10_ml_nested_pooled.csv", index=False)
    print(by.filter(regex="cycle|AUROC|strategy").round(3).T.to_string())
    print(pd.DataFrame(pooled).round(4).to_string(index=False))
