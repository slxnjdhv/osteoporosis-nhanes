"""Supplement to analysis.py: a usable linear version of the race-free short form.

The selected model (LR-spline) cannot be written as one simple equation, so for TRIPOD+AI
item 22 we also fit plain logistic regression on the short-form predictors, report its
coefficients on the original scale (after Platt recalibration) and its temporal validation.
A 9-predictor variant without alkaline phosphatase (no blood test) is also reported.
Run after analysis.py:  python src/short_form_linear.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from analysis import (bootstrap_metrics, calibration_slope_intercept, delong_test, fit_recalibrator,
                      make_pipeline, youden_threshold)
from config import (CV_FOLDS, DATASET_CSV, PRIMARY_OUTCOME, SEED, TABLE_DIR, TEST_CYCLES,
                    TRAIN_CYCLES)

df = pd.read_csv(DATASET_CSV)
train = df[df.cycle.isin(TRAIN_CYCLES)].reset_index(drop=True)
test = df[df.cycle.isin(TEST_CYCLES)].reset_index(drop=True)
y_tr, y_te = train[PRIMARY_OUTCOME].values, test[PRIMARY_OUTCOME].values
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)

short = pd.read_csv(TABLE_DIR / "table8b_short_form_tests.csv").short_form_features[0].split(";")
variants = {"short_form_linear": short,
            "short_form_linear_no_lab": [f for f in short if f != "alk_phos"]}

# OST probability (same recalibration as analysis.py) for comparison
med = train[["weight", "age"]].median()
ost = lambda d: 0.2 * (d["weight"].fillna(med["weight"]) - d["age"].fillna(med["age"])).values
ost_lr = LogisticRegression(C=1e6, max_iter=1000).fit(ost(train).reshape(-1, 1), y_tr)
p_ost = ost_lr.predict_proba(ost(test).reshape(-1, 1))[:, 1]

summary, perf = [], []
for label, cols in variants.items():
    pipe = make_pipeline("LR", "none", 1.0, {}, cols=cols)
    raw = cross_val_predict(pipe, train[cols], y_tr, cv=cv, method="predict_proba")[:, 1]
    recal = fit_recalibrator(y_tr, raw)
    thr = youden_threshold(y_tr, recal(raw))
    pipe.fit(train[cols], y_tr)
    p = recal(pipe.predict_proba(test[cols])[:, 1])

    sc, lrm = pipe.named_steps["scale"], pipe.named_steps["model"]
    beta = lrm.coef_[0] / sc.scale_
    b0 = lrm.intercept_[0] - np.sum(lrm.coef_[0] * sc.mean_ / sc.scale_)
    coef = pd.DataFrame({"term": ["(intercept)"] + cols,
                         "coefficient": np.r_[recal.a + recal.b * b0, recal.b * beta]})
    coef["odds_ratio_per_unit"] = np.where(coef.term == "(intercept)", np.nan, np.exp(coef.coefficient))
    coef.to_csv(TABLE_DIR / f"tableS_coefficients_{label}.csv", index=False)

    # check: the equation reproduces the pipeline's predictions (imputation aside)
    Xc = test[cols].dropna()
    lp = coef.coefficient[0] + Xc.values @ coef.coefficient[1:].values
    assert np.allclose(1 / (1 + np.exp(-lp)), recal(pipe.predict_proba(Xc)[:, 1]), atol=1e-6)

    _, _, p_vs_ost = delong_test(y_te, p, p_ost)
    slope, intercept = calibration_slope_intercept(y_te, p)
    b = bootstrap_metrics(y_te, p, thr).assign(model=label, data="temporal_test")
    perf.append(b)
    summary.append({"model": label, "predictors": ";".join(cols), "threshold": thr,
                    "AUROC": roc_auc_score(y_te, p), "AUPRC": average_precision_score(y_te, p),
                    "calib_slope": slope, "calib_intercept": intercept, "DeLong_p_vs_OST": p_vs_ost})
    print(label, json.dumps(summary[-1], indent=1, default=float))

pd.concat(perf).to_csv(TABLE_DIR / "tableS_short_form_linear_performance.csv", index=False)
pd.DataFrame(summary).to_csv(TABLE_DIR / "tableS_short_form_linear_summary.csv", index=False)
