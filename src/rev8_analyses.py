"""Revision-8 analyses (run after analysis.py; reads its outputs).

1. Internal-external cross-validation (IECV) by cycle with the penalty of every LR model
   re-tuned inside each fold (nested; Optuna, same budget as analysis.py), for the full model,
   the model without race, and the model without race, income and education; plus the age-weight-
   sex model, OST, sex-specific OST and the tuned ML families (hyperparameters fixed at their
   development-set values; stacking omitted for run time).
2. Pooling by DerSimonian-Laird with Hartung-Knapp-Sidik-Jonkman (HKSJ) CIs; equivalence
   (TOST, +/-0.02) of the full model vs OST and vs age-weight-sex, and of each ML family vs LR.
3. Cases detected per 1,000 screened at fixed referral rates on the pooled out-of-cycle
   predictions (439 cases), referral within each cycle, bootstrap stratified by cycle.
4. Refit bootstrap with 1,000 replicates (hyperparameters fixed).
Outputs: outputs/tables/rev8_*.csv / .json
"""
from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from analysis import (calibration_slope_intercept, crossfit_recal, fit_recalibrator, make_pipeline,
                      youden_threshold)
from config import (CV_FOLDS, DATASET_CSV, N_JOBS, OPTUNA_TRIALS, PRIMARY_OUTCOME, RACE_FEATURES,
                    SEED, TABLE_DIR, TEST_CYCLES, TRAIN_CYCLES)

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_BOOT = 1000
MARGIN = 0.02
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
cfg = json.load(open(TABLE_DIR / "final_model_config.json"))
FEATS = cfg["features"]
NORACE = [f for f in FEATS if f not in RACE_FEATURES]
NORACE_SES = [f for f in NORACE if f not in ("pir", "education")]
LR3 = ["age", "weight", "female"]
tuned = pd.read_csv(TABLE_DIR / "table4b_tuned_families_cv.csv")
ML = {}
for _, r in tuned.iterrows():
    if r.model in ("LR", "Stacking"):
        continue
    ML[r.model] = (r.strategy, json.loads(r.params))

df = pd.read_csv(DATASET_CSV)
df = df[df.cycle.isin(TRAIN_CYCLES + TEST_CYCLES)].reset_index(drop=True)
CYC = ["D", "E", "F", "H", "J"]


def develop(cols, model, strategy, params, tr, te):
    y = tr[PRIMARY_OUTCOME].values
    pw = (y == 0).sum() / max(y.sum(), 1)
    pipe = make_pipeline(model, strategy, pw, params, cols=cols)
    raw = cross_val_predict(pipe, tr[cols], y, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
    rc = fit_recalibrator(y, raw)
    pipe.fit(tr[cols], y)
    return rc(pipe.predict_proba(te[cols])[:, 1])


def tune_C(cols, tr):
    y = tr[PRIMARY_OUTCOME].values

    def obj(t):
        pipe = make_pipeline("LR", "none", 1.0, {"C": t.suggest_float("C", 1e-3, 100, log=True)}, cols=cols)
        return cross_val_score(pipe, tr[cols], y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=OPTUNA_TRIALS)
    return st.best_params["C"]


def ost(d, med):
    return 0.2 * (d["weight"].fillna(med["weight"]) - d["age"].fillna(med["age"])).values


def score_prob(s_tr, s_te, y):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s_tr, y)
    return lr.predict_proba(s_te)[:, 1]


def boot_diff(y, p1, p2, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), float(np.std(d, ddof=1))


def pool(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float); k = len(est)
    w = 1 / se ** 2; fe = np.sum(w * est) / np.sum(w); q = np.sum(w * (est - fe) ** 2)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    wr = 1 / (se ** 2 + tau2); mu = np.sum(wr * est) / np.sum(wr); se_dl = np.sqrt(1 / np.sum(wr))
    q_h = np.sum(wr * (est - mu) ** 2) / (k - 1)
    se_h = np.sqrt(max(q_h, 1.0) * se_dl ** 2)       # HKSJ with the ad hoc floor (variance not below DL)
    t95, t90 = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.95, k - 1)
    pi = np.sqrt(tau2 + se_dl ** 2) * stats.t.ppf(0.975, k - 2)
    return {"pooled": mu, "dl_low": mu - 1.96 * se_dl, "dl_high": mu + 1.96 * se_dl,
            "hksj_low": mu - t95 * se_h, "hksj_high": mu + t95 * se_h,
            "hksj90_low": mu - t90 * se_h, "hksj90_high": mu + t90 * se_h,
            "tau2": tau2, "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
            "pi_low": mu - pi, "pi_high": mu + pi}


# ------------------------------------------------------------------ 1. nested IECV
preds, rows, Cs = [], [], {}
for c in CYC:
    tr = df[df.cycle != c].reset_index(drop=True); te = df[df.cycle == c].reset_index(drop=True)
    y, yv = tr[PRIMARY_OUTCOME].values, te[PRIMARY_OUTCOME].values
    med = tr[["weight", "age"]].median()
    P = {}
    for nm, cols in [("Full", FEATS), ("No race", NORACE), ("No race, income, education", NORACE_SES)]:
        C = tune_C(cols, tr); Cs[f"{c}:{nm}"] = C
        P[nm] = develop(cols, "LR", "none", {"C": C}, tr, te)
    P["Full (C fixed)"] = develop(FEATS, "LR", "none", cfg["params"], tr, te)
    P["Age, weight, sex"] = develop(LR3, "LR", "none", {}, tr, te)
    o_tr, o_te = ost(tr, med), ost(te, med)
    P["OST"] = score_prob(-o_tr.reshape(-1, 1), -o_te.reshape(-1, 1), y)
    f_tr, f_te = tr.female.fillna(0).values, te.female.fillna(0).values
    P["OST, sex-specific"] = score_prob(np.column_stack([o_tr, f_tr, o_tr * f_tr]),
                                        np.column_stack([o_te, f_te, o_te * f_te]), y)
    for m, (stg, prm) in ML.items():
        P[m] = develop(FEATS, m, stg, prm, tr, te)
    out = pd.DataFrame({k: v for k, v in P.items()}); out["y"] = yv; out["cycle"] = c
    preds.append(out)
    r = {"cycle": c, "n": len(te), "events": int(yv.sum())}
    for k, v in P.items():
        r[f"AUROC_{k}"] = roc_auc_score(yv, v)
    r["slope_Full"] = calibration_slope_intercept(yv, P["Full"])[0]
    r["O_E_Full"] = yv.mean() / P["Full"].mean()
    comps = [("Full", "OST"), ("Full", "Age, weight, sex"), ("Full", "OST, sex-specific"),
             ("No race", "OST"), ("No race", "Age, weight, sex"),
             ("No race, income, education", "OST"), ("No race, income, education", "Age, weight, sex"),
             ("OST, sex-specific", "OST")] + [(m, "Full") for m in ML]
    for a, b in comps:
        d, s = boot_diff(yv, P[a], P[b]); r[f"d[{a}|{b}]"] = d; r[f"se[{a}|{b}]"] = s
    rows.append(r)
    print(c, {k: round(v, 3) for k, v in r.items() if k.startswith("AUROC")}, flush=True)

by = pd.DataFrame(rows); by.to_csv(TABLE_DIR / "rev8_iecv_nested_by_cycle.csv", index=False)
pr = pd.concat(preds, ignore_index=True); pr.to_csv(TABLE_DIR / "rev8_iecv_predictions.csv", index=False)
json.dump(Cs, open(TABLE_DIR / "rev8_iecv_tuned_C.json", "w"), indent=1)

pooled = []
for col in [c for c in by.columns if c.startswith("d[")]:
    key = col[2:-1]
    p_ = pool(by[col], by["se" + col[1:]])
    a, b = key.split("|")
    p_.update({"comparison": f"{a} minus {b}",
               "equivalent_pm0.02": bool(p_["hksj90_low"] > -MARGIN and p_["hksj90_high"] < MARGIN)})
    pooled.append(p_)
for col in [c for c in by.columns if c.startswith("AUROC_")]:
    a = by[col].values; n1 = by.events.values; n0 = (by.n - by.events).values
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / (n1 * n0))
    p_ = pool(a, se); p_["comparison"] = col; pooled.append(p_)
pd.DataFrame(pooled).to_csv(TABLE_DIR / "rev8_iecv_nested_pooled.csv", index=False)
print(pd.DataFrame(pooled)[["comparison", "pooled", "dl_low", "dl_high", "hksj_low", "hksj_high",
                            "hksj90_low", "hksj90_high", "I2"]].round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 3. cases per 1,000 on pooled IECV predictions
TOOLS = ["Full", "No race, income, education", "Age, weight, sex", "OST, sex-specific", "OST"]


def detected(d, tool, q):
    tot = 0
    for _, g in d.groupby("cycle"):
        k = int(round(q * len(g)))
        tot += g.y.values[np.argsort(-g[tool].values, kind="stable")[:k]].sum()
    return tot


rates = [0.15, 0.20, 0.25, 0.30]
groups = {c: np.where(pr.cycle.values == c)[0] for c in CYC}
rng = np.random.default_rng(SEED)
bidx = [np.concatenate([rng.choice(ix, len(ix)) for ix in groups.values()]) for _ in range(N_BOOT)]


def one_boot(ix):
    d = pr.iloc[ix]
    return [[detected(d, t, q) for t in TOOLS] for q in rates]


B = np.array(Parallel(n_jobs=N_JOBS)(delayed(one_boot)(ix) for ix in bidx))   # boot x rate x tool
N = len(pr); rows_c = []
for qi, q in enumerate(rates):
    r = {"referred_pct": q * 100, "cases_total": int(pr.y.sum()), "cases_per_1000_total": pr.y.sum() / N * 1000}
    for ti, t in enumerate(TOOLS):
        det = detected(pr, t, q)
        r[f"per1000_{t}"] = det / N * 1000; r[f"sens_{t}"] = det / pr.y.sum()
    for ti, t in enumerate(TOOLS[1:], start=1):
        bd = (B[:, qi, 0] - B[:, qi, ti]) / N * 1000
        r[f"full_minus_{t}"] = r["per1000_Full"] - r[f"per1000_{t}"]
        r[f"full_minus_{t}_low"], r[f"full_minus_{t}_high"] = np.quantile(bd, [0.025, 0.975])
        r[f"full_minus_{t}_relpct"] = 100 * (r["per1000_Full"] / r[f"per1000_{t}"] - 1)
        r[f"scans_per_extra_case_{t}"] = (1000 * q) / max(r[f"full_minus_{t}"], 1e-9)
    rows_c.append(r)
pd.DataFrame(rows_c).to_csv(TABLE_DIR / "rev8_iecv_cases_per_1000.csv", index=False)
print(pd.DataFrame(rows_c).round(2).T.to_string(), flush=True)

# ------------------------------------------------------------------ 4. refit bootstrap, 1,000 replicates
train = df[df.cycle.isin(TRAIN_CYCLES)].reset_index(drop=True)
test = df[df.cycle.isin(TEST_CYCLES)].reset_index(drop=True)
y_tr, y_te = train[PRIMARY_OUTCOME].values, test[PRIMARY_OUTCOME].values
med = train[["weight", "age"]].median(); o_te = -ost(test, med)


def refit(s):
    rg = np.random.default_rng(s)
    i = rg.integers(0, len(train), len(train)); j = rg.integers(0, len(test), len(test))
    trb, yb = train.iloc[i], y_tr[i]
    pf = make_pipeline("LR", "none", 1.0, cfg["params"], cols=FEATS).fit(trb[FEATS], yb)
    p3 = make_pipeline("LR", "none", 1.0, {}, cols=LR3).fit(trb[LR3], yb)
    yv = y_te[j]
    a_f = roc_auc_score(yv, pf.predict_proba(test.iloc[j][FEATS])[:, 1])
    a_3 = roc_auc_score(yv, p3.predict_proba(test.iloc[j][LR3])[:, 1])
    return a_f, a_3, roc_auc_score(yv, o_te[j])


rb = np.array(Parallel(n_jobs=N_JOBS)(delayed(refit)(SEED + 1 + k) for k in range(N_BOOT)))
q = lambda v, a: [float(np.quantile(v, a[0])), float(np.quantile(v, a[1]))]
rb_out = {"n_boot": len(rb), "AUROC_full_95": q(rb[:, 0], (0.025, 0.975)),
          "diff_vs_OST_95": q(rb[:, 0] - rb[:, 2], (0.025, 0.975)), "diff_vs_OST_90": q(rb[:, 0] - rb[:, 2], (0.05, 0.95)),
          "diff_vs_LR3_95": q(rb[:, 0] - rb[:, 1], (0.025, 0.975)), "diff_vs_LR3_90": q(rb[:, 0] - rb[:, 1], (0.05, 0.95)),
          "p_diff_vs_OST_le0": float((rb[:, 0] - rb[:, 2] <= 0).mean()),
          "p_diff_vs_LR3_le0": float((rb[:, 0] - rb[:, 1] <= 0).mean())}
json.dump(rb_out, open(TABLE_DIR / "rev8_refit_bootstrap_1000.json", "w"), indent=1)
print(rb_out)
print("Done.")
