"""Revision-9 analyses (run after rev8_analyses.py; reads its IECV predictions).

1. IECV (nested tuning, as rev8) for a clinical-only model: no laboratory tests, no race/ethnicity,
   no income or education. Pooled AUROC and differences (DerSimonian-Laird + HKSJ).
2. Cases per 1,000 at fixed referral rates on the pooled out-of-cycle predictions, with random
   tie-breaking (averaged over 200 draws), for full, clinical + laboratory (race/SES-free),
   clinical-only, age-weight-sex, sex-specific OST and OST.
3. The same in guideline strata and in the population not routinely screened (all except women >=65).
4. Survey-weighted cases per 1,000 (pooled examination weights; and 2017-2018 alone).
Outputs: outputs/tables/rev9_*.csv
"""
from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from analysis import calibration_slope_intercept, fit_recalibrator, make_pipeline
from config import (CV_FOLDS, DATASET_CSV, LAB_FEATURES, N_JOBS, OPTUNA_TRIALS, PRIMARY_OUTCOME,
                    RACE_FEATURES, SEED, TABLE_DIR, TEST_CYCLES, TRAIN_CYCLES)

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_BOOT, N_TIE = 1000, 200
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
FEATS = json.load(open(TABLE_DIR / "final_model_config.json"))["features"]
CLIN = [f for f in FEATS if f not in LAB_FEATURES + RACE_FEATURES + ["pir", "education"]]
CYC = ["D", "E", "F", "H", "J"]
df = pd.read_csv(DATASET_CSV)
df = df[df.cycle.isin(TRAIN_CYCLES + TEST_CYCLES)].reset_index(drop=True)
pr = pd.read_csv(TABLE_DIR / "rev8_iecv_predictions.csv")
meta = pd.concat([df[df.cycle == c] for c in CYC], ignore_index=True)
assert (meta[PRIMARY_OUTCOME].values == pr.y.values).all() and (meta.cycle.values == pr.cycle.values).all()
print("clinical-only predictors:", CLIN, flush=True)


def tune_C(cols, tr):
    y = tr[PRIMARY_OUTCOME].values

    def obj(t):
        pipe = make_pipeline("LR", "none", 1.0, {"C": t.suggest_float("C", 1e-3, 100, log=True)}, cols=cols)
        return cross_val_score(pipe, tr[cols], y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=OPTUNA_TRIALS)
    return st.best_params["C"]


def develop(cols, C, tr, te):
    y = tr[PRIMARY_OUTCOME].values
    pipe = make_pipeline("LR", "none", 1.0, {"C": C}, cols=cols)
    raw = cross_val_predict(pipe, tr[cols], y, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
    rc = fit_recalibrator(y, raw)
    pipe.fit(tr[cols], y)
    return rc(pipe.predict_proba(te[cols])[:, 1])


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
    se_h = np.sqrt(max(np.sum(wr * (est - mu) ** 2) / (k - 1), 1.0) * se_dl ** 2)
    t95, t90 = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.95, k - 1)
    return {"pooled": mu, "hksj_low": mu - t95 * se_h, "hksj_high": mu + t95 * se_h,
            "hksj90_low": mu - t90 * se_h, "hksj90_high": mu + t90 * se_h}


# ---------------------------------------------------------------- 1. clinical-only model in IECV
clin, rows, Cs = [], [], {}
for c in CYC:
    tr = df[df.cycle != c].reset_index(drop=True); te = df[df.cycle == c].reset_index(drop=True)
    C = tune_C(CLIN, tr); Cs[c] = C
    p = develop(CLIN, C, tr, te); clin.append(p)
    yv = te[PRIMARY_OUTCOME].values; P = pr[pr.cycle == c]
    r = {"cycle": c, "C": C, "AUROC_Clinical": roc_auc_score(yv, p),
         "slope_Clinical": calibration_slope_intercept(yv, p)[0]}
    for b in ["OST", "Age, weight, sex", "Full", "No race, income, education"]:
        d, s = boot_diff(yv, p if b != "Full" and b != "No race, income, education" else P[b].values,
                         P[b].values if b != "Full" and b != "No race, income, education" else p)
        r[f"d[{b}]"], r[f"se[{b}]"] = d, s
    rows.append(r); print(c, round(C, 3), round(r["AUROC_Clinical"], 3), flush=True)
pr["Clinical"] = np.concatenate(clin)
pr.to_csv(TABLE_DIR / "rev9_iecv_predictions.csv", index=False)
by = pd.DataFrame(rows); by.to_csv(TABLE_DIR / "rev9_clinical_iecv_by_cycle.csv", index=False)
lab = {"OST": "Clinical minus OST", "Age, weight, sex": "Clinical minus Age, weight, sex",
       "Full": "Full minus Clinical", "No race, income, education": "Clinical + laboratory minus Clinical"}
pooled = [{**pool(by[f"d[{k}]"], by[f"se[{k}]"]), "comparison": v} for k, v in lab.items()]
a = by.AUROC_Clinical.values
ev = pr.groupby("cycle", sort=False).y.agg(["sum", "count"]).loc[CYC]
n1, n0 = ev["sum"].values, (ev["count"] - ev["sum"]).values
q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / (n1 * n0))
pooled.append({**pool(a, se), "comparison": "AUROC_Clinical"})
pd.DataFrame(pooled).to_csv(TABLE_DIR / "rev9_clinical_iecv_pooled.csv", index=False)
print(pd.DataFrame(pooled).round(4).to_string(index=False), flush=True)

# ---------------------------------------------------------------- 2-4. cases per 1,000 (random tie-breaking)
TOOLS = ["Full", "No race, income, education", "Clinical", "Age, weight, sex", "OST, sex-specific", "OST"]
pr["female"] = meta.female.fillna(0).values; pr["age"] = meta.age.values
pr["w"] = meta.wt_pooled.values; pr["w_mec"] = meta.wt_mec.values
cyc_idx = {c: np.where(pr.cycle.values == c)[0] for c in CYC}


def detected(ix, tool, q, rng, weight=None):
    """Cases (or weighted cases) detected when the top q share (by count or weight) within each
    cycle is referred; ties broken at random."""
    tot = 0.0
    for c in CYC:
        g = ix[pr.cycle.values[ix] == c]
        if len(g) == 0:
            continue
        s = pr[tool].values[g]; u = rng.random(len(g))
        order = np.lexsort((u, -s))
        yy = pr.y.values[g][order]
        if weight is None:
            k = int(round(q * len(g))); tot += yy[:k].sum()
        else:
            ww = pr[weight].values[g][order]; cw = np.cumsum(ww) / ww.sum()
            tot += (yy * ww)[cw <= q].sum() / ww.sum() * ww.sum()
    return tot


def per1000(ix, tool, q, rng, weight=None):
    if weight is None:
        return detected(ix, tool, q, rng) / len(ix) * 1000
    return detected(ix, tool, q, rng, weight) / pr[weight].values[ix].sum() * 1000


rates = [0.15, 0.20, 0.25, 0.30]


def table(ix, name, weight=None, rates_=rates):
    rng0 = np.random.default_rng(SEED)
    out = []
    boot_ix = []
    rb = np.random.default_rng(SEED + 7)
    for _ in range(N_BOOT):
        boot_ix.append(np.concatenate([rb.choice(ix[pr.cycle.values[ix] == c], (pr.cycle.values[ix] == c).sum())
                                       for c in CYC if (pr.cycle.values[ix] == c).any()]))

    def one(bix, s):
        r_ = np.random.default_rng(s)
        return [[per1000(bix, t, q, r_, weight) for t in TOOLS] for q in rates_]
    B = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(b, SEED + i) for i, b in enumerate(boot_ix)))
    yw = pr.y.values[ix] if weight is None else pr.y.values[ix] * pr[weight].values[ix]
    tot = yw.sum() / (len(ix) if weight is None else pr[weight].values[ix].sum()) * 1000
    for qi, q in enumerate(rates_):
        r = {"population": name, "weighted": weight or "none", "n": len(ix), "cases": int(pr.y.values[ix].sum()),
             "referred_pct": q * 100, "cases_per_1000_total": tot}
        est = {t: np.mean([per1000(ix, t, q, rng0, weight) for _ in range(N_TIE)]) for t in TOOLS}
        for ti, t in enumerate(TOOLS):
            r[f"per1000_{t}"] = est[t]
        for ti, t in enumerate(TOOLS[1:], start=1):
            bd = B[:, qi, 0] - B[:, qi, ti]
            r[f"full_minus_{t}"] = est["Full"] - est[t]
            r[f"full_minus_{t}_low"], r[f"full_minus_{t}_high"] = np.quantile(bd, [0.025, 0.975])
            r[f"full_minus_{t}_relpct"] = 100 * (est["Full"] / est[t] - 1)
        out.append(r)
    return out


allix = np.arange(len(pr))
fem, age = pr.female.values == 1, pr.age.values
strata = {"All": allix,
          "Not routinely screened (all except women >=65)": allix[~(fem & (age >= 65))],
          "Women 50-64": allix[fem & (age < 65)], "Women >=65": allix[fem & (age >= 65)],
          "Men 50-69": allix[~fem & (age < 70)], "Men >=70": allix[~fem & (age >= 70)]}
res = []
for nm, ix in strata.items():
    res += table(ix, nm, rates_=rates if nm in ("All", "Not routinely screened (all except women >=65)") else [0.20])
    print("done", nm, flush=True)
res += table(allix, "All, survey-weighted (pooled weights)", weight="w")
jix = allix[pr.cycle.values == "J"]
res += table(jix, "2017-2018, survey-weighted", weight="w_mec")
res += table(jix, "2017-2018, unweighted")
pd.DataFrame(res).to_csv(TABLE_DIR / "rev9_cases_per_1000.csv", index=False)

# AUROCs in strata (pooled out-of-cycle predictions)
srows = []
for nm, ix in strata.items():
    yy = pr.y.values[ix]
    r = {"population": nm, "n": len(ix), "cases": int(yy.sum())}
    for t in TOOLS + ["SVM"]:
        r[f"AUROC_{t}"] = roc_auc_score(yy, pr[t].values[ix])
    rng = np.random.default_rng(SEED); d = []
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(ix[pr.cycle.values[ix] == c], (pr.cycle.values[ix] == c).sum()) for c in CYC])
        d.append(roc_auc_score(pr.y.values[b], pr.Full.values[b]) - roc_auc_score(pr.y.values[b], pr.OST.values[b]))
    r["full_minus_OST"] = r["AUROC_Full"] - r["AUROC_OST"]
    r["full_minus_OST_low"], r["full_minus_OST_high"] = np.quantile(d, [0.025, 0.975])
    srows.append(r)
pd.DataFrame(srows).to_csv(TABLE_DIR / "rev9_strata_iecv.csv", index=False)
print(pd.DataFrame(srows).round(3).T.to_string())
print(pd.DataFrame(res).round(2).T.to_string())
print("Done.")
