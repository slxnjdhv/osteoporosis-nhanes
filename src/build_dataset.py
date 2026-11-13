"""Step 2: merge NHANES files, derive the osteoporosis outcome and predictors,
apply inclusion criteria, and save one analysis-ready CSV plus a flow-chart log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (AGE_CAP, CYCLES, DATASET_CSV, FEATURES, FN_REF_MEAN, FN_REF_SD, LS_REF_MEAN,
                    LS_REF_SD, MIN_AGE, MIN_VERTEBRAE, PROC_DIR, RAW_DIR, T_SCORE_CUTOFF, TABLE_DIR,
                    TH_REF_MEAN, TH_REF_SD)

# NHANES variable -> analysis name
RENAME = {
    "RIDAGEYR": "age", "INDFMPIR": "pir", "DMDEDUC2": "education",
    "BMXBMI": "bmi", "BMXWT": "weight", "BMXHT": "height", "BMXWAIST": "waist",
    "LBXSCA": "calcium", "LBXSPH": "phosphorus", "LBXSAPSI": "alk_phos",
    "LBXSCR": "creatinine", "LBXSAL": "albumin", "LBXSUA": "uric_acid",
    "LBXSGL": "glucose", "LBXSTP": "total_protein", "LBXSCH": "cholesterol",
    "DXXNKBMD": "fn_bmd", "DXXOFBMD": "th_bmd",
}


def read_xpt(name: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{name}.xpt"
    if not path.exists():
        return None
    df = pd.read_sas(path, format="xport")
    # SAS transport stores 0 as a tiny float (e.g. 5.4e-79): restore exact zeros
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    df["SEQN"] = df["SEQN"].astype(int)
    return df


def yes_no(s: pd.Series) -> pd.Series:
    """NHANES 1=yes, 2=no, 7/9=refused/don't know -> 1/0/NaN."""
    return s.map({1: 1.0, 2: 0.0})


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


BONE_DRUGS = ["ALENDRONATE", "RISEDRONATE", "IBANDRONATE", "ZOLEDRONIC", "ETIDRONATE",
              "DENOSUMAB", "RALOXIFENE", "TERIPARATIDE", "ABALOPARATIDE", "CALCITONIN",
              "ROMOSOZUMAB", "BAZEDOXIFENE"]


def bone_drug_users(suffix: str) -> set:
    """SEQNs reporting current use of an osteoporosis medication (RXQ_RX, past 30 days)."""
    path = RAW_DIR / f"RXQ_RX_{suffix}.xpt"
    if not path.exists():
        return set()
    rx = pd.read_sas(path, format="xport", encoding="latin1")
    name = rx["RXDDRUG"].fillna("").astype(str).str.upper()
    hit = name.apply(lambda n: any(d in n for d in BONE_DRUGS))
    return set(rx.loc[hit, "SEQN"].astype(int))


def build_cycle(suffix: str) -> pd.DataFrame | None:
    demo = read_xpt(f"DEMO_{suffix}")
    if demo is None:
        print(f"[warn] DEMO_{suffix} missing - cycle skipped")
        return None
    df = demo
    for comp in ["BMX", "DXXFEM", "DXXSPN", "BIOPRO", "VID", "SMQ", "DIQ", "MCQ", "OSQ", "RHQ", "ALQ"]:
        part = read_xpt(f"{comp}_{suffix}")
        if part is not None:
            df = df.merge(part, on="SEQN", how="left", suffixes=("", f"_{comp}"))

    out = pd.DataFrame({"SEQN": df["SEQN"], "cycle": suffix,
                        "cycle_years": CYCLES[suffix]})
    for src, dst in RENAME.items():
        out[dst] = col(df, src)
    # 25(OH)D: LBXVIDMS (2007+) or LBDVIDMS (2005-06, LC-MS/MS-equivalent revised values)
    out["vitamin_d"] = col(df, "LBXVIDMS").fillna(col(df, "LBDVIDMS"))
    # Harmonise age top-coding (85 in 2005-06, 80 thereafter)
    out["age"] = out["age"].clip(upper=AGE_CAP)
    # NCHS-recommended recalibration of 2005-2006 serum creatinine to the standardised method
    if suffix == "D":
        out["creatinine"] = -0.016 + 0.978 * out["creatinine"]

    # Demographics
    out["female"] = (col(df, "RIAGENDR") == 2).astype(float)
    out["education"] = out["education"].where(out["education"].between(1, 5))
    race = col(df, "RIDRETH1")  # 1 Mex-Am, 2 Other Hisp, 3 NH White, 4 NH Black, 5 Other
    out["race"] = race.map({1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
                            4: "NH Black", 5: "Other/Multi"})
    out["race_mexican"] = (race == 1).astype(float)
    out["race_other_hisp"] = (race == 2).astype(float)
    out["race_black"] = (race == 4).astype(float)
    out["race_other"] = (race == 5).astype(float)
    # Non-Hispanic Asian participants (RIDRETH3 = 6; identifiable and oversampled from 2011)
    out["asian"] = (col(df, "RIDRETH3") == 6).astype(int)

    # Smoking: never / former / current
    ever = yes_no(col(df, "SMQ020"))
    now = col(df, "SMQ040")  # 1 every day, 2 some days, 3 not at all
    out["smoke_current"] = np.where(ever.isna(), np.nan, ((ever == 1) & now.isin([1, 2])).astype(float))
    out["smoke_former"] = np.where(ever.isna(), np.nan, ((ever == 1) & (now == 3)).astype(float))

    # Conditions (self-reported, doctor told)
    out["diabetes"] = col(df, "DIQ010").map({1: 1.0, 2: 0.0, 3: 0.0})  # 3 = borderline
    out["arthritis"] = yes_no(col(df, "MCQ160A"))
    out["chd"] = yes_no(col(df, "MCQ160C"))
    out["stroke"] = yes_no(col(df, "MCQ160F"))
    out["cancer"] = yes_no(col(df, "MCQ220"))
    # Rheumatoid arthritis (type of arthritis; only for the SCORE comparator, not a predictor).
    # 2005-08 MCQ190 and 2009-10 MCQ191: 1 = rheumatoid; 2013-18 MCQ195: 2 = rheumatoid.
    ra_code = {"D": ("MCQ190", 1), "E": ("MCQ190", 1), "F": ("MCQ191", 1), "H": ("MCQ195", 2), "J": ("MCQ195", 2)}
    v_, k_ = ra_code[suffix]
    out["rheumatoid"] = (col(df, v_) == k_).astype(float)
    # number of fracture sites (hip, wrist, spine) for SCORE
    out["n_fracture_sites"] = pd.concat([col(df, v) for v in ["OSQ010A", "OSQ010B", "OSQ010C"]], axis=1).eq(1).sum(1)

    # Lumbar spine BMD. Complete scans: total L1-L4 BMD. Partial scans (some vertebrae
    # invalid): BMC-weighted BMD of the valid vertebrae if >= MIN_VERTEBRAE are valid.
    bmc = pd.concat([col(df, f"DXXL{i}BMC") for i in range(1, 5)], axis=1)
    area = pd.concat([col(df, f"DXXL{i}A") for i in range(1, 5)], axis=1)
    bmd_v = pd.concat([col(df, f"DXXL{i}BMD") for i in range(1, 5)], axis=1)
    valid = bmd_v.notna().values & bmc.notna().values & area.notna().values
    n_valid = valid.sum(axis=1)
    partial_bmd = np.where(valid, bmc.values, 0).sum(1) / np.where(valid, area.values, 0).sum(1).clip(min=1e-9)
    total = col(df, "DXXOSBMD")
    out["ls_bmd"] = np.where(total.notna(), total,
                             np.where(n_valid >= MIN_VERTEBRAE, partial_bmd, np.nan))
    out["ls_n_vertebrae"] = np.where(total.notna(), 4, n_valid)
    out["ls_partial"] = (total.isna() & (n_valid >= MIN_VERTEBRAE)).astype(int)

    # FRAX-type clinical risk factors (osteoporosis questionnaire). OSQ060/OSQ070
    # (told / treated for osteoporosis) are deliberately NOT used: outcome leakage.
    fx = pd.concat([col(df, v) for v in ["OSQ010A", "OSQ010B", "OSQ010C"]], axis=1)
    out["prior_fracture"] = np.where((fx == 1).any(axis=1), 1.0,
                                     np.where((fx == 2).any(axis=1), 0.0, np.nan))
    par = pd.concat([col(df, "OSQ170"), col(df, "OSQ200")], axis=1)
    out["parent_hip_fracture"] = np.where((par == 1).any(axis=1), 1.0,
                                          np.where((par == 2).any(axis=1), 0.0, np.nan))
    out["steroid_use"] = yes_no(col(df, "OSQ130"))
    # Previously told by a doctor they have osteoporosis: not screening candidates (excluded
    # from the primary cohort; kept for a sensitivity analysis)
    out["known_osteo"] = (col(df, "OSQ060") == 1).astype(int)
    out["bone_drug"] = out["SEQN"].isin(bone_drug_users(suffix)).astype(int)

    # Reproductive history (women); men coded 0
    female = out["female"] == 1
    out["postmenopausal"] = np.where(female, col(df, "RHQ031").map({2: 1.0, 1: 0.0}), 0.0)
    out["hormone_use"] = np.where(female, yes_no(col(df, "RHQ540")), 0.0)

    # Alcohol: average drinks/day = drinks per drinking day x drinking days per year / 365.
    # FRAX threshold = 3 or more units daily. Non-drinkers coded 0.
    drinks = col(df, "ALQ130").where(col(df, "ALQ130") < 777)
    # 2005-2014: ALQ120Q times per ALQ120U (1 week, 2 month, 3 year)
    q = col(df, "ALQ120Q").where(col(df, "ALQ120Q") < 777)
    per = col(df, "ALQ120U").map({1: 52.0, 2: 12.0, 3: 1.0})
    days_old = q * per
    # 2017-2018: ALQ121 frequency category -> approximate drinking days per year
    days_new = col(df, "ALQ121").map({0: 0, 1: 365, 2: 286, 3: 182, 4: 104, 5: 52,
                                      6: 30, 7: 12, 8: 9, 9: 4.5, 10: 1.5})
    days = days_old.fillna(days_new)
    daily = drinks * days / 365.0
    nondrinker = ((col(df, "ALQ101") == 2) | (col(df, "ALQ120Q") == 0) |
                  (col(df, "ALQ111") == 2) | (col(df, "ALQ121") == 0))
    out["alcohol_3plus"] = np.where(daily.notna(), (daily >= 3).astype(float),
                                    np.where(nondrinker, 0.0, np.nan))

    # Survey design (for weighted descriptive statistics)
    out["wt_mec"] = col(df, "WTMEC2YR")
    out["psu"] = col(df, "SDMVPSU")
    out["strata"] = col(df, "SDMVSTRA")
    return out


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = [f for s in CYCLES if (f := build_cycle(s)) is not None]
    df = pd.concat(frames, ignore_index=True)

    flow = [("All NHANES participants in selected cycles", len(df))]
    df = df[df["age"] >= MIN_AGE]
    flow.append((f"Aged >= {MIN_AGE} years", len(df)))
    # keep the aged-50+ participants without femoral DXA to compare with the cohort
    df[df["fn_bmd"].isna()].to_csv(PROC_DIR / "excluded_no_dxa.csv", index=False)
    df = df[df["fn_bmd"].notna()]
    flow.append(("With valid femoral neck DXA", len(df)))

    # Outcomes. PRIMARY: femoral neck OR total hip T <= -2.5 (WHO/ISCD reference standard,
    # measured identically in all cycles). Spine-based definitions = sensitivity analyses.
    df["fn_t"] = (df["fn_bmd"] - FN_REF_MEAN) / FN_REF_SD
    df["th_t"] = (df["th_bmd"] - TH_REF_MEAN) / TH_REF_SD
    df["ls_t"] = (df["ls_bmd"] - LS_REF_MEAN) / LS_REF_SD
    hip = (df["fn_t"] <= T_SCORE_CUTOFF) | (df["th_t"] <= T_SCORE_CUTOFF)
    complete_spine = (df["ls_n_vertebrae"] == 4) & (df["ls_partial"] == 0)
    df["osteoporosis"] = hip.astype(int)
    df["osteo_spine_complete"] = (hip | (complete_spine & (df["ls_t"] <= T_SCORE_CUTOFF))).astype(int)
    df["osteo_spine_partial"] = (hip | (df["ls_t"] <= T_SCORE_CUTOFF)).astype(int)

    # Combined MEC weight across pooled cycles (NCHS guidance: divide by number of cycles)
    df["wt_pooled"] = df["wt_mec"] / df["cycle"].nunique()

    # Drop participants missing more than half of the predictors
    frac_missing = df[FEATURES].isna().mean(axis=1)
    df = df[frac_missing <= 0.5]
    flow.append(("With <= 50% of predictors missing", len(df)))
    # Screening population: exclude those already told they have osteoporosis
    df.to_csv(PROC_DIR / "nhanes_osteoporosis_incl_diagnosed.csv", index=False)
    excl = (df.known_osteo == 1) | (df.bone_drug == 1)
    print(f"Previously diagnosed: {df.known_osteo.sum()}; osteoporosis drug users: {df.bone_drug.sum()} "
          f"(not diagnosed: {((df.bone_drug == 1) & (df.known_osteo == 0)).sum()}); excluded in total "
          f"{excl.sum()}, of whom {df[excl].osteoporosis.sum()} meet the hip outcome")
    df = df[~excl]
    flow.append(("Not diagnosed with or treated for osteoporosis (final cohort)", len(df)))

    # Fail loudly if any predictor is entirely missing in a cycle (renamed NHANES variable)
    gone = [(c, v) for c, g in df.groupby("cycle") for v in FEATURES if g[v].isna().all()]
    if gone:
        raise RuntimeError(f"Predictor entirely missing within a cycle: {gone}")
    miss = df.groupby("cycle_years")[FEATURES].apply(lambda g: g.isna().mean() * 100).T.round(1)
    miss.to_csv(TABLE_DIR / "missingness_by_cycle.csv")
    df.to_csv(DATASET_CSV, index=False)
    pd.DataFrame(flow, columns=["step", "n"]).to_csv(TABLE_DIR / "flow_chart.csv", index=False)

    print("\nParticipant flow:")
    for step, n in flow:
        print(f"  {n:>7,}  {step}")
    print("\nBy cycle:\n", df.groupby("cycle_years")[["osteoporosis", "osteo_spine_complete",
                                                     "osteo_spine_partial", "ls_partial"]].mean().round(3))
    print("\nMissing % per predictor:\n", (df[FEATURES].isna().mean() * 100).round(1).sort_values(ascending=False).head(10))
    print(f"\nSaved {DATASET_CSV}")


if __name__ == "__main__":
    main()
