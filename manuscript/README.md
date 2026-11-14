# Manuscript (LaTeX)

`main.tex` contains the full paper (revision 9): title page and declarations, structured abstract,
Introduction, Methods, Results (Tables 1-5, Figures 1-4), Discussion, Conclusion, references, and
Supplementary Methods, Supplementary Results, Tables S1-S29 (including the record of every analysis
version and the TRIPOD+AI checklist) and Figures S1-S4.
Earlier versions are kept as `main_v1_before_fixes.tex` ... `main_v8_before_rev9.tex`.

Revision 9 changes: pooled IECV difference from OST declared the primary estimand (2017-2018

Revision 9b (text only, no new analyses): corrected number slips (330-430 people per extra case; sex-specific OST -0.1 to 1.0 cases), described the fixed penalty of the three-variable model, stated the tie rule of the 2017-2018 case table, added Supplementary Table iecvclin (clinical model by held-out cycle), explained one-stage vs meta-analysed pooled AUROCs, labelled the study exploratory, and qualified the claim that the clinical items are available in health records.

Revision 9c (text only): second-review fixes - weighted case-count attribution, calibration CI, IECV-based ML claim in abstract, stale temporal-split wording, net-benefit units, imputation of binary predictors and ordinal spline term stated, cycle-to-cycle calibration discussed.

Revision 10: ML families re-tuned inside each IECV development set (imbalance strategy re-selected, 40 Optuna trials per family per fold). Run `python src/rev10_ml_nested.py D` (and E, F, H, J; they can run in parallel, about 3 hours each on 8 cores with one thread per worker, e.g. OMP_NUM_THREADS=1) after rev8_analyses.py, then `python src/rev10_ml_nested.py pool`.

Revision 10b (text only): third-review fixes - abstract and introduction wording on which families were compared, no-race IECV results reported, CI methods stated in captions, IECV-first framing (findings, comparison with previous studies, keywords, workflow figure), USPSTF 2025 recommendation cited, word count recounted.

Revision 10c (length): main text cut from about 4,460 to 3,040 words; supplement reduced from 30 tables and 4 figures to 22 and 2. Removed from the supplement (all still produced by the code in outputs/tables): default-configuration comparison, 2017-2018 case counts by tie order, re-implementation feature lists, SHAP importance table, dependence and waterfall plots, short-form equation and short-form-by-sex tables, OST by sex, and survey-weighted performance (key numbers kept in the text). No analysis changed.
reported as one fold, P values descriptive); three-way interpretation of gains (relevant / superior
but uncertain relevance / equivalent); clinical-only model (no blood tests, race or income) in IECV;
cases per 1,000 with random tie-breaking, in guideline strata, in the population not routinely
screened and with survey weights (`src/rev9_analyses.py`, run after `rev8_analyses.py`); limitation
that no untouched validation data exist (no 2021-2023 femur DXA release); new title; fixes to
tuning-table values, captions, supplement headings and wording.

Revision 8 changes: internal-external cross-validation with the penalty re-tuned inside each fold
and Hartung-Knapp-Sidik-Jonkman CIs (`src/rev8_analyses.py`); equivalence margin applied to the full
model vs OST and vs age-weight-sex (not equivalent); cases per 1,000 from all 439 IECV cases, with
relative gain and people screened per extra case; race/SES-free model; ML families in IECV; refit
bootstrap with 1,000 replicates; penalty type, untuned stacking, citation and number fixes;
supplementary items numbered in order of citation. Run `python src/rev8_analyses.py` after
`python src/analysis.py` (about 2 hours).

Revision 7 changes: main text shortened to about 3,800 words with 5 tables; internal-external
cross-validation presented as the main evidence and the testing hierarchy labelled post hoc; secondary
family reduced to four (race-free model now a sensitivity analysis); sex-specific OST; ORAI and SCORE in
women only; cases detected per 1,000 screened at fixed DXA capacity; decision curves restricted to 2-15%;
survey-weight distribution and trimming; scanner-shift sensitivity analysis; short form moved to the
supplement and not recommended; FRAX rationale; 14 new references; tuning, incremental-value, subgroup,
SHAP and short-form tables moved to the supplement.

Revision 6 changes: explicit testing hierarchy (primary: full model vs OST; secondary, Holm-adjusted:
ORAI, SCORE, age-weight-sex LR, Boruta + XGBoost, race-free model; ML vs LR as equivalence within
+/-0.02 AUROC); ORAI and SCORE benchmarks; internal-external cross-validation over the five cycles
with random-effects pooling; refit bootstrap (development + validation resampling); guideline-defined
clinical strata; 40 Optuna trials and wider search spaces for every family, with a search-space table;
SHAP on the development sample with an importance table; short form by sex; loess calibration with
bootstrap band; abstract cut to 250 words; workflow figure, 24-configuration table and operating-points
table moved to the supplement.

Revision 5 changes: cohort also excludes osteoporosis-drug users; every model family tuned with the
same Optuna budget; bootstrap P values with Holm adjustment across 11 comparisons; DXA non-response
weighting; OST by sex; operating-points table; loess calibration; non-Hispanic Asian subgroup and
same-scanner sensitivity analyses; clinically obtainable, race-free short form.

Regenerate every number: `python src/build_dataset.py`, `python src/analysis.py`,
(the full run takes several hours). Figure 1 (participant flow) and Supplementary Figure S1 (workflow) are drawn in TikZ. All other figures
are in `figures/`.

## Build option A - Overleaf (no install)
1. Go to overleaf.com -> New Project -> Upload Project.
2. Upload `manuscript.zip`.
3. Click Recompile (the default pdfLaTeX compiler works).

## Build option B - on this Mac
Install Tectonic, a single-binary LaTeX engine (needs no admin rights):

```bash
mkdir -p ~/.local/bin && curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz" | tar -xz -C ~/.local/bin
```

Then build (the first run downloads the LaTeX packages it needs):

```bash
cd "/Users/sulaxan/Claude Work/Project 2/osteoporosis_nhanes/manuscript" && ~/.local/bin/tectonic main.tex
```

Output: `main.pdf`.

## Before submission
- Title-page details (author, affiliation, ORCID, funding, contributions, code link, word count) are filled in; check the funding statement and word count against the journal's requirements.
- Journal of Medical Systems uses the Springer Nature LaTeX template (`sn-jnl`). Text, tables and
  `thebibliography` can be pasted into it unchanged; switch the reference style if the journal asks.
