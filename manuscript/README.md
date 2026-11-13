# Manuscript (LaTeX)

`main.tex` contains the full paper (revision 6): title page and declarations, structured abstract,
Introduction, Methods, Results (Tables 1-9, Figures 1-5), Discussion (including Implementation
considerations), Conclusion, references, and supplementary Tables S1-S17 (including the record of
every analysis version and the TRIPOD+AI checklist) and Figures S1-S3.
Earlier versions are kept as `main_v1_before_fixes.tex`, `main_v2_before_rev3.tex`,
`main_v3_before_rev4.tex`, `main_v4_before_rev5.tex` and `main_v5_before_rev6.tex`.

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
- Fill in the bracketed placeholders: authors, affiliations, funding, code link, contributions, word count.
- Journal of Medical Systems uses the Springer Nature LaTeX template (`sn-jnl`). Text, tables and
  `thebibliography` can be pasted into it unchanged; switch the reference style if the journal asks.
