# Explainable, fair, temporally validated ML for osteoporosis screening (NHANES)

Extends: Zhang D et al., *J Med Syst* 2025, https://doi.org/10.1007/s10916-025-02203-1

## Run

```bash
pip install -r requirements.txt
python src/download_nhanes.py      # ~45 CDC .xpt files -> data/raw
python src/build_dataset.py        # cohort + outcome -> data/processed/nhanes_osteoporosis.csv
python src/analysis.py             # all models, tables, figures -> outputs/
```

Quick smoke test (3-fold CV, 50 bootstraps, 3 tuning trials): `QUICK=1 python src/analysis.py`

**macOS:** XGBoost/LightGBM need OpenMP. Install it with `brew install libomp`.

## Design

| Item | Choice (edit in `src/config.py`) |
|---|---|
| Cycles | 2005-06, 2007-08, 2009-10, 2013-14 (development), 2017-18 (temporal test) |
| Population | Age >= 50 with valid femoral-neck DXA |
| Outcome | Osteoporosis = femoral neck **or** lumbar spine T-score <= -2.5 (NHANES III young white female reference) |
| Predictors | Demographics, body measures, serum biochemistry, vitamin D, smoking, diabetes, comorbidities (BMD never used) |
| Missing data | Iterative (MICE-style) imputation, fitted inside each CV fold |
| Feature selection | Boruta, L1-logistic (Lasso), XGBoost gain (on development data only) |

## Outputs

| File | Paper item |
|---|---|
| `tables/flow_chart.csv` | Figure 1 participant flow |
| `tables/table1_baseline_characteristics.csv` | Table 1 |
| `tables/table2_selected_features.csv` | Table 2 |
| `tables/table3a/3b_*` | Base-paper reproduction (10-fold CV, split ratios) |
| `tables/table4_model_imbalance_comparison_cv.csv` | 7 models x 4 imbalance strategies |
| `tables/final_model_config.json` | Chosen model, strategy, tuned parameters |
| `tables/table5_final_performance_bootstrap.csv`, `table5b_delong_test.csv` | Internal + temporal performance with 95% CI, DeLong vs base paper |
| `tables/table6_fairness_subgroups.csv` | Sex / age / race subgroup performance |
| `tables/table7_shap_importance.csv` | SHAP ranking |
| `tables/table8_full_vs_short_form.csv` | Short-form screening model |
| `figures/fig3..fig8` | ROC, calibration, decision curve, SHAP beeswarm/dependence/waterfall |

## Before publishing — verify
- T-score reference means/SDs in `config.py` match the reference you cite.
- Variable recodes in `build_dataset.py` against the NHANES codebooks for every cycle.
- Report with the TRIPOD+AI checklist.
- Limitation: feature selection is done once on the development set (not nested in CV), so internal CV is slightly optimistic; the temporal test set is fully independent.
