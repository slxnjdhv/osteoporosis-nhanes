"""Central configuration for the NHANES osteoporosis ML study.

Edit this file to change cycles, features, reference values or run settings.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
DATASET_CSV = PROC_DIR / "nhanes_osteoporosis.csv"

# ---------------------------------------------------------------- NHANES cycles
# suffix -> cycle years. Cycles with femur DXA data.
CYCLES = {
    "D": "2005-2006",
    "E": "2007-2008",
    "F": "2009-2010",
    "H": "2013-2014",
    "J": "2017-2018",
}
TRAIN_CYCLES = ["D", "E", "F", "H"]   # development data
TEST_CYCLES = ["J"]                   # held-out temporal validation

# NHANES data files to download for each cycle (file name = COMPONENT_SUFFIX)
COMPONENTS = ["DEMO", "BMX", "DXXFEM", "DXXSPN", "BIOPRO", "VID", "SMQ", "DIQ", "MCQ", "OSQ", "RHQ", "ALQ", "RXQ_RX"]

# ---------------------------------------------------------------- outcome
MIN_AGE = 50
AGE_CAP = 80        # NHANES top-codes age at 85 (2005-06) and 80 (later cycles): harmonise
# T-score references (verified in published NHANES analyses):
#   femoral neck & total hip: NHANES III non-Hispanic white women aged 20-29 (Looker 1998)
#   lumbar spine L1-L4: Hologic white-female reference
FN_REF_MEAN, FN_REF_SD = 0.858, 0.120     # femoral neck BMD, g/cm2
TH_REF_MEAN, TH_REF_SD = 0.942, 0.122     # total hip BMD, g/cm2
LS_REF_MEAN, LS_REF_SD = 1.047, 0.110     # lumbar spine (L1-L4) BMD, g/cm2
T_SCORE_CUTOFF = -2.5
MIN_VERTEBRAE = 2   # for the partial-scan sensitivity outcome only
# Outcome definitions (column names in the dataset). PRIMARY = hip only, which uses the
# WHO/ISCD reference standard and is measured identically in every cycle.
PRIMARY_OUTCOME = "osteoporosis"            # femoral neck or total hip T <= -2.5
SENSITIVITY_OUTCOMES = {
    "osteo_spine_complete": "Hip or complete-scan lumbar spine (L1-L4)",
    "osteo_spine_partial": "Hip or lumbar spine from >=2 valid vertebrae",
}

# ---------------------------------------------------------------- predictors
# Explicit whitelist: BMD / DXA variables are NEVER used as predictors (leakage).
NUMERIC_FEATURES = [
    "age", "pir", "education",
    "bmi", "weight", "height", "waist",
    "calcium", "phosphorus", "alk_phos", "creatinine", "albumin",
    "uric_acid", "glucose", "total_protein", "cholesterol", "vitamin_d",
]
BINARY_FEATURES = [
    "female", "diabetes", "arthritis", "stroke", "chd", "cancer",
    "smoke_former", "smoke_current",
    # FRAX-type clinical risk factors (OSQ / RHQ / ALQ questionnaires)
    "prior_fracture", "parent_hip_fracture", "steroid_use",
    "postmenopausal", "hormone_use", "alcohol_3plus",
    "race_mexican", "race_other_hisp", "race_black", "race_other",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
RACE_FEATURES = ["race_mexican", "race_other_hisp", "race_black", "race_other"]
LAB_FEATURES = ["calcium", "phosphorus", "alk_phos", "creatinine", "albumin", "uric_acid",
                "glucose", "total_protein", "cholesterol", "vitamin_d"]
# Readable names for figures
PRETTY = {"age": "Age", "pir": "Income-to-poverty ratio", "education": "Education", "bmi": "BMI",
          "weight": "Weight", "height": "Height", "waist": "Waist circumference", "calcium": "Calcium",
          "phosphorus": "Phosphorus", "alk_phos": "Alkaline phosphatase", "creatinine": "Creatinine",
          "albumin": "Albumin", "uric_acid": "Uric acid", "glucose": "Glucose",
          "total_protein": "Total protein", "cholesterol": "Cholesterol", "vitamin_d": "Vitamin D",
          "female": "Female sex", "diabetes": "Diabetes", "arthritis": "Arthritis", "stroke": "Stroke",
          "chd": "Coronary heart disease", "cancer": "Cancer", "smoke_former": "Former smoker",
          "smoke_current": "Current smoker", "prior_fracture": "Prior fracture",
          "parent_hip_fracture": "Parental hip fracture", "steroid_use": "Glucocorticoid use",
          "postmenopausal": "Postmenopausal", "hormone_use": "Hormone therapy (ever)",
          "alcohol_3plus": "Alcohol >=3/day", "race_mexican": "Mexican American",
          "race_other_hisp": "Other Hispanic", "race_black": "Non-Hispanic Black", "race_other": "Other race"}

# Nested blocks for the incremental-value analysis
INCREMENTAL_BLOCKS = [
    ("Age, weight, sex", ["age", "weight", "female"]),
    ("+ other body measures and socio-demographics", ["height", "bmi", "waist", "pir", "education"]),
    ("+ clinical risk factors and history",
     ["prior_fracture", "parent_hip_fracture", "steroid_use", "postmenopausal", "hormone_use",
      "alcohol_3plus", "smoke_former", "smoke_current", "diabetes", "arthritis", "stroke",
      "chd", "cancer"]),
    ("+ laboratory tests", LAB_FEATURES),
    ("+ race/ethnicity", RACE_FEATURES),
]

# Short-form screening model: number of top SHAP features to keep
SHORT_FORM_K = 10

# ---------------------------------------------------------------- run settings
SEED = 42
CV_FOLDS_BASELINE = 10      # base paper used 10-fold CV
CV_FOLDS = 5                # for the extended model comparison
N_BOOT = 1000               # bootstrap resamples for 95% CIs
OPTUNA_TRIALS = 40          # hyper-parameter tuning trials (0 = skip tuning)
N_JOBS = -1

# Quick smoke-test mode:  QUICK=1 python src/analysis.py
import os  # noqa: E402
if os.environ.get("QUICK") == "1":
    CV_FOLDS_BASELINE, CV_FOLDS, N_BOOT, OPTUNA_TRIALS = 3, 3, 50, 3
