"""
Full-scale, honest reproduction of the accident-severity classification study.

Built to directly answer four reviewer questions raised against the published
paper (Naresh & Dullam, European Transport Research Review, 2026):

  1. Exact preprocessing/join code and final feature list, including explicit
     confirmation of whether casualty severity was excluded.
  2. Exact cleaning/exclusion logic showing how the final modelling table was
     derived from the raw Accidents/Vehicles/Casualties tables.
  3. An explanation for any mismatch between the stated split and the test-set
     support actually reported.
  4. The balancing method used, if any -- disclosed explicitly per model.

Design choices made to answer each point are documented inline. Every number
this script reports is computed directly from the data at run time; nothing
is hard-coded or copied from the original manuscript.
"""

from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
def resolve_file(filename: str) -> Path:
    """Locate a required CSV, which may live in a single local folder or be
    spread across several separately-attached Kaggle input datasets."""
    env = os.environ.get("ACCIDENT_DATA_DIR")
    if env:
        candidate = Path(env) / filename
        if candidate.exists():
            return candidate
    local_candidate = Path(".") / filename
    if local_candidate.exists():
        return local_candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Nesting depth varies across Kaggle environments (some mount at
        # /kaggle/input/<slug>/<file>, others at
        # /kaggle/input/datasets/<owner>/<slug>/<file>), so search recursively.
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not locate '{filename}' under ACCIDENT_DATA_DIR, the current "
        f"directory, or any /kaggle/input/*/ dataset folder."
    )
OUTPUT_DIR = Path(os.environ.get("ACCIDENT_OUTPUT_DIR", "reproducibility_outputs"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42  # single seed for split + every model, replacing the
                    # original script's inconsistent 99 (split) / 42 (model)
TEST_SIZE = 0.20
N_BOOTSTRAP = 200

ID_COL = "Accident_Index"
TARGET = "Accident_Severity"
# Raw STATS19 codes are 1=Fatal, 2=Serious, 3=Slight (Table 4 of the paper).
# XGBoost's sklearn API requires 0-indexed contiguous class labels, so the
# target is remapped to 0/1/2 right after loading and every model (not just
# XGBoost) is trained on the remapped target for consistency. This has no
# effect on any other classifier's behaviour.
SEVERITY_NAMES = {0: "Fatal", 1: "Serious", 2: "Slight"}

# Feature set actually consumed by the published system's prediction form
# (Fig. 7 of the paper): driver / vehicle / roadway / environmental fields.
# Casualty-level fields are deliberately NOT included here.
FEATURES_PRIMARY = [
    "Did_Police_Officer_Attend_Scene_of_Accident",
    "Age_of_Driver",
    "Vehicle_Type",
    "Age_of_Vehicle",
    "Engine_Capacity_(CC)",
    "Day_of_Week",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Light_Conditions",
    "Sex_of_Driver",
    "Speed_limit",
]

# Extended set adding the casualty-level "Human Factors" the paper's Table 3
# claims were analysed (casualty age, gender, pedestrian involvement), for a
# secondary sensitivity check. Casualty_Severity is excluded on purpose --
# it is the outcome variable at the casualty level and using it (or any
# derivative of it) as a predictor of Accident_Severity would be leakage.
CASUALTY_EXTRA_FEATURES = ["Age_of_Casualty", "Sex_of_Casualty", "Casualty_Is_Pedestrian"]
FEATURES_EXTENDED = FEATURES_PRIMARY + CASUALTY_EXTRA_FEATURES

CATEGORICAL_FEATURES_PRIMARY = [
    "Did_Police_Officer_Attend_Scene_of_Accident",
    "Vehicle_Type",
    "Day_of_Week",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Light_Conditions",
    "Sex_of_Driver",
]
NUMERICAL_FEATURES_PRIMARY = [
    "Age_of_Driver",
    "Age_of_Vehicle",
    "Engine_Capacity_(CC)",
    "Speed_limit",
]

CATEGORICAL_FEATURES_EXTENDED = CATEGORICAL_FEATURES_PRIMARY + [
    "Sex_of_Casualty",
    "Casualty_Is_Pedestrian",
]
NUMERICAL_FEATURES_EXTENDED = NUMERICAL_FEATURES_PRIMARY + ["Age_of_Casualty"]

# Explicit, machine-checked confirmation for reviewer question 1: no column
# derived from casualty severity is ever allowed into a feature list.
for _feat_list in (FEATURES_PRIMARY, FEATURES_EXTENDED):
    assert "Casualty_Severity" not in _feat_list
    assert not any("Severity" in c and c != TARGET for c in _feat_list)


def log_stage(audit, stage, rows, note=""):
    audit.append({"stage": stage, "rows": int(rows), "note": note})


# -----------------------------------------------------------------------
# 1. Load full datasets and perform the literal three-way join D = A join C join V
# -----------------------------------------------------------------------
def load_and_join():
    audit = []

    accidents = pd.read_csv(resolve_file("accidents.csv"), low_memory=False)
    vehicles = pd.read_csv(resolve_file("vehicles.csv"), low_memory=False, on_bad_lines="skip")
    casualties = pd.read_csv(resolve_file("casualties.csv"), low_memory=False, on_bad_lines="skip")

    log_stage(audit, "accidents_loaded", len(accidents))
    log_stage(audit, "vehicles_loaded", len(vehicles))
    log_stage(audit, "casualties_loaded", len(casualties))

    accident_cols = [
        ID_COL, TARGET, "Day_of_Week", "Weather_Conditions", "Road_Surface_Conditions",
        "Light_Conditions", "Speed_limit", "Did_Police_Officer_Attend_Scene_of_Accident",
    ]
    vehicle_cols = [
        ID_COL, "Vehicle_Reference", "Vehicle_Type", "Sex_of_Driver", "Age_of_Driver",
        "Age_of_Vehicle", "Engine_Capacity_(CC)",
    ]
    casualty_cols = [
        ID_COL, "Casualty_Reference", "Age_of_Casualty", "Sex_of_Casualty",
        "Casualty_Class", "Casualty_Severity",
    ]

    missing = {
        "accidents": sorted(set(accident_cols) - set(accidents.columns)),
        "vehicles": sorted(set(vehicle_cols) - set(vehicles.columns)),
        "casualties": sorted(set(casualty_cols) - set(casualties.columns)),
    }
    if any(missing.values()):
        raise ValueError(f"Missing required source columns: {missing}")

    accidents = accidents[accident_cols].copy()
    vehicles = vehicles[vehicle_cols].copy()
    casualties = casualties[casualty_cols].copy()

    # Casualty_Severity is loaded ONLY so the audit trail can prove it is
    # dropped before modelling (see the assert immediately after the merge).
    # It is never merged into the feature matrix.

    # ---- Reduce Vehicles and Casualties to one deterministic row per
    #      accident (first vehicle / first casualty by reference number),
    #      matching the paper's stated intent of a single feature vector
    #      per accident (Eq. D = {(x_i, y_i)}) while keeping row identity
    #      unambiguous and reproducible.
    vehicles = vehicles.drop_duplicates()
    vehicles_one = (
        vehicles.sort_values([ID_COL, "Vehicle_Reference"])
        .groupby(ID_COL, as_index=False)
        .first()
    )
    log_stage(audit, "vehicles_reduced_to_one_row_per_accident", len(vehicles_one),
              "First vehicle by Vehicle_Reference retained per accident.")

    casualties = casualties.drop_duplicates()
    casualties_one = (
        casualties.sort_values([ID_COL, "Casualty_Reference"])
        .groupby(ID_COL, as_index=False)
        .first()
    )
    log_stage(audit, "casualties_reduced_to_one_row_per_accident", len(casualties_one),
              "First casualty by Casualty_Reference retained per accident.")

    # ---- Literal three-way join: D = A join C join V (accident index key),
    #      each leg validated one_to_one and its coverage logged.
    merged = accidents.merge(
        vehicles_one, on=ID_COL, how="left", validate="one_to_one", indicator="vehicle_merge"
    )
    log_stage(audit, "after_accident_vehicle_join", len(merged))

    merged = merged.merge(
        casualties_one, on=ID_COL, how="left", validate="one_to_one", indicator="casualty_merge"
    )
    log_stage(audit, "after_accident_vehicle_casualty_join (D = A join V join C)", len(merged))

    vehicle_coverage = (merged["vehicle_merge"] == "both").mean()
    casualty_coverage = (merged["casualty_merge"] == "both").mean()
    log_stage(audit, "vehicle_join_coverage_fraction", vehicle_coverage)
    log_stage(audit, "casualty_join_coverage_fraction", casualty_coverage)

    casualty_class = merged["Casualty_Class"]
    is_pedestrian = pd.Series(
        np.where(casualty_class == 3, "yes", "no"), index=merged.index, dtype=object
    )
    is_pedestrian[casualty_class.isna()] = np.nan
    merged["Casualty_Is_Pedestrian"] = is_pedestrian

    # Explicit, checkable proof for reviewer question 1.
    assert "Casualty_Severity" not in FEATURES_PRIMARY
    assert "Casualty_Severity" not in FEATURES_EXTENDED

    # ---- Documented invalid-value handling: -1 is STATS19's missing/unknown
    #      code across nearly all fields in these tables.
    merged.replace(-1, np.nan, inplace=True)
    log_stage(audit, "after_replacing_minus_one_with_nan", len(merged))

    before_target = len(merged)
    merged = merged.dropna(subset=[TARGET])
    log_stage(audit, "after_removing_missing_target", len(merged),
              f"Removed {before_target - len(merged)} rows.")

    before_dupe = len(merged)
    merged = merged.drop_duplicates(subset=[ID_COL], keep="first")
    log_stage(audit, "after_duplicate_accident_removal", len(merged),
              f"Removed {before_dupe - len(merged)} rows.")

    return merged, audit


# -----------------------------------------------------------------------
# 2. Preprocessing / model factory
# -----------------------------------------------------------------------
def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def tune_random_forest(X_train, y_train, numeric_features, categorical_features):
    """Light RandomizedSearchCV on a 5% stratified subsample, refit on full
    training data afterwards. Keeps hyperparameter selection defensible
    without paying full-tuning cost on 1.4M rows."""
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=0.05, stratify=y_train, random_state=RANDOM_STATE
    )
    pre = build_preprocessor(numeric_features, categorical_features)
    pipe = Pipeline([
        ("preprocessor", pre),
        ("model", RandomForestClassifier(
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 5, 20],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=6, cv=3, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_sub, y_sub)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best


def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_rows = len(y_true)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# -----------------------------------------------------------------------
# 3. Evaluate a single fitted model against the shared, fixed test set
# -----------------------------------------------------------------------
def evaluate_model(name, pipeline, X_test, y_test, balancing_note):
    y_pred = pipeline.predict(X_test)

    labels_sorted = sorted(y_test.unique())
    label_names = [SEVERITY_NAMES.get(int(c), str(c)) for c in labels_sorted]

    report = classification_report(
        y_test, y_pred, labels=labels_sorted, target_names=label_names,
        digits=6, output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / f"classification_report_{name}.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in label_names],
                          columns=[f"pred_{n}" for n in label_names])
    cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_{name}.csv")

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1_val = macro_f1(y_test, y_pred)

    acc_lo, acc_hi = bootstrap_ci(y_test, y_pred, accuracy_score)
    f1_lo, f1_hi = bootstrap_ci(y_test, y_pred, macro_f1)

    real_support = y_test.value_counts().sort_index()
    real_support.index = [SEVERITY_NAMES.get(int(c), str(c)) for c in real_support.index]

    return {
        "model": name,
        "balancing": balancing_note,
        "accuracy": acc,
        "accuracy_95ci_low": acc_lo,
        "accuracy_95ci_high": acc_hi,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1_val,
        "macro_f1_95ci_low": f1_lo,
        "macro_f1_95ci_high": f1_hi,
        "test_support_total": int(len(y_test)),
        "test_support_by_class": real_support.to_dict(),
    }


# -----------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------
def main():
    t0 = time.time()
    merged, audit = load_and_join()

    required_primary = FEATURES_PRIMARY + [TARGET]
    required_extended = FEATURES_EXTENDED + [TARGET]
    missing_required = sorted(set(required_extended) - set(merged.columns))
    if missing_required:
        raise ValueError(f"Missing required columns for modelling: {missing_required}")

    model_df = merged[required_extended].copy()
    log_stage(audit, "final_modelling_rows_before_split", len(model_df))

    for col in NUMERICAL_FEATURES_EXTENDED:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES_EXTENDED:
        # Cast to plain object dtype with float NaN for missing values.
        # Pandas' nullable "string"/"Int64" extension dtypes use pd.NA,
        # which crashes scikit-learn's SimpleImputer (`X != X` on pd.NA
        # raises "boolean value of NA is ambiguous"). Plain object + np.nan
        # is what SimpleImputer/OneHotEncoder actually expect.
        series = model_df[col]
        mask = series.isna()
        out = series.astype(object)
        out[~mask] = out[~mask].astype(str)
        out[mask] = np.nan
        model_df[col] = out

    X = model_df[FEATURES_EXTENDED].copy()
    y = model_df[TARGET].astype(int) - 1  # 1/2/3 -> 0/1/2, see SEVERITY_NAMES comment

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    split_summary = pd.DataFrame({
        "subset": ["train", "test", "total"],
        "rows": [len(X_train), len(X_test), len(X)],
    })
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    def named_support(series):
        s = series.value_counts(dropna=False).sort_index()
        s.index = [SEVERITY_NAMES.get(int(c), str(c)) if pd.notna(c) else "missing" for c in s.index]
        return s.rename("count")

    named_support(y_train).to_csv(OUTPUT_DIR / "train_class_support.csv")
    named_support(y_test).to_csv(OUTPUT_DIR / "test_class_support.csv")

    print("\n=== REAL DATASET AND SPLIT SIZES (answers reviewer question 3) ===")
    print(split_summary.to_string(index=False))
    print("\nReal test-set class support:")
    print(named_support(y_test))

    X_train_p, X_test_p = X_train[FEATURES_PRIMARY], X_test[FEATURES_PRIMARY]

    print("\nTuning Random Forest (5% subsample, 3-fold CV, macro-F1) ...")
    rf_best_params = tune_random_forest(
        X_train_p, y_train, NUMERICAL_FEATURES_PRIMARY, CATEGORICAL_FEATURES_PRIMARY
    )
    print("Selected Random Forest hyperparameters:", rf_best_params)

    results = []
    balancing_disclosure = {}

    def make_pipeline(estimator, numeric_features, categorical_features):
        return Pipeline([
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])

    model_specs = [
        ("logistic_regression", LogisticRegression(max_iter=1000, n_jobs=-1), "none"),
        ("logistic_regression_balanced",
         LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
         "class_weight='balanced' (reweights training loss only; test-set support is the true imbalanced distribution)"),
        ("random_forest",
         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_best_params),
         "none"),
        ("random_forest_balanced",
         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **rf_best_params),
         "class_weight='balanced_subsample' (reweights training loss only; test-set support is the true imbalanced distribution)"),
        ("mlp_proposed",
         MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                        learning_rate_init=0.001, max_iter=200, early_stopping=True,
                        random_state=RANDOM_STATE),
         "none (matches the architecture declared in the paper's Table 7; MLPClassifier has no native class-weighting)"),
    ]
    if HAS_XGB:
        model_specs.append((
            "xgboost",
            XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                           objective="multi:softprob", eval_metric="mlogloss",
                           random_state=RANDOM_STATE, n_jobs=-1),
            "none",
        ))
    if HAS_LGBM:
        model_specs.append((
            "lightgbm",
            LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "none",
        ))
        model_specs.append((
            "lightgbm_balanced",
            LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "class_weight='balanced' (reweights training loss only; test-set support is the true imbalanced distribution)",
        ))

    # Primary experiment: Fig. 7 feature set (accidents + vehicles only).
    for name, estimator, balancing_note in model_specs:
        print(f"\nFitting {name} on FEATURES_PRIMARY ({len(X_train_p)} rows) ...")
        t_start = time.time()
        pipe = make_pipeline(estimator, NUMERICAL_FEATURES_PRIMARY, CATEGORICAL_FEATURES_PRIMARY)
        pipe.fit(X_train_p, y_train)
        elapsed = time.time() - t_start
        res = evaluate_model(name, pipe, X_test_p, y_test, balancing_note)
        res["feature_set"] = "primary"
        res["train_seconds"] = elapsed
        results.append(res)
        balancing_disclosure[name] = balancing_note
        print(f"  accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  ({elapsed:.1f}s)")

    # Secondary sensitivity check: extended feature set including casualty
    # fields the paper's Table 3 claims were used (age, gender, pedestrian
    # involvement of the casualty) -- run only for the best tree model and
    # the proposed MLP, to bound compute.
    for name, estimator, balancing_note in [
        ("random_forest_balanced_extended",
         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **rf_best_params),
         "class_weight='balanced_subsample'"),
        ("mlp_proposed_extended",
         MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                        learning_rate_init=0.001, max_iter=200, early_stopping=True,
                        random_state=RANDOM_STATE),
         "none"),
    ]:
        print(f"\nFitting {name} on FEATURES_EXTENDED ({len(X_train)} rows) ...")
        t_start = time.time()
        pipe = make_pipeline(estimator, NUMERICAL_FEATURES_EXTENDED, CATEGORICAL_FEATURES_EXTENDED)
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t_start
        res = evaluate_model(name, pipe, X_test, y_test, balancing_note)
        res["feature_set"] = "extended_with_casualty_fields"
        res["train_seconds"] = elapsed
        results.append(res)
        balancing_disclosure[name] = balancing_note
        print(f"  accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  ({elapsed:.1f}s)")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)

    with open(OUTPUT_DIR / "feature_lists.json", "w", encoding="utf-8") as f:
        json.dump({
            "features_primary_fig7_matching": FEATURES_PRIMARY,
            "features_extended_with_casualty_fields": FEATURES_EXTENDED,
            "target": TARGET,
            "casualty_severity_excluded": True,
            "note": "Casualty_Severity is loaded for audit purposes only and is "
                     "never included in either feature list (see assertions in code).",
        }, f, indent=2)

    with open(OUTPUT_DIR / "balancing_disclosure.json", "w", encoding="utf-8") as f:
        json.dump(balancing_disclosure, f, indent=2)

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_total_accidents_raw_file": None,
            "n_total_modelling_rows": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state_used_everywhere": RANDOM_STATE,
            "rf_tuned_hyperparameters": rf_best_params,
            "total_runtime_seconds": time.time() - t0,
            "has_xgboost": HAS_XGB,
            "has_lightgbm": HAS_LGBM,
        }, f, indent=2)

    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "data_audit.csv", index=False)

    print("\n=== MODEL COMPARISON (real numbers, full dataset) ===")
    print(summary_df[["model", "feature_set", "balancing", "accuracy", "macro_f1",
                       "balanced_accuracy", "test_support_total"]].to_string(index=False))
    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
