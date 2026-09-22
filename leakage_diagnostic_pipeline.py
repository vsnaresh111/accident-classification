"""
DIAGNOSTIC / NEGATIVE-CONTROL RUN -- NOT A CANDIDATE MODEL.

Purpose: test whether including Casualty_Severity (the injury outcome at the
casualty level) as a predictor of Accident_Severity reproduces the kind of
inflated accuracy (~96%) an independent reviewer reported when probing the
published paper's claimed 91.2% MLP result. This is the leading hypothesis
for how the original paper's numbers were produced -- accidentally or
otherwise leaking an outcome-adjacent variable into the feature set.

This script is a controlled variant of full_reproduction_pipeline.py with
ONE deliberate change: Casualty_Severity is added to the feature list. Every
other choice (join, cleaning, split, seed, models) is identical, so any jump
in accuracy can be attributed specifically to that one variable.

Do not read the numbers from this script as a legitimate model to report as
"our result" -- they exist only to explain and quantify a data-integrity
problem for the corrigendum.
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_file(filename: str) -> Path:
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
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate '{filename}'.")


OUTPUT_DIR = Path(os.environ.get("ACCIDENT_OUTPUT_DIR", "leakage_diagnostic_outputs"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "Accident_Index"
TARGET = "Accident_Severity"
SEVERITY_NAMES = {0: "Fatal", 1: "Serious", 2: "Slight"}

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

# The ONE deliberate change versus the legitimate pipeline: add the
# casualty-level outcome variable as a feature, purely to quantify its
# leakage effect.
FEATURES_LEAKAGE = FEATURES_PRIMARY + ["Casualty_Severity_Feature"]

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
CATEGORICAL_FEATURES_LEAKAGE = CATEGORICAL_FEATURES_PRIMARY + ["Casualty_Severity_Feature"]


def log_stage(audit, stage, rows, note=""):
    audit.append({"stage": stage, "rows": int(rows), "note": note})


def to_categorical_object(series):
    mask = series.isna()
    out = series.astype(object)
    out[~mask] = out[~mask].astype(str)
    out[mask] = np.nan
    return out


def load_and_join():
    audit = []
    accidents = pd.read_csv(resolve_file("accidents.csv"), low_memory=False)
    vehicles = pd.read_csv(resolve_file("vehicles.csv"), low_memory=False, on_bad_lines="skip")
    casualties = pd.read_csv(resolve_file("casualties.csv"), low_memory=False, on_bad_lines="skip")

    accident_cols = [
        ID_COL, TARGET, "Day_of_Week", "Weather_Conditions", "Road_Surface_Conditions",
        "Light_Conditions", "Speed_limit", "Did_Police_Officer_Attend_Scene_of_Accident",
    ]
    vehicle_cols = [
        ID_COL, "Vehicle_Reference", "Vehicle_Type", "Sex_of_Driver", "Age_of_Driver",
        "Age_of_Vehicle", "Engine_Capacity_(CC)",
    ]
    casualty_cols = [ID_COL, "Casualty_Reference", "Casualty_Severity"]

    accidents = accidents[accident_cols].copy()
    vehicles = vehicles[vehicle_cols].copy()
    casualties = casualties[casualty_cols].copy()

    vehicles = vehicles.drop_duplicates()
    vehicles_one = (
        vehicles.sort_values([ID_COL, "Vehicle_Reference"]).groupby(ID_COL, as_index=False).first()
    )
    casualties = casualties.drop_duplicates()
    casualties_one = (
        casualties.sort_values([ID_COL, "Casualty_Reference"]).groupby(ID_COL, as_index=False).first()
    )
    log_stage(audit, "accidents_loaded", len(accidents))
    log_stage(audit, "vehicles_reduced_to_one_row_per_accident", len(vehicles_one))
    log_stage(audit, "casualties_reduced_to_one_row_per_accident", len(casualties_one))

    merged = accidents.merge(vehicles_one, on=ID_COL, how="left", validate="one_to_one")
    merged = merged.merge(casualties_one, on=ID_COL, how="left", validate="one_to_one")
    log_stage(audit, "after_three_way_join", len(merged))

    # Rename here, deliberately, so it is unmistakable in every output file
    # that this is Casualty_Severity being used as a feature -- not an
    # accidental column-name collision.
    merged["Casualty_Severity_Feature"] = merged["Casualty_Severity"]

    merged.replace(-1, np.nan, inplace=True)
    before_target = len(merged)
    merged = merged.dropna(subset=[TARGET])
    log_stage(audit, "after_removing_missing_target", len(merged),
              f"Removed {before_target - len(merged)} rows.")
    merged = merged.drop_duplicates(subset=[ID_COL], keep="first")
    log_stage(audit, "final_modelling_rows_before_split", len(merged))
    return merged, audit


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


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    labels_sorted = sorted(y_test.unique())
    label_names = [SEVERITY_NAMES.get(int(c), str(c)) for c in labels_sorted]

    report = classification_report(
        y_test, y_pred, labels=labels_sorted, target_names=label_names,
        digits=6, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f"classification_report_{name}.csv")

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=[f"true_{n}" for n in label_names],
                 columns=[f"pred_{n}" for n in label_names]).to_csv(
        OUTPUT_DIR / f"confusion_matrix_{name}.csv"
    )

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = macro_f1(y_test, y_pred)
    return {"model": name, "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1,
            "test_support_total": int(len(y_test))}


def main():
    t0 = time.time()
    merged, audit = load_and_join()

    model_df = merged[FEATURES_LEAKAGE + [TARGET]].copy()
    for col in NUMERICAL_FEATURES_PRIMARY:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES_LEAKAGE:
        model_df[col] = to_categorical_object(model_df[col])

    X = model_df[FEATURES_LEAKAGE].copy()
    y = model_df[TARGET].astype(int) - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Diagnostic run: {len(X)} total rows, {len(X_train)} train / {len(X_test)} test")
    print("Feature set (includes Casualty_Severity_Feature as a deliberate leak):", FEATURES_LEAKAGE)

    def make_pipeline(estimator):
        return Pipeline([
            ("preprocessor", build_preprocessor(NUMERICAL_FEATURES_PRIMARY, CATEGORICAL_FEATURES_LEAKAGE)),
            ("model", estimator),
        ])

    # Random Forest hyperparameters reused from the legitimate full run
    # (full_reproduction_pipeline.py), where they were selected by
    # RandomizedSearchCV on a 5% subsample: {'n_estimators': 100,
    # 'min_samples_leaf': 5, 'max_depth': None}. Not re-tuned here since the
    # point is to isolate the effect of adding one feature, not to re-derive
    # hyperparameters.
    model_specs = [
        ("logistic_regression_LEAKAGE", LogisticRegression(max_iter=1000)),
        ("random_forest_LEAKAGE",
         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 n_estimators=100, min_samples_leaf=5, max_depth=None)),
        ("mlp_proposed_LEAKAGE",
         MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                        learning_rate_init=0.001, max_iter=200, early_stopping=True,
                        random_state=RANDOM_STATE)),
    ]

    results = []
    for name, estimator in model_specs:
        print(f"\nFitting {name} ...")
        t_start = time.time()
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t_start
        res = evaluate_model(name, pipe, X_test, y_test)
        res["train_seconds"] = elapsed
        results.append(res)
        print(f"  accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  ({elapsed:.1f}s)")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_DIR / "leakage_diagnostic_summary.csv", index=False)
    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "leakage_diagnostic_audit.csv", index=False)

    print("\n=== LEAKAGE DIAGNOSTIC RESULTS (Casualty_Severity included as a feature) ===")
    print("This is NOT a legitimate model. It exists only to test whether including")
    print("the casualty-level outcome variable explains the paper's inflated accuracy.")
    print(summary_df.to_string(index=False))
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
