"""Train and persist five classifiers on the UCI Bank Marketing dataset.

Running this script end-to-end regenerates everything the Streamlit app needs:

    model/*.joblib      one fitted Pipeline per classifier (preprocessing baked in)
    model/metadata.joblib
    test_data.csv       the held-out split, raw feature values + target
    reports/metrics.csv the comparison table reproduced in README.md

Usage:  python train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 2026
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT_ROOT / "data" / "bank-full.csv"
MODEL_DIR = PROJECT_ROOT / "model"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "subscribed"
POSITIVE_LABEL = "yes"

# `duration` is the length of the last contact call. It is only known *after* the
# call has finished, by which point the outcome is already known too, so leaving
# it in leaks the label and inflates every metric. The UCI dataset description
# says as much. We drop it and keep the remaining 15 predictors.
LEAKY_COLUMNS = ["duration"]

# Support Vector Machines scale roughly quadratically with the number of training
# rows, and Streamlit Community Cloud's free tier is memory-constrained. A
# stratified subsample keeps the five models comparable on identical data while
# staying inside both budgets.
SUBSAMPLE_SIZE = 12_000
TEST_FRACTION = 0.2


def load_dataset() -> pd.DataFrame:
    """Read the semicolon-delimited UCI file and apply project-wide renames."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Expected the UCI source file at {SOURCE_CSV}. "
            "Download bank.zip from https://archive.ics.uci.edu/dataset/222/bank+marketing "
            "and place bank-full.csv in the data/ folder."
        )

    frame = pd.read_csv(SOURCE_CSV, sep=";", quotechar='"')
    frame = frame.drop(columns=LEAKY_COLUMNS)
    # `y` is an unhelpfully terse name once it reaches a UI; rename it once, here.
    frame = frame.rename(columns={"y": TARGET_COLUMN})
    return frame


def stratified_subsample(frame: pd.DataFrame) -> pd.DataFrame:
    """Shrink the 45k-row file while preserving the ~12% positive rate."""
    if len(frame) <= SUBSAMPLE_SIZE:
        return frame.reset_index(drop=True)

    subsample, _ = train_test_split(
        frame,
        train_size=SUBSAMPLE_SIZE,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return subsample.reset_index(drop=True)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Scale the numeric columns, one-hot the categorical ones.

    Every classifier gets the identical transformer so that differences in the
    comparison table come from the algorithms rather than from the features.
    """
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in features.columns if c not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_classifiers() -> dict[str, object]:
    """The five required algorithms, with the imbalance handled where supported."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def positive_class_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Continuous ranking scores for the positive class, for the AUC calculation.

    Most estimators here expose `predict_proba`. SVC is deliberately fitted
    without `probability=True` -- Platt scaling would add an internal 5-fold
    refit for no benefit, since ROC AUC only needs a monotonic ranking and
    `decision_function` already provides one.
    """
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        positive_index = list(pipeline.classes_).index(POSITIVE_LABEL)
        return probabilities[:, positive_index]
    return pipeline.decision_function(features)


def score_model(
    pipeline: Pipeline, features: pd.DataFrame, truth: pd.Series
) -> dict[str, float]:
    """All six required metrics for one fitted pipeline."""
    predictions = pipeline.predict(features)
    scores = positive_class_scores(pipeline, features)

    binary_truth = (truth == POSITIVE_LABEL).astype(int)

    return {
        "Accuracy": accuracy_score(truth, predictions),
        "AUC": roc_auc_score(binary_truth, scores),
        "Precision": precision_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(truth, predictions),
    }


def slugify(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def write_markdown_table(comparison: pd.DataFrame, destination: Path) -> None:
    """Emit the comparison table as Markdown, pasted straight into the README.

    Hand-rolled rather than via `DataFrame.to_markdown`, which would pull in
    `tabulate` as an extra deployment dependency for four lines of formatting.
    """
    header = ["ML Model Name", *comparison.columns]
    rows = [
        [name, *(f"{value:.4f}" for value in record)]
        for name, record in zip(comparison.index, comparison.to_numpy())
    ]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    frame = load_dataset()
    print(f"Loaded {len(frame):,} rows and {frame.shape[1] - 1} predictors from the UCI file.")

    frame = stratified_subsample(frame)
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    print(
        f"Training on {len(train_features):,} rows, holding out {len(test_features):,} "
        f"({test_target.value_counts().get(POSITIVE_LABEL, 0)} positives)."
    )

    # The held-out split is what the Streamlit app consumes. Writing it before
    # training makes it obvious that no model ever saw these rows.
    holdout = test_features.copy()
    holdout[TARGET_COLUMN] = test_target.values
    holdout.to_csv(PROJECT_ROOT / "test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(holdout):,} rows).")

    results: list[dict[str, object]] = []

    for model_name, classifier in build_classifiers().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(train_features)),
                ("classify", classifier),
            ]
        )
        pipeline.fit(train_features, train_target)

        metrics = score_model(pipeline, test_features, test_target)
        results.append({"Model": model_name, **metrics})

        destination = MODEL_DIR / f"{slugify(model_name)}.joblib"
        joblib.dump(pipeline, destination)
        print(
            f"  {model_name:<24} acc={metrics['Accuracy']:.3f} "
            f"auc={metrics['AUC']:.3f} mcc={metrics['MCC']:.3f}  -> {destination.name}"
        )

    comparison = pd.DataFrame(results).set_index("Model").round(4)
    comparison.to_csv(REPORT_DIR / "metrics.csv")

    joblib.dump(
        {
            "target_column": TARGET_COLUMN,
            "positive_label": POSITIVE_LABEL,
            "feature_columns": features.columns.tolist(),
            "class_labels": sorted(target.unique().tolist()),
            "model_files": {name: f"{slugify(name)}.joblib" for name in build_classifiers()},
            "random_state": RANDOM_STATE,
            "training_rows": len(train_features),
            "holdout_rows": len(test_features),
        },
        MODEL_DIR / "metadata.joblib",
    )

    print("\nHold-out comparison table")
    print(comparison.to_string())
    print(f"\nSaved {len(results)} pipelines to {MODEL_DIR} and metrics to {REPORT_DIR / 'metrics.csv'}.")

    write_markdown_table(comparison, REPORT_DIR / "metrics.md")


if __name__ == "__main__":
    main()
