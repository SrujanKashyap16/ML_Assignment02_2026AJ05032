"""Streamlit front-end for the Bank Marketing classifier comparison.

The app is strictly an *evaluation* surface: it loads the five pipelines that
`train_models.py` already fitted and scores them against whatever hold-out CSV
is supplied. No estimator is ever refitted here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
BUNDLED_TEST_CSV = PROJECT_ROOT / "test_data.csv"

st.set_page_config(
    page_title="Term Deposit Classifier Bench",
    page_icon="📞",
    layout="wide",
)

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading saved pipelines…")
def load_artifacts() -> tuple[dict, dict]:
    """Read the metadata manifest and every pipeline it points at."""
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    pipelines = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metadata["model_files"].items()
    }
    return metadata, pipelines


@st.cache_data(show_spinner=False)
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def positive_class_scores(pipeline, features: pd.DataFrame, positive_label: str) -> np.ndarray:
    """Ranking scores for the positive class, mirroring the training script.

    SVC was fitted without Platt scaling, so it exposes `decision_function`
    instead of `predict_proba`. Either is a valid input to ROC AUC.
    """
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        return probabilities[:, list(pipeline.classes_).index(positive_label)]
    return pipeline.decision_function(features)


def evaluate(pipeline, features, truth, positive_label) -> dict:
    predictions = pipeline.predict(features)
    scores = positive_class_scores(pipeline, features, positive_label)
    binary_truth = (truth == positive_label).astype(int)

    return {
        "predictions": predictions,
        "scores": scores,
        "metrics": {
            "Accuracy": accuracy_score(truth, predictions),
            "AUC": roc_auc_score(binary_truth, scores),
            "Precision": precision_score(truth, predictions, pos_label=positive_label, zero_division=0),
            "Recall": recall_score(truth, predictions, pos_label=positive_label, zero_division=0),
            "F1": f1_score(truth, predictions, pos_label=positive_label, zero_division=0),
            "MCC": matthews_corrcoef(truth, predictions),
        },
    }


def validate_columns(frame: pd.DataFrame, metadata: dict) -> list[str]:
    """Return a list of human-readable problems with the uploaded file."""
    problems = []
    target = metadata["target_column"]

    if target not in frame.columns:
        problems.append(
            f"The target column **`{target}`** is missing. "
            "The uploaded CSV must carry the ground-truth label so metrics can be computed."
        )

    missing = [c for c in metadata["feature_columns"] if c not in frame.columns]
    if missing:
        problems.append("Missing feature column(s): " + ", ".join(f"`{c}`" for c in missing))

    return problems


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def confusion_matrix_figure(truth, predictions, labels):
    matrix = confusion_matrix(truth, predictions, labels=labels)
    figure, axes = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="BuPu",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    figure.tight_layout()
    return figure


def roc_figure(binary_truth, scores, model_name, auc_value):
    false_positive_rate, true_positive_rate, _ = roc_curve(binary_truth, scores)
    figure, axes = plt.subplots(figsize=(4.2, 3.4))
    axes.plot(false_positive_rate, true_positive_rate, lw=2, label=f"AUC = {auc_value:.3f}")
    axes.plot([0, 1], [0, 1], "--", lw=1, color="grey", label="Chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(model_name, fontsize=10)
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

metadata, pipelines = load_artifacts()
TARGET = metadata["target_column"]
POSITIVE = metadata["positive_label"]
LABELS = metadata["class_labels"]

st.title("📞 Term Deposit Classifier Bench")
st.caption(
    "Five classifiers trained on the UCI Bank Marketing campaign data, scored side by side "
    "on a hold-out set. Models are loaded from disk — nothing is retrained in the browser."
)

with st.sidebar:
    st.header("Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type="csv",
        help=f"Needs the {len(metadata['feature_columns'])} feature columns plus the "
             f"`{TARGET}` target column.",
    )

    if uploaded is not None:
        data = read_csv(uploaded)
        data_origin = f"uploaded file — `{uploaded.name}`"
    else:
        data = read_csv(BUNDLED_TEST_CSV)
        data_origin = "bundled `test_data.csv` (the hold-out split)"
        st.info("No file uploaded — showing the hold-out split shipped with the repo.")

    st.divider()
    st.header("Model")
    selected_model = st.selectbox("Classifier", list(pipelines.keys()))

    st.divider()
    st.caption(
        f"Trained on {metadata['training_rows']:,} rows · "
        f"random_state={metadata['random_state']}"
    )

problems = validate_columns(data, metadata)
if problems:
    st.error("The test file cannot be scored:")
    for problem in problems:
        st.markdown(f"- {problem}")
    st.stop()

features = data[metadata["feature_columns"]]
truth = data[TARGET]
binary_truth = (truth == POSITIVE).astype(int)

top = st.columns(4)
top[0].metric("Rows scored", f"{len(data):,}")
top[1].metric("Features", len(metadata["feature_columns"]))
top[2].metric(f"Positives (`{POSITIVE}`)", f"{int(binary_truth.sum()):,}")
top[3].metric("Positive rate", f"{binary_truth.mean():.1%}")
st.caption(f"Source: {data_origin}")

single_tab, compare_tab, data_tab = st.tabs(
    ["Selected model", "All five models", "Test data"]
)

# -- Selected model ---------------------------------------------------------

with single_tab:
    result = evaluate(pipelines[selected_model], features, truth, POSITIVE)
    st.subheader(selected_model)

    metric_columns = st.columns(6)
    for column, (label, value) in zip(metric_columns, result["metrics"].items()):
        column.metric(label, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_matrix_figure(truth, result["predictions"], LABELS))
    with right:
        st.markdown("**ROC curve**")
        st.pyplot(
            roc_figure(binary_truth, result["scores"], selected_model, result["metrics"]["AUC"])
        )

    st.markdown("**Classification report**")
    report = classification_report(
        truth, result["predictions"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

# -- All models -------------------------------------------------------------

with compare_tab:
    st.subheader("Comparison across all five classifiers")

    comparison = pd.DataFrame(
        {
            name: evaluate(pipeline, features, truth, POSITIVE)["metrics"]
            for name, pipeline in pipelines.items()
        }
    ).transpose()
    comparison.index.name = "ML Model Name"

    st.dataframe(
        comparison.style.format("{:.4f}").background_gradient(cmap="BuPu", axis=0),
        width="stretch",
    )

    st.markdown("**Metric profile by model**")
    melted = comparison.reset_index().melt(
        id_vars="ML Model Name", var_name="Metric", value_name="Value"
    )
    figure, axes = plt.subplots(figsize=(9, 3.6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="ML Model Name", ax=axes)
    axes.set_ylim(0, 1)
    axes.set_xlabel("")
    axes.legend(fontsize=7, ncol=3, loc="upper right")
    figure.tight_layout()
    st.pyplot(figure)

    best = comparison["MCC"].idxmax()
    st.success(
        f"Best Matthews correlation on this file: **{best}** "
        f"(MCC {comparison.loc[best, 'MCC']:.4f}, AUC {comparison.loc[best, 'AUC']:.4f}). "
        "MCC is the fairest single summary here because the classes are heavily imbalanced."
    )

    st.download_button(
        "Download comparison as CSV",
        comparison.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# -- Test data --------------------------------------------------------------

with data_tab:
    st.subheader("Rows being scored")
    st.dataframe(data.head(200), width="stretch")
    st.caption(f"Showing the first {min(200, len(data)):,} of {len(data):,} rows.")

    st.markdown("**Predictions from the selected model**")
    preview = data.copy()
    preview["predicted"] = pipelines[selected_model].predict(features)
    preview["correct"] = preview["predicted"] == preview[TARGET]
    st.dataframe(
        preview[[TARGET, "predicted", "correct"]].head(200), width="stretch"
    )
    st.download_button(
        "Download predictions as CSV",
        preview.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
