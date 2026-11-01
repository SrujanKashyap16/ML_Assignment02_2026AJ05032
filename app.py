"""Streamlit front-end for the Bank Marketing classifier comparison.

The app is strictly an *evaluation* surface: it loads the five pipelines that
`train_models.py` already fitted and scores them against whatever hold-out CSV
is supplied. No estimator is ever refitted here.
"""

from __future__ import annotations

import html
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import theme

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
BUNDLED_TEST_CSV = PROJECT_ROOT / "test_data.csv"

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

METRIC_HELP = {
    "Accuracy": "Share of all rows classified correctly. Misleading when classes are imbalanced.",
    "AUC": "Probability a random subscriber is ranked above a random non-subscriber.",
    "Precision": "Of the clients flagged as subscribers, how many actually subscribed.",
    "Recall": "Of all real subscribers, how many the model found.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between prediction and truth across all four cells. Robust to imbalance.",
}

st.set_page_config(
    page_title="Term Deposit Classifier Bench",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.apply_matplotlib_theme()


# --------------------------------------------------------------------------
# Page styling
# --------------------------------------------------------------------------


def inject_styles() -> None:
    """Page chrome: typography, KPI cards and panels.

    Everything here targets our own class names rather than Streamlit's
    generated ones, which change between releases.
    """
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }}
          #MainMenu, footer {{ visibility: hidden; }}

          .masthead {{ border-bottom: 1px solid {theme.BORDER}; padding-bottom: 1.1rem; margin-bottom: 1.6rem; }}
          .masthead .eyebrow {{
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.11em;
            text-transform: uppercase; color: {theme.ACCENT}; margin-bottom: 0.45rem;
          }}
          .masthead h1 {{
            font-size: 1.85rem; font-weight: 680; letter-spacing: -0.021em;
            color: {theme.INK}; margin: 0 0 0.4rem 0; line-height: 1.15;
          }}
          .masthead p {{
            font-size: 0.93rem; color: {theme.INK_SECONDARY};
            margin: 0; max-width: 68ch; line-height: 1.55;
          }}

          .kpi-row {{ display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
          .kpi {{
            flex: 1 1 0; min-width: 140px; background: {theme.SURFACE};
            border: 1px solid {theme.BORDER}; border-radius: 10px; padding: 0.8rem 0.95rem;
          }}
          .kpi .label {{
            font-size: 0.68rem; font-weight: 650; letter-spacing: 0.07em;
            text-transform: uppercase; color: {theme.INK_MUTED}; margin-bottom: 0.3rem;
          }}
          .kpi .value {{
            font-size: 1.5rem; font-weight: 640; color: {theme.INK};
            line-height: 1.1; font-variant-numeric: tabular-nums;
          }}
          .kpi .sub {{ font-size: 0.73rem; color: {theme.INK_MUTED}; margin-top: 0.2rem; }}
          .kpi.is-lead {{ border-color: {theme.ACCENT}; box-shadow: inset 3px 0 0 {theme.ACCENT}; }}

          .panel {{
            background: {theme.SURFACE_SUNKEN}; border: 1px solid {theme.BORDER};
            border-left: 3px solid {theme.ACCENT}; border-radius: 8px;
            padding: 0.85rem 1.05rem; margin: 0.4rem 0 1rem 0;
            font-size: 0.87rem; color: {theme.INK_SECONDARY}; line-height: 1.55;
          }}
          .panel strong {{ color: {theme.INK}; font-weight: 650; }}

          .section-title {{
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
            text-transform: uppercase; color: {theme.INK_MUTED};
            margin: 1.5rem 0 0.7rem 0; padding-bottom: 0.4rem;
            border-bottom: 1px solid {theme.BORDER};
          }}

          .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {theme.BORDER}; }}
          .stTabs [data-baseweb="tab"] {{
            height: 2.6rem; padding: 0; font-size: 0.89rem; font-weight: 550;
            color: {theme.INK_SECONDARY};
          }}
          .stTabs [aria-selected="true"] {{ color: {theme.ACCENT}; font-weight: 650; }}

          section[data-testid="stSidebar"] {{ border-right: 1px solid {theme.BORDER}; }}
          section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
          .side-head {{
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
            text-transform: uppercase; color: {theme.INK_MUTED}; margin: 0 0 0.45rem 0;
          }}
          .side-note {{
            font-size: 0.74rem; color: {theme.INK_MUTED};
            line-height: 1.5; border-top: 1px solid {theme.BORDER};
            padding-top: 0.75rem; margin-top: 0.4rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    """Render a row of stat tiles. `cards` take label / value / sub / lead."""
    blocks = []
    for card in cards:
        classes = "kpi is-lead" if card.get("lead") else "kpi"
        sub = f"<div class='sub'>{html.escape(card.get('sub', ''))}</div>" if card.get("sub") else ""
        blocks.append(
            f"<div class='{classes}'>"
            f"<div class='label'>{html.escape(card['label'])}</div>"
            f"<div class='value'>{html.escape(card['value'])}</div>{sub}</div>"
        )
    st.markdown(f"<div class='kpi-row'>{''.join(blocks)}</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{html.escape(title)}</div>", unsafe_allow_html=True)


def panel(markup: str) -> None:
    st.markdown(f"<div class='panel'>{markup}</div>", unsafe_allow_html=True)


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


def confusion_matrix_figure(truth, predictions, labels, positive_label):
    """2x2 matrix shaded by row share, annotated with counts and quadrant names."""
    matrix = confusion_matrix(truth, predictions, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)

    figure, axes = plt.subplots(figsize=(4.4, 3.5))
    axes.imshow(shares, cmap=theme.BLUE_RAMP, vmin=0, vmax=1)

    quadrant = {
        (0, 0): "TN", (0, 1): "FP",
        (1, 0): "FN", (1, 1): "TP",
    }
    # Row/column order follows `labels`; flip the quadrant names if the positive
    # class is not the second label, so TP always marks the right cell.
    if labels[1] != positive_label:
        quadrant = {(0, 0): "TP", (0, 1): "FN", (1, 0): "FP", (1, 1): "TN"}

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            # White ink once the fill is dark enough to swallow black text.
            ink = "#ffffff" if shares[row, column] > 0.55 else theme.INK
            axes.text(
                column, row - 0.13, f"{matrix[row, column]:,}",
                ha="center", va="center", fontsize=15, fontweight="bold", color=ink,
            )
            axes.text(
                column, row + 0.19,
                f"{quadrant[(row, column)]} · {shares[row, column]:.0%} of row",
                ha="center", va="center", fontsize=7.5, color=ink, alpha=0.85,
            )

    axes.set_xticks(range(len(labels)), [f"predicted\n{label}" for label in labels])
    axes.set_yticks(range(len(labels)), [f"actual\n{label}" for label in labels])
    axes.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    # A surface-coloured minor grid separates the fills without drawing a border.
    axes.grid(which="minor", color=theme.SURFACE, linewidth=2.5)
    axes.tick_params(which="minor", length=0)
    theme.strip_chrome(axes, keep=())
    figure.tight_layout()
    return figure


def roc_figure(binary_truth, scores, auc_value):
    """Single-series ROC. No legend box -- the curve is labelled in place."""
    false_positive_rate, true_positive_rate, _ = roc_curve(binary_truth, scores)

    figure, axes = plt.subplots(figsize=(4.4, 3.5))
    axes.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), lw=1.2, color=theme.BORDER_STRONG)
    axes.text(0.70, 0.62, "chance", fontsize=7.5, color=theme.INK_MUTED, rotation=35, ha="center")

    axes.fill_between(false_positive_rate, true_positive_rate, alpha=0.1, color=theme.ACCENT, lw=0)
    axes.plot(false_positive_rate, true_positive_rate, lw=2, color=theme.ACCENT, solid_capstyle="round")

    axes.text(
        0.97, 0.08, f"AUC {auc_value:.3f}", ha="right", fontsize=11,
        fontweight="bold", color=theme.ACCENT,
    )
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.grid(axis="both", color=theme.BORDER, lw=0.7)
    axes.set_axisbelow(True)
    theme.strip_chrome(axes)
    figure.tight_layout()
    return figure


def comparison_figure(comparison: pd.DataFrame, colors: dict[str, str]):
    """Small multiples: one panel per metric, five directly-labelled bars.

    One grouped chart of 30 bars would need a legend and a lot of squinting.
    Faceting by metric keeps every panel on a shared 0-1 scale and lets each
    bar carry its own value, which also supplies the contrast relief the
    palette requires.
    """
    models = list(comparison.index)
    positions = np.arange(len(models))

    figure, axes_grid = plt.subplots(2, 3, figsize=(10.5, 5.0), sharex=True)

    for index, (axes, metric) in enumerate(zip(axes_grid.ravel(), METRIC_ORDER)):
        values = comparison[metric].to_numpy()
        best = int(np.argmax(values))

        axes.barh(
            positions, values, height=0.62,
            color=[colors[model] for model in models], zorder=3,
        )
        for position, value in enumerate(values):
            leads = position == best
            axes.text(
                value + 0.03, position, f"{value:.3f}", va="center", fontsize=8,
                fontweight="bold" if leads else "normal",
                color=theme.INK if leads else theme.INK_SECONDARY,
            )

        axes.set_title(metric, loc="left", pad=7)
        axes.set_xlim(0, 1.28)
        # Model names label the left-hand panel of each row; the row order is
        # identical across panels, so identity never rests on colour alone.
        axes.set_yticks(positions, models if index % 3 == 0 else [""] * len(models))
        axes.invert_yaxis()
        axes.set_xticks([0, 0.5, 1.0])
        axes.grid(axis="x", color=theme.BORDER, lw=0.7)
        axes.set_axisbelow(True)
        theme.strip_chrome(axes, keep=("left",))

    figure.tight_layout(h_pad=2.0, w_pad=1.4)
    return figure


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

inject_styles()

metadata, pipelines = load_artifacts()
TARGET = metadata["target_column"]
POSITIVE = metadata["positive_label"]
LABELS = metadata["class_labels"]
COLORS = theme.model_colors(pipelines.keys())

st.markdown(
    """
    <div class="masthead">
      <div class="eyebrow">UCI Bank Marketing &middot; Binary classification</div>
      <h1>Term Deposit Classifier Bench</h1>
      <p>Five classifiers trained on a Portuguese bank's direct-marketing campaign,
         scored side by side on held-out data. Models are loaded from disk as fitted
         pipelines &mdash; nothing is retrained in the browser.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='side-head'>Test data</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type="csv",
        label_visibility="collapsed",
        help=f"Needs the {len(metadata['feature_columns'])} feature columns plus the "
             f"`{TARGET}` target column.",
    )

    if uploaded is not None:
        data = read_csv(uploaded)
        data_origin = f"uploaded file — {uploaded.name}"
    else:
        data = read_csv(BUNDLED_TEST_CSV)
        data_origin = "bundled test_data.csv — the hold-out split"
        st.caption("No file uploaded. Showing the hold-out split shipped with the repository.")

    st.markdown("<div class='side-head' style='margin-top:1.4rem'>Model</div>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Classifier", list(pipelines.keys()), label_visibility="collapsed"
    )

    st.markdown(
        f"<div class='side-note'>Trained on {metadata['training_rows']:,} rows · "
        f"{len(metadata['feature_columns'])} features · seed {metadata['random_state']}<br>"
        f"Preprocessing is fitted inside each pipeline.</div>",
        unsafe_allow_html=True,
    )

problems = validate_columns(data, metadata)
if problems:
    st.error("This file cannot be scored:")
    for problem in problems:
        st.markdown(f"- {problem}")
    st.stop()

features = data[metadata["feature_columns"]]
truth = data[TARGET]
binary_truth = (truth == POSITIVE).astype(int)
majority_baseline = max(binary_truth.mean(), 1 - binary_truth.mean())

kpi_row([
    {"label": "Rows scored", "value": f"{len(data):,}", "sub": data_origin},
    {"label": "Features", "value": str(len(metadata["feature_columns"])), "sub": "after dropping duration"},
    {"label": f"Positives ({POSITIVE})", "value": f"{int(binary_truth.sum()):,}",
     "sub": f"{binary_truth.mean():.1%} of rows"},
    {"label": "Majority baseline", "value": f"{majority_baseline:.4f}",
     "sub": "accuracy if always predicting no"},
])

single_tab, compare_tab, data_tab = st.tabs(
    ["Selected model", "Model comparison", "Test data"]
)

# -- Selected model ---------------------------------------------------------

with single_tab:
    result = evaluate(pipelines[selected_model], features, truth, POSITIVE)
    metrics = result["metrics"]

    section(f"{selected_model} — hold-out performance")
    kpi_row([
        {"label": name, "value": f"{metrics[name]:.4f}", "lead": name == "MCC"}
        for name in METRIC_ORDER
    ])
    st.caption(
        "Precision, recall and F1 are reported for the positive class — the clients who "
        "subscribed. MCC is highlighted because it is the only metric here that accounts "
        "for all four confusion-matrix cells."
    )

    if metrics["Accuracy"] < majority_baseline:
        panel(
            f"<strong>Accuracy sits below the majority baseline.</strong> This model scores "
            f"{metrics['Accuracy']:.4f} against {majority_baseline:.4f} for always predicting "
            f"“{LABELS[0]}”. That is the intended trade: balanced class weights push it to "
            f"find {metrics['Recall']:.0%} of actual subscribers, which a baseline never does."
        )

    section("Confusion matrix and ROC")
    left, right = st.columns(2, gap="large")
    with left:
        st.pyplot(confusion_matrix_figure(truth, result["predictions"], LABELS, POSITIVE))
    with right:
        st.pyplot(roc_figure(binary_truth, result["scores"], metrics["AUC"]))

    section("Classification report")
    report = pd.DataFrame(
        classification_report(truth, result["predictions"], output_dict=True, zero_division=0)
    ).transpose()
    st.dataframe(
        report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                             "f1-score": "{:.4f}", "support": "{:,.0f}"}),
        width="stretch",
    )

# -- Comparison -------------------------------------------------------------

with compare_tab:
    comparison = pd.DataFrame(
        {
            name: evaluate(pipeline, features, truth, POSITIVE)["metrics"]
            for name, pipeline in pipelines.items()
        }
    ).transpose()[METRIC_ORDER]
    comparison.index.name = "ML Model Name"

    leader = comparison["MCC"].idxmax()
    accuracy_leader = comparison["Accuracy"].idxmax()

    section("All five classifiers on this file")
    st.dataframe(
        comparison.style
        .format("{:.4f}")
        .background_gradient(cmap=theme.BLUE_RAMP, axis=0, vmin=0, vmax=1),
        width="stretch",
        column_config={
            metric: st.column_config.Column(help=METRIC_HELP[metric]) for metric in METRIC_ORDER
        },
    )

    if leader != accuracy_leader:
        panel(
            f"<strong>{leader}</strong> leads on MCC ({comparison.loc[leader, 'MCC']:.4f}) and "
            f"AUC ({comparison.loc[leader, 'AUC']:.4f}), while <strong>{accuracy_leader}</strong> "
            f"tops accuracy at {comparison.loc[accuracy_leader, 'Accuracy']:.4f} — barely above the "
            f"{majority_baseline:.4f} you get by predicting “{LABELS[0]}” every time, and at a recall "
            f"of just {comparison.loc[accuracy_leader, 'Recall']:.1%}. Ranking by accuracy on an "
            f"imbalanced problem selects the wrong model."
        )
    else:
        panel(
            f"<strong>{leader}</strong> leads on MCC ({comparison.loc[leader, 'MCC']:.4f}) "
            f"and AUC ({comparison.loc[leader, 'AUC']:.4f})."
        )

    section("Metric by metric")
    st.pyplot(comparison_figure(comparison, COLORS))
    st.caption("Shared 0–1 scale across panels. The strongest value in each panel is bolded.")

    st.download_button(
        "Download comparison (CSV)",
        comparison.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# -- Test data --------------------------------------------------------------

with data_tab:
    section("Rows being scored")
    st.dataframe(data.head(200), width="stretch", height=320)
    st.caption(f"Showing the first {min(200, len(data)):,} of {len(data):,} rows.")

    section(f"Predictions — {selected_model}")
    preview = data.copy()
    preview["predicted"] = pipelines[selected_model].predict(features)
    preview["correct"] = preview["predicted"] == preview[TARGET]

    hits = int(preview["correct"].sum())
    kpi_row([
        {"label": "Correct", "value": f"{hits:,}", "sub": f"{hits / len(preview):.1%} of rows"},
        {"label": "Incorrect", "value": f"{len(preview) - hits:,}",
         "sub": f"{1 - hits / len(preview):.1%} of rows"},
    ])

    st.dataframe(
        preview[[TARGET, "predicted", "correct"]].head(200), width="stretch", height=320
    )
    st.download_button(
        "Download predictions (CSV)",
        preview.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
