# Term Deposit Classifier Bench

Comparing five classification algorithms on the UCI **Bank Marketing** dataset, with an
interactive Streamlit front-end that scores saved models against an uploaded test CSV.

> M.Tech (AIML) — Machine Learning, Assignment 2 (BITS Pilani WILP)

---

## a. Problem statement

A Portuguese retail bank ran a direct-marketing campaign over the telephone, calling
existing clients and asking them to open a **term deposit**. Most people said no: only
about 11.7% of calls converted. Every call costs agent time, so blanket-dialling the whole
client list wastes most of the budget.

The task is framed as **binary classification**: given what the bank knows about a client
*before* the call is placed — demographics, existing products, and the history of previous
campaign contacts — predict whether that client will subscribe to a term deposit
(`subscribed` ∈ {`yes`, `no`}). A usable model lets the bank rank the call list and work
down it, spending the same agent-hours on a far denser pool of likely subscribers.

Two consequences shape everything below:

1. **The classes are heavily imbalanced (≈ 88 / 12).** Plain accuracy is close to useless
   here — a model that predicts `no` for every single client scores 88.3% and identifies
   zero customers. The evaluation therefore leans on **AUC**, **recall** and
   **MCC**, which cannot be gamed that way.
2. **Recall matters more than precision.** A false positive costs one wasted phone call.
   A false negative is a customer the bank never calls and a deposit it never books. The
   models that support it are fitted with `class_weight="balanced"` for this reason.

### A note on the `duration` column

The raw dataset includes `duration`, the length of the last contact call in seconds. It is
the single most predictive column — and it is **unusable**. Its value is only known once
the call has already ended, at which point the outcome is known too; the UCI dataset
description says plainly that it "should be discarded if the intention is to have a
realistic predictive model."

Keeping it inflates every AUC in the table below to roughly 0.90. **This project drops
it.** The numbers reported here are lower than many published notebooks on the same
dataset, and deliberately so — they reflect what is actually knowable at the moment the
bank has to decide whether to dial. 15 honest predictors remain.

---

## b. Dataset description

| Property | Value |
| --- | --- |
| Name | Bank Marketing (`bank-full.csv`) |
| Source | UCI Machine Learning Repository — [dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Donated by | S. Moro, P. Rita, P. Cortez (2014) |
| Instances in source file | **45,211** |
| Features used | **15** (16 in the source, minus the leaky `duration`) |
| Target | `subscribed` (renamed from `y`) — binary, `yes` / `no` |
| Class balance | 88.3% `no` / 11.7% `yes` |
| Missing values | None encoded as NaN; unknowns appear as the literal category `"unknown"` |

Both the assignment minimums are met comfortably: **15 features ≥ 12** and
**45,211 instances ≥ 500**.

### Features

**Client demographics (numeric + categorical)**

| Column | Type | Description |
| --- | --- | --- |
| `age` | numeric | Client age in years |
| `job` | categorical | Occupation — 12 levels (`admin.`, `blue-collar`, `management`, …) |
| `marital` | categorical | `married`, `single`, `divorced` |
| `education` | categorical | `primary`, `secondary`, `tertiary`, `unknown` |

**Existing banking relationship**

| Column | Type | Description |
| --- | --- | --- |
| `default` | categorical | Has credit in default? |
| `balance` | numeric | Average yearly balance, in euros |
| `housing` | categorical | Has a housing loan? |
| `loan` | categorical | Has a personal loan? |

**Current campaign contact**

| Column | Type | Description |
| --- | --- | --- |
| `contact` | categorical | Contact channel — `cellular`, `telephone`, `unknown` |
| `day` | numeric | Day of month of last contact |
| `month` | categorical | Month of last contact (`jan` … `dec`) |
| `campaign` | numeric | Number of contacts made during this campaign for this client |

**Previous campaign history**

| Column | Type | Description |
| --- | --- | --- |
| `pdays` | numeric | Days since the client was last contacted in a previous campaign (`-1` = never) |
| `previous` | numeric | Number of contacts before this campaign |
| `poutcome` | categorical | Outcome of the previous campaign — `success`, `failure`, `other`, `unknown` |

**Target:** `subscribed` — did the client subscribe to a term deposit?

### Sampling and the train/test split

Support Vector Machines scale roughly quadratically in the number of training rows, and
Streamlit Community Cloud's free tier is memory-constrained. A **stratified subsample of
12,000 rows** (`random_state=2026`) is drawn from the 45,211, preserving the original
positive rate exactly. That subsample is split **80 / 20, stratified**:

- **9,600 rows** train all five models
- **2,400 rows** are held out and written to [`test_data.csv`](test_data.csv) — 281 positives

Every metric in this README comes from that hold-out set, which no model saw during
fitting. The file is written to disk *before* training begins.

### Preprocessing

All five classifiers share one identical `ColumnTransformer`, so differences in the
comparison table come from the algorithms rather than from the features:

- **Numeric columns** → `StandardScaler`. Essential for KNN and SVM, which are
  distance-based: without it, `balance` (range ≈ −8,000 to 100,000) would drown out `age`.
- **Categorical columns** → `OneHotEncoder(handle_unknown="ignore")`. The `ignore`
  setting means an unseen category in an uploaded CSV degrades to an all-zero block
  instead of crashing the app.

Each transformer is fitted on training data only and saved **inside** the model pipeline,
so the Streamlit app applies exactly the same transformation at scoring time and no
preprocessing artifact can drift out of sync.

---

## c. GitHub repository link

<https://github.com/SrujanKashyap16/ML_Assignment02_2026AJ05032>

**Live Streamlit app:** <https://ml-assignment-02-2026aj05032-d69cklzmzkbevcqqwdc5gc.streamlit.app>

### Repository layout

```
ML_Assignment02_2026AJ05032/
├── app.py                  Streamlit application (loads saved models, never retrains)
├── theme.py                Design tokens + Matplotlib styling shared by every figure
├── train_models.py         Source code: trains + persists all five pipelines
├── requirements.txt        Deployment dependencies
├── .streamlit/
│   └── config.toml         Pinned light theme, so charts and page share one surface
├── README.md               This file
├── test_data.csv           2,400-row hold-out split (15 features + target)
├── model/
│   ├── logistic_regression.joblib      fitted Pipeline (preprocessing + estimator)
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── metadata.joblib                 column manifest, class labels, split sizes
├── data/
│   ├── bank-full.csv       Original UCI source file (45,211 rows)
│   └── bank-names.txt      Original UCI column documentation
└── reports/
    ├── metrics.csv         Comparison table, machine-readable
    └── metrics.md          Comparison table, Markdown (reproduced below)
```

Each `.joblib` is a complete `sklearn.pipeline.Pipeline` — preprocessing and estimator in
one object — so there are no loose preprocessing artifacts to lose.

### Reproducing the results

```bash
git clone https://github.com/SrujanKashyap16/ML_Assignment02_2026AJ05032.git
cd ML_Assignment02_2026AJ05032

python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate  on Linux/macOS
pip install -r requirements.txt

python train_models.py          # regenerates model/, test_data.csv, reports/
streamlit run app.py            # opens the app at http://localhost:8501
```

`train_models.py` is seeded with `random_state=2026` throughout, so the subsample, the
split and every metric below reproduce exactly.

---

## d. Models used

Five classifiers, each wrapped in the shared preprocessing pipeline described above and
fitted on the same 9,600 training rows.

| Model | Key hyperparameters | Rationale |
| --- | --- | --- |
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` | Linear baseline; interpretable coefficients |
| Decision Tree | `max_depth=8`, `min_samples_leaf=25`, `class_weight="balanced"` | Depth and leaf size capped to stop it memorising the training set |
| KNN | `n_neighbors=25`, `weights="distance"` | k=25 smooths the sparse minority class; distance weighting favours close neighbours |
| Naive Bayes | `GaussianNB` (default) | Gaussian rather than Multinomial — the scaled numeric columns take negative values, which Multinomial cannot accept |
| Support Vector Machine | `SVC(kernel="rbf", C=1.0, gamma="scale")`, `class_weight="balanced"` | RBF kernel for non-linear boundaries |

**AUC calculation.** Logistic Regression, Decision Tree, KNN and Naive Bayes all expose
`predict_proba`. SVC is deliberately fitted **without** `probability=True`: Platt scaling
would trigger an internal 5-fold refit for no benefit, since ROC AUC only needs a
monotonic ranking and `decision_function` already supplies one. Both routes are valid
inputs to `roc_auc_score`.

**Class-weighting is not uniform.** `class_weight="balanced"` is applied to Logistic
Regression, the Decision Tree and the SVM. KNN and GaussianNB have no equivalent
parameter in scikit-learn. That asymmetry is itself one of the clearest findings below.

### Comparison table

All figures computed on the **2,400-row hold-out set** (281 positives). Precision, recall
and F1 are reported for the **positive class (`yes`)**, which is the class the bank
actually cares about.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7388 | 0.7137 | 0.2379 | 0.5587 | 0.3337 | 0.2314 |
| Decision Tree | 0.7742 | 0.7104 | 0.2682 | 0.5374 | 0.3578 | 0.2602 |
| KNN | 0.8883 | 0.7213 | 0.6444 | 0.1032 | 0.1779 | 0.2267 |
| Naive Bayes | 0.8304 | 0.7005 | 0.3190 | 0.3950 | 0.3529 | 0.2586 |
| **Support Vector Machine** | 0.8162 | **0.7296** | 0.3268 | 0.5374 | **0.4065** | **0.3185** |

**Baseline for reference:** predicting `no` for every client scores **0.8829 accuracy**,
0.5 AUC, and **0.0 on recall, F1 and MCC**. Read the accuracy column against that number,
not against zero.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | The **highest recall in the table (0.5587)** — it finds more subscribers than anything else — bought at the cost of the **lowest accuracy (0.7388)** and worst precision (0.2379). `class_weight="balanced"` pushes the decision threshold far toward `yes`, so it flags roughly one client in three and is wrong about three-quarters of those. Its AUC (0.7137) is mid-pack, which is the more telling number: the *ranking* is competitive, and the weak precision is a threshold artefact rather than a modelling failure. The real ceiling is the linear decision boundary — it cannot express interactions like "contacted in March **and** `poutcome=success`", which is exactly where the signal in this dataset lives. Fast to train, and its coefficients are directly readable, so it remains the sensible explainability baseline. |
| **Decision Tree** | Sits mid-table almost everywhere: accuracy 0.7742, recall 0.5374, MCC 0.2602 — a modest but genuine improvement on Logistic Regression on every imbalance-aware metric, confirming that the non-linear splits capture real structure. Its weak spot is **AUC (0.7104, the second-lowest)**, and the cause is structural rather than a tuning miss: a depth-8 tree assigns only one probability per leaf, so its score distribution is a coarse step function with few distinct values, and the ROC curve it traces is correspondingly blocky. In other words it classifies acceptably but *ranks* poorly — a real limitation when the business goal is to sort a call list. Capping `max_depth=8` and `min_samples_leaf=25` was necessary: unconstrained, it drove training accuracy toward 1.0 and hold-out MCC sharply down. |
| **KNN** | **The most instructive result in the table, and a textbook accuracy paradox.** Its 0.8883 accuracy is the highest of all five models and is very nearly worthless — it barely clears the 0.8829 you get by answering `no` every time. The recall of **0.1032** exposes why: it identifies roughly **one subscriber in ten**. With no `class_weight` parameter available and an 88% majority, almost every 25-neighbour ball around a query point is dominated by `no`, so the vote almost never tips. The flip side is the **best precision in the table (0.6444)** — on the rare occasion it does say `yes` it is right about two times in three — but a campaign tool that ignores 90% of its prospects cannot be deployed. Its **lowest MCC (0.2267)** correctly ranks it last overall, which is precisely the point of reporting MCC alongside accuracy. Also the most expensive model to ship: 3.9 MB, since it stores the entire training set. |
| **Naive Bayes** | The **weakest AUC (0.7005)**, and the conditional-independence assumption is the clear culprit — it is violated twice over here. One-hot encoding turns each categorical column into a block of mutually exclusive dummies that are strongly *negatively* correlated by construction, and the genuine real-world correlations (`job` with `education`, `pdays` with `previous` and `poutcome`) compound the problem. GaussianNB additionally assumes each feature is normally distributed within a class, which is badly wrong for the skewed `balance` and for `pdays`, where the `-1` sentinel for "never contacted" forms a large spike no Gaussian can fit. Despite all that it lands a respectable MCC of 0.2586 — better than KNN — because it is the only unweighted model that still produces a usable recall (0.3950). Near-instant to train and 7 KB on disk; a reasonable fallback when compute is the binding constraint, not accuracy. |
| **Support Vector Machine** | **The strongest model on this dataset.** It takes the top spot on the three metrics that matter under imbalance — **AUC 0.7296, F1 0.4065, MCC 0.3185** — and it is the only model that avoids a glaring weakness anywhere in the row. It matches the Decision Tree's recall (0.5374) while ranking distinctly better (AUC 0.7296 vs 0.7104), and it reaches that recall at higher precision than Logistic Regression (0.3268 vs 0.2379). The RBF kernel supplies the non-linear boundary the linear model lacks, while `class_weight="balanced"` keeps the minority class in view — the combination the other four each miss one half of. The costs are real and worth stating: it was by far the slowest to fit (the reason for the 12,000-row subsample), the model is 2.6 MB of support vectors, and its `decision_function` margins are not calibrated probabilities. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF kernel).** It leads on AUC, F1 and MCC simultaneously, and MCC is the right tie-breaker here because it is the only one of the six metrics that accounts for all four confusion-matrix cells and so cannot be inflated by the 88% majority class. The accuracy column deserves an explicit warning: **KNN "wins" accuracy at 0.8883 and is the worst model in the table**, catching one subscriber in ten against the SVM's five. Ranking by accuracy on this dataset would select exactly the wrong model — the single most important lesson the comparison teaches. In deployment terms the SVM roughly **quadruples the hit rate** of untargeted dialling (32.7% precision against an 11.7% base rate) while still reaching over half of all subscribers. If probability calibration or sub-millisecond inference were hard requirements, Logistic Regression is the pragmatic second choice: comparable AUC, trivial to serve, and its coefficients explain themselves to a business stakeholder. |

---

## e. Streamlit application

**Live app:** <https://ml-assignment-02-2026aj05032-d69cklzmzkbevcqqwdc5gc.streamlit.app>

The app loads the five `.joblib` pipelines from `model/` and scores them against the test
CSV. **No model is retrained in the browser** — `train_models.py` is the only thing that
ever calls `.fit()`.

### Required features

| # | Requirement | Where it lives |
| --- | --- | --- |
| a | **Dataset upload (CSV)** | Sidebar file uploader. Falls back to the bundled `test_data.csv` so the app is never empty on first load. |
| b | **Model selection dropdown** | Sidebar `selectbox` listing all five classifiers; the whole *Selected model* tab reacts to it. |
| c | **Display of evaluation metrics** | Six metric tiles — Accuracy, AUC, Precision, Recall, F1, MCC — plus a full five-model comparison table on its own tab. |
| d | **Confusion matrix / classification report** | Both: an annotated confusion-matrix heatmap *and* the per-class classification report. |

### Beyond the minimum

- **"All five models" tab** — every classifier scored on the uploaded file in one
  gradient-shaded table, with a grouped bar chart of all six metrics side by side, so the
  comparison is visible on the deployed app rather than only in this README.
- **ROC curve** per model, drawn from the same scores used for the AUC figure.
- **Upload validation** — a missing target column or missing features produces a specific,
  named error listing exactly which columns are absent, instead of a stack trace.
- **Automatic winner callout** — highlights the best MCC on whatever file is loaded, with
  a note on why MCC is the right summary under imbalance.
- **Downloadable outputs** — the comparison table and per-row predictions export as CSV.
- **Caching** — `@st.cache_resource` for the pipelines, `@st.cache_data` for the CSV, so
  switching models re-scores without re-reading from disk.

### Uploading your own test file

The CSV must contain all **15 feature columns plus the `subscribed` target column**, with
the same names and categorical spellings as `test_data.csv`. The easiest valid upload is
`test_data.csv` itself, or any row subset of it.

---

## Deployment notes

Deployed on **Streamlit Community Cloud** — sign in with GitHub, *New app*, select this
repository, branch `main`, main file `app.py`.

Two things worth flagging, since missing dependencies are the most common deployment
failure:

- `requirements.txt` pins **`scikit-learn>=1.9`**. The pipelines were pickled with 1.9.1,
  and joblib artifacts are not guaranteed to load across major scikit-learn versions — an
  older resolved version is the likeliest cause of a deployment-time load error.
- **Every chart is plain Matplotlib**, with no seaborn dependency. Fonts are pinned to
  DejaVu Sans, which ships inside Matplotlib, so the figures render identically on the
  deployment host without relying on any system font being installed.

Total repository size is ~11 MB, well inside the free tier.

---

## Tech stack

Python 3.13 · scikit-learn · pandas · NumPy · Streamlit · Matplotlib · joblib

## Dataset citation

> S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank
> Telemarketing.* Decision Support Systems, Elsevier, 62:22-31, June 2014.
> UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
