import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ---------- Streamlit page config ----------
st.set_page_config(layout="wide", page_title="Mini AutoML")

# ---------- Cached CSV reader ----------
@st.cache_data
def read_csv(file):
    """Read a CSV (or gzipped CSV) into a DataFrame with caching."""
    return pd.read_csv(file, low_memory=False)

# ---------- Small helpers ----------
def is_classification(y: pd.Series) -> bool:
    """Heuristic: treat as classification if object dtype or <=10 unique values."""
    if y.dtype == "O":
        return True
    return y.dropna().nunique() <= 10

def split_cols(df, target):
    """Split into feature, numeric, and categorical column lists."""
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return feats, num, cat

def preprocessor(num_cols, cat_cols, scale_numeric=True):
    """Build a preprocessing ColumnTransformer for numeric and categorical columns."""
    num_pipe_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_pipe_steps.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_pipe_steps)

    # Keep it simple & widely compatible
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

def get_models(task):
    """Return a small, fast model zoo for the selected task."""
    if task == "classification":
        return {
            "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }

def safe_folds_for_data(task, y, n_folds):
    """
    Make CV folds safe for the data size/class balance to avoid errors:
    - For classification, folds cannot exceed the minority class size.
    - For regression, folds cannot exceed n_samples.
    """
    n_folds = int(n_folds)
    if task == "classification":
        vc = y.value_counts()
        if len(vc) < 2:
            return None
        max_folds = int(vc.min())
        return max(2, min(n_folds, max_folds))
    else:
        return max(2, min(n_folds, len(y)))

def compute_auc_for_fold(pipe, X_te, y_te):
    """
    AUC for binary/multiclass:
    - Binary: use proba of the positive class.
    - Multiclass: macro AUC (OvR) with explicit labels to align columns.
    Returns None if predict_proba is unavailable.
    """
    try:
        if not hasattr(pipe, "predict_proba"):
            return None
        proba = pipe.predict_proba(X_te)
        classes = pipe.named_steps["model"].classes_

        if len(classes) == 2:
            pos_label = classes[1]
            pos_idx = list(classes).index(pos_label)
            y_true_bin = (pd.Series(y_te) == pos_label).astype(int).values
            auc = roc_auc_score(y_true_bin, proba[:, pos_idx])
        else:
            auc = roc_auc_score(y_te, proba, multi_class="ovr",
                                average="macro", labels=classes)
        return float(auc)
    except Exception:
        return None

def plot_grouped_metrics_altair(res_df, metrics, title="", fixed_domain_01=False):
    """
    One grouped bar chart for all models × metrics.
    Hover a metric to isolate it (others fade). Works on Altair v5; falls back for v4.
    """
    plot_df = res_df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    ).dropna()

    hover_sel = alt.selection_point(
        fields=["Metric"],
        on="mouseover",
        clear="mouseout",
        nearest=False,
        empty=True,
    )

    y_scale = alt.Scale(domain=[0, 1]) if fixed_domain_01 else alt.Undefined

    base = alt.Chart(plot_df).encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0, title=None)),
        xOffset=alt.XOffset("Metric:N"),
        y=alt.Y("Value:Q", scale=y_scale, title=None),
        color=alt.Color("Metric:N", legend=alt.Legend(orient="top")),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Value:Q", format=".3f"),
        ],
    )

    bars = base.mark_bar(size=22).encode(
        opacity=alt.condition(hover_sel, alt.value(1.0), alt.value(0.15))
    ).add_params(hover_sel)

    chart = bars.properties(title=title, height=420)
    try:
        st.altair_chart(chart, use_container_width=True)  # Altair v5
    except Exception:
        # Fallback for Altair v4
        hover_sel_v4 = alt.selection_single(
            fields=["Metric"], on="mouseover", empty="all", nearest=False
        )
        bars_v4 = base.mark_bar(size=22).encode(
            opacity=alt.condition(hover_sel_v4, alt.value(1.0), alt.value(0.15))
        ).add_selection(hover_sel_v4)
        st.altair_chart(bars_v4.properties(title=title, height=420),
                        use_container_width=True)

# ---------- UI ----------
st.title("Welcome to Mini AutoML")
st.markdown("""
    Meet **Mini AutoML** — your no-fuss, one-screen machine-learning helper.
    
    Upload a CSV (or .csv.gz), pick your target, and Mini AutoML does the rest: it auto-detects whether you’re doing **classification** or **regression**, builds a clean preprocessing pipeline (missing-value imputation, one-hot encoding, numeric scaling), and runs **three classic models** out of the box: **Logistic Regression, Random Forest,** and **Gradient Boosting**.
    
    Under the hood, it uses **K-fold cross-validation** and shows crisp, comparable metrics:
    
    * Classification: **Accuracy, Precision (macro), Recall (macro), F1 (macro), AUC**
    * Regression: **RMSE, MAE, R²**
    
    Results land in a tidy table plus a **single interactive grouped bar chart** (hover a metric to spotlight it!). Smart safeguards keep things smooth: folds adjust automatically for tiny classes, AUC appears only when it’s valid, and errors come with clear guidance.
    
    Built with **Streamlit + scikit-learn + Altair**, Mini AutoML turns “which model should I use?” into a quick, visual decision. Upload, run, compare — and walk away with a best model you can defend.

    """)

st.divider()
file = st.file_uploader("Upload a CSV (or .csv.gz)", type=["csv", "gz"])
if not file:
    st.info("Please upload a dataset to begin.")
    st.stop()

df = read_csv(file)
st.caption(f"Rows: {len(df):,}  |  Cols: {len(df.columns)}")
st.dataframe(df.head(200), use_container_width=True)

# Target selection
default_target = df.columns[-1]
target = st.selectbox("Target column", options=df.columns, index=len(df.columns)-1)

# Task detection with override
auto_task = "classification" if is_classification(df[target]) else "regression"
task = st.radio("Task type", ["classification", "regression"],
                index=(0 if auto_task == "classification" else 1), horizontal=True)

# CV folds
n_folds = st.slider("Cross-validation folds", 3, 10, 5, step=1)

# ----- Prepare data -----
mask = df[target].notna()
data = df.loc[mask].copy()

feats, num_cols, cat_cols = split_cols(data, target)
if len(feats) == 0:
    st.error("No feature columns available (only the target is present). Pick a different target.")
    st.stop()

X = data[feats]
y = data[target]

if task == "classification":
    # Use string labels so StratifiedKFold behaves as expected
    y = y.astype(str).str.strip()

eff_folds = safe_folds_for_data(task, y, n_folds)
if eff_folds is None:
    st.error("Classification needs at least 2 classes in the target. Please select a different target.")
    st.stop()
if eff_folds != n_folds:
    st.warning(f"Adjusted folds to {eff_folds} to match dataset size/class balance.")

prep = preprocessor(num_cols, cat_cols, scale_numeric=True)
models = get_models(task)

splitter = StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=42) if task == "classification" \
           else KFold(n_splits=eff_folds, shuffle=True, random_state=42)

# Run once automatically + button to run again
run_clicked = st.button("Run")
if run_clicked or "auto_ran" not in st.session_state:
    st.session_state["auto_ran"] = True

    with st.spinner(f"Running {eff_folds}-fold cross-validation..."):
        rows = []

        for name, est in models.items():
            pipe = Pipeline([("pre", prep), ("model", est)])

            if task == "classification":
                accs, precs, recs, f1s, aucs = [], [], [], [], []

                for tr, te in splitter.split(X, y):
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                    y_pred = pipe.predict(X.iloc[te])
                    y_true = y.iloc[te].values

                    accs.append(accuracy_score(y_true, y_pred))
                    precs.append(precision_score(y_true, y_pred, average="macro", zero_division=0))
                    recs.append(recall_score(y_true, y_pred, average="macro", zero_division=0))
                    f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

                    auc_val = compute_auc_for_fold(pipe, X.iloc[te], y_true)
                    if auc_val is not None:
                        aucs.append(auc_val)

                rows.append({
                    "Model": name,
                    "Accuracy": float(np.mean(accs)),
                    "Precision_macro": float(np.mean(precs)),
                    "Recall_macro": float(np.mean(recs)),
                    "F1_macro": float(np.mean(f1s)),
                    "AUC_macro": (float(np.mean(aucs)) if len(aucs) > 0 else np.nan),
                })

            else:  # regression
                rmses, maes, r2s = [], [], []
                for tr, te in splitter.split(X, y):
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                    y_pred = pipe.predict(X.iloc[te])
                    y_true = y.iloc[te].values

                    # RMSE = sqrt(MSE) for older sklearn compatibility
                    rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
                    maes.append(float(mean_absolute_error(y_true, y_pred)))
                    r2s.append(float(r2_score(y_true, y_pred)))

                rows.append({
                    "Model": name,
                    "RMSE": float(np.mean(rmses)),
                    "MAE": float(np.mean(maes)),
                    "R2": float(np.mean(r2s)),
                })

    # ----- Results table + plots -----
    res = pd.DataFrame(rows)
    st.subheader("Cross-validated metrics (mean over folds)")

    if task == "classification":
        cols_order = ["Model", "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "AUC_macro"]
        present = [c for c in cols_order if c in res.columns]
        res = res[present].round(4)
        st.dataframe(res, use_container_width=True)

        # Pick the winner by F1 (macro); fall back to Accuracy if F1 is missing
        if "F1_macro" in res.columns and res["F1_macro"].notna().any():
            best_row = res.sort_values("F1_macro", ascending=False).iloc[0]
            best_msg = (f"Best model by F1 (macro): **{best_row['Model']}** "
                        f"(F1={best_row['F1_macro']:.3f}, "
                        f"Acc={best_row.get('Accuracy', float('nan')):.3f}, "
                        f"Prec={best_row.get('Precision_macro', float('nan')):.3f}, "
                        f"Rec={best_row.get('Recall_macro', float('nan')):.3f}"
                        f"{', AUC='+str(round(best_row.get('AUC_macro', float('nan')),3)) if 'AUC_macro' in res.columns and not pd.isna(best_row.get('AUC_macro', np.nan)) else ''})")
        else:
            best_row = res.sort_values("Accuracy", ascending=False).iloc[0]
            best_msg = (f"Best model by Accuracy: **{best_row['Model']}** "
                        f"(Acc={best_row['Accuracy']:.3f})")

        # One combined chart with interactive highlight
        metrics_to_plot = [m for m in ["F1_macro", "Accuracy", "Precision_macro", "Recall_macro", "AUC_macro"] if m in res.columns]
        st.subheader("Model comparison")
        plot_grouped_metrics_altair(
            res, metrics_to_plot,
            title="Models vs. Metrics (hover a metric to isolate it)",
            fixed_domain_01=True
        )

        st.success(best_msg)

    else:
        cols_order = ["Model", "RMSE", "MAE", "R2"]
        res = res[cols_order].round(4)
        st.dataframe(res, use_container_width=True)

        best_row = res.sort_values("RMSE", ascending=True).iloc[0]
        st.subheader("Model comparison")
        plot_grouped_metrics_altair(
            res, ["RMSE", "MAE", "R2"],
            title="Models vs. Metrics (hover a metric to isolate it)",
            fixed_domain_01=False
        )

        st.success(f"Best model by RMSE: **{best_row['Model']}** "
                   f"(RMSE={best_row['RMSE']:.4f}, MAE={best_row['MAE']:.4f}, R²={best_row['R2']:.4f})")
