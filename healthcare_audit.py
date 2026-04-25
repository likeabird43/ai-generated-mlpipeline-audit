import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


# =========================
# Config
# =========================

RANDOM_STATE = 42

BASE = Path("/Users/j/Desktop/AI_Projects/Kpop_project")
HEALTH_DATA = BASE / "data" / "heart_disease_uci.csv"

OUT = BASE / "results"
OUT.mkdir(exist_ok=True)


# =========================
# Data Loading
# =========================

def load_health_data():
    df = pd.read_csv(HEALTH_DATA)

    # Recode physiologically invalid values as missing.
    # In this dataset, chol=0 and trestbps=0 are not clinically plausible.
    if "chol" in df.columns:
        df.loc[df["chol"] == 0, "chol"] = np.nan

    if "trestbps" in df.columns:
        df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

    if "num" not in df.columns:
        raise ValueError("Dataset must contain target column: num")

    # Proxy target:
    # heart_disease = 1 if num > 0.
    # This combines different disease severities and should not be interpreted
    # as a clean clinical endpoint.
    df["heart_disease"] = (df["num"] > 0).astype(int)

    return df


# =========================
# Pipeline
# =========================

def build_pipeline(X):
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols)
        ]
    )

    # Simple baseline model.
    # The goal is not model optimization, but auditing evaluation reliability.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipe


# =========================
# Evaluation Helpers
# =========================

def evaluate_predictions(y_true, probs, preds):
    return {
        "n": len(y_true),
        "positive_rate": y_true.mean(),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        # Threshold-dependent metric.
        # Included only as a secondary diagnostic.
        "balanced_accuracy": balanced_accuracy_score(y_true, preds)
    }


def save_coefficients(pipe, out_path):
    """
    Save logistic regression coefficients for interpretability audit.

    Important:
    Coefficients are diagnostic only. They should not be over-interpreted
    as causal or clinically stable feature importance.
    """
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.coef_[0]))]

    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(out_path, index=False)


# =========================
# Audits
# =========================

def random_split_evaluation(df):
    drop_cols = ["num", "heart_disease"]

    # Do not use ID-like columns.
    if "id" in df.columns:
        drop_cols.append("id")

    X = df.drop(columns=drop_cols)
    y = df["heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]

    # Threshold-dependent; secondary diagnostic only.
    preds = (probs >= 0.5).astype(int)

    metrics = evaluate_predictions(y_test, probs, preds)
    metrics["evaluation"] = "random_split"
    metrics["held_out_site"] = "mixed"
    metrics["note"] = (
        "Naive random split. May overestimate performance due to cohort mixing."
    )

    save_coefficients(
        pipe,
        OUT / "healthcare_random_split_coefficients.csv"
    )

    return metrics


def site_heldout_evaluation(df):
    if "dataset" not in df.columns:
        raise ValueError("Site-held-out evaluation requires a 'dataset' column.")

    results = []

    for site in sorted(df["dataset"].dropna().unique()):
        train_df = df[df["dataset"] != site].copy()
        test_df = df[df["dataset"] == site].copy()

        drop_cols = ["num", "heart_disease"]

        if "id" in df.columns:
            drop_cols.append("id")

        # IMPORTANT:
        # Remove dataset from features for site-held-out evaluation.
        # Otherwise the model can learn cohort identity directly.
        if "dataset" in df.columns:
            drop_cols.append("dataset")

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["heart_disease"]

        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df["heart_disease"]

        if y_train.nunique() < 2:
            print(f"Skipping {site}: only one class present in training set.")
            continue

        if y_test.nunique() < 2:
            print(f"Skipping {site}: only one class present in test set.")
            continue

        pipe = build_pipeline(X_train)
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]

        # Threshold-dependent; secondary diagnostic only.
        preds = (probs >= 0.5).astype(int)

        metrics = evaluate_predictions(y_test, probs, preds)
        metrics["evaluation"] = "site_heldout"
        metrics["held_out_site"] = site
        metrics["note"] = (
            "Site-held-out validation. More realistic test of cross-cohort generalization."
        )

        results.append(metrics)

        safe_site_name = str(site).replace(" ", "_").replace("/", "_")
        save_coefficients(
            pipe,
            OUT / f"healthcare_site_heldout_coefficients_{safe_site_name}.csv"
        )

    return results


def missingness_audit(df):
    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )

    missing["missing_rate"] = missing["missing_count"] / len(df)
    missing = missing.sort_values("missing_rate", ascending=False)

    missing.to_csv(
        OUT / "healthcare_missingness_audit.csv",
        index=False
    )

    if "dataset" in df.columns:
        # Version-safe approach across pandas releases.
        site_missing = df.groupby("dataset", dropna=False).apply(
            lambda x: x.isna().mean()
        )
        site_missing.index.name = "dataset"
        site_missing = site_missing.reset_index()

        site_missing.to_csv(
            OUT / "healthcare_site_missingness_audit.csv",
            index=False
        )


# =========================
# Main
# =========================

def main():
    print("Loading healthcare dataset...")
    df = load_health_data()

    print("\nDataset shape:", df.shape)

    print("\nTarget definition:")
    print("heart_disease = 1 if num > 0")
    print(
        "Note: This is a proxy target and may combine different disease severities."
    )

    print("\nTarget distribution:")
    print(df["heart_disease"].value_counts(normalize=True).rename("rate"))

    if "dataset" in df.columns:
        print("\nPositive rate by source cohort:")
        print(df.groupby("dataset")["heart_disease"].mean())

    print("\nRunning missingness audit...")
    missingness_audit(df)

    all_metrics = []

    print("\nRunning random split evaluation...")
    random_metrics = random_split_evaluation(df)
    all_metrics.append(random_metrics)

    print("Random split:")
    print(random_metrics)

    print("\nRunning site-held-out evaluation...")
    site_metrics = site_heldout_evaluation(df)
    all_metrics.extend(site_metrics)

    for m in site_metrics:
        print(m)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df = metrics_df[
        [
            "evaluation",
            "held_out_site",
            "n",
            "positive_rate",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "note"
        ]
    ]

    metrics_df.to_csv(
        OUT / "healthcare_audit_metrics.csv",
        index=False
    )

    print("\nSaved:")
    print("- results/healthcare_audit_metrics.csv")
    print("- results/healthcare_missingness_audit.csv")
    if "dataset" in df.columns:
        print("- results/healthcare_site_missingness_audit.csv")
    print("- results/healthcare_random_split_coefficients.csv")
    print("- results/healthcare_site_heldout_coefficients_[site].csv")

    print("\nSummary:")
    print(metrics_df.to_string(index=False))

    print("\nInterpretation note:")
    print(
        "ROC-AUC and PR-AUC are emphasized because they are threshold-independent. "
        "Balanced accuracy is reported only as a secondary diagnostic."
    )


if __name__ == "__main__":
    main()