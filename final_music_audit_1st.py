import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


BASE = Path("/Users/j/Desktop/AI_Projects/Kpop_project")

KPOP_DISCO = BASE / "data" / "kpopfullspotifydiscography" / "single_album_track_data.csv"
KPOP_HITS = BASE / "data" / "KPopHits2021.csv"

OUT = BASE / "results"
OUT.mkdir(exist_ok=True)


FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "key", "mode", "time_signature"
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\(.*?\)", "", x)
    x = re.sub(r"\[.*?\]", "", x)
    x = re.sub(r"feat\.|ft\.|featuring", "", x)
    x = re.sub(r"[^a-z0-9가-힣\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def split_artists(x):
    if pd.isna(x):
        return []
    raw = str(x).lower().strip()
    parts = re.split(r",|&|/|\+|\bx\b|\band\b", raw)
    normalized_parts = [normalize_text(p) for p in parts]
    return [p for p in normalized_parts if p]


def load_data():
    disco = pd.read_csv(KPOP_DISCO)
    hits = pd.read_csv(KPOP_HITS)

    disco["title_norm"] = disco["Track_Title"].apply(normalize_text)
    disco["artist_norm"] = disco["Artist"].apply(normalize_text)

    hits["title_norm"] = hits["title"].apply(normalize_text)
    hits["artist_list"] = hits["artist/s"].apply(split_artists)

    return disco, hits


def create_kpop_labels(disco, hits):
    df = disco.copy()

    hit_pairs = set()
    for _, row in hits.iterrows():
        for artist in row["artist_list"]:
            hit_pairs.add((row["title_norm"], artist))

    df["hit_strict"] = df.apply(
        lambda r: int((r["title_norm"], r["artist_norm"]) in hit_pairs),
        axis=1
    )

    hit_titles = set(hits["title_norm"])
    df["hit_loose"] = df["title_norm"].isin(hit_titles).astype(int)

    return df


def safe_mean(scores, key):
    if scores is None:
        return np.nan
    return float(np.nanmean(scores[key]))


def evaluate_label(df, label_col, label_name):
    use_features = [f for f in FEATURES if f in df.columns]

    required_cols = use_features + [label_col, "artist_norm"]
    data = df.dropna(subset=required_cols).copy()
    data = data.reset_index(drop=True)

    X = data[use_features]
    y = data[label_col].astype(int)
    groups = data["artist_norm"]

    positive_rate = y.mean()
    n_pos = int(y.sum())
    n_total = len(y)

    if n_pos < 10:
        print(f"Skipping {label_name}: too few positives ({n_pos})")
        return None, None

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        data.index,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    groups_train = groups.iloc[idx_train]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # WRONG pipeline: SMOTE before CV
    smote = SMOTE(random_state=42, k_neighbors=min(3, max(1, n_pos - 1)))
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_smote_scaled = scaler.fit_transform(X_smote)

    wrong_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    wrong_scores = cross_validate(
        wrong_model,
        X_smote_scaled,
        y_smote,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        n_jobs=-1,
    )

    # CORRECT pipeline: SMOTE inside CV
    correct_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    correct_scores = cross_validate(
        correct_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        n_jobs=-1,
    )

    # GROUP CV: artist-held-out evaluation
    group_cv_roc_auc = np.nan
    group_cv_pr_auc = np.nan

    if groups_train.nunique() >= 5:
        group_cv = GroupKFold(n_splits=5)

        try:
            group_scores = cross_validate(
                correct_pipe,
                X_train,
                y_train,
                cv=group_cv.split(X_train, y_train, groups_train),
                scoring=["roc_auc", "average_precision"],
                n_jobs=-1,
            )

            group_cv_roc_auc = safe_mean(group_scores, "test_roc_auc")
            group_cv_pr_auc = safe_mean(group_scores, "test_average_precision")

        except ValueError as e:
            print(f"GroupKFold skipped for {label_name}: {e}")

    # Held-out test
    correct_pipe.fit(X_train, y_train)
    probs = correct_pipe.predict_proba(X_test)[:, 1]

    test_roc = roc_auc_score(y_test, probs)
    test_pr = average_precision_score(y_test, probs)

    top_k = min(50, len(y_test))
    top_idx = np.argsort(probs)[::-1][:top_k]
    precision_at_50 = y_test.iloc[top_idx].mean()

    model = correct_pipe.named_steps["model"]
    importances = pd.DataFrame({
        "label": label_name,
        "feature": use_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics = {
        "label": label_name,
        "n_total": n_total,
        "n_positive": n_pos,
        "positive_rate": positive_rate,

        "wrong_cv_roc_auc": safe_mean(wrong_scores, "test_roc_auc"),
        "wrong_cv_pr_auc": safe_mean(wrong_scores, "test_average_precision"),

        "correct_cv_roc_auc": safe_mean(correct_scores, "test_roc_auc"),
        "correct_cv_pr_auc": safe_mean(correct_scores, "test_average_precision"),

        "group_cv_roc_auc": group_cv_roc_auc,
        "group_cv_pr_auc": group_cv_pr_auc,

        "heldout_roc_auc": test_roc,
        "heldout_pr_auc": test_pr,

        "random_pr_baseline": y_test.mean(),
        "precision_at_50": precision_at_50,
    }

    return metrics, importances


def matching_audit(df):
    loose_hits = df[df["hit_loose"] == 1].copy()

    collisions = (
        loose_hits.groupby("title_norm")["Artist"]
        .nunique()
        .reset_index(name="n_artists")
        .query("n_artists > 1")
        .sort_values("n_artists", ascending=False)
    )

    examples = loose_hits[
        loose_hits["title_norm"].isin(collisions["title_norm"].head(20))
    ][["Artist", "Track_Title", "title_norm", "hit_strict", "hit_loose"]]

    collisions.to_csv(OUT / "title_collision_summary.csv", index=False)
    examples.to_csv(OUT / "title_collision_examples.csv", index=False)

    print("\nSaved matching audit:")
    print("- title_collision_summary.csv")
    print("- title_collision_examples.csv")


def main():
    print("Loading data...")
    disco, hits = load_data()

    print("Creating labels...")
    kpop = create_kpop_labels(disco, hits)

    print("\nK-pop label counts:")
    print(kpop[["hit_strict", "hit_loose"]].sum())

    all_metrics = []
    all_importances = []

    experiments = [
        (kpop, "hit_strict", "kpop_strict_artist_title"),
        (kpop, "hit_loose", "kpop_loose_title_only"),
    ]

    for df, label_col, label_name in experiments:
        print(f"\n=== Running experiment: {label_name} ===")
        metrics, importances = evaluate_label(df, label_col, label_name)

        if metrics is not None:
            all_metrics.append(metrics)
            all_importances.append(importances)

            print("Positive rate:", round(metrics["positive_rate"], 4))
            print("Wrong CV PR-AUC:", round(metrics["wrong_cv_pr_auc"], 4))
            print("Correct CV PR-AUC:", round(metrics["correct_cv_pr_auc"], 4))
            print("Group CV PR-AUC:", round(metrics["group_cv_pr_auc"], 4))
            print("Held-out PR-AUC:", round(metrics["heldout_pr_auc"], 4))
            print("Precision@50:", round(metrics["precision_at_50"], 4))

    metrics_df = pd.DataFrame(all_metrics)

    if all_importances:
        importance_df = pd.concat(all_importances, ignore_index=True)
    else:
        importance_df = pd.DataFrame(columns=["label", "feature", "importance"])

    metrics_df.to_csv(OUT / "final_audit_metrics.csv", index=False)
    importance_df.to_csv(OUT / "final_feature_importance.csv", index=False)

    matching_audit(kpop)

    print("\nSaved:")
    print("- final_audit_metrics.csv")
    print("- final_feature_importance.csv")

    print("\nSummary:")
    print(metrics_df[
        [
            "label",
            "positive_rate",
            "wrong_cv_pr_auc",
            "correct_cv_pr_auc",
            "group_cv_pr_auc",
            "heldout_pr_auc",
            "random_pr_baseline",
            "precision_at_50",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()