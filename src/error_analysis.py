from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


TEAM_RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}


FEATURE_COLUMNS = [
    "request_text",
    "channel",
    "product_family",
    "warranty_status",
]


def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")

    df["created_at_ist"] = pd.to_datetime(
        df["created_at_ist"],
        errors="coerce",
    )

    df["target"] = df["team_label"].replace(
        TEAM_RENAME_MAP
    )

    return df


def build_model():
    categorical_features = [
        "channel",
        "product_family",
        "warranty_status",
    ]

    features = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
                "request_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=50000,
                ),
                "request_text",
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    df = load_data()

    train_df = df[
        df["created_at_ist"] < "2026-04-01"
    ].copy()

    val_df = df[
        df["created_at_ist"] >= "2026-04-01"
    ].copy()

    model = build_model()

    print("\nTraining V4 for error analysis...")

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["target"],
    )

    predictions = model.predict(
        val_df[FEATURE_COLUMNS]
    )

    decision_scores = model.decision_function(
        val_df[FEATURE_COLUMNS]
    )

    val_df["predicted_team"] = predictions

    val_df["correct"] = (
        val_df["target"]
        == val_df["predicted_team"]
    )

    # Highest and second-highest SVM scores
    sorted_scores = pd.DataFrame(
        decision_scores
    ).apply(
        lambda row: sorted(row, reverse=True),
        axis=1,
    )

    val_df["top_score"] = [
        values[0]
        for values in sorted_scores
    ]

    val_df["second_score"] = [
        values[1]
        for values in sorted_scores
    ]

    val_df["margin"] = (
        val_df["top_score"]
        - val_df["second_score"]
    )

    errors = val_df[
        ~val_df["correct"]
    ].copy()

    print("\n========== ERROR SUMMARY ==========")

    print("Validation rows:", len(val_df))
    print("Correct:", val_df["correct"].sum())
    print("Errors:", len(errors))

    print(
        "Error rate:",
        round(len(errors) / len(val_df), 4),
    )

    print("\n========== ERRORS BY TRUE TEAM ==========")

    print(
        errors["target"]
        .value_counts()
    )

    print("\n========== CONFUSION PAIRS ==========")

    confusion_pairs = (
        errors.groupby(
            ["target", "predicted_team"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    print(
        confusion_pairs
        .head(20)
        .to_string(index=False)
    )

    print("\n========== LOW-MARGIN ERRORS ==========")

    columns = [
        "request_id",
        "target",
        "predicted_team",
        "channel",
        "product_family",
        "warranty_status",
        "margin",
        "request_text",
    ]

    print(
        errors
        .sort_values("margin")
        [columns]
        .head(20)
        .to_string(index=False)
    )

    print("\n========== HIGH-MARGIN ERRORS ==========")

    print(
        errors
        .sort_values(
            "margin",
            ascending=False,
        )
        [columns]
        .head(15)
        .to_string(index=False)
    )

    print("\n========== ERROR ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    main()