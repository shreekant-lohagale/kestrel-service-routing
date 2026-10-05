from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


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


def build_model():
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
                [
                    "channel",
                    "product_family",
                    "warranty_status",
                ],
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
    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    resolution = pd.read_csv(
        DATA_DIR / "resolution_log.csv"
    )

    train["created_at_ist"] = pd.to_datetime(
        train["created_at_ist"],
        errors="coerce",
    )

    df = train.merge(
        resolution[
            [
                "request_id",
                "final_team",
            ]
        ],
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    df["target"] = (
        df["final_team"]
        .replace(TEAM_RENAME_MAP)
    )

    train_df = df[
        df["created_at_ist"] < "2026-04-01"
    ].copy()

    val_df = df[
        df["created_at_ist"] >= "2026-04-01"
    ].copy()

    model = build_model()

    print("\nTraining outcome model...")

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["target"],
    )

    predictions = model.predict(
        val_df[FEATURE_COLUMNS]
    )

    scores = model.decision_function(
        val_df[FEATURE_COLUMNS]
    )

    sorted_scores = np.sort(
        scores,
        axis=1,
    )[:, ::-1]

    margins = (
        sorted_scores[:, 0]
        - sorted_scores[:, 1]
    )

    val_df["prediction"] = predictions
    val_df["margin"] = margins
    val_df["correct"] = (
        val_df["prediction"]
        == val_df["target"]
    )

    overall_accuracy = accuracy_score(
        val_df["target"],
        predictions,
    )

    print(
        "\n========== OVERALL =========="
    )

    print(
        f"Validation accuracy: "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Validation rows: "
        f"{len(val_df)}"
    )

    print(
        "\n========== THRESHOLD ANALYSIS =========="
    )

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
    ]

    rows = []

    for threshold in thresholds:

        review = val_df[
            val_df["margin"] < threshold
        ]

        auto = val_df[
            val_df["margin"] >= threshold
        ]

        review_count = len(review)
        auto_count = len(auto)

        review_error_rate = (
            1 - review["correct"].mean()
            if review_count
            else 0
        )

        auto_accuracy = (
            auto["correct"].mean()
            if auto_count
            else 0
        )

        review_share = (
            review_count / len(val_df)
        )

        rows.append(
            {
                "threshold": threshold,
                "review_rows": review_count,
                "review_share": review_share,
                "review_error_rate":
                    review_error_rate,
                "auto_rows": auto_count,
                "auto_accuracy":
                    auto_accuracy,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "review_share":
                    lambda x: f"{x:.2%}",

                "review_error_rate":
                    lambda x: f"{x:.2%}",

                "auto_accuracy":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print(
        "\n========== MARGIN BY CORRECTNESS =========="
    )

    print(
        val_df.groupby("correct")[
            "margin"
        ].describe()
    )

    result.to_csv(
        OUTPUT_DIR
        / "margin_threshold_analysis.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/margin_threshold_analysis.csv"
    )

    print(
        "\n========== COMPLETE =========="
    )


if __name__ == "__main__":
    main()