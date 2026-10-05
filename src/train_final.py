from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


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


ALLOWED_TEAMS = {
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims",
}


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
    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    resolution = pd.read_csv(
        DATA_DIR / "resolution_log.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test_unlabelled.csv"
    )

    sample = pd.read_csv(
        DATA_DIR / "sample_submission.csv"
    )

    # Join actual resolution outcome
    train = train.merge(
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

    train["target"] = (
        train["final_team"]
        .replace(TEAM_RENAME_MAP)
    )

    print("\n========== FINAL TRAINING ==========")

    print(
        "Training rows:",
        len(train),
    )

    print(
        "Test rows:",
        len(test),
    )

    print(
        "Target teams:",
        sorted(
            train["target"].unique()
        ),
    )

    if train["target"].isna().any():
        raise ValueError(
            "Missing final_team values found."
        )

    unknown_targets = (
        set(train["target"].unique())
        - ALLOWED_TEAMS
    )

    if unknown_targets:
        raise ValueError(
            f"Unexpected target teams: "
            f"{unknown_targets}"
        )

    model = build_model()

    print(
        "\nTraining final outcome model..."
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["target"],
    )

    print(
        "Training complete."
    )

    # Save trained model
    model_path = (
        MODEL_DIR
        / "routing_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Saved model: {model_path}"
    )

    # Predict hidden test rows
    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    # Build exact submission shape
    submission = sample.copy()

    submission["team"] = predictions

    # ----------------------------
    # VALIDATION CHECKS
    # ----------------------------

    if len(submission) != len(test):
        raise ValueError(
            "Submission row count "
            "does not match test."
        )

    if not submission[
        "request_id"
    ].equals(test["request_id"]):
        raise ValueError(
            "Submission request IDs "
            "do not match test order."
        )

    if submission[
        "request_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate request IDs found."
        )

    invalid_predictions = (
        set(submission["team"].unique())
        - ALLOWED_TEAMS
    )

    if invalid_predictions:
        raise ValueError(
            f"Invalid predicted teams: "
            f"{invalid_predictions}"
        )

    output_path = (
        ROOT / "predictions.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n========== PREDICTIONS =========="
    )

    print(
        "Rows:",
        len(submission),
    )

    print(
        "\nPredicted team distribution:"
    )

    print(
        submission["team"]
        .value_counts()
    )

    print(
        "\nFirst 10 predictions:"
    )

    print(
        submission.head(10)
        .to_string(index=False)
    )

    print(
        "\nSaved:"
    )

    print(
        "predictions.csv"
    )

    print(
        "\n========== FINAL MODEL COMPLETE =========="
    )


if __name__ == "__main__":
    main()