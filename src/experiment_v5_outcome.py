from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
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


def load_data():
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
        resolution,
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    # Old bot routing decision
    df["initial_target"] = (
        df["team_label"]
        .replace(TEAM_RENAME_MAP)
    )

    # Actual team that resolved the request
    df["final_target"] = (
        df["final_team"]
        .replace(TEAM_RENAME_MAP)
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

    model = Pipeline(
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

    return model


def main():
    df = load_data()

    train_df = df[
        df["created_at_ist"] < "2026-04-01"
    ].copy()

    val_df = df[
        df["created_at_ist"] >= "2026-04-01"
    ].copy()

    print("\n========== V5 DATA ==========")

    print(
        "Training rows:",
        len(train_df),
    )

    print(
        "Validation rows:",
        len(val_df),
    )

    # --------------------------------
    # Existing bot vs actual outcome
    # --------------------------------

    old_bot_accuracy = accuracy_score(
        val_df["final_target"],
        val_df["initial_target"],
    )

    print(
        "\n========== OLD BOT VS FINAL OUTCOME =========="
    )

    print(
        f"Old bot final-team accuracy: "
        f"{old_bot_accuracy:.4f}"
    )

    print(
        f"Old bot error rate: "
        f"{1 - old_bot_accuracy:.4f}"
    )

    # --------------------------------
    # Train outcome-aware model
    # --------------------------------

    model = build_model()

    print(
        "\nTraining model on FINAL TEAM..."
    )

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["final_target"],
    )

    predictions = model.predict(
        val_df[FEATURE_COLUMNS]
    )

    accuracy = accuracy_score(
        val_df["final_target"],
        predictions,
    )

    macro_f1 = f1_score(
        val_df["final_target"],
        predictions,
        average="macro",
    )

    error_rate = 1 - accuracy

    print(
        "\n========== V5 OUTCOME RESULTS =========="
    )

    print(
        f"Accuracy:   {accuracy:.4f}"
    )

    print(
        f"Macro F1:   {macro_f1:.4f}"
    )

    print(
        f"Error rate: {error_rate:.4f}"
    )

    improvement = (
        accuracy - old_bot_accuracy
    )

    print(
        "\n========== VS OLD BOT =========="
    )

    print(
        f"Old bot → final team: "
        f"{old_bot_accuracy:.4f}"
    )

    print(
        f"New model → final team: "
        f"{accuracy:.4f}"
    )

    print(
        f"Improvement: "
        f"{improvement:+.4f}"
    )

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            val_df["final_target"],
            predictions,
            digits=4,
        )
    )

    results = val_df[
        [
            "request_id",
            "request_text",
            "channel",
            "product_family",
            "warranty_status",
            "initial_target",
            "final_target",
        ]
    ].copy()

    results[
        "predicted_final_team"
    ] = predictions

    results[
        "correct"
    ] = (
        results["final_target"]
        == results["predicted_final_team"]
    )

    results.to_csv(
        OUTPUT_DIR
        / "v5_outcome_validation.csv",
        index=False,
    )

    metrics = {
        "experiment": "V5 outcome model",
        "target": "final_team",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "old_bot_final_accuracy":
            round(
                float(old_bot_accuracy),
                6,
            ),
        "model_final_accuracy":
            round(
                float(accuracy),
                6,
            ),
        "macro_f1":
            round(
                float(macro_f1),
                6,
            ),
        "error_rate":
            round(
                float(error_rate),
                6,
            ),
        "improvement_over_old_bot":
            round(
                float(improvement),
                6,
            ),
    }

    with open(
        OUTPUT_DIR
        / "v5_outcome_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        "\n========== V5 COMPLETE =========="
    )


if __name__ == "__main__":
    main()