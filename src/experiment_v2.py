from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TEAM_RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}


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


def temporal_split(df):
    train_part = df[
        df["created_at_ist"] < "2026-04-01"
    ].copy()

    validation_part = df[
        df["created_at_ist"] >= "2026-04-01"
    ].copy()

    return train_part, validation_part


def main():
    df = load_data()

    train_df, val_df = temporal_split(df)

    print("\n========== V2 DATA ==========")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))

    print(
        "Training dates:",
        train_df["created_at_ist"].min(),
        "to",
        train_df["created_at_ist"].max(),
    )

    print(
        "Validation dates:",
        val_df["created_at_ist"].min(),
        "to",
        val_df["created_at_ist"].max(),
    )

    categorical_features = [
        "channel",
        "product_family",
        "warranty_status",
    ]

    features = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
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
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    feature_columns = [
        "request_text",
        "channel",
        "product_family",
        "warranty_status",
    ]

    print("\nTraining V2 model...")

    model.fit(
        train_df[feature_columns],
        train_df["target"],
    )

    predictions = model.predict(
        val_df[feature_columns]
    )

    accuracy = accuracy_score(
        val_df["target"],
        predictions,
    )

    macro_f1 = f1_score(
        val_df["target"],
        predictions,
        average="macro",
    )

    error_rate = 1 - accuracy

    print("\n========== V2 RESULTS ==========")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Error rate: {error_rate:.4f}")

    print("\n========== VS V1 ==========")

    v1_metrics_path = OUTPUT_DIR / "v1_metrics.json"

    if v1_metrics_path.exists():
        with open(
            v1_metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            v1 = json.load(f)

        accuracy_change = (
            accuracy - v1["accuracy"]
        )

        f1_change = (
            macro_f1 - v1["macro_f1"]
        )

        print(
            f"V1 accuracy: {v1['accuracy']:.4f}"
        )
        print(
            f"V2 accuracy: {accuracy:.4f}"
        )
        print(
            f"Accuracy change: {accuracy_change:+.4f}"
        )

        print(
            f"V1 Macro F1: {v1['macro_f1']:.4f}"
        )
        print(
            f"V2 Macro F1: {macro_f1:.4f}"
        )
        print(
            f"Macro F1 change: {f1_change:+.4f}"
        )

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            val_df["target"],
            predictions,
            digits=4,
        )
    )

    labels = sorted(df["target"].unique())

    cm = confusion_matrix(
        val_df["target"],
        predictions,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm_df)

    results = val_df[
        [
            "request_id",
            "created_at_ist",
            "channel",
            "product_family",
            "warranty_status",
            "request_text",
            "team_label",
            "target",
        ]
    ].copy()

    results["predicted_team"] = predictions

    results["correct"] = (
        results["target"]
        == results["predicted_team"]
    )

    results.to_csv(
        OUTPUT_DIR / "v2_validation_predictions.csv",
        index=False,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "v2_confusion_matrix.csv"
    )

    metrics = {
        "experiment": "V2",
        "model": "Word TF-IDF + metadata + Logistic Regression",
        "features": [
            "request_text",
            "channel",
            "product_family",
            "warranty_status",
        ],
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "error_rate": round(float(error_rate), 6),
    }

    with open(
        OUTPUT_DIR / "v2_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nSaved:")
    print("outputs/v2_validation_predictions.csv")
    print("outputs/v2_confusion_matrix.csv")
    print("outputs/v2_metrics.json")

    print("\n========== V2 COMPLETE ==========")


if __name__ == "__main__":
    main()