from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

    print("\n========== V4 DATA ==========")
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

    print("\nTraining V4 model...")

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["target"],
    )

    predictions = model.predict(
        val_df[FEATURE_COLUMNS]
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

    print("\n========== V4 RESULTS ==========")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Error rate: {error_rate:.4f}")

    print("\n========== VS V3 ==========")

    v3_metrics_path = OUTPUT_DIR / "v3_metrics.json"

    if v3_metrics_path.exists():
        with open(
            v3_metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            v3 = json.load(f)

        accuracy_change = accuracy - v3["accuracy"]
        f1_change = macro_f1 - v3["macro_f1"]

        print(f"V3 accuracy: {v3['accuracy']:.4f}")
        print(f"V4 accuracy: {accuracy:.4f}")
        print(f"Accuracy change: {accuracy_change:+.4f}")

        print(f"V3 Macro F1: {v3['macro_f1']:.4f}")
        print(f"V4 Macro F1: {macro_f1:.4f}")
        print(f"Macro F1 change: {f1_change:+.4f}")

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
        OUTPUT_DIR / "v4_validation_predictions.csv",
        index=False,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "v4_confusion_matrix.csv"
    )

    metrics = {
        "experiment": "V4",
        "model": (
            "Word + Character TF-IDF "
            "+ metadata + LinearSVC"
        ),
        "features": FEATURE_COLUMNS,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "error_rate": round(float(error_rate), 6),
    }

    with open(
        OUTPUT_DIR / "v4_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nSaved:")
    print("outputs/v4_validation_predictions.csv")
    print("outputs/v4_confusion_matrix.csv")
    print("outputs/v4_metrics.json")

    print("\n========== V4 COMPLETE ==========")


if __name__ == "__main__":
    main()