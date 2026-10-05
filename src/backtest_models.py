from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
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

    df["initial_target"] = (
        df["team_label"]
        .replace(TEAM_RENAME_MAP)
    )

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


def evaluate_fold(
    df,
    train_end,
    val_start,
    val_end,
    fold_name,
):
    train_df = df[
        df["created_at_ist"] < train_end
    ].copy()

    val_df = df[
        (df["created_at_ist"] >= val_start)
        & (df["created_at_ist"] < val_end)
    ].copy()

    print(
        f"\n========== {fold_name} =========="
    )

    print(
        "Training rows:",
        len(train_df),
    )

    print(
        "Validation rows:",
        len(val_df),
    )

    # -----------------------------
    # MODEL 1: imitate old bot
    # -----------------------------

    initial_model = build_model()

    initial_model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["initial_target"],
    )

    initial_predictions = (
        initial_model.predict(
            val_df[FEATURE_COLUMNS]
        )
    )

    initial_accuracy = accuracy_score(
        val_df["initial_target"],
        initial_predictions,
    )

    initial_f1 = f1_score(
        val_df["initial_target"],
        initial_predictions,
        average="macro",
    )

    # -----------------------------
    # OLD BOT vs actual outcome
    # -----------------------------

    old_bot_outcome_accuracy = (
        accuracy_score(
            val_df["final_target"],
            val_df["initial_target"],
        )
    )

    # -----------------------------
    # MODEL 2: predict final team
    # -----------------------------

    outcome_model = build_model()

    outcome_model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["final_target"],
    )

    outcome_predictions = (
        outcome_model.predict(
            val_df[FEATURE_COLUMNS]
        )
    )

    outcome_accuracy = accuracy_score(
        val_df["final_target"],
        outcome_predictions,
    )

    outcome_f1 = f1_score(
        val_df["final_target"],
        outcome_predictions,
        average="macro",
    )

    print(
        f"Label-match model accuracy: "
        f"{initial_accuracy:.4f}"
    )

    print(
        f"Label-match Macro F1: "
        f"{initial_f1:.4f}"
    )

    print(
        f"Old bot → final team: "
        f"{old_bot_outcome_accuracy:.4f}"
    )

    print(
        f"Outcome model → final team: "
        f"{outcome_accuracy:.4f}"
    )

    print(
        f"Outcome Macro F1: "
        f"{outcome_f1:.4f}"
    )

    print(
        f"Outcome improvement over old bot: "
        f"{outcome_accuracy - old_bot_outcome_accuracy:+.4f}"
    )

    return {
        "fold": fold_name,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "label_match_accuracy": initial_accuracy,
        "label_match_macro_f1": initial_f1,
        "old_bot_final_accuracy":
            old_bot_outcome_accuracy,
        "outcome_model_accuracy":
            outcome_accuracy,
        "outcome_model_macro_f1":
            outcome_f1,
        "outcome_improvement":
            outcome_accuracy
            - old_bot_outcome_accuracy,
    }


def main():
    df = load_data()

    folds = [
        {
            "name": "Fold 1 - Oct-Dec 2025",
            "train_end": "2025-10-01",
            "val_start": "2025-10-01",
            "val_end": "2026-01-01",
        },
        {
            "name": "Fold 2 - Jan-Mar 2026",
            "train_end": "2026-01-01",
            "val_start": "2026-01-01",
            "val_end": "2026-04-01",
        },
        {
            "name": "Fold 3 - Apr-Jun 2026",
            "train_end": "2026-04-01",
            "val_start": "2026-04-01",
            "val_end": "2026-07-01",
        },
    ]

    results = []

    for fold in folds:
        result = evaluate_fold(
            df=df,
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            fold_name=fold["name"],
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n========== BACKTEST SUMMARY =========="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n========== AVERAGES =========="
    )

    print(
        "Average label-match accuracy:",
        round(
            results_df[
                "label_match_accuracy"
            ].mean(),
            4,
        ),
    )

    print(
        "Average old-bot final accuracy:",
        round(
            results_df[
                "old_bot_final_accuracy"
            ].mean(),
            4,
        ),
    )

    print(
        "Average outcome-model accuracy:",
        round(
            results_df[
                "outcome_model_accuracy"
            ].mean(),
            4,
        ),
    )

    print(
        "Latest-quarter outcome accuracy:",
        round(
            results_df.iloc[-1][
                "outcome_model_accuracy"
            ],
            4,
        ),
    )

    results_df.to_csv(
        OUTPUT_DIR / "backtest_results.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/backtest_results.csv"
    )

    print(
        "\n========== BACKTEST COMPLETE =========="
    )


if __name__ == "__main__":
    main()