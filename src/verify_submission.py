from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

PREDICTIONS_PATH = (
    ROOT / "predictions.csv"
)

TEST_PATH = (
    DATA_DIR / "test_unlabelled.csv"
)

MODEL_PATH = (
    MODEL_DIR / "routing_model.joblib"
)


ALLOWED_TEAMS = {
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims",
}


def fail(message):
    raise AssertionError(message)


def main():
    print(
        "\n========== SUBMISSION VERIFICATION =========="
    )

    # ------------------------------------------------
    # Required files
    # ------------------------------------------------

    if not PREDICTIONS_PATH.exists():
        fail(
            "predictions.csv is missing."
        )

    if not TEST_PATH.exists():
        fail(
            "test_unlabelled.csv is missing."
        )

    if not MODEL_PATH.exists():
        fail(
            "routing_model.joblib is missing."
        )

    print("✓ Required files exist")

    # ------------------------------------------------
    # Load
    # ------------------------------------------------

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    # ------------------------------------------------
    # Exact columns
    # ------------------------------------------------

    expected_columns = [
        "request_id",
        "team",
    ]

    if list(predictions.columns) != expected_columns:
        fail(
            f"Wrong prediction columns. "
            f"Expected {expected_columns}, "
            f"got {list(predictions.columns)}"
        )

    print(
        "✓ Correct columns: "
        "request_id, team"
    )

    # ------------------------------------------------
    # Row count
    # ------------------------------------------------

    if len(predictions) != len(test):
        fail(
            "Prediction row count does not "
            "match test set."
        )

    print(
        f"✓ Correct row count: "
        f"{len(predictions)}"
    )

    # ------------------------------------------------
    # Request IDs
    # ------------------------------------------------

    if predictions[
        "request_id"
    ].duplicated().any():

        fail(
            "Duplicate request IDs found "
            "in predictions.csv."
        )

    if not predictions[
        "request_id"
    ].equals(test["request_id"]):

        fail(
            "Prediction request IDs do not "
            "match the test set in order."
        )

    print(
        "✓ Request IDs match test set"
    )

    # ------------------------------------------------
    # Nulls
    # ------------------------------------------------

    if predictions.isna().any().any():
        fail(
            "Missing values found in "
            "predictions.csv."
        )

    print(
        "✓ No missing values"
    )

    # ------------------------------------------------
    # Valid teams
    # ------------------------------------------------

    predicted_teams = set(
        predictions["team"].unique()
    )

    invalid_teams = (
        predicted_teams
        - ALLOWED_TEAMS
    )

    if invalid_teams:
        fail(
            f"Invalid team names found: "
            f"{invalid_teams}"
        )

    print(
        "✓ All predictions use valid "
        "current team names"
    )

    # ------------------------------------------------
    # All seven teams present
    # ------------------------------------------------

    missing_teams = (
        ALLOWED_TEAMS
        - predicted_teams
    )

    if missing_teams:
        print(
            "Warning: no test predictions "
            f"for teams: {missing_teams}"
        )
    else:
        print(
            "✓ All seven teams appear "
            "in predictions"
        )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model_size_mb = (
        MODEL_PATH.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"✓ Model exists "
        f"({model_size_mb:.2f} MB)"
    )

    # ------------------------------------------------
    # Distribution
    # ------------------------------------------------

    print(
        "\nPredicted team distribution:"
    )

    print(
        predictions["team"]
        .value_counts()
        .to_string()
    )

    print(
        "\n========== VERIFICATION PASSED =========="
    )

    print(
        "predictions.csv is ready "
        "for submission."
    )


if __name__ == "__main__":
    main()