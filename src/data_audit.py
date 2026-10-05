from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


TEAM_RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    resolution = pd.read_csv(DATA_DIR / "resolution_log.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    return train, test, resolution, teams, sample


def main():
    train, test, resolution, teams, sample = load_data()

    # Parse dates
    train["created_at_ist"] = pd.to_datetime(
        train["created_at_ist"], errors="coerce"
    )

    test["created_at_ist"] = pd.to_datetime(
        test["created_at_ist"], errors="coerce"
    )

    resolution["resolved_at"] = pd.to_datetime(
        resolution["resolved_at"], errors="coerce"
    )

    print("\n========== DATASET SIZE ==========")

    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Resolution rows:", len(resolution))
    print("Teams:", len(teams))
    print("Sample submission rows:", len(sample))

    print("\n========== DATE RANGE ==========")

    print(
        "Train:",
        train["created_at_ist"].min(),
        "to",
        train["created_at_ist"].max(),
    )

    print(
        "Test:",
        test["created_at_ist"].min(),
        "to",
        test["created_at_ist"].max(),
    )

    print("\n========== MISSING VALUES ==========")

    print("\nTrain:")
    print(train.isna().sum())

    print("\nTest:")
    print(test.isna().sum())

    print("\n========== DUPLICATE IDS ==========")

    print(
        "Train duplicate request_ids:",
        train["request_id"].duplicated().sum(),
    )

    print(
        "Test duplicate request_ids:",
        test["request_id"].duplicated().sum(),
    )

    print(
        "Resolution duplicate request_ids:",
        resolution["request_id"].duplicated().sum(),
    )

    print("\n========== RAW TEAM LABELS ==========")

    print(train["team_label"].value_counts())

    print("\nUnique raw labels:")
    print(sorted(train["team_label"].unique()))

    # Normalize renamed teams
    train["team_label_canonical"] = train["team_label"].replace(
        TEAM_RENAME_MAP
    )

    resolution["first_team_canonical"] = resolution[
        "first_team"
    ].replace(TEAM_RENAME_MAP)

    resolution["final_team_canonical"] = resolution[
        "final_team"
    ].replace(TEAM_RENAME_MAP)

    print("\n========== CURRENT TEAM LABELS ==========")

    print(
        sorted(train["team_label_canonical"].unique())
    )

    print(
        "Number of canonical teams:",
        train["team_label_canonical"].nunique(),
    )

    print("\n========== RESOLUTION LOG AUDIT ==========")

    merged = train.merge(
        resolution,
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    print(
        "Train rows matched to resolution log:",
        merged["final_team"].notna().sum(),
        "/",
        len(train),
    )

    print(
        "team_label == first_team:",
        (
            merged["team_label"]
            == merged["first_team"]
        ).mean(),
    )

    merged["team_label_canonical"] = merged[
        "team_label"
    ].replace(TEAM_RENAME_MAP)

    merged["first_team_canonical"] = merged[
        "first_team"
    ].replace(TEAM_RENAME_MAP)

    merged["final_team_canonical"] = merged[
        "final_team"
    ].replace(TEAM_RENAME_MAP)

    final_match_rate = (
        merged["first_team_canonical"]
        == merged["final_team_canonical"]
    ).mean()

    print(
        "First team == final team:",
        round(final_match_rate, 4),
    )

    print(
        "Requests ending in another team:",
        (
            merged["first_team_canonical"]
            != merged["final_team_canonical"]
        ).sum(),
    )

    print(
        "Requests with at least one transfer:",
        (merged["transfers"] > 0).sum(),
    )

    print(
        "Total transfers:",
        merged["transfers"].sum(),
    )

    print("\n========== TEST / SUBMISSION CHECK ==========")

    print(
        "Sample rows == test rows:",
        len(sample) == len(test),
    )

    print(
        "Sample request IDs match test order:",
        sample["request_id"].equals(test["request_id"]),
    )

    print("\nTest source distribution:")
    print(test["source"].value_counts())

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()