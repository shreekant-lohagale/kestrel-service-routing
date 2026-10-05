from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "routing_model.joblib"
)


FEATURE_COLUMNS = [
    "request_text",
    "channel",
    "product_family",
    "warranty_status",
]


REVIEW_MARGIN_THRESHOLD = 0.20


app = FastAPI(
    title="Kestrel Home Service Router",
    version="1.1.0",
    description=(
        "Routes a Kestrel Home service request "
        "to the predicted resolving team."
    ),
)


class ServiceRequest(BaseModel):
    request_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Customer opening message "
            "or IVR transcript"
        ),
    )

    channel: str = Field(
        ...,
        description=(
            "ivr | chat | whatsapp | email"
        ),
    )

    product_family: str = Field(
        ...,
        description="Kestrel product family",
    )

    warranty_status: str = Field(
        ...,
        description=(
            "in_warranty | shield | "
            "out_of_warranty"
        ),
    )


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. "
            "Run `python src/train_final.py` first."
        )

    return joblib.load(MODEL_PATH)


MODEL = load_model()


def clean_feature_name(name: str) -> str:
    """
    Convert sklearn feature names into
    employee-readable text.
    """

    if name.startswith("word_tfidf__"):
        term = name.replace(
            "word_tfidf__",
            "",
            1,
        )

        return f'text phrase "{term}"'

    if name.startswith("char_tfidf__"):
        term = name.replace(
            "char_tfidf__",
            "",
            1,
        )

        return f'text pattern "{term}"'

    if name.startswith(
        "categorical__channel_"
    ):
        value = name.replace(
            "categorical__channel_",
            "",
            1,
        )

        return f"channel is {value}"

    if name.startswith(
        "categorical__product_family_"
    ):
        value = name.replace(
            "categorical__product_family_",
            "",
            1,
        )

        return (
            f"product family is {value}"
        )

    if name.startswith(
        "categorical__warranty_status_"
    ):
        value = name.replace(
            "categorical__warranty_status_",
            "",
            1,
        )

        return (
            f"warranty status is {value}"
        )

    return name


def explain_prediction(
    row: pd.DataFrame,
    predicted_team: str,
):
    """
    Return readable reasons for the selected team.

    The explanations come from positive model
    feature contributions. Character n-grams are
    used by the classifier but hidden from the
    employee-facing explanation because they are
    harder to interpret.
    """

    feature_pipeline = (
        MODEL.named_steps["features"]
    )

    classifier = (
        MODEL.named_steps["classifier"]
    )

    transformed = (
        feature_pipeline.transform(row)
    )

    feature_names = (
        feature_pipeline
        .get_feature_names_out()
    )

    classes = classifier.classes_

    class_index = np.where(
        classes == predicted_team
    )[0][0]

    class_weights = (
        classifier.coef_[class_index]
    )

    feature_values = (
        transformed.toarray()[0]
    )

    contributions = (
        feature_values
        * class_weights
    )

    ranked_indices = np.argsort(
        contributions
    )[::-1]

    reasons = []

    weak_terms = {
        "working",
        "showing",
        "help",
        "please",
        "sir",
        "hello",
        "hi",
        "team",
        "thanks",
        "kindly",
        "good morning",
        "good evening",
        "asap",
    }

    # Prefer useful word-level and metadata
    # explanations over character fragments.
    for index in ranked_indices:
        contribution = contributions[index]

        if contribution <= 0:
            continue

        raw_name = feature_names[index]

        if raw_name.startswith(
            "char_tfidf__"
        ):
            continue

        if raw_name.startswith(
            "word_tfidf__"
        ):
            phrase = raw_name.replace(
                "word_tfidf__",
                "",
                1,
            ).strip()

            if phrase in weak_terms:
                continue

        reason = clean_feature_name(
            raw_name
        )

        if reason not in reasons:
            reasons.append(reason)

        if len(reasons) >= 2:
            break

    # Always add operational context where useful.
    product_reason = (
        f"product family is "
        f"{row.iloc[0]['product_family']}"
    )

    warranty_reason = (
        f"warranty status is "
        f"{row.iloc[0]['warranty_status']}"
    )

    channel_reason = (
        f"channel is "
        f"{row.iloc[0]['channel']}"
    )

    if (
        product_reason not in reasons
        and len(reasons) < 3
    ):
        reasons.append(
            product_reason
        )

    if (
        warranty_reason not in reasons
        and len(reasons) < 3
    ):
        reasons.append(
            warranty_reason
        )

    if (
        channel_reason not in reasons
        and len(reasons) < 3
    ):
        reasons.append(
            channel_reason
        )

    # Final fallback
    if not reasons:
        reasons = [
            product_reason,
            warranty_reason,
        ]

    return reasons[:3]


@app.get("/")
def root():
    return {
        "service":
            "Kestrel Home Service Router",

        "status":
            "running",

        "docs":
            "/docs",

        "health":
            "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "review_margin_threshold":
            REVIEW_MARGIN_THRESHOLD,
    }


@app.post("/predict")
def predict(
    request: ServiceRequest,
):
    try:
        row = pd.DataFrame(
            [
                {
                    "request_text":
                        request.request_text.strip(),

                    "channel":
                        request.channel,

                    "product_family":
                        request.product_family,

                    "warranty_status":
                        request.warranty_status,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        predicted_team = (
            MODEL.predict(row)[0]
        )

        scores = (
            MODEL.decision_function(row)[0]
        )

        sorted_scores = np.sort(
            scores
        )[::-1]

        margin = float(
            sorted_scores[0]
            - sorted_scores[1]
        )

        needs_human_review = (
            margin
            < REVIEW_MARGIN_THRESHOLD
        )

        routing_status = (
            "human_review"
            if needs_human_review
            else "auto_route"
        )

        reasons = explain_prediction(
            row=row,
            predicted_team=predicted_team,
        )

        return {
            "predicted_team":
                predicted_team,

            "routing_status":
                routing_status,

            "needs_human_review":
                needs_human_review,

            "decision_margin":
                round(margin, 4),

            "review_threshold":
                REVIEW_MARGIN_THRESHOLD,

            "reasons":
                reasons,

            "note": (
                "Requests below the review threshold "
                "are flagged for human review. "
                "The decision margin is an SVM "
                "ranking margin, not a probability."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )