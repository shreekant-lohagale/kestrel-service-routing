import os

import requests
import streamlit as st


API_URL = os.getenv(
    "KESTREL_API_URL",
    "http://127.0.0.1:8000",
)


PRODUCT_FAMILIES = [
    "Air Fryer",
    "Mixer Grinder",
    "Water Purifier",
    "Robot Vacuum",
    "Induction Cooktop",
    "Ceiling Fan",
    "Room Heater",
]


CHANNELS = [
    "chat",
    "whatsapp",
    "email",
    "ivr",
]


WARRANTY_OPTIONS = {
    "In warranty": "in_warranty",
    "Kestrel Shield": "shield",
    "Out of warranty": "out_of_warranty",
}


st.set_page_config(
    page_title="Kestrel Service Router",
    page_icon="↗",
    layout="centered",
)


st.title("Kestrel Home Service Router")

st.caption(
    "Route a new service request to the team "
    "most likely to resolve it."
)


# --------------------------------------------------
# API STATUS
# --------------------------------------------------

api_online = False
review_threshold = None

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=2,
    )

    if health_response.status_code == 200:
        api_online = True

        health_data = health_response.json()

        review_threshold = health_data.get(
            "review_margin_threshold"
        )

except requests.RequestException:
    api_online = False


if api_online:
    st.success(
        "Routing service online",
        icon="✅",
    )

    if review_threshold is not None:
        st.caption(
            f"Validated human-review threshold: "
            f"{review_threshold:.2f}"
        )

else:
    st.warning(
        "Routing service is not available. "
        "Start the FastAPI service first.",
        icon="⚠️",
    )


st.divider()


# --------------------------------------------------
# REQUEST FORM
# --------------------------------------------------

st.subheader("New service request")


request_text = st.text_area(
    "Customer request",
    placeholder=(
        "Example: My water purifier stopped "
        "working and is showing error code E2"
    ),
    height=140,
)


col1, col2 = st.columns(2)


with col1:
    product_family = st.selectbox(
        "Product",
        PRODUCT_FAMILIES,
    )


with col2:
    channel = st.selectbox(
        "Channel",
        CHANNELS,
    )


warranty_label = st.selectbox(
    "Warranty status",
    list(WARRANTY_OPTIONS.keys()),
)

warranty_status = (
    WARRANTY_OPTIONS[warranty_label]
)


# --------------------------------------------------
# ROUTE REQUEST
# --------------------------------------------------

if st.button(
    "Route request",
    type="primary",
    use_container_width=True,
):

    if not request_text.strip():

        st.error(
            "Enter the customer's request first."
        )

    elif not api_online:

        st.error(
            "The routing API is offline. "
            "Start FastAPI and try again."
        )

    else:

        payload = {
            "request_text":
                request_text.strip(),

            "channel":
                channel,

            "product_family":
                product_family,

            "warranty_status":
                warranty_status,
        }


        try:
            with st.spinner(
                "Finding the best team..."
            ):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=10,
                )


            if response.status_code == 200:

                result = response.json()

                predicted_team = result.get(
                    "predicted_team"
                )

                routing_status = result.get(
                    "routing_status"
                )

                needs_human_review = result.get(
                    "needs_human_review",
                    False,
                )

                margin = result.get(
                    "decision_margin",
                    0.0,
                )

                threshold = result.get(
                    "review_threshold",
                    review_threshold,
                )

                reasons = result.get(
                    "reasons",
                    [],
                )


                # ----------------------------------
                # RESULT
                # ----------------------------------

                st.divider()

                st.subheader(
                    "Routing recommendation"
                )


                st.metric(
                    "Send to",
                    predicted_team,
                )


                # ----------------------------------
                # ROUTING DECISION
                # ----------------------------------

                if needs_human_review:

                    st.warning(
                        "Human review recommended",
                        icon="⚠️",
                    )

                    st.write(
                        "The routing decision is "
                        "too close to auto-route "
                        "safely based on the "
                        "validated margin threshold."
                    )

                else:

                    st.success(
                        "Auto-routing recommended",
                        icon="✅",
                    )


                # ----------------------------------
                # DECISION MARGIN
                # ----------------------------------

                metric_col1, metric_col2 = (
                    st.columns(2)
                )


                with metric_col1:

                    st.metric(
                        "Decision margin",
                        f"{margin:.4f}",
                    )


                with metric_col2:

                    if threshold is not None:

                        st.metric(
                            "Review threshold",
                            f"{threshold:.2f}",
                        )

                    else:

                        st.metric(
                            "Routing status",
                            routing_status,
                        )


                # ----------------------------------
                # EXPLANATION
                # ----------------------------------

                st.markdown(
                    "### Why this route"
                )


                if reasons:

                    for reason in reasons:

                        st.write(
                            f"• {reason}"
                        )

                else:

                    st.write(
                        "No readable explanation "
                        "features were available."
                    )


                # ----------------------------------
                # MODEL NOTE
                # ----------------------------------

                st.info(
                    "The decision margin is the gap "
                    "between the top two LinearSVC "
                    "scores. It is not a probability."
                )


                if needs_human_review:

                    st.caption(
                        "Requests below the validated "
                        "review threshold are not "
                        "recommended for automatic "
                        "assignment."
                    )

                else:

                    st.caption(
                        "This request is above the "
                        "validated review threshold "
                        "and is eligible for "
                        "automatic routing."
                    )


            else:

                st.error(
                    "The routing service returned "
                    f"HTTP {response.status_code}."
                )

                st.code(
                    response.text
                )


        except requests.RequestException as exc:

            st.error(
                "Could not reach the routing API."
            )

            st.code(
                str(exc)
            )


st.divider()


st.caption(
    "Local model · No paid model API calls · "
    "Word + character TF-IDF · LinearSVC · "
    "Designed for Kestrel Home service routing"
)