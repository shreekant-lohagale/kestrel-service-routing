# Kestrel Home Service Router

A local machine-learning service for routing Kestrel Home Appliances service requests to the team most likely to resolve them.

The project was built for the Kestrel Home service-request routing assessment.

The current vendor routing bot costs approximately **₹3.2 lakh per year**. The goal of this project is to evaluate whether a lightweight local model can replace or reduce dependence on that bot without introducing paid model/API costs.

---

## Business decision

The client initially asked for a model that could reproduce historical routing labels at **90%+ accuracy**.

During analysis, I found that the supplied `team_label` is the **initial queue chosen by the existing routing bot**, not necessarily the team that ultimately resolved the request.

Across historical requests:

- Existing bot initial route matched the eventual resolving team only about **77.23%** of the time in forward-in-time backtesting.
- A model trained to reproduce the existing routing labels achieved approximately **96.72% average accuracy**.
- A separate model trained on the actual `final_team` achieved approximately **84.12% average accuracy** across three forward-in-time validation periods.
- On the latest validation quarter, the final-team model achieved **84.36% accuracy**.

Because the assessment asks for the team a request **should go to**, the submitted predictions are generated from the outcome-oriented model trained on `final_team`, rather than simply copying the old routing bot.

---

## Human-review safeguard

The final model uses the margin between the top two LinearSVC scores as an uncertainty signal.

The margin is **not a probability**.

Using the latest 2,135-request validation period:

| Review threshold | Requests sent to review | Auto-routed accuracy |
|---:|---:|---:|
| 0.10 | 5.53% | 88.45% |
| **0.20** | **9.93%** | **91.26%** |
| 0.30 | 12.88% | 93.49% |
| 0.40 | 14.61% | 94.73% |
| 0.50 | 16.16% | 95.81% |

The application therefore uses:

```text
decision margin < 0.20
→ human review
```

At this threshold, roughly **90% of requests remain eligible for automatic routing**, and those auto-routed requests achieved approximately **91.26% validation accuracy**.

This does not mean the overall model accuracy is 91.26%. Overall latest-quarter accuracy is **84.36%**.

---

## Model

The final routing model uses:

```text
Request text
    ↓
Word TF-IDF
+
Character TF-IDF
+
Channel
+
Product family
+
Warranty status
    ↓
LinearSVC
    ↓
Predicted resolving team
```

Character-level TF-IDF was added because some historical legacy records contain malformed text and spelling variation.

The final model does not require:

- OpenAI
- Gemini
- Claude
- any paid LLM
- a vector database
- a model API key

Paid model/API cost per prediction is therefore:

```text
₹0
```

At approximately 700 requests per month:

```text
700 × ₹0 = ₹0/month
```

This excludes ordinary infrastructure/hosting costs if the service is deployed to production.

---

## Experiment history

The model was developed incrementally using a forward-in-time validation split.

| Version | Approach | Accuracy |
|---|---|---:|
| V1 | Word TF-IDF + Logistic Regression | 92.97% |
| V2 | Word TF-IDF + metadata + Logistic Regression | 93.40% |
| V3 | Word TF-IDF + metadata + LinearSVC | 94.94% |
| V4 | Word + character TF-IDF + metadata + LinearSVC | 96.39% |
| V5 | Outcome model trained on actual final team | 84.36% |

V1–V4 measure the ability to reproduce historical routing labels.

V5 measures the harder and more operationally useful problem: predicting the team that actually resolves the request.

---

## Forward-in-time backtesting

To estimate future performance more realistically, I used three expanding-window temporal validation periods instead of relying on a random split.

| Validation period | Label-match model | Existing bot → final team | Outcome model → final team |
|---|---:|---:|---:|
| Oct–Dec 2025 | 96.82% | 77.54% | 83.49% |
| Jan–Mar 2026 | 96.96% | 77.49% | 84.50% |
| Apr–Jun 2026 | 96.39% | 76.67% | 84.36% |
| **Average** | **96.72%** | **77.23%** | **84.12%** |

Based on this, I expect approximately **84% accuracy** on unseen future outcomes.

Accuracy is the primary metric because every request has one routing destination and the client's original success criterion was expressed as percentage correctly routed.

Macro F1 is also monitored to make sure performance is not driven only by the largest team.

---

## Routing teams

The model predicts one of the seven current Kestrel teams:

```text
Billing
Filters & Consumables
Installs & Demo
Product Advice
Repairs
Returns & Replacement
Warranty Claims
```

Historical names are normalized as follows:

```text
Consumables
→ Filters & Consumables

Installations
→ Installs & Demo
```

---

# Running locally

## 1. Requirements

Recommended:

```text
Python 3.10+
```

Clone the repository and enter the project:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd kestrel-service-routing
```

Create a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Add the supplied assessment data

The assessment data is intentionally **not included in this public repository**.

Place the supplied files inside:

```text
data/
```

Expected structure:

```text
data/
├── train.csv
├── test_unlabelled.csv
├── resolution_log.csv
├── teams.csv
├── sample_submission.csv
├── README.txt
├── email-thread.txt
└── ops-policy.pdf
```

The public repository contains only:

```text
data/.gitkeep
```

---

## 3. Audit the data

Run:

```bash
python src/data_audit.py
```

This checks:

- row counts
- date ranges
- missing values
- duplicate request IDs
- historical team names
- current team normalization
- resolution-log joins
- transfer counts
- submission/test alignment

---

## 4. Train the final model

Run:

```bash
python src/train_final.py
```

This will create:

```text
models/routing_model.joblib
predictions.csv
```

The model is trained using all historical labelled requests and the actual resolving `final_team`.

---

## 5. Verify the submission file

Run:

```bash
python src/verify_submission.py
```

Expected result:

```text
========== VERIFICATION PASSED ==========
predictions.csv is ready for submission.
```

The verifier checks:

- exactly one row per test request
- correct request order
- no duplicate IDs
- no missing predictions
- only valid team names
- all required columns
- trained model exists

---

# Run the API

Start FastAPI:

```bash
python -m uvicorn src.api:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```json
{
  "request_text": "My water purifier stopped working and is showing error code E2",
  "channel": "chat",
  "product_family": "Water Purifier",
  "warranty_status": "in_warranty"
}
```

Example response:

```json
{
  "predicted_team": "Repairs",
  "routing_status": "auto_route",
  "needs_human_review": false,
  "decision_margin": 3.1325,
  "review_threshold": 0.2,
  "reasons": [
    "text phrase \"purifier stopped\"",
    "text phrase \"showing error\"",
    "product family is Water Purifier"
  ]
}
```

---

# Run the screen

Keep the FastAPI service running.

Open another terminal, activate the same virtual environment and run:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit application calls the FastAPI `/predict` endpoint rather than loading the routing model directly.

The screen shows:

```text
Predicted team
Routing recommendation
Auto-route / human-review status
Decision margin
Validated review threshold
Human-readable routing reasons
```

---

# Run tests

The model must be trained before running the API tests.

Run:

```bash
python -m pytest -q -p no:cacheprovider
```

Current test result:

```text
5 passed
```

Tests cover:

- health endpoint
- clear auto-routing case
- ambiguous human-review case
- invalid empty request
- required API response fields

---

# Reproduce the experiments

Baseline:

```bash
python src/experiment_v1.py
```

Add metadata:

```bash
python src/experiment_v2.py
```

Switch to LinearSVC:

```bash
python src/experiment_v3.py
```

Add character TF-IDF:

```bash
python src/experiment_v4.py
```

Train against actual resolution outcome:

```bash
python src/experiment_v5_outcome.py
```

Run forward-in-time backtests:

```bash
python src/backtest_models.py
```

Analyse errors:

```bash
python src/error_analysis.py
```

Analyse routing-margin thresholds:

```bash
python src/margin_analysis.py
```

---

# Important finding about the supplied labels

The historical `team_label` is the queue selected by the vendor routing bot when a request first arrived.

After joining the training set with the resolution log:

```text
team_label == first_team
for 100% of training requests
```

However, the first team matched the final resolving team for only about:

```text
77%
```

This means achieving 90%+ agreement with `team_label` does not automatically imply 90% correct operational routing.

The final solution therefore separates:

```text
Historical-label reproduction
from
Actual resolution routing
```

This is the main reason the project uses `final_team` for the submitted outcome-oriented model.

---

# Transfer cost context

Historical analysis found:

```text
2,696 requests with at least one transfer
3,902 total transfers
```

Kestrel's policy values:

```text
Transfer handling cost = ₹305
Extra customer contact after a misroute = ₹260
```

Observed historical handling cost associated with these transferred requests is approximately:

```text
3,902 × ₹305
= ₹11.90 lakh

2,696 × ₹260
= ₹7.01 lakh

Combined
≈ ₹18.91 lakh
```

This does **not** mean the new model automatically saves ₹18.91 lakh.

It demonstrates that reducing actual misrouting can have a larger operational value than merely avoiding the ₹3.2 lakh routing-bot licence.

---

# Known limitations

The final-team model does not currently achieve 90% overall accuracy.

Latest-quarter overall accuracy is approximately:

```text
84.36%
```

The human-review mechanism improves the reliability of automatically routed cases but reduces automation coverage.

At the selected `0.20` threshold:

```text
~90% auto-routed
~10% human review
91.26% accuracy on the auto-routed subset
```

Other limitations:

- historical routing labels contain noise
- some requests contain multiple issues
- vague requests cannot always be routed from the opening message alone
- legacy text contains encoding corruption
- final team is an operational outcome, but it is not guaranteed to be perfect semantic ground truth
- the SVM margin is not a calibrated probability
- the service has no authentication or production database
- the project does not include live CRM integration
- production infrastructure cost is not included in the ₹0 model-call estimate

---

# Deliberate scope cuts

I deliberately did not build:

```text
LLM routing
RAG
Chatbot
Vector database
Agent framework
Paid model API integration
Authentication
Production database
Live Kestrel CRM integration
```

The routing problem could be solved more cheaply and reproducibly with local text classification.

The priority was:

```text
correct target definition
→ temporal validation
→ error analysis
→ reproducible local model
→ working endpoint
→ working screen
```

rather than adding AI components that did not improve the business decision.

---

# Data privacy

Kestrel's supplied operational data is excluded from the public repository.

The following are ignored by Git:

```text
data/*
outputs/*
predictions.csv
models/*
*.joblib
```

Do not commit or publish the assessment data.

---

# Project structure

```text
kestrel-service-routing/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_audit.py
│   ├── experiment_v1.py
│   ├── experiment_v2.py
│   ├── experiment_v3.py
│   ├── experiment_v4.py
│   ├── experiment_v5_outcome.py
│   ├── error_analysis.py
│   ├── backtest_models.py
│   ├── margin_analysis.py
│   ├── train_final.py
│   └── verify_submission.py
│
└── tests/
    └── test_api.py
```

---

## Summary

The project demonstrates two different routing questions:

```text
Can we reproduce the existing routing bot?
≈ 96.7% accuracy

Can we predict the team that actually resolves the request?
≈ 84.1% forward-in-time accuracy
```

For operational use, the second number is the more important one.

The submitted service therefore predicts the resolving team and adds an evidence-based human-review safeguard instead of blindly automating every request.