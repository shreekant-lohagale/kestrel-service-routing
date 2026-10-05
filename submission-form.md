# Kestrel Home — Task 2 Submission Form

## What did you build, and what business decision does it support? State the number and the rupees.

I built a local service-request routing system for Kestrel Home. It includes a trained text-classification model, a FastAPI `/predict` endpoint, a Streamlit screen that calls the endpoint, human-readable routing reasons, and a human-review safeguard for ambiguous requests.

The business decision is whether Kestrel should replace the current routing bot, which costs ₹3.2 lakh per year.

A model trained to reproduce the old bot's historical routing labels achieves about 96.7% accuracy, which clears the requested 90% target. However, I found that those labels represent the old bot's initial queue, not necessarily the team that finally resolves the request.

When I trained against the actual resolving team, forward-in-time accuracy averaged about 84.1%.

My recommendation is therefore not to switch the current bot off only because the 90% label-match target is met. I would first run the new router in shadow mode, auto-route clearer cases, and keep ambiguous cases for human review.

The submitted model uses no paid model/API calls, so the paid model-call cost is ₹0.


## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

I expect approximately 84% accuracy on the hidden outcomes.

I use accuracy as the primary metric because every request has one routing destination, and the client's success criterion is expressed as the percentage of requests routed to the correct team.

I estimated this using three forward-in-time validation periods rather than a random split:

- Oct–Dec 2025: 83.49%
- Jan–Mar 2026: 84.50%
- Apr–Jun 2026: 84.36%

The average was 84.12%.

I also tracked Macro F1 so that performance was not being driven only by the largest queue.

I would not use the 96.7% historical-label score as the expected hidden outcome score because that measures how well the model copies the old routing bot, rather than how well it predicts the team that eventually resolves the request.


## How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

I validated the model using expanding forward-in-time splits, so every validation period occurred after the data used for training.

The outcome-model accuracies were:

- 83.49%
- 84.50%
- 84.36%

On the latest Apr–Jun 2026 validation set:

- Validation rows: 2,135
- Accuracy: 84.36%
- Error rate: 15.64%
- Macro F1: 83.70%

I also analysed the LinearSVC decision margin.

Using a human-review threshold of 0.20:

- 9.93% of validation requests were flagged for human review
- 90.07% remained eligible for automatic routing
- The auto-routed subset achieved 91.26% accuracy

The model gets requests wrong most often when the opening message is vague, contains multiple different issues, or could reasonably belong to more than one team.

For example, a request such as "Please call me about my purifier" does not contain enough information for a reliable automatic route, so the system flags it for human review.

There is also noise in the historical routing data. Some requests have an initial routing label that does not match the content of the message or the team that eventually resolved the case.


## Did you change, narrow, or push back on the client's ask? What, when, and why.

Yes.

The client asked for 90%+ agreement with the historical `team_label` and described those labels as ground truth.

During the data audit, I joined the training data with the resolution log and found that `team_label` is exactly the same as the first team selected by the old routing bot for 100% of the training rows.

After normalising the renamed teams, the first team matched the final resolving team only about 77% of the time.

I therefore separated the problem into two questions:

1. Can we reproduce the old routing bot?
2. Can we predict the team that actually resolves the request?

The first problem reaches about 96.7% accuracy.

The second, more operationally useful problem reaches about 84.1% average forward-in-time accuracy.

Because the requested `predictions.csv` asks for the team a request should go to and the hidden evaluation refers to outcomes, I used the outcome-oriented `final_team` model for the submitted predictions instead of simply reproducing the old bot.

I also normalised the historical team names:

- `Consumables` → `Filters & Consumables`
- `Installations` → `Installs & Demo`

because their responsibilities did not change when the teams were renamed.


## What is wrong with what you are handing us, or with the data we handed you? Be specific: bugs, shortcuts, columns you did not trust, rows that looked wrong.

The main limitation is that the final outcome model does not achieve 90% overall accuracy. Its latest-quarter accuracy is about 84.36%.

The 0.20 human-review threshold improves reliability for automatically routed requests, but it means roughly 10% of requests still require manual review.

I also do not treat `team_label` as perfect ground truth. It records the existing bot's first routing decision, and many requests were later transferred.

`final_team` is a better operational target, but it is not guaranteed to be perfect semantic ground truth either. A request can eventually be closed by a team for workflow reasons that are not completely visible in the opening message.

Some legacy Zoho request text also contains encoding corruption and malformed characters. Character TF-IDF helps reduce sensitivity to this, but I did not manually repair every historical message.

The LinearSVC decision margin is not a calibrated probability, so I deliberately expose it as a ranking margin instead of calling it confidence.

The service also does not currently include authentication, a production database, live CRM integration, production monitoring, or automatic model retraining.

The public repository excludes private assessment data, generated analysis containing customer text, the trained model artifact, and `predictions.csv`. The supplied assessment files are required to reproduce these locally.


## What did you deliberately leave out, and why that rather than something else?

I deliberately left out LLM routing, RAG, a chatbot, vector databases, agent frameworks, paid model APIs, authentication, a production database, and live CRM integration.

The core task is supervised text classification with seven known routing destinations.

TF-IDF plus LinearSVC performed strongly, runs locally, is reproducible, and has ₹0 paid model-call cost.

I preferred spending the limited time on:

- understanding the target correctly
- data-quality checks
- temporal validation
- comparing multiple models
- error analysis
- human-review logic
- a working API
- a working screen
- reproducibility

rather than adding AI components that did not improve the routing decision.


## Anything you built or found that nobody asked for?

Yes.

The main finding was that the supplied historical routing labels are not the same as actual resolution outcomes.

The old bot's first route matched the final resolving team only about 77.2% of the time in forward-in-time backtesting.

I also found:

- 2,696 historical requests had at least one transfer
- 3,902 total transfers occurred

Using Kestrel's planning costs:

3,902 × ₹305 = approximately ₹11.90 lakh in transfer handling cost

2,696 × ₹260 = approximately ₹7.01 lakh in additional customer-contact cost

Combined, this is approximately ₹18.91 lakh of observed transfer and additional-contact handling cost across the supplied historical period.

I am not claiming the new model automatically saves ₹18.91 lakh. It shows that improving real routing quality could be financially more important than only saving the ₹3.2 lakh annual routing-bot licence.

I also added an evidence-based human-review mechanism.

Using a LinearSVC margin threshold of 0.20 sends about 10% of requests for review, while the remaining automatically routed subset achieved 91.26% validation accuracy.


## What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

I used ChatGPT as a development assistant for understanding the brief, discussing modelling choices, reviewing experiments, debugging Python/FastAPI/Streamlit code, and structuring the validation workflow, README, and memo.

Inside the submitted product, I did not use a paid LLM or model API.

The actual routing system uses:

- word TF-IDF
- character TF-IDF
- one-hot encoded operational metadata
- scikit-learn LinearSVC

I tested the project incrementally:

- V1: Word TF-IDF + Logistic Regression — 92.97% label-match accuracy
- V2: Text + metadata + Logistic Regression — 93.40%
- V3: Text + metadata + LinearSVC — 94.94%
- V4: Word + character TF-IDF + metadata + LinearSVC — 96.39%
- V5: Same modelling approach trained on actual final-team outcomes — 84.36% latest-quarter accuracy

The most useful modelling changes were switching from Logistic Regression to LinearSVC and adding character-level TF-IDF.

The most important idea I discarded was treating 96% historical-label agreement as 96% correct routing. The resolution log showed that this would have been misleading.

I also considered more complex approaches such as LLM routing, RAG, and chatbot-style features, but discarded them because they added cost and complexity without earning their place in this classification problem.

Three-minute screen recording:

https://drive.google.com/file/d/13mtrWJ0ZGbjsr4VH2M-49QyuMNJxo4Qg/view?usp=drive_link


## Your Public Google Drive Link

https://drive.google.com/drive/folders/1qJPFwIvJ7WC43jvBVJKBqbTNHizd6a6U


## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. The public GitHub repository does not contain Kestrel's private assessment data. Put the supplied files into the `data/` directory and run `python src/train_final.py` to rebuild the trained model and `predictions.csv`.

2. Do not confuse the two main performance numbers. About 96.7% is historical routing-label reproduction. About 84.1% is average forward-in-time accuracy against the actual resolving team. The submitted predictions use the outcome-oriented approach.

3. The 0.20 LinearSVC decision-margin threshold is intentional. Requests below it should be sent for human review. On the latest validation quarter it flagged about 10% of requests, while the remaining automatically routed subset achieved 91.26% accuracy.


## Honest hours spent. One number.

6


## Github Repo Link

https://github.com/shreekant-lohagale/kestrel-service-routing


## What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? Show the arithmetic. If you used no paid calls, say so.

The submitted system uses no paid model or API calls.

Paid model/API cost per prediction:

₹0

At approximately 700 requests per month:

700 × ₹0 = ₹0 per month

So the paid model-call cost is ₹0 per prediction and ₹0 per month at the stated volume.

This excludes normal production hosting, compute, and infrastructure costs if Kestrel deploys the service in production.