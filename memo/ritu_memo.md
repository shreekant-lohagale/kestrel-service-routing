# Memo to Ritu Deshpande
## Kestrel Home Service Routing

### Decision

I would not switch off the existing routing bot solely because a new model can reproduce its historical labels.

The replacement model can match the old bot's routing labels at about 96.7% accuracy, which clears the requested 90% bar. However, the historical labels are the old bot's first routing decision, not necessarily the team that finally resolved the request.

When measured against the actual resolving team, the existing bot was correct about 77.2% of the time in forward-in-time backtesting.

A model trained directly on the resolving team improved that to about 84.1%.

My recommendation is therefore to use the new model in a controlled rollout rather than treating 96.7% label agreement as 96.7% correct routing.

---

### The number

Across three forward-in-time validation periods, the outcome-oriented model achieved:

- 83.49% accuracy for Oct–Dec 2025
- 84.50% accuracy for Jan–Mar 2026
- 84.36% accuracy for Apr–Jun 2026

Average outcome accuracy was approximately 84.12%.

I also added a human-review safeguard based on the model's routing margin.

Using a threshold of 0.20 on the latest validation quarter:

- about 90% of requests remained eligible for automatic routing
- about 10% were sent for human review
- the automatically routed subset achieved 91.26% accuracy

This gives Kestrel a practical way to automate clearer requests while holding ambiguous ones for an agent.

---

### The rupees

The current routing bot licence costs approximately:

**₹3.2 lakh per year**

The replacement model runs locally and uses no paid model or LLM API.

Model-call cost:

**₹0 per prediction**

At approximately 700 requests per month:

**700 × ₹0 = ₹0 per month in paid model/API cost**

This excludes normal production hosting and infrastructure costs.

There is also a larger operational reason to improve routing quality.

Historical data contains:

- 2,696 requests with at least one transfer
- 3,902 total transfers

Using Kestrel's planning costs:

- 3,902 transfers × ₹305 = approximately ₹11.90 lakh
- 2,696 transferred requests × ₹260 additional contact cost = approximately ₹7.01 lakh

That is about ₹18.91 lakh of observed transfer and additional-contact handling cost across the supplied historical period.

This is not a claim that the new model will save ₹18.91 lakh. It shows that reducing misrouting could be worth more than the software licence alone.

---

### What I would do next week

I would run the new router in shadow mode beside the current process for one week.

Requests with a routing margin of 0.20 or higher can be compared with the existing assignment and eventual resolving team.

Requests below 0.20 should remain with human review.

At the end of the week I would measure:

1. percentage routed correctly to the eventual resolving team
2. transfer rate
3. human-review rate
4. performance by team
5. handling cost created by misroutes

If the live results are consistent with validation, Kestrel can move the high-margin traffic to the local model first and keep human review for ambiguous cases.

I would not recommend a full switch based only on the historical-label score.