# Project Management And Future Operator App

This project has two different needs:

- Development management: how to build, track, test, and improve the project professionally.
- Future operator system: how to use the finished project day to day when it reliably finds opportunities.

These should start separate. External tools are best for managing development. A custom app makes sense later for the unique Vinted workflow.

## Recommended Development Stack

Use external tools first:

- GitHub: source of truth for code, branches, issues, pull requests, and project boards.
- GitHub Projects: task board, roadmap, priorities, status, and simple charts tied to issues and pull requests.
- Notion: strategy, ideas, experiment notes, decision logs, business model notes, and future app specs.
- MLflow: model experiment tracking, metrics, parameters, artifacts, and model comparison.
- Streamlit: internal dashboards for model results, live run status, plots, price-band reports, photo-quality scores, and review queues.
- Telegram: phone notifications when a promising listing is found.

Linear can be useful later if GitHub Issues becomes too messy, but for a solo project GitHub plus Notion is enough to start.

## Future Operator System

The final daily-use system can become a custom web/mobile app, but it should not be built too early. First, make the workflow visible and repeatable.

Start with:

- Telegram alerts for strong candidates.
- Streamlit dashboard for candidate review and performance monitoring.
- SQLite or Postgres database for predictions, decisions, outcomes, purchases, and sales.
- Manual approval for any action involving offers, messages, or purchases.

Later, build a real app with:

- Candidate inbox.
- Candidate states: interested, ignore, watch, bought, sold.
- Offer drafts.
- Inventory tracking.
- Purchase cost, shipping, fees, sale price, and profit.
- Model score history.
- Seller and item details.
- Photo-improvement opportunity queue.
- Analytics by search, brand, price range, model, and threshold.

## Suggested Roadmap

1. Use GitHub Issues and GitHub Projects for all engineering tasks.
2. Use Notion for big-picture planning and experiment writeups.
3. Add MLflow to all model experiments.
4. Build a Streamlit dashboard reading existing experiment outputs.
5. Add Telegram alerts for high-confidence candidates.
6. Create a central database for listings, scores, decisions, and outcomes.
7. Only then design the custom operator app.

## Guiding Principle

Do not build a beautiful app around an unstable workflow.

First make the workflow measurable, visible, and repeatable. Then turn repeated actions into product features.

