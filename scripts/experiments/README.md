# Experiment Code Layout

The real experiment implementations now live under:

- `current/`: live workflow packages and active support code.
- `old/`: historical standalone experiments kept for reproducibility.

The package folders still present directly under `scripts/experiments/` are
compatibility wrappers. They keep old imports such as
`experiments.deal_finder.model_sweep` and old direct script commands working
while new code moves to the canonical paths:

- `experiments.current.<package>`
- `experiments.old.<package>`

Data and model artifacts remain under `data/experiments/`; this layout change
only affects source code.

