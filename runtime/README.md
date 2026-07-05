# Runtime Artifacts

This folder is for local process output, not source code or experiment-owned data.

- `logs/` contains live scraper and telemetry logs.
- `pids/` contains process ID files.
- `archive/` contains local rotated runtime history.

Runtime contents are ignored by git. Stable scrape data remains under `data/simple_scrape/`, and experiment artifacts remain under each experiment's own `data/` folder.
