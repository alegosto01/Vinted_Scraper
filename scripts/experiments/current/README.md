# Current Experiment Packages

These packages are active in the current live workflow or are imported by active
code paths.

## Live path

- `basic_5_giant_model`: scores the live collector output and sends Telegram
  recommendations.
- `giant_basic_visual`: shadow-only test of adding main-image-only visual
  features to the Basic5 giant model.
- `time_to_sell`: collects high-score/control cohorts and rechecks sale status.

## Current support code

- `teacher_student_basic_filter`: stage-1 model source for the collector.
- `full_scrape_model`: full item/seller and feature-modality model source.
- `deal_finder`: legacy modeling and paper-trading utilities still imported by
  current code.
- `benchmark_basic_to_full`: cascade utilities reused by the collector.
- `photo_arbitrage`: photo quality and DINO feature extraction utilities.
- `reselling_process`: Telegram/reselling description tooling.
- `tracking`: experiment tracking helpers.

Old import paths under `scripts/experiments/<package>` remain as compatibility
wrappers during the transition.
