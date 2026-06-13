#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.photo_arbitrage.modeling import prepare_labeled_frame, save_model, train_photo_quality_model
from experiments.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    normalize_methods,
)
from experiments.photo_arbitrage.paths import (
    LABELS_DIR,
    MODELS_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first local bad-photo classifier.")
    parser.add_argument("--labels", default=str(LABELS_DIR / "photo_quality_label_sheet.csv"))
    parser.add_argument("--methods", default="simple", help="Comma-separated methods: simple,pyiqa,aesthetic,dino,all")
    parser.add_argument("--pyiqa-model", default=DEFAULT_PYIQA_MODEL)
    parser.add_argument("--aesthetic-model", default=DEFAULT_AESTHETIC_MODEL)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--max-images-per-item", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Label sheet not found: {labels_path}")
        return 1
    labels = pd.read_csv(labels_path, low_memory=False)
    method_config = MethodConfig(
        methods=normalize_methods(args.methods),
        pyiqa_model=args.pyiqa_model,
        aesthetic_model=args.aesthetic_model,
        dino_model=args.dino_model,
        max_images_per_item=args.max_images_per_item,
        device=args.device,
    )
    model, metadata = train_photo_quality_model(labels, method_config=method_config)
    model_path, metadata_path = save_model(model, metadata, model_dir=MODELS_DIR)
    usable = prepare_labeled_frame(labels, method_config=method_config)
    write_manifest(
        MODELS_DIR / "photo_quality_v1_train_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "labels": str(labels_path),
            "usable_training_rows": int(len(usable)),
            "methods": list(method_config.methods),
            "model_path": str(model_path) if model_path else "",
            "metadata_path": str(metadata_path),
            "status": metadata.get("status"),
        },
    )
    if model is None:
        print(f"Model not trained yet: {metadata.get('reason', 'not enough usable labels')}")
        print(f"Metadata written to {metadata_path}")
        return 0
    print(f"Trained {metadata['model_version']} with {metadata['training_rows']} rows")
    print(f"Model written to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
