#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.old.basic_5_stacking._deps.photo_arbitrage.modeling import LABEL_SOURCE_MANUAL, LABEL_SOURCES, label_readiness_summary, prepare_labeled_frame, save_model, train_photo_quality_model
from experiments.old.basic_5_stacking._deps.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_FASHIONCLIP_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    normalize_methods,
)
from experiments.old.basic_5_stacking._deps.photo_arbitrage.paths import (
    LABELS_DIR,
    MODELS_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first local bad-photo classifier.")
    parser.add_argument("--labels", default=str(LABELS_DIR / "photo_quality_label_sheet.csv"))
    parser.add_argument("--methods", default="simple", help="Comma-separated methods: simple,pyiqa,aesthetic,fashionclip,dino,all")
    parser.add_argument("--pyiqa-model", default=DEFAULT_PYIQA_MODEL)
    parser.add_argument("--aesthetic-model", default=DEFAULT_AESTHETIC_MODEL)
    parser.add_argument("--fashionclip-model", default=DEFAULT_FASHIONCLIP_MODEL)
    parser.add_argument(
        "--allow-fashionclip-downloads",
        action="store_true",
        help="Allow Transformers to fetch the FashionCLIP model if it is not already cached locally.",
    )
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--max-images-per-item", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--label-source",
        choices=sorted(LABEL_SOURCES),
        default=LABEL_SOURCE_MANUAL,
        help="manual uses reviewed manual_label values; fashionclip_pseudo trains an explicit weak-label baseline from FashionClipPseudoLabel.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print label readiness and exit without training or writing model artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Label sheet not found: {labels_path}")
        return 1
    labels = pd.read_csv(labels_path, low_memory=False)
    readiness = label_readiness_summary(labels, label_source=args.label_source)
    if args.check_only:
        print(f"Label source: {args.label_source}")
        print(f"Readiness: {readiness['status']}")
        print(f"Rows: total={readiness['total_rows']} trainable={readiness['trainable_rows']}")
        print(f"Good labels: {readiness['good_rows']}")
        print(f"Bad labels: {readiness['bad_rows']}")
        print(f"Blank labels: {readiness['blank_rows']}")
        print(readiness["message"])
        return 0 if readiness["status"] == "ready" else 2
    method_config = MethodConfig(
        methods=normalize_methods(args.methods),
        pyiqa_model=args.pyiqa_model,
        aesthetic_model=args.aesthetic_model,
        fashionclip_model=args.fashionclip_model,
        fashionclip_local_files_only=not bool(args.allow_fashionclip_downloads),
        dino_model=args.dino_model,
        max_images_per_item=args.max_images_per_item,
        device=args.device,
    )
    model, metadata = train_photo_quality_model(labels, method_config=method_config, label_source=args.label_source)
    model_path, metadata_path = save_model(model, metadata, model_dir=MODELS_DIR)
    usable = prepare_labeled_frame(labels, method_config=method_config, label_source=args.label_source)
    write_manifest(
        MODELS_DIR / "photo_quality_v1_train_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "labels": str(labels_path),
            "label_source": args.label_source,
            "usable_training_rows": int(len(usable)),
            "methods": list(method_config.methods),
            "fashionclip_model": method_config.fashionclip_model,
            "fashionclip_local_files_only": bool(method_config.fashionclip_local_files_only),
            "model_path": str(model_path) if model_path else "",
            "metadata_path": str(metadata_path),
            "status": metadata.get("status"),
            "evaluation": metadata.get("evaluation", {}),
        },
    )
    if model is None:
        print(f"Model not trained yet: {metadata.get('reason', 'not enough usable labels')}")
        print(f"Metadata written to {metadata_path}")
        return 0
    print(f"Trained {metadata['model_version']} with {metadata['training_rows']} rows")
    evaluation = metadata.get("evaluation") or {}
    if evaluation.get("status") == "cross_validated":
        print(
            "Cross-validation: "
            f"folds={evaluation.get('folds')} "
            f"auc={evaluation.get('auc_bad_vs_good')} "
            f"accuracy={evaluation.get('accuracy')}"
        )
    print(f"Model written to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
