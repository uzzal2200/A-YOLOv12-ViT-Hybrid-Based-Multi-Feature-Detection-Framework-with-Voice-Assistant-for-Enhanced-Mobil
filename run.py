#!/usr/bin/env python3
# ============================================================
# run.py  —  Main Entry Point
# ============================================================
# Pipeline logic:
#
#   currency        → train yolov5n, yolov8n, yolov11n, yolov12n, yolov12_vit
#   footpath        → train yolov5n, yolov8n, yolov11n, yolov12n, yolov12_vit
#   blind_assistant → train yolov5n, yolov8n, yolov11n, yolov12n, yolov12_vit
#   visually_impaired → TEST ONLY with yolov12_vit (public dataset,
#                       validates our proposed model independently)
#
# Usage:
#   python run.py                             # full pipeline
#   python run.py --datasets currency         # one dataset
#   python run.py --models yolov12_vit        # one model
#   python run.py --epochs 50 --batch 8       # override hyperparams
#   python run.py --skip-download             # skip Roboflow download
#   python run.py --test-weights path/best.pt # custom weights for test-only
# ============================================================

import argparse
import sys
import traceback
from pathlib import Path

from configs.datasets import DATASETS
from configs.models import MODELS, TRAIN_CONFIG
from datasets.downloader import download_dataset
from trainers.trainer import train_model
from trainers.tester import test_model
from utils.logger import ResultLogger


# ── Models run on our own datasets ──────────────────────────
OWN_DATASET_MODELS = ["yolov5n", "yolov8n", "yolov11n", "yolov12n", "yolov12_vit"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv5n/v8n/v11n/v12n/v12-ViT Multi-Dataset Training & Testing Pipeline"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=OWN_DATASET_MODELS,
        choices=list(MODELS.keys()),
        help="Models to train on own datasets (default: all 5)",
    )
    parser.add_argument("--epochs",  type=int,   default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch",   type=int,   default=TRAIN_CONFIG["batch"])
    parser.add_argument("--imgsz",   type=int,   default=TRAIN_CONFIG["imgsz"])
    parser.add_argument("--device",  default=str(TRAIN_CONFIG["device"]))
    parser.add_argument("--workers", type=int,   default=TRAIN_CONFIG.get("workers", 4))
    parser.add_argument("--output",  default="results", help="Root output directory")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Roboflow download (data already local)",
    )
    parser.add_argument(
        "--test-weights", default=None,
        help="Path to best.pt for test-only datasets (optional)",
    )
    return parser.parse_args()


def _download(dataset_key, dataset_cfg, skip):
    """Download and fix yaml. Returns yaml_path or None on failure."""
    if skip:
        yaml_path = str(Path(dataset_cfg["yaml_path"]).resolve())
        print(f"\n[Skip] Using existing YAML: {yaml_path}")
        return yaml_path
    try:
        return download_dataset(
            dataset_key=dataset_key,
            dataset_cfg=dataset_cfg,
            output_dir=".",
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to download '{dataset_key}': {e}")
        traceback.print_exc()
        return None


def main():
    args = parse_args()

    train_cfg = {
        "epochs":      args.epochs,
        "batch":       args.batch,
        "imgsz":       args.imgsz,
        "device":      args.device,
        "workers":     args.workers,
        "conf_thresh": TRAIN_CONFIG["conf_thresh"],
    }

    logger  = ResultLogger(output_dir=args.output)
    failed  = []

    # ── Print run plan ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  YOLO Multi-Model Multi-Dataset Pipeline")
    print("=" * 65)
    for ds_key in args.datasets:
        cfg = DATASETS[ds_key]
        if cfg.get("test_only"):
            models_str = ", ".join(cfg["test_models"]) + "  [TEST ONLY]"
        else:
            models_str = ", ".join(args.models)
        print(f"  {ds_key:<22} → {models_str}")
    print(f"\n  Epochs: {train_cfg['epochs']}  Batch: {train_cfg['batch']}  "
          f"ImgSz: {train_cfg['imgsz']}  Device: {train_cfg['device']}")
    print("=" * 65)

    # ── Main loop ────────────────────────────────────────────
    for dataset_key in args.datasets:
        dataset_cfg = DATASETS[dataset_key]
        is_test_only = dataset_cfg.get("test_only", False)

        # 1. Download / verify dataset
        yaml_path = _download(dataset_key, dataset_cfg, args.skip_download)
        if yaml_path is None:
            test_models = dataset_cfg.get("test_models", []) if is_test_only else args.models
            for m in test_models:
                failed.append((dataset_key, m))
            continue

        # 2a. TEST-ONLY dataset (e.g. visually_impaired)
        if is_test_only:
            for model_key in dataset_cfg["test_models"]:
                print(f"\n[TEST ONLY] Dataset={dataset_key}  Model={model_key}")
                try:
                    metrics = test_model(
                        model_key=model_key,
                        model_cfg=MODELS[model_key],
                        dataset_key=dataset_key,
                        yaml_path=yaml_path,
                        train_cfg=train_cfg,
                        output_dir=args.output,
                        weights_path=args.test_weights,
                    )
                    logger.log(dataset_key, model_key, metrics)
                except Exception as e:
                    print(f"\n[ERROR] Test failed for {dataset_key}/{model_key}: {e}")
                    traceback.print_exc()
                    failed.append((dataset_key, model_key))
            continue

        # 2b. Own datasets — train all selected models
        for model_key in args.models:
            print(f"\n[TRAIN] Dataset={dataset_key}  Model={model_key}")
            try:
                metrics = train_model(
                    model_key=model_key,
                    model_cfg=MODELS[model_key],
                    dataset_key=dataset_key,
                    yaml_path=yaml_path,
                    train_cfg=train_cfg,
                    output_dir=args.output,
                )
                logger.log(dataset_key, model_key, metrics)
            except Exception as e:
                print(f"\n[ERROR] Training failed for {dataset_key}/{model_key}: {e}")
                traceback.print_exc()
                failed.append((dataset_key, model_key))

    # ── Summary ──────────────────────────────────────────────
    logger.print_summary()

    if failed:
        print(f"\n[WARNING] {len(failed)} run(s) failed:")
        for ds, md in failed:
            print(f"  ✗ dataset={ds}  model={md}")
        sys.exit(1)
    else:
        print("\n[Done] All runs completed successfully!")


if __name__ == "__main__":
    main()
