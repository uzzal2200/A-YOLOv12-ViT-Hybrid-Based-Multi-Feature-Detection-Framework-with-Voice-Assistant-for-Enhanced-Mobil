# ============================================================
# trainers/trainer.py
# Train, validate, and test a YOLO model on a dataset
# ============================================================

import os
from pathlib import Path
from ultralytics import YOLO
from models.builder import build_model


def train_model(
    model_key: str,
    model_cfg: dict,
    dataset_key: str,
    yaml_path: str,
    train_cfg: dict,
    output_dir: str = "results",
) -> dict:
    """
    Full pipeline: build → train → validate → return metrics.

    Args:
        model_key:    e.g. 'yolov12_vit'
        model_cfg:    from MODELS[model_key]
        dataset_key:  e.g. 'currency'
        yaml_path:    path to fixed data.yaml
        train_cfg:    from TRAIN_CONFIG
        output_dir:   root folder to save results

    Returns:
        metrics dict with box precision, recall, mAP50, mAP50-95
    """
    run_name = f"{dataset_key}__{model_key}"
    project_dir = str(Path(output_dir) / dataset_key)

    print(f"\n{'='*60}")
    print(f"  Training : {model_cfg['name']}")
    print(f"  Dataset  : {dataset_key}")
    print(f"  Run      : {run_name}")
    print(f"{'='*60}")

    model = build_model(model_key, model_cfg)

    model.train(
        data=yaml_path,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        workers=train_cfg.get("workers", 4),
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )

    # Load best weights for validation
    best_weights = Path(project_dir) / run_name / "weights" / "best.pt"
    print(f"\n[Trainer] Validating with best weights: {best_weights}")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=yaml_path,
        device=train_cfg["device"],
        project=project_dir,
        name=f"{run_name}_val",
        exist_ok=True,
    )

    result = {
        "run_name": run_name,
        "best_weights": str(best_weights),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
    }

    print(f"\n[Result] {run_name}")
    print(f"         Precision : {result['precision']:.4f}")
    print(f"         Recall    : {result['recall']:.4f}")
    print(f"         mAP50     : {result['mAP50']:.4f}")
    print(f"         mAP50-95  : {result['mAP50_95']:.4f}")

    return result
