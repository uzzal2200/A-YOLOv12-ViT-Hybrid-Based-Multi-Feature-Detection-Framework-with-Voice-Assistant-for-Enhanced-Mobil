# ============================================================
# trainers/tester.py
# Test-only pipeline: load pre-trained best.pt → val on
# a public/external dataset (no training).
#
# Used for: visually_impaired dataset with yolov12_vit
# Purpose : validate the proposed model on an independent
#           publicly available benchmark.
# ============================================================

import glob
from pathlib import Path
from ultralytics import YOLO
from models.builder import build_model


def test_model(
    model_key: str,
    model_cfg: dict,
    dataset_key: str,
    yaml_path: str,
    train_cfg: dict,
    output_dir: str = "results",
    weights_path: str = None,
) -> dict:
    """
    Validate a model on a dataset WITHOUT training.

    If weights_path is None, uses the base pretrained weights
    (yolo12n.pt etc.) directly for evaluation — useful when we
    want to test the ViT-hybrid architecture zero-shot on a
    public dataset, OR you can pass a custom best.pt path.

    Args:
        model_key:    e.g. 'yolov12_vit'
        model_cfg:    from MODELS[model_key]
        dataset_key:  e.g. 'visually_impaired'
        yaml_path:    path to fixed data.yaml
        train_cfg:    from TRAIN_CONFIG (uses device, conf_thresh)
        output_dir:   root folder to save results
        weights_path: optional path to a best.pt to load

    Returns:
        metrics dict
    """
    run_name = f"{dataset_key}__{model_key}__test_only"
    project_dir = str(Path(output_dir) / dataset_key)

    print(f"\n{'='*60}")
    print(f"  [TEST ONLY] Model  : {model_cfg['name']}")
    print(f"              Dataset: {dataset_key}")
    print(f"              Run    : {run_name}")
    print(f"{'='*60}")

    if weights_path and Path(weights_path).exists():
        print(f"[Tester] Loading custom weights: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"[Tester] Loading base weights: {model_cfg['weights']}")
        model = build_model(model_key, model_cfg)

    metrics = model.val(
        data=yaml_path,
        device=train_cfg["device"],
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )

    # Run inference on test split and save visual results
    test_img_dir = _get_test_dir(yaml_path)
    if test_img_dir and Path(test_img_dir).exists():
        print(f"\n[Tester] Running inference on test split: {test_img_dir}")
        model.predict(
            source=test_img_dir,
            conf=train_cfg.get("conf_thresh", 0.25),
            save=True,
            project=project_dir,
            name=f"{run_name}_predict",
            exist_ok=True,
        )

    result = {
        "run_name":  run_name,
        "mode":      "test_only",
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
    }

    print(f"\n[Test Result] {run_name}")
    print(f"              Precision : {result['precision']:.4f}")
    print(f"              Recall    : {result['recall']:.4f}")
    print(f"              mAP50     : {result['mAP50']:.4f}")
    print(f"              mAP50-95  : {result['mAP50_95']:.4f}")

    return result


def _get_test_dir(yaml_path: str) -> str:
    """Parse test image directory from data.yaml."""
    try:
        import ruamel.yaml
        y = ruamel.yaml.YAML()
        with open(yaml_path) as f:
            data = y.load(f)
        return data.get("test", "")
    except Exception:
        return ""
