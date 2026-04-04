# ============================================================
# utils/visualizer.py
# Plot training curves, confusion matrices, sample predictions
# ============================================================

import glob
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_results(run_dir: str, run_name: str):
    """
    Display all standard training plots from a completed run.

    Args:
        run_dir:  e.g. 'results/currency/currency__yolov12_vit'
        run_name: label for titles
    """
    run_path = Path(run_dir)
    plots = {
        "Results": "results.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Confusion Matrix (Normalized)": "confusion_matrix_normalized.png",
        "Box F1 Curve": "BoxF1_curve.png",
        "Box PR Curve": "BoxPR_curve.png",
        "Box Precision Curve": "BoxP_curve.png",
        "Box Recall Curve": "BoxR_curve.png",
        "Labels": "labels.jpg",
    }

    available = {k: run_path / v for k, v in plots.items() if (run_path / v).exists()}

    if not available:
        print(f"[Visualizer] No plots found in {run_dir}")
        return

    cols = 2
    rows = (len(available) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5), dpi=150)
    axes = axes.flatten()

    for i, (title, path) in enumerate(available.items()):
        axes[i].imshow(plt.imread(str(path)))
        axes[i].set_title(f"{run_name}\n{title}", fontsize=9)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = run_path / "summary_plots.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.show()
    print(f"[Visualizer] Summary saved to {out_path}")


def show_sample_predictions(image_paths: list, model, conf: float = 0.25, max_images: int = 9):
    """
    Run inference on sample images and display a grid.

    Args:
        image_paths: list of image file paths
        model: loaded YOLO model
        conf: confidence threshold
        max_images: max images to show (default 9 = 3x3 grid)
    """
    sample = image_paths[:max_images]
    cols = 3
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), dpi=150)
    axes = axes.flatten()

    for i, img_path in enumerate(sample):
        result = model(img_path, conf=conf, verbose=False)
        plotted = result[0].plot()
        axes[i].imshow(plotted[..., ::-1])  # BGR → RGB
        axes[i].set_title(Path(img_path).name, fontsize=7)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
