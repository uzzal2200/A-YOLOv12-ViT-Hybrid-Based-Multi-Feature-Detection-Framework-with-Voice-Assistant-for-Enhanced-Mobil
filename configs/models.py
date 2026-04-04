# ============================================================
# configs/models.py
# Model configurations - yolov5n, yolov8n, yolov11n, yolov12n, yolov12-vit
# ============================================================

MODELS = {
    "yolov5n": {
        "name": "YOLOv5n",
        "weights": "yolov5n.pt",
        "type": "standard",
    },
    "yolov8n": {
        "name": "YOLOv8n",
        "weights": "yolov8n.pt",
        "type": "standard",
    },
    "yolov11n": {
        "name": "YOLOv11n",
        "weights": "yolo11n.pt",
        "type": "standard",
    },
    "yolov12n": {
        "name": "YOLOv12n",
        "weights": "yolo12n.pt",
        "type": "standard",
    },
    "yolov12_vit": {
        "name": "YOLOv12n + ViT Hybrid",
        "weights": "yolo12n.pt",
        "type": "vit_hybrid",
        "vit": {
            "dim": 256,
            "heads": 4,
            "mlp_ratio": 4,
            "inject_layer": 6,  # Mid-backbone stable feature map
        },
    },
}

# ============================================================
# Training hyperparameters (shared across all runs)
# ============================================================
TRAIN_CONFIG = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,         # Adjust based on your GPU VRAM
    "device": 0,         # 0 = first GPU, 'cpu' = CPU
    "conf_thresh": 0.25,
    "workers": 4,
}
