# ============================================================
# configs/datasets.py
# Dataset configurations
#
# test_only=True  → skip training, only run val/test with
#                   the best weights of the specified model(s)
# test_models     → which model keys to test on this dataset
# ============================================================

DATASETS = {
    # ── Our own datasets (train ALL 5 models) ────────────────
    "currency": {
        "name": "Bangladeshi Currency Detection",
        "roboflow": {
            "api_key": "1LFy5jaNTlJG9JXeIGz2",
            "workspace": "tazwarul",
            "project": "bangladeshi-currency-detection-8gb5c",
            "version": 2,
            "format": "yolov12",
        },
        "local_dir": "Bangladeshi-Currency-Detection-2",
        "yaml_path": "Bangladeshi-Currency-Detection-2/data.yaml",
        "test_only": False,
    },
    "footpath": {
        "name": "Footpath Occupancy Detection",
        "roboflow": {
            "api_key": "QPnypz6jTqSK3sauF1hK",
            "workspace": "footpath-detection",
            "project": "footpath-occupancy-detection",
            "version": 11,
            "format": "yolov12",
        },
        "local_dir": "footpath-occupancy-detection-11",
        "yaml_path": "footpath-occupancy-detection-11/data.yaml",
        "test_only": False,
    },
    "blind_assistant": {
        "name": "Blind Assistant (Custom Object Detection)",
        "roboflow": {
            "api_key": "QPnypz6jTqSK3sauF1hK",
            "workspace": "md-uzzal-mia",
            "project": "blind-assistant-vbhah-03g8u",
            "version": 1,
            "format": "yolov12",
        },
        "local_dir": "Blind-Assistant-1",
        "yaml_path": "Blind-Assistant-1/data.yaml",
        "test_only": False,
    },

    # ── Public dataset — TEST ONLY with proposed YOLOv12-ViT ─
    # Purpose: validate our proposed model on an independent
    # publicly available benchmark (no training here).
    "visually_impaired": {
        "name": "Visually Impaired Dataset (Public — Test Only)",
        "roboflow": {
            "api_key": "QPnypz6jTqSK3sauF1hK",
            "workspace": "all-mix",
            "project": "visually-impaired-dataset",
            "version": 2,
            "format": "yolov12",
        },
        "local_dir": "Visually-impaired-dataset-2",
        "yaml_path": "Visually-impaired-dataset-2/data.yaml",
        "test_only": True,
        "test_models": ["yolov12_vit"],  # Only proposed model is evaluated here
    },
}
