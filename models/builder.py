# ============================================================
# models/builder.py
# Builds YOLO model (standard or ViT-hybrid) from config
# ============================================================

from ultralytics import YOLO
from models.vit_block import ViTBlock


def build_model(model_key: str, model_cfg: dict) -> YOLO:
    """
    Build and return a YOLO model based on config.

    Args:
        model_key: e.g. 'yolov12_vit'
        model_cfg: dict from configs/models.py MODELS[model_key]

    Returns:
        model: Ultralytics YOLO instance (with ViT hook if applicable)
    """
    weights = model_cfg["weights"]
    model_type = model_cfg["type"]

    print(f"\n[Builder] Loading {model_cfg['name']} from '{weights}' ...")
    model = YOLO(weights)

    if model_type == "vit_hybrid":
        model = _inject_vit(model, model_cfg["vit"])

    return model


def _inject_vit(model: YOLO, vit_cfg: dict) -> YOLO:
    """
    Inject a ViTBlock into the YOLO backbone via a forward hook.

    The hook is attached to backbone layer `inject_layer` (default=6),
    which corresponds to the mid-backbone 20×20 / 40×40 feature map.
    """
    dim = vit_cfg["dim"]
    heads = vit_cfg["heads"]
    mlp_ratio = vit_cfg["mlp_ratio"]
    layer_idx = vit_cfg["inject_layer"]

    vit = ViTBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio)

    def _vit_hook(module, inp, out):
        return vit(out)

    target_layer = model.model.model[layer_idx]
    target_layer.register_forward_hook(_vit_hook)

    print(
        f"[Builder] ViT injected at backbone layer {layer_idx} "
        f"(dim={dim}, heads={heads}, mlp_ratio={mlp_ratio})"
    )
    return model
