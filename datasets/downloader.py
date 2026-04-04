# ============================================================
# datasets/downloader.py
# Downloads dataset from Roboflow and fixes data.yaml paths
# ============================================================

import os
from pathlib import Path
import ruamel.yaml
from roboflow import Roboflow


def download_dataset(dataset_key: str, dataset_cfg: dict, output_dir: str = ".") -> str:
    """
    Download dataset from Roboflow if not already present.

    Args:
        dataset_key: e.g. 'currency'
        dataset_cfg: dict from configs/datasets.py
        output_dir: directory to download into

    Returns:
        yaml_path: path to fixed data.yaml
    """
    local_dir = Path(output_dir) / dataset_cfg["local_dir"]
    yaml_path = Path(output_dir) / dataset_cfg["yaml_path"]
    fixed_yaml = Path(output_dir) / f"data_{dataset_key}.yaml"

    if local_dir.exists():
        print(f"[Dataset] '{dataset_cfg['name']}' already downloaded at {local_dir}")
    else:
        print(f"[Dataset] Downloading '{dataset_cfg['name']}' from Roboflow ...")
        rf_cfg = dataset_cfg["roboflow"]
        rf = Roboflow(api_key=rf_cfg["api_key"])
        project = rf.workspace(rf_cfg["workspace"]).project(rf_cfg["project"])
        version = project.version(rf_cfg["version"])
        version.download(rf_cfg["format"], location=str(local_dir))

    _fix_yaml(yaml_path, fixed_yaml, local_dir)
    return str(fixed_yaml)


def _fix_yaml(src_yaml: Path, dst_yaml: Path, base_dir: Path):
    """
    Fix train/val/test paths in data.yaml to absolute paths.
    Saves fixed yaml to dst_yaml.
    """
    yaml = ruamel.yaml.YAML()

    with open(src_yaml) as fp:
        data = yaml.load(fp)

    data["train"] = str((base_dir / "train" / "images").resolve())
    data["val"]   = str((base_dir / "valid" / "images").resolve())
    data["test"]  = str((base_dir / "test"  / "images").resolve())

    with open(dst_yaml, "w") as fp:
        yaml.dump(data, fp)

    print(f"[Dataset] YAML fixed → {dst_yaml}")
    print(f"          train : {data['train']}")
    print(f"          val   : {data['val']}")
    print(f"          test  : {data['test']}")
