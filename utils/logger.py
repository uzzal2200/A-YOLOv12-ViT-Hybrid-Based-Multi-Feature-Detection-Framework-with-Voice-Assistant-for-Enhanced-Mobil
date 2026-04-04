# ============================================================
# utils/logger.py
# Save and display all training/testing results
# ============================================================

import json
import csv
from pathlib import Path
from datetime import datetime


class ResultLogger:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, dataset_key: str, model_key: str, metrics: dict):
        record = {
            "timestamp": self.timestamp,
            "dataset":   dataset_key,
            "model":     model_key,
            **metrics,
        }
        self.records.append(record)
        self._save_json()
        self._save_csv()

    def _save_json(self):
        path = self.output_dir / f"results_{self.timestamp}.json"
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def _save_csv(self):
        path = self.output_dir / f"results_{self.timestamp}.csv"
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def print_summary(self):
        if not self.records:
            print("[Logger] No results to display.")
            return

        # Separate train vs test-only records
        train_records = [r for r in self.records if r.get("mode") != "test_only"]
        test_records  = [r for r in self.records if r.get("mode") == "test_only"]

        print("\n" + "=" * 82)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 82)

        # ── Training results ─────────────────────────────────
        if train_records:
            print("\n  [OWN DATASETS — Trained Models]")
            header = (f"  {'Dataset':<22} {'Model':<18} {'Precision':>10} "
                      f"{'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
            print(header)
            print("  " + "-" * 76)
            for r in train_records:
                print(
                    f"  {r['dataset']:<22} {r['model']:<18} "
                    f"{r.get('precision',0):>10.4f} "
                    f"{r.get('recall',0):>8.4f} "
                    f"{r.get('mAP50',0):>8.4f} "
                    f"{r.get('mAP50_95',0):>10.4f}"
                )

        # ── Test-only results ─────────────────────────────────
        if test_records:
            print("\n  [PUBLIC DATASET — Proposed Model Validation (Test Only)]")
            header = (f"  {'Dataset':<22} {'Model':<18} {'Precision':>10} "
                      f"{'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
            print(header)
            print("  " + "-" * 76)
            for r in test_records:
                print(
                    f"  {r['dataset']:<22} {r['model']:<18} "
                    f"{r.get('precision',0):>10.4f} "
                    f"{r.get('recall',0):>8.4f} "
                    f"{r.get('mAP50',0):>8.4f} "
                    f"{r.get('mAP50_95',0):>10.4f}"
                )

        print("\n" + "=" * 82)
        print(f"  Results saved → {self.output_dir}/results_{self.timestamp}.json / .csv")
