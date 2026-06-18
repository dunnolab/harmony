from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Docking model training from a YAML config.")
    parser.add_argument("config", type=Path, help="Path to a Docking model YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from docking_model.workflows.train import run_training

    history = run_training(str(args.config))
    if int(os.environ.get("RANK", "0")) != 0:
        return

    for entry in history:
        epoch = entry["epoch"]
        train = entry["train"]
        val = entry["val"]
        message = f"epoch={epoch} train_loss={train.loss:.6g} train_steps={train.steps}"
        if val is not None:
            message += f" val_loss={val.loss:.6g} val_steps={val.steps}"
        inference = entry.get("inference")
        inference_batches = entry.get("inference_batches")
        if inference_batches is not None and inference_batches > 0:
            message += f" val_inference_batches={inference_batches}"
        if inference is not None:
            message += f" val_inference_batches={len(inference)}"
        for key, value in sorted((entry.get("inference_metrics") or {}).items()):
            message += f" {key}={value:.6g}"
        print(message)


if __name__ == "__main__":
    main()
