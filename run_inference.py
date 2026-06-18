from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Docking model inference from a YAML config.")
    parser.add_argument("config", type=Path, help="Path to a Docking model YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides as key=value, e.g. inference.input_csv=/path/input.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from docking_model.workflows.infer import run_inference

    run_inference(str(args.config), overrides=args.overrides or None, show_progress=True)


if __name__ == "__main__":
    main()
