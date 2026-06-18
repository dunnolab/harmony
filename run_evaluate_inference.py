from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved Docking model inference outputs from a YAML config.")
    parser.add_argument("config", help="Path to the inference YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides as key=value, e.g. inference.output_dir=/path/predictions.",
    )
    args = parser.parse_args()

    from docking_model.workflows.infer import evaluate_saved_inference

    evaluate_saved_inference(str(args.config), overrides=args.overrides or None, show_progress=True)


if __name__ == "__main__":
    main()
