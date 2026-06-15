import argparse

from experiment import (
    calibrate_only,
    run_experiment_directory,
    run_experiments,
    train_only,
)
from web_service import DEFAULT_CONFIG_PATH, launch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-output conformal prediction entrypoint.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    experiment_parser = subparsers.add_parser(
        "experiment", help="Run experiments defined in a YAML config."
    )
    experiment_parser.add_argument("config", help="Path to the experiment YAML file.")

    directory_parser = subparsers.add_parser(
        "experiment_directory",
        help="Run every experiment YAML under a directory sequentially.",
    )
    directory_parser.add_argument("directory", help="Directory containing experiment YAMLs.")

    train_parser = subparsers.add_parser(
        "train", help="Train the model only (save the checkpoint under its artifacts dir)."
    )
    train_parser.add_argument("config", help="Path to the YAML config.")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate only, using the model already trained under its artifacts dir.",
    )
    calibrate_parser.add_argument("config", help="Path to the YAML config.")

    web_parser = subparsers.add_parser(
        "web_service", help="Launch the Gradio web service."
    )
    web_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the web service config (default: {DEFAULT_CONFIG_PATH}).",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "experiment":
        run_experiments(args.config)
    elif args.mode == "experiment_directory":
        run_experiment_directory(args.directory)
    elif args.mode == "train":
        train_only(args.config)
    elif args.mode == "calibrate":
        calibrate_only(args.config)
    elif args.mode == "web_service":
        launch(args.config)


if __name__ == "__main__":
    main()
