import argparse
from pathlib import Path

from benchmark.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run processing benchmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML config file (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
