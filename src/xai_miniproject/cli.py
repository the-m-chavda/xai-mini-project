from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Iterator, TextIO

from xai_miniproject.analyze import run_analysis
from xai_miniproject.config import Config, load_config
from xai_miniproject.explain import run_explanations
from xai_miniproject.train import run_training


class Tee:
    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xai-mini")
    parser.add_argument(
        "--config",
        default="configs/aifb.yaml",
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where terminal logs are saved.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing a copy of terminal output to a log file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze", help="Compute RDF and task statistics.")
    subparsers.add_parser("train", help="Train and evaluate the R-GCN classifier.")
    subparsers.add_parser("explain", help="Run CELOE/EvoLearner explanations from GNN predictions.")
    subparsers.add_parser("run-all", help="Run analyze, train, and explain in sequence.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.no_log:
        run_command(args.command, config, parser)
        return

    with terminal_log(config, args.command, Path(args.log_dir)):
        run_command(args.command, config, parser)


def run_command(command: str, config: Config, parser: argparse.ArgumentParser) -> None:
    if command == "analyze":
        run_analysis(config)
    elif command == "train":
        run_training(config)
    elif command == "explain":
        run_explanations(config)
    elif command == "run-all":
        run_analysis(config)
        run_training(config)
        run_explanations(config)
    else:
        parser.error(f"Unknown command: {command}")


@contextmanager
def terminal_log(config: Config, command: str, log_root: Path) -> Iterator[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_log_dir = log_root / f"{config.dataset.name}_log"
    dataset_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = dataset_log_dir / f"{timestamp}_{command}.log"
    start_time = time.perf_counter()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)
        print(f"Log file: {log_path}")
        print(f"Command: {command}")
        print(f"Config: {config.path}")
        print(f"Dataset: {config.dataset.name}")
        print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
        print("-" * 80)
        try:
            yield log_path
        except BaseException:
            print("-" * 80)
            print("Run failed with exception:")
            traceback.print_exc()
            raise
        finally:
            duration = time.perf_counter() - start_time
            print("-" * 80)
            print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Duration seconds: {duration:.2f}")
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
