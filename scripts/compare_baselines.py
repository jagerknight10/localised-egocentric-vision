"""Compare saved frame and causal-Transformer baseline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, default=Path("outputs/frame_baseline"))
    parser.add_argument("--transformer-dir", type=Path, default=Path("outputs/causal_transformer"))
    return parser


def run(args: argparse.Namespace) -> None:
    rows = []
    for subject in ("S1", "S2", "S3", "S4"):
        frame_path = args.frame_dir / f"held_out_{subject}.json"
        transformer_path = args.transformer_dir / f"held_out_{subject}.json"
        frame = json.loads(frame_path.read_text())
        transformer = json.loads(transformer_path.read_text())
        rows.append((subject, frame, transformer))

    print("subject | frame acc | transformer acc | frame balanced | transformer balanced")
    print("--------|-----------:|-----------------:|---------------:|---------------------:")
    for subject, frame, transformer in rows:
        print(
            f"{subject:7} | {frame['test_accuracy']:.4f}     | "
            f"{transformer['test_accuracy']:.4f}           | "
            f"{frame['balanced_accuracy']:.4f}         | "
            f"{transformer['balanced_accuracy']:.4f}"
        )

    for key in ("test_accuracy", "balanced_accuracy"):
        frame_mean = sum(row[1][key] for row in rows) / len(rows)
        transformer_mean = sum(row[2][key] for row in rows) / len(rows)
        print(f"mean {key}: frame={frame_mean:.4f} transformer={transformer_mean:.4f}")


if __name__ == "__main__":
    run(build_parser().parse_args())
