"""Summarize paired full-frame and restricted-input evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, default=Path("outputs/transformer_features_paired_full"))
    parser.add_argument("--restricted-dir", type=Path, default=Path("outputs/transformer_features_restricted"))
    return parser


def run(args: argparse.Namespace) -> None:
    rows = []
    for subject in ("S1", "S2", "S3", "S4"):
        full = json.loads((args.full_dir / f"held_out_{subject}.json").read_text())
        restricted = json.loads((args.restricted_dir / f"held_out_{subject}.json").read_text())
        rows.append((subject, full, restricted))

    print("subject | test frames | full acc | restricted acc | full balanced | restricted balanced")
    print("--------|------------:|---------:|----------------:|---------------:|-------------------:")
    for subject, full, restricted in rows:
        print(
            f"{subject:7} | {restricted['test_frames']:11d} | "
            f"{full['test_accuracy']:.4f}   | {restricted['test_accuracy']:.4f}          | "
            f"{full['balanced_accuracy']:.4f}        | {restricted['balanced_accuracy']:.4f}"
        )

    for key in ("test_accuracy", "balanced_accuracy"):
        full_mean = sum(row[1][key] for row in rows) / len(rows)
        restricted_mean = sum(row[2][key] for row in rows) / len(rows)
        print(f"macro mean {key}: full={full_mean:.4f} restricted={restricted_mean:.4f}")

    total = sum(row[2]["test_frames"] for row in rows)
    print(f"total test frames: {total}")


if __name__ == "__main__":
    run(build_parser().parse_args())
