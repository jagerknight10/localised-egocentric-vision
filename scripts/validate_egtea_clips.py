"""Validate EGTEA trimmed-clip names against official annotations."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from egovision.data.egtea import build_clip_manifest, parse_action_labels_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--clip-archive", type=Path)
    parser.add_argument("--clips", type=Path)
    return parser


def run(args: argparse.Namespace) -> None:
    if bool(args.clip_archive) == bool(args.clips):
        raise ValueError("Provide exactly one of --clip-archive or --clips")
    annotations = parse_action_labels_csv(args.annotations)
    if args.clip_archive:
        with tarfile.open(args.clip_archive) as archive:
            names = tuple(
                Path(member.name)
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".mp4")
            )
    else:
        names = tuple(args.clips.rglob("*.mp4"))
    manifest = build_clip_manifest(names, annotations)
    print(f"annotations={len(annotations)}")
    print(f"clips={len(names)}")
    print(f"validated={len(manifest)}")
    print(f"sessions={len({item.metadata.video_session for item in manifest})}")


if __name__ == "__main__":
    run(build_parser().parse_args())
