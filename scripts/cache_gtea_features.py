"""Extract and cache DINOv2 features for the GTEA video collection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from egovision.data.annotations import (
    align_frame_labels,
    build_global_label_map,
    encode_action_segments,
    parse_action_annotations,
)
from egovision.data.video import read_video
from egovision.device import get_device
from egovision.features.cache import save_feature_cache
from egovision.features.dinov2 import DinoV2FeatureExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/raw/gtea/videos"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/gtea/annotations/labels"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/features"))
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    video_paths = tuple(sorted(args.videos.glob("*.mp4")))
    annotation_paths = tuple(sorted(args.annotations.glob("*.txt")))
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found in {args.videos}")
    if not annotation_paths:
        raise FileNotFoundError(f"No annotation files found in {args.annotations}")

    label_map = build_global_label_map(annotation_paths)
    device = get_device(args.device)
    extractor = DinoV2FeatureExtractor(device=device)
    extractor.load()

    print(f"videos={len(video_paths)} classes={len(label_map)} device={device}")
    for number, video_path in enumerate(video_paths, start=1):
        cache_path = args.output / f"{video_path.stem}.pt"
        label_path = args.annotations / f"{video_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing annotation for {video_path.name}: {label_path}")
        if cache_path.exists() and not args.overwrite:
            print(f"[{number}/{len(video_paths)}] skip {video_path.name}")
            continue

        video = read_video(video_path, stride=args.stride)
        named_segments = parse_action_annotations(label_path)
        segments, _ = encode_action_segments(named_segments, label_map)
        frame_indices = torch.from_numpy(video.frame_indices)
        labels = align_frame_labels(frame_indices, segments, overlap_policy="later")
        features = extractor.extract(video.frames)

        save_feature_cache(
            cache_path,
            features,
            labels,
            frame_indices,
            {
                "video_id": video_path.stem,
                "fps": video.metadata.fps,
                "stride": args.stride,
                "model_id": extractor.model_id,
                "feature_dim": features.shape[1],
                "label_map": label_map,
                "overlap_policy": "later",
            },
        )
        print(f"[{number}/{len(video_paths)}] saved {cache_path} shape={tuple(features.shape)}")


if __name__ == "__main__":
    run(build_parser().parse_args())
