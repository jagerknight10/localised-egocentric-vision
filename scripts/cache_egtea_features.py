"""Extract frozen DINOv2 features from EGTEA trimmed action clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from egovision.data.egtea import (
    action_ids_by_name,
    build_clip_manifest,
    parse_action_classes_csv,
    parse_action_labels_csv,
)
from egovision.data.video import read_video
from egovision.device import get_device
from egovision.features.cache import save_feature_cache
from egovision.features.dinov2 import DinoV2FeatureExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--action-labels", type=Path, required=True)
    parser.add_argument("--class-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    clip_paths = tuple(sorted(args.clips.rglob("*.mp4")))
    if args.limit is not None:
        clip_paths = clip_paths[: args.limit]
    annotations = parse_action_labels_csv(args.action_labels)
    classes = parse_action_classes_csv(args.class_index)
    action_ids = action_ids_by_name(classes)
    manifest = build_clip_manifest(clip_paths, annotations)
    device = get_device(args.device)
    extractor = DinoV2FeatureExtractor(device=device)
    extractor.load()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"clips={len(manifest)} classes={len(action_ids)} device={device}")

    for number, record in enumerate(manifest, 1):
        cache_path = args.output / f"{record.metadata.clip_name}.pt"
        if cache_path.exists() and not args.overwrite:
            print(f"[{number}/{len(manifest)}] skip {record.metadata.clip_name}")
            continue
        video = read_video(record.path, stride=args.stride)
        features = extractor.extract(video.frames, batch_size=args.batch_size)
        label = action_ids.get(record.annotation.action_name)
        if label is None:
            raise ValueError(f"Unknown action class: {record.annotation.action_name}")
        labels = torch.full((features.shape[0],), label, dtype=torch.long)
        frame_indices = torch.from_numpy(video.frame_indices)
        save_feature_cache(
            cache_path,
            features,
            labels,
            frame_indices,
            {
                "dataset": "EGTEA",
                "clip_name": record.metadata.clip_name,
                "video_session": record.metadata.video_session,
                "action_name": record.annotation.action_name,
                "action_id": label,
                "fps": video.metadata.fps,
                "stride": args.stride,
                "model_id": extractor.model_id,
                "feature_dim": features.shape[1],
                "input_type": "full_frame",
            },
        )
        print(f"[{number}/{len(manifest)}] saved {cache_path.name} shape={tuple(features.shape)}")


if __name__ == "__main__":
    run(build_parser().parse_args())
