"""Cache DINOv2 features for exact GTEA hand-mask frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from egovision.data.annotations import align_frame_labels, build_global_label_map, encode_action_segments, parse_action_annotations
from egovision.data.hand_masks import crop_frame, index_hand_masks
from egovision.data.video import read_video
from egovision.device import get_device
from egovision.features.cache import save_feature_cache
from egovision.features.dinov2 import DinoV2FeatureExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/raw/gtea/videos"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/gtea/annotations/labels"))
    parser.add_argument("--masks", type=Path, default=Path("data/raw/gtea/hand_masks/GTEA/Annotations"))
    parser.add_argument("--output", type=Path, default=Path("data/features_restricted"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    label_paths = tuple(sorted(args.labels.glob("*.txt")))
    label_map = build_global_label_map(label_paths)
    masks = index_hand_masks(args.masks, margin=args.margin)
    extractor = DinoV2FeatureExtractor(device=get_device(args.device))
    extractor.load()
    for video_path in sorted(args.videos.glob("*.mp4")):
        key = video_path.stem.lower().removesuffix("_c1")
        available = masks.get(key, {})
        if not available:
            print(f"skip {video_path.name}: no masks")
            continue
        output_path = args.output / f"{video_path.stem}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"skip {video_path.name}: cache exists")
            continue
        video = read_video(video_path, stride=1)
        selected = [(i, available[i]) for i in video.frame_indices.tolist() if i in available]
        if not selected:
            print(f"skip {video_path.name}: no exact decoded mask frames")
            continue
        frames = torch.from_numpy(video.frame_indices)
        selected_indices = torch.tensor([i for i, _ in selected], dtype=torch.int64)
        crops = __import__("numpy").stack([crop_frame(video.frames[i], mask) for i, mask in selected])
        named = parse_action_annotations(args.labels / f"{video_path.stem}.txt")
        segments, _ = encode_action_segments(named, label_map)
        labels = align_frame_labels(selected_indices, segments, overlap_policy="later")
        features = extractor.extract(crops)
        save_feature_cache(output_path, features, labels, selected_indices, {
            "video_id": video_path.stem,
            "model_id": extractor.model_id,
            "feature_dim": features.shape[1],
            "label_map": label_map,
            "input_type": "oracle_hand_mask_crop",
            "margin": args.margin,
            "num_annotated_frames": len(selected_indices),
        })
        print(f"saved {output_path} shape={tuple(features.shape)}")


if __name__ == "__main__":
    run(build_parser().parse_args())
