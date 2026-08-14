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
from egovision.features.cache import load_feature_cache, save_feature_cache
from egovision.features.dinov2 import DinoV2FeatureExtractor

video_path = Path("data/raw/gtea/videos/S1_Cheese_C1.mp4")
label_path = Path("data/raw/gtea/annotations/labels/S1_Cheese_C1.txt")
annotation_paths = tuple(sorted(label_path.parent.glob("*.txt")))
cache_path = Path("data/features/S1_Cheese_C1.pt")

video = read_video(video_path, stride=15)

print(label_path)

named_segments = parse_action_annotations(label_path)
label_map = build_global_label_map(annotation_paths)
segments, label_map = encode_action_segments(named_segments, label_map)
frame_indices = torch.from_numpy(video.frame_indices)
labels = align_frame_labels(frame_indices, segments, overlap_policy="later")

extractor = DinoV2FeatureExtractor(device=get_device())
extractor.load()
features = extractor.extract(video.frames)

metadata = {
    "video_id": video_path.stem,
    "fps": video.metadata.fps,
    "stride": 15,
    "model_id": extractor.model_id,
    "feature_dim": features.shape[1],
    "label_map": label_map,
}

save_feature_cache(
    cache_path,
    features,
    labels,
    frame_indices,
    metadata,
)

cache = load_feature_cache(cache_path)

print("features:", cache["features"].shape)
print("labels:", cache["labels"].shape)
print("frame_indices:", cache["frame_indices"].shape)
print("metadata:", cache["metadata"])
print("cache:", cache_path)
