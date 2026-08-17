"""Dataset interfaces and development datasets."""

from .annotations import (
    ActionSegment,
    NamedActionSegment,
    align_frame_labels,
    build_global_label_map,
    encode_action_segments,
    parse_action_annotations,
)
from .cached import CachedFrameDataset, CachedWindowDataset, split_by_subject, subject_id_from_cache
from .hand_masks import HandMask, crop_frame, index_hand_masks, parse_hand_mask
from .egtea import (
    EGTEAActionAnnotation,
    EGTEAActionClass,
    EGTEAClipMetadata,
    EGTEAClipRecord,
    EGTEASplitRecord,
    action_ids_by_name,
    build_clip_manifest,
    cache_paths_for_split,
    parse_action_classes_csv,
    parse_action_labels_csv,
    parse_clip_metadata,
    parse_split_file,
    validate_split_classes,
)
from .video import SampledVideo, VideoMetadata, read_video, read_video_at_indices, sample_frame_indices

__all__ = [
    "ActionSegment",
    "CachedFrameDataset",
    "CachedWindowDataset",
    "HandMask",
    "NamedActionSegment",
    "SampledVideo",
    "VideoMetadata",
    "align_frame_labels",
    "build_global_label_map",
    "encode_action_segments",
    "parse_action_annotations",
    "split_by_subject",
    "subject_id_from_cache",
    "crop_frame",
    "index_hand_masks",
    "parse_hand_mask",
    "read_video",
    "read_video_at_indices",
    "sample_frame_indices",
    "EGTEAActionAnnotation",
    "EGTEAActionClass",
    "EGTEAClipMetadata",
    "EGTEAClipRecord",
    "EGTEASplitRecord",
    "action_ids_by_name",
    "build_clip_manifest",
    "cache_paths_for_split",
    "parse_action_classes_csv",
    "parse_action_labels_csv",
    "parse_clip_metadata",
    "parse_split_file",
    "validate_split_classes",
]
