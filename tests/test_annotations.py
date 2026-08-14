import pytest
import torch

from egovision.data.annotations import (
    ActionSegment,
    align_frame_labels,
    build_global_label_map,
    encode_action_segments,
    parse_action_annotations,
)


def test_alignment_uses_inclusive_segment_bounds() -> None:
    frames = torch.tensor([0, 2, 4, 6, 8])
    segments = (ActionSegment(2, 4, 7), ActionSegment(6, 6, 3))
    assert align_frame_labels(frames, segments).tolist() == [-1, 7, 7, 3, -1]


def test_overlapping_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        align_frame_labels(
            torch.tensor([3]),
            (ActionSegment(1, 3, 1), ActionSegment(3, 5, 2)),
        )


def test_later_segment_can_resolve_transition_boundary() -> None:
    labels = align_frame_labels(
        torch.tensor([3]),
        (ActionSegment(1, 3, 1), ActionSegment(3, 5, 2)),
        overlap_policy="later",
    )
    assert labels.tolist() == [2]


def test_gtea_parser_ignores_object_presence(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("<open><cheese> (2-4) [1]\n<cheese> (1-5)\n")
    named = parse_action_annotations(path)
    encoded, label_map = encode_action_segments(named)
    assert [segment.name for segment in named] == ["open cheese"]
    assert label_map == {"open cheese": 0}
    assert encoded[0] == ActionSegment(2, 4, 0)


def test_gtea_parser_flattens_multiple_nouns(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("<pour><mayonnaise,cheese,bread> (10-20) [1]\n")
    named = parse_action_annotations(path)
    assert named[0].name == "pour mayonnaise cheese bread"


def test_global_label_map_is_shared_across_files(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("<take><bread> (1-2) [0]\n")
    second.write_text("<open><cheese> (3-4) [1]\n")
    label_map = build_global_label_map((first, second))
    assert set(label_map) == {"take bread", "open cheese"}
    encoded, same_map = encode_action_segments(
        parse_action_annotations(first), label_map
    )
    assert same_map == label_map
    assert encoded[0].label == label_map["take bread"]
