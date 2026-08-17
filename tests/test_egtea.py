from pathlib import Path

import pytest

from egovision.data.egtea import (
    parse_action_classes_csv,
    parse_action_labels_csv,
    parse_clip_metadata,
    parse_split_file,
    validate_split_classes,
)


def test_parse_egtea_action_csv(tmp_path: Path) -> None:
    path = tmp_path / "action_labels.csv"
    path.write_text(
        "# header\n"
        "1;P01-R01-PastaSalad-100-200-F0001-F0008;P01-R01-PastaSalad;100;200;"
        "Transfer cucumber,cutting_board,bowl;Transfer;cucumber,cutting_board,bowl\n"
    )
    record = parse_action_labels_csv(path)[0]
    assert record.video_session == "P01-R01-PastaSalad"
    assert record.start_ms == 100
    assert record.nouns == ("cucumber", "cutting_board", "bowl")


def test_parse_egtea_class_index_and_validate_split(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    classes.write_text("# header\n0;Cut tomato;Cut;tomato\n")
    split = tmp_path / "train.txt"
    split.write_text("P01-R01-PastaSalad-100-200-F0001-F0008 1 2 3\n")
    parsed_classes = parse_action_classes_csv(classes)
    records = parse_split_file(split)
    validate_split_classes(records, parsed_classes)
    assert records[0].action_id == 0
    assert records[0].verb_id == 1
    assert records[0].noun_ids == (2,)


def test_parse_trimmed_clip_metadata() -> None:
    metadata = parse_clip_metadata(
        "P21-R04-ContinentalBreakfast-291175-294945-F006979-F007088.mp4"
    )
    assert metadata.video_session == "P21-R04-ContinentalBreakfast"
    assert metadata.start_ms == 291175
    assert metadata.end_frame == 7088


def test_invalid_trimmed_clip_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid EGTEA trimmed-clip"):
        parse_clip_metadata("not-a-trimmed-clip.mp4")
