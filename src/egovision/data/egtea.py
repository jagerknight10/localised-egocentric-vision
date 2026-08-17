"""EGTEA Gaze+ action metadata and official split parsing."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EGTEAActionAnnotation:
    """One dense action interval from ``action_labels.csv``."""

    clip_id: int
    clip_prefix: str
    video_session: str
    start_ms: int
    end_ms: int
    action_name: str
    verb: str
    nouns: tuple[str, ...]


@dataclass(frozen=True)
class EGTEAActionClass:
    """One action class from ``cls_label_index.csv``."""

    action_id: int
    action_name: str
    verb: str
    nouns: tuple[str, ...]


@dataclass(frozen=True)
class EGTEASplitRecord:
    """One clip and zero-based class IDs from an official split.

    EGTEA split files store action, verb, and noun IDs as one-based values;
    this parser normalizes them to zero-based IDs for PyTorch.
    """

    clip_name: str
    action_id: int
    verb_id: int
    noun_ids: tuple[int, ...]


@dataclass(frozen=True)
class EGTEAClipMetadata:
    """Metadata encoded in a trimmed-clip filename."""

    clip_name: str
    video_session: str
    start_ms: int
    end_ms: int
    start_frame: int
    end_frame: int


_CLIP_PATTERN = re.compile(
    r"^(?P<session>.+)-(?P<start_ms>\d+)-(?P<end_ms>\d+)"
    r"-F(?P<start_frame>\d+)-F(?P<end_frame>\d+)$"
)


def _nouns(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_action_labels_csv(path: str | Path) -> tuple[EGTEAActionAnnotation, ...]:
    """Parse EGTEA's semicolon-separated dense action CSV."""
    records: list[EGTEAActionAnnotation] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) != 8:
                raise ValueError(f"Expected 8 columns in {path}, got {len(row)}")
            records.append(
                EGTEAActionAnnotation(
                    clip_id=int(row[0].strip()),
                    clip_prefix=row[1].strip(),
                    video_session=row[2].strip(),
                    start_ms=int(row[3].strip()),
                    end_ms=int(row[4].strip()),
                    action_name=row[5].strip(),
                    verb=row[6].strip(),
                    nouns=_nouns(row[7]),
                )
            )
    if not records:
        raise ValueError(f"No EGTEA action rows found in {path}")
    return tuple(records)


def parse_action_classes_csv(path: str | Path) -> tuple[EGTEAActionClass, ...]:
    """Parse EGTEA's semicolon-separated 106-class index."""
    records: list[EGTEAActionClass] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) != 4:
                raise ValueError(f"Expected 4 columns in {path}, got {len(row)}")
            records.append(
                EGTEAActionClass(
                    action_id=int(row[0].strip()),
                    action_name=row[1].strip(),
                    verb=row[2].strip(),
                    nouns=_nouns(row[3]),
                )
            )
    if not records:
        raise ValueError(f"No EGTEA action classes found in {path}")
    return tuple(records)


def parse_split_file(path: str | Path) -> tuple[EGTEASplitRecord, ...]:
    """Parse an official EGTEA split file."""
    records: list[EGTEASplitRecord] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(f"Invalid split row {line_number} in {path}")
        records.append(
            EGTEASplitRecord(
                clip_name=fields[0],
                action_id=int(fields[1]) - 1,
                verb_id=int(fields[2]) - 1,
                noun_ids=tuple(int(value) - 1 for value in fields[3:]),
            )
        )
    if not records:
        raise ValueError(f"No EGTEA split rows found in {path}")
    return tuple(records)


def parse_clip_metadata(clip_name: str) -> EGTEAClipMetadata:
    """Parse session, millisecond, and padded-frame fields from a clip name."""
    stem = Path(clip_name).stem
    match = _CLIP_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Invalid EGTEA trimmed-clip name: {clip_name}")
    values = match.groupdict()
    start_ms = int(values["start_ms"])
    end_ms = int(values["end_ms"])
    start_frame = int(values["start_frame"])
    end_frame = int(values["end_frame"])
    if end_ms < start_ms or end_frame < start_frame:
        raise ValueError(f"Invalid interval in EGTEA clip name: {clip_name}")
    return EGTEAClipMetadata(
        clip_name=clip_name,
        video_session=values["session"],
        start_ms=start_ms,
        end_ms=end_ms,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def validate_split_classes(
    records: tuple[EGTEASplitRecord, ...],
    classes: tuple[EGTEAActionClass, ...],
) -> None:
    """Raise if split action IDs are absent from the class index."""
    valid_ids = {item.action_id for item in classes}
    unknown = {item.action_id for item in records} - valid_ids
    if unknown:
        raise ValueError(f"Split contains unknown action IDs: {sorted(unknown)}")
