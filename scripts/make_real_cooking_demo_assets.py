"""Generate real GTEA cooking demo assets for the academic project page.

Creates, under demo/assets/real-cooking/:
  1. cooking-source-clip.mp4      raw egocentric clip, no overlays
  2. cooking-localized-stream.mp4 full frame + localized crop + causal strip
  3. cooking-demo.json            full provenance metadata
  4. ATTRIBUTION.md               dataset attribution

Design notes (research decisions, not boilerplate):
  * GTEA hand-mask XMLs exist only every 40 original frames. We linearly
    interpolate each hand box corner between adjacent anchors and HOLD each
    anchor box outside the anchor range. This is conservative: the overlay
    never drifts far from an annotated position. The policy is recorded in
    cooking-demo.json.
  * XML frame N (1-based) corresponds to MP4 decoder index N-1 (verified by
    pixel diff against the dataset reference images).
  * The XML canvas is 720x405 while the decoded video is 720x404. We clamp
    the bottom coordinate to row 403 (the last video row) and record it in
    the metadata.
  * The "causal" temporal strip contains only past/current localized crops;
    a hatched block marks the future as visually unavailable.
  * Every displayed label comes from the dataset action annotations. No
    model predictions, confidences, or invented labels are shown.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# --- Layout (all coordinates are output-canvas pixels) -------------------
CANVAS_W, CANVAS_H = 1280, 720
HEADER_H = 50
PANEL_CAP_Y = 72          # baseline of panel captions
PANELS_Y = 80
PANEL_W = 628
PANEL_H = 388
PANEL_GAP = 24
TL_CAP_Y = 490            # baseline of timeline caption
TIMELINE_Y, TIMELINE_H = 498, 37
STRIP_CAP_Y = 560
STRIP_Y, STRIP_H = 568, 64
STRIP_X0 = 16
STRIP_CELLS = 220         # 220 frames of history at 15 fps (~14.7 s)
STRIP_CELL_W = 4          # strip width = 880 px
FUTURE_X0 = 16 + STRIP_CELLS * STRIP_CELL_W + 12
FOOTER_Y = 660

INK = (38, 40, 46)
MUTED = (105, 110, 120)
LINE = (220, 222, 228)
ACCENT = (23, 105, 170)
BOX_COLOR = (0, 153, 76)  # BGR green, restrained
PANEL_BG = (246, 247, 249)
FUTURE_BG = (238, 239, 242)
SEG_COLORS = [(38, 40, 46), (58, 94, 133), (86, 130, 102)]  # restrained


def parse_hand_boxes(xml_path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Return {hand_name: (x0, y0, x1, y1)} from one GTEA annotation XML."""
    root = ET.parse(xml_path).getroot()
    hands: dict[str, tuple[int, int, int, int]] = {}
    for obj in root.findall(".//object"):
        name = (obj.findtext("name") or "hand").strip()
        pts = [(int(p.findtext("x")), int(p.findtext("y"))) for p in obj.findall(".//polygon/pt")]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        hands[name] = (min(xs), min(ys), max(xs), max(ys))
    if not hands:
        raise ValueError(f"No hand polygons in {xml_path}")
    return hands


def load_anchor_boxes(ann_dir: Path, video_stem: str, anchor_frames: list[int]) -> dict[int, dict[str, tuple[int, int, int, int]]]:
    boxes: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    for f in anchor_frames:
        path = ann_dir / f"{video_stem}_{f:010d}.xml"
        if not path.exists():
            raise FileNotFoundError(path)
        boxes[f] = parse_hand_boxes(path)
    return boxes


def lerp_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int], t: float) -> tuple[int, int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))  # type: ignore[return-value]


def box_at_frame(
    anchors: dict[int, dict[str, tuple[int, int, int, int]]],
    frame: int,
    clip_h: int,
) -> tuple[int, int, int, int]:
    """Interpolated union box of both hands at one original frame.

    Conservative policy: linear interpolation between adjacent anchors,
    hold the nearest anchor outside the anchor range, clamp to the decoded
    video size (720x404).
    """
    fs = sorted(anchors)
    if frame <= fs[0]:
        hands = anchors[fs[0]]
    elif frame >= fs[-1]:
        hands = anchors[fs[-1]]
    else:
        fa = max(f for f in fs if f <= frame)
        fb = min(f for f in fs if f >= frame)
        if fa == fb:
            hands = anchors[fa]
        else:
            t = (frame - fa) / (fb - fa)
            hands = {name: lerp_box(anchors[fa][name], anchors[fb][name], t) for name in anchors[fa]}
    x0 = min(b[0] for b in hands.values())
    y0 = min(b[1] for b in hands.values())
    x1 = max(b[2] for b in hands.values())
    y1 = max(b[3] for b in hands.values())
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(719, x1), min(clip_h - 1, y1)
    return (x0, y0, x1, y1)


def text_bg(img: np.ndarray, xy: tuple[int, int], text: str, scale: float, color) -> None:
    """Draw text with a white halo for legibility over video."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_crop(crop: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    x0, y0 = (panel_w - nw) // 2, (panel_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = small
    return canvas


def draw_future_hatch(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Diagonal hatch marking the unavailable-future region."""
    canvas[y0:y1 + 1, x0:x1 + 1] = FUTURE_BG
    for x in range(x0 - (y1 - y0), x1, 14):
        xa = max(x0, x)
        ya = y1 - (xa - x)
        xb = min(x1, x + (y1 - y0))
        yb = ya - (xb - xa)
        cv2.line(canvas, (xa, ya), (xb, yb), (214, 216, 222), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), LINE, 1)


def build_interval(seg: dict, clip_end: int) -> dict:
    """Render one action interval for metadata, clipped to the clip range."""
    out = {
        "action": seg["action"],
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "start_time_seconds": round(seg["start_frame"] / 15.0, 4),
        "end_time_seconds": round(min(seg["end_frame"], clip_end) / 15.0, 4),
        "clipped_to_clip": seg["end_frame"] > clip_end,
    }
    if out["clipped_to_clip"]:
        out["clipped_note"] = (
            f"original interval ends at frame {seg['end_frame']}, beyond the clip; "
            f"the timeline shows it only up to frame {clip_end}."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/raw/gtea"))
    ap.add_argument("--out", type=Path, default=Path("demo/assets/real-cooking"))
    ap.add_argument("--start-frame", type=int, default=6, help="First original (1-based) GTEA frame in the clip.")
    ap.add_argument("--end-frame", type=int, default=226, help="Last original (1-based) GTEA frame in the clip.")
    ap.add_argument("--anchors", type=int, nargs="+", default=[20, 60, 100, 140, 180])
    args = ap.parse_args()

    ann_dir = args.data / "hand_masks/GTEA/Annotations"
    video_path = args.data / "videos/S1_Cheese_C1.mp4"
    video_stem = "s1_cheese"

    anchors = load_anchor_boxes(ann_dir, video_stem, args.anchors)
    # Action intervals from the GTEA action annotations for S1_Cheese_C1.
    action_segments = [
        {"action": "take bread", "start_frame": 13, "end_frame": 71},
        {"action": "take cheese", "start_frame": 100, "end_frame": 172},
        {"action": "open cheese", "start_frame": 172, "end_frame": 270},
    ]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 15.0) > 0.01:
        raise SystemExit(f"Expected 15 fps, got {fps}")
    clip_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = args.end_frame - args.start_frame + 1

    args.out.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    w_clip = cv2.VideoWriter(str(args.out / "cooking-source-clip.mp4"), fourcc, fps, (clip_w, clip_h))
    w_stream = cv2.VideoWriter(str(args.out / "cooking-localized-stream.mp4"), fourcc, fps, (CANVAS_W, CANVAS_H))
    if not w_clip.isOpened() or not w_stream.isOpened():
        raise SystemExit("Could not open H.264 writers")

    # XML frame N (1-based) == decoder index N-1. The clip starts at decoder
    # index start_frame-1, so we drop the first (start_frame-1) decoded frames.
    drop = args.start_frame - 1
    px_per_frame = (CANVAS_W - 2 * STRIP_X0) / n_frames
    crop_hist: list[np.ndarray] = []
    boxes_meta: list[dict] = []

    for i in range(n_frames + drop):
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"Decode failed at decoder index {i}")
        if i < drop:
            continue
        ci = i - drop                      # clip index, 0-based
        gtea_frame = args.start_frame + ci  # original 1-based frame number

        w_clip.write(img)

        box = box_at_frame(anchors, gtea_frame, clip_h)
        boxes_meta.append({"frame": gtea_frame, "box_xyxy": list(box)})
        x0, y0, x1, y1 = box
        crop = img[y0 : y1 + 1, x0 : x1 + 1]

        canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        # header bar
        canvas[:HEADER_H] = (250, 250, 251)
        cv2.line(canvas, (0, HEADER_H), (CANVAS_W, HEADER_H), LINE, 1)
        cv2.putText(canvas, "S1_Cheese_C1  |  GTEA egocentric cooking video", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, INK, 2, cv2.LINE_AA)
        t = gtea_frame / 15.0
        cv2.putText(canvas, f"t = {t:5.2f} s   (frame {gtea_frame})", (CANVAS_W - 330, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, MUTED, 2, cv2.LINE_AA)

        # left panel: full frame + annotated box
        left = fit_crop(img, PANEL_W, PANEL_H)
        canvas[PANELS_Y:PANELS_Y + PANEL_H, 0:PANEL_W] = left
        cv2.rectangle(canvas, (0, PANELS_Y), (PANEL_W, PANELS_Y + PANEL_H), LINE, 1)
        scale = min(PANEL_W / clip_w, PANEL_H / clip_h)
        ox, oy = (PANEL_W - int(clip_w * scale)) // 2, (PANEL_H - int(clip_h * scale)) // 2
        bx0 = ox + int(x0 * scale); by0 = oy + int(y0 * scale)
        bx1 = ox + int(x1 * scale); by1 = oy + int(y1 * scale)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BOX_COLOR, 2, cv2.LINE_AA)
        text_bg(canvas, (10, PANEL_CAP_Y), "Original frame - task-relevant region (GTEA hand-mask annotation)", 0.52, MUTED)

        # right panel: localized crop
        rx = PANEL_W + PANEL_GAP
        right = fit_crop(crop, PANEL_W, PANEL_H)
        canvas[PANELS_Y:PANELS_Y + PANEL_H, rx:rx + PANEL_W] = right
        cv2.rectangle(canvas, (rx, PANELS_Y), (rx + PANEL_W, PANELS_Y + PANEL_H), LINE, 1)
        text_bg(canvas, (rx + 10, PANEL_CAP_Y), "Localized crop - hand-object region only", 0.52, MUTED)

        # action timeline bar (dataset annotations only)
        tx0, tx1 = STRIP_X0, CANVAS_W - STRIP_X0
        cv2.rectangle(canvas, (tx0, TIMELINE_Y), (tx1, TIMELINE_Y + TIMELINE_H), (240, 241, 244), -1)
        for si, seg in enumerate(action_segments):
            s0 = max(seg["start_frame"], args.start_frame) - args.start_frame
            s1 = min(seg["end_frame"], args.end_frame) - args.start_frame
            if s1 < 0 or s0 >= n_frames:
                continue
            s0, s1 = max(0, s0), min(n_frames - 1, s1)
            xa = int(tx0 + s0 * px_per_frame)
            xb = int(tx0 + (s1 + 1) * px_per_frame)
            cv2.rectangle(canvas, (xa, TIMELINE_Y), (xb, TIMELINE_Y + TIMELINE_H), (255, 255, 255), -1)
            cv2.rectangle(canvas, (xa, TIMELINE_Y), (xb, TIMELINE_Y + TIMELINE_H), SEG_COLORS[si % 3], 1)
            cv2.putText(canvas, f" {seg['action']} ", (xa + 5, TIMELINE_Y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, SEG_COLORS[si % 3], 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (tx0, TIMELINE_Y), (tx1, TIMELINE_Y + TIMELINE_H), LINE, 1)
        # playhead
        ph = int(tx0 + ci * px_per_frame)
        cv2.line(canvas, (ph, TIMELINE_Y - 3), (ph, TIMELINE_Y + TIMELINE_H + 3), ACCENT, 2, cv2.LINE_AA)
        text_bg(canvas, (tx0, TL_CAP_Y), "Action annotations (original GTEA labels)", 0.5, MUTED)

        # causal temporal strip: right-aligned history of past/current crops
        strip_w = STRIP_CELLS * STRIP_CELL_W
        strip_x1 = STRIP_X0 + strip_w
        cv2.rectangle(canvas, (STRIP_X0, STRIP_Y), (strip_x1, STRIP_Y + STRIP_H), (255, 255, 255), -1)
        for k in range(STRIP_CELLS):
            j = ci - (STRIP_CELLS - 1 - k)
            if j < 0:
                continue
            src = crop if j == ci else crop_hist[j]
            cell = cv2.resize(src, (STRIP_CELL_W, STRIP_H), interpolation=cv2.INTER_AREA)
            canvas[STRIP_Y : STRIP_Y + STRIP_H,
                   STRIP_X0 + k * STRIP_CELL_W : STRIP_X0 + (k + 1) * STRIP_CELL_W] = cell
        cv2.rectangle(canvas, (STRIP_X0, STRIP_Y), (strip_x1, STRIP_Y + STRIP_H), LINE, 1)
        # current-frame marker: the filled history is right-aligned, so the
        # current frame always sits at the right edge of the strip.
        cx = STRIP_X0 + strip_w - 2
        cv2.rectangle(canvas, (cx - 2, STRIP_Y), (cx + 2, STRIP_Y + STRIP_H), ACCENT, -1)
        # future region: unavailable, hatched
        draw_future_hatch(canvas, FUTURE_X0, STRIP_Y, CANVAS_W - STRIP_X0, STRIP_Y + STRIP_H)
        text_bg(canvas, (FUTURE_X0 + 14, STRIP_Y + 38), "future - unavailable", 0.55, MUTED)
        text_bg(canvas, (STRIP_X0, STRIP_CAP_Y), "Causal context - past and current localized evidence only", 0.52, MUTED)

        # footer
        cv2.line(canvas, (0, FOOTER_Y), (CANVAS_W, FOOTER_Y), LINE, 1)
        cv2.putText(canvas,
                    "GTEA dataset (Li, Liu & Rehg, ECCV 2018)  |  boxes: GTEA hand-mask annotations  |  "
                    "labels: GTEA action annotations",
                    (16, FOOTER_Y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)

        w_stream.write(canvas)
        crop_hist.append(crop)

    cap.release()
    w_clip.release()
    w_stream.release()

    meta = {
        "source_clip_identifier": "GTEA:S1_Cheese_C1",
        "source_video": "GTEA raw video S1_Cheese_C1.mp4 (720x404 @ 15 fps)",
        "clip_start_frame": args.start_frame,
        "clip_end_frame": args.end_frame,
        "clip_frame_count": n_frames,
        "clip_duration_seconds": round(n_frames / 15.0, 4),
        "fps": 15.0,
        "original_dimensions": [clip_w, clip_h],
        "xml_frame_ids_used": [a for a in args.anchors if args.start_frame <= a <= args.end_frame],
        "xml_all_anchor_frames_in_clip": [a for a in args.anchors if args.start_frame <= a <= args.end_frame],
        "frame_numbering": "Original 1-based GTEA frame numbers are preserved in all metadata. "
                           "XML frame N aligns with MP4 decoder index N-1 (verified by pixel "
                           "diff against the dataset reference images).",
        "pixel_size_adjustment": "XML canvas is 720x405 (rows 0-404); decoded video is 720x404 "
                                 "(rows 0-403). Annotation row 404 does not exist in the video, "
                                 "so bottom box coordinates are clamped to row 403 (up to a "
                                 "2-pixel vertical adjustment on the bottom edge only).",
        "annotation_policy": {
            "density": "Hand-mask XML exists only every 40 original frames; 5 anchors (20, 60, "
                       "100, 140, 180) fall inside the clip.",
            "interpolation": "Per-corner linear interpolation of each hand bounding box between "
                             "adjacent anchors; hold the nearest anchor outside the anchor range; "
                             "clamp to the decoded video size.",
            "box_definition": "Union of the annotated hand boxes (left and right hand) at each "
                              "frame; this is the task-relevant region.",
            "anchors_outside_clip": "Clip frames 6-19 hold anchor 20; frames 181-226 hold anchor 180.",
        },
        "boxes_per_rendered_frame": boxes_meta,
        "action_intervals": [build_interval(s, args.end_frame) for s in action_segments],
        "action_transition_timestamps_seconds": [round(13 / 15.0, 4), round(100 / 15.0, 4), round(172 / 15.0, 4)],
        "label_provenance": "All displayed labels and intervals are taken directly from the GTEA "
                            "action annotations for S1_Cheese_C1. No model predictions, "
                            "confidences, or invented labels are shown.",
        "files": {
            "source_clip": "cooking-source-clip.mp4",
            "localized_stream": "cooking-localized-stream.mp4",
            "attribution": "ATTRIBUTION.md",
        },
    }
    (args.out / "cooking-demo.json").write_text(json.dumps(meta, indent=2) + "\n")

    attribution = """# Data attribution

Demo assets in this directory use the **GTEA** (Google/Texas Ego-Centric
Action) dataset, subject S1, video `S1_Cheese_C1`.

Required citation:

> Yin Li, Miao Liu, and James M. Rehg. "In the Eye of the Beholder: Joint
> Learning of Gaze and Actions in First Person Video." ECCV, 2018.

- The egocentric footage and action labels are the original dataset data.
- The bounding boxes and localized crops derive from the GTEA **hand-mask
  annotations** (sparse XML polygon annotations, one per 40 original frames).
  Boxes between annotation frames use the conservative interpolation policy
  documented in `cooking-demo.json`.
- No model predictions are shown; all on-screen labels are dataset
  annotations.
"""
    (args.out / "ATTRIBUTION.md").write_text(attribution)

    for name in ["cooking-source-clip.mp4", "cooking-localized-stream.mp4"]:
        p = args.out / name
        print(f"{name}: {p.stat().st_size / 1e6:.2f} MB")
    print("metadata + attribution written to", args.out)


if __name__ == "__main__":
    main()
