"""Save visual previews of exact GTEA hand-mask crops."""

from pathlib import Path

import cv2
import numpy as np

from egovision.data.hand_masks import crop_frame, index_hand_masks
from egovision.data.video import read_video_at_indices


def main() -> None:
    video_path = Path("data/raw/gtea/videos/S1_Cheese_C1.mp4")
    mask_dir = Path("data/raw/gtea/hand_masks/GTEA/Annotations")
    output_dir = Path("outputs/hand_crop_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = index_hand_masks(mask_dir, margin=16)["s1_cheese"]
    indices = np.array(sorted(masks)[:3], dtype=np.int64)
    frames, _ = read_video_at_indices(video_path, indices)

    for frame_index, frame in zip(indices, frames):
        mask = masks[int(frame_index)]
        crop = crop_frame(frame, mask)
        boxed = frame.copy()
        x0, y0, x1, y1 = mask.box
        cv2.rectangle(boxed, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        canvas = np.zeros((max(boxed.shape[0], crop.shape[0]), boxed.shape[1] + crop.shape[1], 3), dtype=np.uint8)
        canvas[:boxed.shape[0], :boxed.shape[1] + 0] = boxed
        canvas[:crop.shape[0], boxed.shape[1]:] = crop
        # OpenCV writes BGR, while our arrays are RGB.
        cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"crop_{frame_index:06d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"frame={frame_index} box={mask.box} crop_shape={crop.shape}")


if __name__ == "__main__":
    main()
