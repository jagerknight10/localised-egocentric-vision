import cv2
import numpy as np

mask_second = 16
video_frame_index = mask_second * 24

frame_path = (
    f"data/raw/egtea_gaze_plus/preview/video_frame_{video_frame_index:06d}.jpg"
)
mask_path = (
    "data/raw/egtea_gaze_plus/hand_masks/extracted/"
    f"Masks/OP01-R01-PastaSalad_{mask_second:06d}.png"
)

frame = cv2.imread(frame_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask = cv2.resize(
    mask,
    (frame.shape[1], frame.shape[0]),
    interpolation=cv2.INTER_NEAREST,
)

binary = mask > 0

ys, xs = np.where(binary)
if len(xs) == 0:
    raise RuntimeError("Mask contains no hand pixels")

x1, x2 = xs.min(), xs.max() + 1
y1, y2 = ys.min(), ys.max() + 1

overlay = frame.copy()
overlay[binary] = (0, 255, 0)
overlay = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

crop = frame[y1:y2, x1:x2]

cv2.imwrite(
    f"data/raw/egtea_gaze_plus/preview/overlay_{mask_second:06d}.jpg",
    overlay,
)
cv2.imwrite(
    f"data/raw/egtea_gaze_plus/preview/crop_{mask_second:06d}.jpg",
    crop,
)

print("frame shape:", frame.shape)
print("mask shape after resize:", mask.shape)
print("mask second:", mask_second)
print("video frame index:", video_frame_index)
print("bounding box:", (x1, y1, x2, y2))
print("crop shape:", crop.shape)

