from PIL import Image, ImageDraw
from pathlib import Path

frame = Image.open(
    "data/raw/egtea_gaze_plus/preview/video_frame_000031.jpg"
).convert("RGB")

mask = Image.open(
    "data/raw/egtea_gaze_plus/hand_masks/extracted/Images/"
    "OP01-R01-PastaSalad_000031.jpg"
).convert("RGB")

canvas = Image.new("RGB", (frame.width * 2, frame.height), "white")
canvas.paste(frame, (0, 0))
canvas.paste(mask.resize(frame.size), (frame.width, 0))
canvas.save("data/raw/egtea_plus_preview_frame_000031.jpg")
print("saved:", "data/raw/egtea_plus_preview_frame_000031.jpg")

