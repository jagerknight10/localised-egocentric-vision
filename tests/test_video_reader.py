from pathlib import Path
from egovision.data.video import read_video

video_path = Path("data/raw/gtea/videos/S1_Cheese_C1.mp4")

sampled = read_video(video_path, stride=15)

print("video:", sampled.metadata.path)
print("frame_count:", sampled.metadata.frame_count)
print("fps:", sampled.metadata.fps)
print("resolution:", sampled.metadata.width, "x", sampled.metadata.height)
print("sampled frame shape:", sampled.frames.shape)
print("sampled indices:", sampled.frame_indices[:10])
