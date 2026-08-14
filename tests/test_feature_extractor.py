from egovision.data.video import read_video
from egovision.device import get_device
from egovision.features.dinov2 import DinoV2FeatureExtractor
from pathlib import Path

video = read_video(
    Path("data/raw/gtea/videos/S1_Cheese_C1.mp4"),
    stride=15,
)

extractor = DinoV2FeatureExtractor(device=get_device())
extractor.load()

features = extractor.extract(video.frames)

print("frames:", video.frames.shape)
print("frame indices:", video.frame_indices.shape)
print("features:", features.shape)
print("feature dtype:", features.dtype)