from egovision.device import get_device
from egovision.features.dinov2 import DinoV2FeatureExtractor

device = get_device()
print("Using device:", device)

extractor = DinoV2FeatureExtractor(
    model_id="facebook/dinov2-small",
    device=device,
)
extractor.load()

print("DINOv2 loaded successfully")
print("Model:", extractor.model_id)