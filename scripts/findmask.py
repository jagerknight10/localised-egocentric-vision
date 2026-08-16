from pathlib import Path
import cv2
import numpy as np

mask_dir = Path(
    "data/raw/egtea_gaze_plus/hand_masks/extracted/Masks"
)

for path in sorted(mask_dir.glob("OP01-R01-PastaSalad_*.png")):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    count = int(np.count_nonzero(mask))
    if count:
        print("first non-empty mask:", path)
        print("nonzero pixels:", count)
        break
else:
    print("No non-empty masks found")
