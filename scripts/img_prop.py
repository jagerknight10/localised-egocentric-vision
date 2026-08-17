
import sys
from PIL import Image
import numpy as np

path = sys.argv[1]
image = Image.open(path)
array = np.asarray(image)

print("path:", path)
print("format:", image.format)
print("mode:", image.mode)
print("size:", image.size)
print("shape:", array.shape)
print("dtype:", array.dtype)
print("min:", array.min())
print("max:", array.max())
print("unique values:", len(np.unique(array)))
print("first values:", np.unique(array)[:20])

