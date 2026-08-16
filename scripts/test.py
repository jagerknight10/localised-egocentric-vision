import cv2
import torch
import transformers

print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
