import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0))
