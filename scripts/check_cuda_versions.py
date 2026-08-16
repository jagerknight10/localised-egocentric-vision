import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
