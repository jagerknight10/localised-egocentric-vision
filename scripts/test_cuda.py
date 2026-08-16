import torch

x = torch.randn(2, 3, 224, 224, device="cuda")
conv = torch.nn.Conv2d(3, 8, 3).cuda()
y = conv(x)

print("CUDA convolution succeeded:", y.shape)
