import torch

data = torch.zeros([8, 16 * 3, 256, 256])

print(data.flatten(-2, -1).permute(0,2,1).shape)
#print(data.chunk(3, dim=1)[0].shape)
