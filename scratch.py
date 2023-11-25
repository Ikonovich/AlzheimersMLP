import torch

x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
w = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0], dtype=torch.long)
output = torch.matmul(x, w.T)

print(output)

