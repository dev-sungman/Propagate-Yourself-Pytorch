import torch
import torch.nn as nn

def PixProLoss(x1, x2, y):



if __name__ == '__main__':
    x1 = torch.randn(8, 256, 7, 7).float()
    x2 = torch.randn(8, 256, 7, 7).float()

    y = torch.randn(8, 256, 7, 7).float()

    loss = PixProLoss(x1, x2, y)

