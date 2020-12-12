import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearBlock, self).__init__()
        self.linear = nn.Conv2d(indim, outdim, kernel_size=1)
        self.bn = nn.BatchNorm2d(outdim)
        #self.bn = nn.SyncBatchNorm(outdim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class PixelPropagationModule(nn.Module):
    def __init__(self, sharpness=2, num_linear=1):
        super(PixelPropagationModule, self).__init__()
        self.sharpness = sharpness
        self.num_linear = num_linear
        
        self.transform_block = self._make_transform_block(LinearBlock, num_linear)

    def _compute_similarity(self, x):
        b, c, h, w = x.shape
        
        x_norm = x.view(b, c*h*w)
        x_norm = torch.linalg.norm(x_norm, 2, dim=1)
        # [B] --> [B, 1, 1]
        x_norm = x_norm.view(-1, 1, 1) 

        x_1 = x.view(b, h*w, c)  # B * HW * C
        x_1 = x_1 / x_norm

        x_2 = x.view(b, c, h*w)  # B * C * HW
        x_2 = x_2 / x_norm
        
        cos = torch.bmm(x_1, x_2) # B * HW * HW
        s = torch.pow(F.relu(cos), self.sharpness)
        return s

    def _make_transform_block(self, block, num_linear):
        layers = []
 
        for _ in range(num_linear):
            layers.append(LinearBlock(indim=256, outdim=256))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        
        vec_sim = self._compute_similarity(x) # B * HW * HW
        
        tr_x = self.transform_block(x)        # C * H * W
        tr_x = tr_x.view(b, h*w, c)           # B * HW * C

        y = torch.bmm(vec_sim, tr_x)          # B * HW * C
        y = y.view(b, c, h, w)                # B * C * H * W
        
        return y

if __name__ == '__main__':
    x = torch.randn(8,256,7,7).float()
    ppm = PixelPropagationModule()

    ppm(x)

