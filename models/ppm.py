import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelPropagationModule(nn.Module):
    def __init__(self, sharpness=2, num_linear=1):
        super(PixelPropagationModule, self).__init__()
        self.sharpness = sharpness
        self.num_linear = num_linear
        
        self.transform_block = self._make_transform_block(num_linear)

    def _compute_similarity(self, x):
        """
        Compute similarity
        x : input matrix (b, c, h, w)
        """
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

    def _make_transform_block(self, num_linear):
        assert num_linear < 3, 'please select num_linear value below 3'
        if num_linear == 0:
            return nn.Identity()
        elif num_linear == 1:
            return nn.Conv2d(256, 256, 1)
        else:
            return nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 1)
                )
            

    def forward(self, x):
        b, c, h, w = x.shape
        
        vec_sim = self._compute_similarity(x) # B * HW * HW
        
        tr_x = self.transform_block(x)        # C * H * W
        tr_x = tr_x.view(b, h*w, c)           # B * HW * C

        y = torch.bmm(vec_sim, tr_x)          # B * HW * C
        y = y.view(b, c, h, w)                # B * C * H * W
        
        return y

##### TEST
if __name__ == '__main__':
    x = torch.randn(8,256,7,7).float()
    ppm = PixelPropagationModule()

    ppm(x)

