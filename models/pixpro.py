import torch
import torch.nn as nn
from ppm import PixelPropagationModule

class PixPro(nn.Module):
    """
    Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/2011.1403
    """

    def __init__(self, encoder, dim1, dim2, momentum, temperature, sharpness, num_linear):
        super(PixPro, self).__init__()

        self.encoder = encoder
        self.dim1 = dim1
        self.dim2 = dim2
        
        # encoder momentum
        self.m = momentum

        # encoder temperature
        self.t = temperature

        self.base_encoder = encoder(dim1=dim1, dim2=dim2)
        self.moment_encoder = encoder(dim1=dim1, dim2=dim2)
        
        for param_base, param_moment in zip(self.base_encoder.parameters(), self.moment_encoder.parameters()):
            param_moment.data.copy_(param_base.data)
            param_moment.requires_grad = False # do not update
        

        self.ppm = PixelPropagationModule(sharpness=sharpness, num_linear=num_linear)

    def forward(self, x1, x2):
        x1 = self.base_encoder(x1)
        y = self.ppm(x1)

        x2 = self.moment_encoder(x2)

        return x1, x2, y
    

    @torch.no_grad()
    def _momentum_update(self):
        for param_base, param_moment in zip(self.base_encoder.parameters(), self.moment_encoder.parameters()):
            param_moment.data = param_moment.data * self.m + param_base.data * (1. - self.m)
        

# for test
if __name__ == '__main__':
    from resnet import resnet50
    model = PixPro(resnet50, 2048, 256, 0.99, 0.03)
    model = model.cuda()

    x = torch.randn(8, 3, 224, 224).float()
    print(x.shape)
    x = x.cuda()

    x = model(x)

    print(x.shape)
