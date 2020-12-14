import torch
import torch.nn as nn
from models.ppm import PixelPropagationModule

import numpy as np
class PixPro(nn.Module):
    """
    Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/2011.1403
    """
    #TODO: synchronized batchnorm
    def __init__(self, encoder, dim1, dim2, momentum, threshold, temperature, sharpness, num_linear):
        super(PixPro, self).__init__()

        self.encoder = encoder
        self.dim1 = dim1
        self.dim2 = dim2
        
        # encoder momentum
        self.m = momentum

        # encoder temperature
        self.t = temperature

        # threshold
        self.threshold = threshold

        self.base_encoder = encoder(dim1=dim1, dim2=dim2)
        self.moment_encoder = encoder(dim1=dim1, dim2=dim2)
        
        for param_base, param_moment in zip(self.base_encoder.parameters(), self.moment_encoder.parameters()):
            param_moment.data.copy_(param_base.data)
            param_moment.requires_grad = False # do not update

        self.ppm = PixelPropagationModule(sharpness=sharpness, num_linear=num_linear)

    def forward(self, x1, x2):
        base = self.base_encoder(x1)
        y = self.ppm(base)
        
        with torch.no_grad():
            self._momentum_update()
            moment = self.moment_encoder(x2)

        return y, moment

    '''
    def _compute_pix_contrast_loss(self, inter_mask, base_feature, moment_feature, A_matrix):
        remain_fm = inter_mask * base_feature
        
        for i in range(base_feature.shape[2]):
            for j in range(base_feature.shape[3]):
                x_i = base_feature[:,:, i, j]

                for k in range(moment_feature.shape[2]):
                    for l in range(moment_feature.shape[3]):
                        if A_matirx[i, j, k, l] == 1:
                            cos_sim = self._compute_cosine_similarity(base_feature[:,:,i,j], moment_feature[:,:,k,l])
                            print(cos_sim)
                            raise
    '''

    @torch.no_grad()
    #TODO: update momeuntum gradually to 1
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
