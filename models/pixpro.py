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

    def forward(self, x1, x2, p1, p2, f1, f2):
        x1 = self.base_encoder(x1)
        y = self.ppm(x1)

        x2 = self.moment_encoder(x2)
        
        inter_area = self._get_inter_area(p1, p2)
        A_matrix = self._get_A_matrix_per_batch(x1, x2, p1, p2)

        return x1, x2, y
    
    def _get_A_matrix_per_batch(self, x1, x2, p1, p2):
        assert x1.shape == x2.shape, 'x1, x2 shape must be same'
        feature_size = (x1.shape[2], x2.shape[3])
        x1, y1, w1, h1 = p1
        x2, y2, w2, h2 = p2
        
        # warp to original matrix
        rw = w1 / feature_size[0]
        rh = h1 / feature_size[1]
        
        base_matrix = []
        moment_matrix = []

        nx1, ny1 = x1, y1
        nx2, ny2 = x2, y2
        for j in range(feature_size[1]):
            nx1 = x1
            nx2 = x2
            for i in range(feature_size[0]):
                base_matrix.append((nx1.float(), ny1.float()))
                moment_matrix.append((nx2.float(), ny2.float()))
                nx1 = nx1 + rw
                nx2 = nx2 + rw
            ny1 = ny1 + rh
            ny2 = ny2 + rh
        
        diag_len = torch.sqrt((w1.float()**2) + (h1.float()**2))
        return self._get_normalized_distance(base_matrix, moment_matrix, diag_len)

    def _get_normalized_distance(self, l1, l2, diag_len):
        A_matrix = []
        for i in l1:
            for j in l2:
                dist = (torch.sqrt(((i[0]-j[0])**2) + ((i[1]-j[1])**2))) / diag_len
                mask = 1 if dist < self.threshold else 0
                A_matrix.append(mask)
        A_matrix = np.array(A_matrix).reshape((len(l1), len(l2)))
        return A_matrix        
    
    def _get_inter_area(self, p1, p2):
        x1, y1, w1, h1 = p1
        x2, y2, w2, h2 = p2
        
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1+w1, x2+w2)
        yB = min(y1+h1, y2+h2)

        return xA, yA, xB, yB



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
