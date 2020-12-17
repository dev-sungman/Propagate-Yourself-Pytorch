import torch
import torch.nn as nn
import numpy as np

import time

class PixproLoss(nn.Module):
    def __init__(self, args):
        super(PixproLoss, self).__init__()
        self.args = args

    def forward(self, base, moment, A_matrix):
        assert base.shape == moment.shape, 'base, moment shape must be same' 
        
        pixpro_loss = -self._get_pixpro_loss(base, moment, A_matrix)

        return pixpro_loss

    def _get_pixpro_loss(self, base, moment, A_matrix):
        """
        base : base matrix (B, C, 7, 7)
        moment : moment matrix (B, C, 7, 7)
        A_matrix : A matrix (B, 49, 49)
        """

        base = base.view(base.shape[0], base.shape[1], 1, -1)
        moment = moment.view(moment.shape[0], moment.shape[1], 1, -1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(base, moment)
        
        A_matrix = A_matrix.type(torch.BoolTensor).cuda()
        return cos_sim.masked_select(A_matrix).mean()

class PixContrastLoss(nn.Module):
    def __init__(self, args):
        super(PixContrastLoss, self).__init__()
        self.args = args

    def forward(self, p1, p2, m1, m2, irect):
        """
        p1 : base position matrix (B, 7, 7, 2)
        p2 : moment position matrix (B, 7, 7, 2)
        irect : intersection rectangle
        m1 : base feature matrix (B, C, 7, 7)
        m2 : moment feature matrix (B, C, 7, 7)
        """
        inter_mask = self._get_intersection_mask(p1, m1, irect)    
        print(inter_mask)

    def _get_intersection_mask(self, p, m, irect):
        """
        p : position matrix (B, 7, 7, 2)
        m : base feature matrix (B, C, 7, 7)
        irect : intersection rectangle
        """
        ix1, iy1, ix2, iy2 = irect[0], irect[1], irect[2], irect[3]
        print(p[:,:,1])
        print(p[:,:,0])
        inter_mask = torch.where((p[:,:,0] >= iy1) & (p[:,:,0] <= iy2) &
                                  (p[:,:,1] >= ix1) & (p[:,:,1] <= ix2), 1., 0.)
        print(inter_mask.shape)
        return inter_mask 


# for test
if __name__ == '__main__':
    
    m1 = torch.randn(256, 7, 7) 
    m2 = torch.randn(256, 7, 7) 

    x, y, w, h = 10, 15, 240, 198

    size = 7
    matrix1 = torch.zeros((size, size, 2))
    matrix1[:, :, 1] = torch.stack([torch.linspace(x, x+w, size)]*size, 0)
    matrix1[:, :, 0] = torch.stack([torch.linspace(y, y+h, size)]*size, 1)
    
    #matrix1[:, :, 0] = torch.stack([torch.linspace(x, x+w, size)]*size, 1)
    #matrix1[:, :, 1] = torch.stack([torch.linspace(y, y+h, size)]*size, 0)
    
    
    x, y, w, h = 26, 5, 210, 231
    matrix2 = torch.zeros((7, 7, 2))
    matrix2[:, :, 1] = torch.stack([torch.linspace(x, x+w, size)]*size, 1)
    matrix2[:, :, 0] = torch.stack([torch.linspace(y, y+h, size)]*size, 0)
   
    irect = [26, 15, 236, 213]

    pixcontrast_loss = PixContrastLoss('test')
    
    pc_loss = pixcontrast_loss(matrix1, matrix2, m1, m2, irect)

    print(matrix1)

