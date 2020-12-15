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

        base = base.view(base.shape[0], base.shape[1], -1, 1)
        moment = moment.view(moment.shape[0], moment.shape[1], 1, -1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(base, moment)
        
        A_matrix = A_matrix.type(torch.BoolTensor).cuda()
        return cos_sim.masked_select(A_matrix).mean()

# for test
if __name__ == '__main__':
    base = torch.randn((512, 256, 7, 7))
    moment = torch.randn((512, 256, 7, 7))
    
    p = torch.zeros((3, 3, 2))
    print(p)



