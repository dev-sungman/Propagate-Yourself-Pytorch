import torch
import torch.nn as nn
import numpy as np

import time

class PixproLoss(nn.Module):
    def __init__(self, args):
        super(PixproLoss, self).__init__()
        self.args = args

    def forward(self, base, moment, p_base, p_moment, f_base, f_moment):
        """
        base      : base feature map    --> (B, C, 7,7)
        moment    : moment feature map  --> (B, C, 7,7)
        p_base    : base position       --> (B, (x, y, w, h))
        p_moment  : moment position     --> (B, (x, y, w, h))
        f_base    : is base fliped ?    --> (B, 1)
        f_moment  : is moment fliped?   --> (B, 1)
        """
        
        assert base.shape == moment.shape, 'base, moment shape must be same' 
        feature_map_size = (base.shape[2], base.shape[3])
        
        # compute loss 
        overall_loss = 0
        iter_ = 0
        for batch in range(base.shape[0]):
            base_batch = base[batch, :, :, :]
            moment_batch = moment[batch, :, :, :]

            p_base_batch = p_base[batch]
            p_moment_batch = p_moment[batch]
            f_base_batch = f_base[batch]
            f_moment_batch = f_moment[batch]
            
            base_matrix = self._warp_affine(p_base_batch)     #position matrix
            moment_matrix = self._warp_affine(p_moment_batch) #position matrix
            inter_rect = self._get_intersection_rect(p_base_batch, p_moment_batch)
            
            if inter_rect is not None:
                if f_base_batch.item() is True:
                    base_matrix = torch.fliplr(base_matrix)
                if f_moment_batch.item() is True:
                    moment_matrix = torch.fliplr(moment_matrix)
                 
                base_A_matrix = self._get_A_matrix(base_matrix, moment_matrix, p_base_batch) 
                moment_A_matrix = self._get_A_matrix(moment_matrix, base_matrix, p_moment_batch)
                

                base_pixpro_loss = self._get_pixpro_loss(base_batch, moment_batch, base_A_matrix)
                moment_pixpro_loss = self._get_pixpro_loss(moment_batch, base_batch, moment_A_matrix)

                overall_loss += (-base_pixpro_loss-moment_pixpro_loss)
                iter_ += 1
        
        return overall_loss / iter_
    
    def _warp_affine(self, p, size=7):
        """
        To get warped matrix
        p : feature map (base)
        size : cropped position in original image space (base)
        """
        x, y, w, h = p
        
        matrix = torch.zeros((size, size, 2)).cuda()
        matrix[:, :, 0] = torch.stack([torch.linspace(x, x+w, size)]*size, 1)
        matrix[:, :, 1] = torch.stack([torch.linspace(y, y+h, size)]*size, 0)
        return matrix
    
    def _get_intersection_rect(self, p1, p2):
        x1, y1, w1, h1 = p1
        x2, y2, w2, h2 = p2
        
        has_intersection = (abs((x1 + w1/2) - (x2 + w2/2)) * 2 < (w1 + w2)) and (abs((y1 + h1/2) - (y2 + h2/2))*2 < (h1 + h2))
        
        if has_intersection:
            xA = max(x1, x2)
            yA = max(y1, y2)
            xB = min(x1+w1, x2+w2)
            yB = min(y1+h1, y2+h2)
            return min(xA, xB), min(yA, yB), max(xA, xB), max(yA, yB)
        else:
            return None
         
    def _get_A_matrix(self, base, moment, point):
        x1, y1, w1, h1 = point
        
        diag_len = torch.sqrt((w1.float()**2) + (h1.float()**2))
        
        A_matrix = self._get_normalized_distance(base, moment, diag_len)
        return A_matrix
    
    def _get_normalized_distance(self, base, moment, diaglen):
        size = base.shape[0]*base.shape[1]

        base_x_matrix = base[:,:,1]
        base_y_matrix = base[:,:,0]
        
        moment_x_matrix = moment[:,:,1]
        moment_y_matrix = moment[:,:,0]

        dist_x_matrix = torch.mm(base_x_matrix.view(-1,1), torch.ones((1,size)).cuda()) - torch.mm(torch.ones((size,1)).cuda(), moment_x_matrix.view(1,-1))
        dist_y_matrix = torch.mm(base_y_matrix.view(-1,1), torch.ones((1,size)).cuda()) - torch.mm(torch.ones((size,1)).cuda(), moment_y_matrix.view(1,-1))
        
        dist_matrix = torch.sqrt(dist_x_matrix**2 + dist_y_matrix**2) / diaglen
        A_matrix = torch.zeros((dist_matrix.shape)).cuda()
        A_matrix[dist_matrix < self.args.threshold] = 1.
        A_matrix[dist_matrix >= self.args.threshold] = 0.
        
        return A_matrix
    
    def _get_pixpro_loss(self, base, moment, A):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(base.view(-1,49), moment.view(-1,49))
        cos_sim = cos_sim * A
        
        pixpro_loss = torch.sum(cos_sim) / torch.count_nonzero(A)
        return pixpro_loss


if __name__ == '__main__':
    base = torch.randn((512, 256, 7, 7))
    moment = torch.randn((512, 256, 7, 7))
    
    p = torch.zeros((3, 3, 2))
    print(p)



