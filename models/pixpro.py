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

    def forward(self, x1, x2, p_base, p_moment, f_base, f_moment):
        """
        base      : base feature map    --> (7,7)
        moment    : moment feature map  --> (7,7)
        p_base    : base position       --> (x, y, w, h)
        p_moment  : moment position     --> (x, y, w, h)
        f_base    : is base fliped ?    --> True / False
        f_moment  : is moment fliped?   --> True / False
        """
        base = self.base_encoder(x1)
        y = self.ppm(base)

        moment = self.moment_encoder(x2)
        assert base.shape == moment.shape, 'base, moment shape must be same' 
        feature_map_size = (base.shape[2], base.shape[3])
        base_position_matrix, moment_position_matrix = self._get_feature_position_matrix(p_base, p_moment, feature_map_size)
        
        # postiion matrix 받아오는 부분을 numpy 로 변환하고 밑에 코드 수정해야함.
        # _get_A_matrix 에서는 문제 없음. 
        inter_rect = self._get_intersection_rect(p_base, p_moment)

        inter_mask_1 = self._get_intersection_features(base_position_matrix, inter_rect, feature_map_size)
        inter_mask_2 = self._get_intersection_features(moment_position_matrix, inter_rect, feature_map_size)
        
        # befor compute A matrix, check the filp flag

        base_A_matrix, moment_A_matrix = self._get_A_matrix(base_position_matrix, moment_position_matrix, p_base, p_moment)

        return base, moment, y
    
    #def _compute_pix_contrast_loss(self, inter_mask, A_matrix
    
    def _get_feature_position_matrix(self, p_base, p_moment, size):
        """
        To get base_matrix, moment_matrix position information, A_matrix for postivie, negative mask
        x1 : feature map (base)
        x2 : feature map (moment)
        p_base : cropped position in original image space (base)
        p_moment : cropped position in original image space (moment)
        """

        feature_size = size
        x1, y1, w1, h1 = p_base
        x2, y2, w2, h2 = p_moment
        
        # warp to original matrix
        rw1 = w1 / (feature_size[0]-1)
        rh1 = h1 / (feature_size[1]-1)
        
        rw2 = w2 / (feature_size[0]-1)
        rh2 = h2 / (feature_size[1]-1)

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
                nx1 = nx1 + rw1
                nx2 = nx2 + rw2
            ny1 = ny1 + rh1
            ny2 = ny2 + rh2
        

        return base_matrix, moment_matrix

    def _get_A_matrix(self, base_pm, moment_pm, p_base, p_moment):
        x1, y1, w1, h1 = p_base
        x2, y2, w2, h2 = p_moment
        
        base_diag_len = torch.sqrt((w1.float()**2) + (h1.float()**2))
        moment_diag_len = torch.sqrt((w2.float()**2) + (h2.float()**2))
        
        base_A_matrix = self._get_normalized_distance(base_pm, moment_pm, base_diag_len)
        moment_A_matrix = self._get_normalized_distance(base_pm, moment_pm, moment_diag_len)

        return base_A_matrix, moment_A_matrix

    def _get_normalized_distance(self, l1, l2, diag_len):
        """
        To get normalized distance between feature maps
        l1 : feature map poistion list (base)
        l2 : feature map position list (moment)
        diag_len : l1's diagonal length for normalization.
        """
        A_matrix = []
        for i in l1:
            for j in l2:
                dist = (torch.sqrt(((i[0]-j[0])**2) + ((i[1]-j[1])**2))) / diag_len
                mask = 1 if dist < self.threshold else 0
                A_matrix.append(mask)
        A_matrix = np.array(A_matrix).reshape((len(l1), len(l2)))
        return A_matrix        
    
    def _get_intersection_rect(self, p_base, p_moment):
        """
        To get intersection area 
        p_base : feature map poistion list (base)
        p_moment : feature map position list (moment)
        """
        x1, y1, w1, h1 = p_base
        x2, y2, w2, h2 = p_moment
        
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1+w1, x2+w2)
        yB = min(y1+h1, y2+h2)

        return min(xA, xB), min(yA, yB), max(xA, xB), max(yA, yB)
    
    def _get_intersection_features(self, x, area, size):
        """
        To get features in intersection
        x : feature map (position)
        area : intersection area (x1, y1, x2, y2)
        """
        x = np.array(x).reshape((size[0], size[1], 2))
        intersection_mask = torch.zeros((x.shape[0], x.shape[1]))
        
        x_list = []
        y_list = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                nx, ny = x[i,j,0], x[i,j,1]
                if nx >= area[0] and nx <= area[2] and ny >= area[1] and ny <= area[3]:
                    x_list.append(nx) # for box
                    y_list.append(ny) # for box
                    intersection_mask[i][j] = 1

        print(intersection_mask) 
        intersection_box = (min(x_list), max(x_list), min(y_list), max(y_list))
        print(intersection_box)
        return intersection_mask 


    @torch.no_grad()
    #TODO: update momeuntum gradually to 1
    def _momentum_update(self):
        for param_base, param_moment in zip(self.base_encoder.parameters(), self.moment_encoder.parameters()):
            param_moment.data = param_moment.data * self.m + param_base.data * (1. - self.m)
    
    ##### FOR DEBUGGING !!!
    def draw_for_debug(self, p1, p2, inter_box, img1, img2, feat1, feat2):
        feat1 = np.array(feat1).reshape((7, 7, 2))
        feat2 = np.array(feat2).reshape((7, 7, 2))
        import torchvision
        import cv2
        box1 = [p1[0], p1[1], p1[0]+p1[2], p1[1]+p1[3]]
        box2 = [p2[0], p2[1], p2[0]+p2[2], p2[1]+p2[3]]
        # box : x1, y1, x2, y2
        torchvision.utils.save_image(img1, 'img1.png')
        torchvision.utils.save_image(img2, 'img2.png')
        
        src = cv2.imread('imgs/0/0.jpeg')
        print_img = src.copy()
        print_img = cv2.rectangle(print_img, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 1)
        print_img = cv2.rectangle(print_img, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 1)
        print_img = cv2.rectangle(print_img, (inter_box[0], inter_box[1]), (inter_box[2], inter_box[3]), (0, 255, 255), 1)

        for i in range(feat1.shape[0]):
            for j in range(feat1.shape[1]):
                nx1, ny1 = feat1[i,j,0], feat1[i,j,1]
                nx2, ny2 = feat2[i,j,0], feat2[i,j,1]
                print_img = cv2.circle(print_img, (nx1,ny1), 2, (0, 255, 0), -1)
                print_img = cv2.circle(print_img, (nx2,ny2), 2, (0, 0, 255), -1)
        
        cv2.imwrite('debug.png', print_img)
        

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
