import torch
import torch.nn as nn
import numpy as np

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

            # warped to the original image space
            base_position_matrix, moment_position_matrix = self._get_feature_position_matrix(p_base_batch, p_moment_batch, feature_map_size)
            inter_rect = self._get_intersection_rect(p_base_batch, p_moment_batch)
            
            base_inter_mask = self._get_intersection_features(base_position_matrix, inter_rect, feature_map_size)
            moment_inter_mask = self._get_intersection_features(moment_position_matrix, inter_rect, feature_map_size)
            
            # befor compute A matrix, check the filp flag
            if base_inter_mask is not None and moment_inter_mask is not None:
                if f_base_batch.item() is True:
                    base_position_matrix = np.fliplr(base_position_matrix)
                if f_moment_batch.item() is True:
                    moment_position_matrix = np.fliplr(moment_position_matrix)
            
                # get A matrix
                base_A_matrix = self._get_A_matrix(base_position_matrix, moment_position_matrix, p_base_batch)
                moment_A_matrix = self._get_A_matrix(moment_position_matrix, base_position_matrix, p_moment_batch)
            
                base_loss = self._compute_pixpro_loss(base_batch, moment_batch, base_A_matrix, moment_A_matrix)
                moment_loss = self._compute_pixpro_loss(moment_batch, base_batch, moment_A_matrix, base_A_matrix)
            
                #self._compute_pix_contrast_loss(base_inter_mask, base, moment, base_A_matrix)

                overall_loss += (-base_loss-moment_loss)
                iter_ += 1

        return overall_loss / iter_

    def _compute_cosine_similarity(self, m1, m2):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(m1, m2)
        return cos_sim


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
        
        base_matrix = np.array(base_matrix).reshape((size[0], size[1], 2))
        moment_matrix = np.array(moment_matrix).reshape((size[0], size[1], 2))

        return base_matrix, moment_matrix

    def _get_A_matrix(self, base_pm, moment_pm, p_base):
        x1, y1, w1, h1 = p_base
        
        base_diag_len = torch.sqrt((w1.float()**2) + (h1.float()**2))
        
        base_A_matrix = self._get_normalized_distance(base_pm, moment_pm, base_diag_len)

        return base_A_matrix

    def _get_normalized_distance(self, base, moment, diag_len):
        """
        To get normalized distance between feature maps
        base : feature map poistion 
        moment : feature map position
        diag_len : l1's diagonal length for normalization.
        """
        

        A_matrix = np.zeros((base.shape[0],base.shape[1],moment.shape[0],moment.shape[1]))
        
        for i in range(base.shape[0]):
            for j in range(base.shape[1]):
                for k in range(moment.shape[0]):
                    for l in range(moment.shape[1]):
                        dist = torch.sqrt(((base[i,j,0]-moment[k,l,0])**2) + ((base[i,j,1]-moment[k,l,1])**2)) / diag_len
                        mask = 1 if dist < self.args.threshold else 0
                        A_matrix[i,j,k,l] = mask
        A_matrix = A_matrix.reshape(base.shape[0], base.shape[1], moment.shape[0], moment.shape[1])
        return A_matrix        
    
    def _get_intersection_rect(self, p_base, p_moment):
        """
        To get intersection area 
        p_base : feature map poistion list (base)
        p_moment : feature map position list (moment)
        """
        has_intersection = True

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
        
        if len(x_list) > 0 and len(y_list) > 0:
            intersection_box = (min(x_list), max(x_list), min(y_list), max(y_list))
        else:
            intersection_box = None
        return intersection_mask 
    
    def _compute_pixpro_loss(self, base_feature, moment_feature, base_A_matrix, moment_A_matrix):
        cos_sim = 0
        for i in range(base_feature.shape[1]):
            for j in range(base_feature.shape[2]):
                y_i = base_feature[:, i, j]

                for k in range(moment_feature.shape[1]):
                    for l in range(moment_feature.shape[2]):
                        x_j = moment_feature[:, k, l]

                        if base_A_matrix[i, j, k, l] == 1:
                            cos_sim += self._compute_cosine_similarity(y_i, x_j)


        return cos_sim / (base_feature.shape[1] * base_feature.shape[2] * moment_feature.shape[1] * moment_feature.shape[2])


if __name__ == '__main__':
    base = torch.randn((7, 7))
    moment = torch.randn((7,7))

    base_col_matrix = torch.ones((base.shape))
    moment_row_matrix = torch.ones(moment.shape))
    
   



