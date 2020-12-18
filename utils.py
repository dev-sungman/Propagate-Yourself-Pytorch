class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


##### FOR VISUAL DEBUGGING !!!
import numpy as np
import torchvision
import cv2
def draw_for_debug(p1, p2, inter_box, img1, img2, feat1, feat2, path, bA, mA):
    feat1 = np.array(feat1).reshape((7, 7, 2))
    feat2 = np.array(feat2).reshape((7, 7, 2))
    import torchvision
    import cv2
    box1 = [p1[0], p1[1], p1[0]+p1[2], p1[1]+p1[3]]
    box2 = [p2[0], p2[1], p2[0]+p2[2], p2[1]+p2[3]]
    
    # box : x1, y1, x2, y2
    torchvision.utils.save_image(img1, 'img1.png')
    torchvision.utils.save_image(img2, 'img2.png')
    
    src = cv2.imread(path)
    print_img = src.copy()
    print_img = cv2.rectangle(print_img, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 1)
    print_img = cv2.rectangle(print_img, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 1)
    print_img = cv2.rectangle(print_img, (int(inter_box[0].item()), int(inter_box[1].item())), 
                                (int(inter_box[2].item()), int(inter_box[3].item())), (0, 255, 255), 1)

    for i in range(feat1.shape[0]):
        for j in range(feat1.shape[1]):
            nx1, ny1 = feat1[i,j,0], feat1[i,j,1]
            nx2, ny2 = feat2[i,j,0], feat2[i,j,1]
            print_img = cv2.circle(print_img, (nx1,ny1), 2, (0, 255, 0), -1)
            print_img = cv2.circle(print_img, (nx2,ny2), 2, (0, 0, 255), -1)
    
    bA = bA.view(7,7,7,7)
    mA = mA.view(7,7,7,7)
    iter_ = 0
    for i in range(bA.shape[0]):
        for j in range(bA.shape[1]):
            A_img = bA[i,j,:,:]
            A_img_cpu = A_img.cpu().detach().numpy()
            A_img_cpu = cv2.resize(A_img_cpu.astype('uint8'), (224,224))
            
            cv2.imwrite('{}.png'.format(iter_), A_img_cpu*255)
            iter_ += 1
    cv2.imwrite('debug.png', print_img)
