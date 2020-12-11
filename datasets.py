from torch.utils.data import Dataset
from transforms import *
import os
import cv2

class PixProDataset(Dataset):
    def __init__(self, root, data_size=(224,224)):
        self.root = root
        self.data_size = data_size
        
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx)
        self.targets = [s[1] for s in self.samples] 
        
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                p=0.6),
            GaussianBlur(prob=0.3, mag=3),
            Solarize(prob=0.3, mag=0.5),
            transforms.ToTensor()
        ])

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx):
        instances = []
        directory = os.path.expanduser(directory)
        
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

        return instances
    
    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self._load_image(path)

        sample1, x1, y1, w1, h1 = RandomResizedCrop(self.data_size)(sample)
        sample1, is_flip1 = RandomHorizontalFlip(p=0.5)(sample1)
        sample1 = self.transform(sample1)

        sample2, x2, y2, w2, h2 = RandomResizedCrop(self.data_size)(sample)
        sample2, is_flip2 = RandomHorizontalFlip(p=0.5)(sample2)
        sample2 = self.transform(sample2)
        
        targets = torch.FloatTensor(np.array([x1, y1, w1, h1, x2, y2, w2, h2, is_flip1, is_flip2]))

        return (sample1, sample2), targets
    

    def __len__(self):
        return len(self.samples)

#### For test
if __name__ == '__main__':
    dataset = PixProDataset(root='imgs')
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1)

    for (i1,i2), (p1,p2), (f1, f2) in dataloader:
        import torchvision        
        torchvision.utils.save_image(i1, 'sample1.png')
        torchvision.utils.save_image(i2, 'sample2.png')

        np_s1 = i1.cpu().detach().numpy()
        np_s2 = i2.cpu().detach().numpy()

        
        src_img = cv2.imread('/workspace/scripts/kakaobrain-homework/imgs/0/0.jpeg')
        print_img = cv2.rectangle(src_img, (p1[0], p1[1]), (p1[0]+p1[2], p1[1]+p1[3]), (0,255,0), 2)
        print_img = cv2.rectangle(print_img, (p2[0], p2[1]), (p2[0]+p2[2], p2[1]+p2[3]), (255, 0, 0), 2)

        cv2.imwrite('origin.png', print_img)


        raise

