import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageFilter

class PixProDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

class BaseTransform(ABC):
    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    @abstractmethod
    def transform(self, img):
        pass

class Solarize(BaseTransform):
    def transform(self, img):
        thres = (1-self.mag) * 255
        return ImageOps.solarize(img, thres)

class GaussianBlur(BaseTransform):
    def transform(self, img):
        kernel_size = self.mag
        gBlur = ImageFilter.GaussianBlur(kernel_size)
        return img.filter(gBlur)
