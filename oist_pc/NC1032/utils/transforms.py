#coding: utf-8
#----- Standard Library -----#
import random 
import numbers
import math

#----- Public Package -----#
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

#----- Module -----#
#None

PAD_ID = -1

# to Tensor
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        tensor = torch.from_numpy(image)
        tensor = tensor.float()

        label = torch.from_numpy(label)

        return tensor, label

# compose
class Compose(object):
    def __init__(self, transforms):
        self.ToTensor = ToTensor()
        self.transforms = transforms

    def __call__(self, img, label):
        img, label = self.ToTensor(img, label)
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


# random horizontal flip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label

#random vertical flip
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label

# random crop
class RandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False):
        """Randam Crop

        Args:
            size (list): Crop size[H,W]
            padding (int, optional): padding size. Defaults to 0.
            pad_if_needed (bool, optional): Are you Need padding? True or False. Defaults to False.
        """        
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):

        assert img.shape[-2:] == label.shape[-2:], 'size of img and label should be the same. %s, %s' % (
            img.shape[-2:], label.shape[-2:])
        if self.padding > 0:
            img = F.pad(img, self.padding)
            label = F.pad(label, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.shape[-1] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.shape[-1]) / 2))
            label = F.pad(label, padding=int((1 + self.size[1] - label.shape[2]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.shape[-2] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.shape[-2]) / 2))
            label = F.pad(label, padding=int((1 + self.size[0] - label.shape[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)


# pad
class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbl):
        h, w = img.shape[-2:]
        ph = math.ceil(h / self.diviser) * self.diviser
        pw = math.ceil(w / self.diviser) * self.diviser
        ph = (ph - h)
        pw = (pw - w)
        
        im = F.pad(img, (0, pw, 0, ph))
        lbl = F.pad(lbl, (0, pw, 0, ph))
        return im, lbl