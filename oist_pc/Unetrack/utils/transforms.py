#coding: utf-8
#----- Standard Library -----#
import random 
import numbers

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

    def __call__(self, img, label ,point):
        img, label = self.ToTensor(img, label)
        for t in self.transforms:
            img, label, point = t(img, label, point)
        return img, label, point


# random horizontal flip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label, point):
        if random.random() < self.p:
            #x flip img=>[2,H,W]
            center = (img.shape[2] - 1) / 2
            point[:, :, 2] = 2 * center - point[:, :, 2]
            
            point[point[:, :, 2] >= img.shape[2]] = PAD_ID
            return F.hflip(img), F.hflip(label), point
        return img, label, point

#random vertical flip
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label, point):
        if random.random() < self.p:
            #y flip img=>[2,H,W]
            center = (img.shape[1] - 1) / 2
            point[:, :, 3] = 2 * center - point[:, :, 3]
            
            point[point[:, :, 3] >= img.shape[1]] = PAD_ID
            return F.vflip(img), F.vflip(label), point
        return img, label, point

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
        h, w = img.shape[1:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label, point):

        assert img.shape[1:] == label.shape[1:], 'size of img and label should be the same. %s, %s' % (
            img.shape[1:], label.shape[1:])
        if self.padding > 0:
            img = F.pad(img, self.padding)
            label = F.pad(label, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.shape[2] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.shape[2]) / 2))
            label = F.pad(label, padding=int((1 + self.size[1] - label.shape[2]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.shape[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.shape[1]) / 2))
            label = F.pad(label, padding=int((1 + self.size[0] - label.shape[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)
        #x crop
        point[:, :, 2] -= j
        point[(point[:, :, 2] < 0) | (point[:, :, 2] >= w)] = PAD_ID
        #y crpo
        point[:, :, 3] -= i
        point[(point[:, :, 3] < 0) | (point[:, :, 3] >= h)] = PAD_ID

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w), point
