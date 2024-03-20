#coding: utf-8
#----- Standard Library -----#
import random
import numbers

#----- Public Package -----#
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

#----- Module -----#
#None

PAD_ID = -100

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
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


# Normalize
class Normalize(object):
    def __init__(self):
        self.norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    def __call__(self, img, label):
        #C,L,H,W => L, C, H, W
        img = self.norm(img.transpose(0,1)).transpose(0,1)
        return img, label

# random horizontal flip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            #x flip img=>[2,H,W]
            return F.hflip(img), F.hflip(label)
        return img, label

#random vertical flip
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            #y flip img=>[2,H,W]
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


# random crop
class RandomCrop_mask(object):
    def __init__(self, size, p=0.8, padding=0, pad_if_needed=False):
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
        self.p = p
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
        
        if random.random() < self.p:
            for _ in range(10):
                i, j, h, w = self.get_params(img, self.size)
                if F.crop(label, i, j, h, w).sum() != 0:
                    break
        else:
            i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)

# random crop
class RandomResize(object):
    def __init__(self, scale):
        """Randam Crop

        Args:
            scale (list): Resize scale[min,max]
        """        
        if isinstance(scale, numbers.Number):
            self.scale = (float(scale), float(scale))
        else:
            self.scale = scale

    def __call__(self, img, label):
        h, w = img.shape[-2:]

        H = random.randint(int(h*self.scale[0]), int(h*self.scale[1]))
        W = random.randint(int(w*self.scale[0]), int(w*self.scale[1]))
        resize_size = [H, W]
        return F.resize(img, resize_size), F.resize(label[None], resize_size)[0]




# resize
class Resize(object):
    def __init__(self, size):
        """Resize

        Args:
            size (number): Resize size[W]
        """        
        self.size = size

    def __call__(self, img, label):
        return F.resize(img, [img.shape[1], self.size]), F.resize(label[None], [img.shape[1], self.size])[0]


# resize
class Padding_div(object):
    def __init__(self, feature_size=32):
        """Resize

        Args:
            size (number): Resize size[W]
        """        
        self.size = feature_size

    def __call__(self, img, label):
        H = (img.shape[1] % self.size)
        if H != 0:
            H = self.size - H
        W = (img.shape[2] % self.size)
        if W != 0:
            W = self.size - W
        return F.pad(img, padding=[0, 0, W, H]), F.pad(label[None], padding=[0, 0, W, H])[0]
