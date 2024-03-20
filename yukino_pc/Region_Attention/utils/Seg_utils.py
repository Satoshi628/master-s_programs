#coding: utf-8
#----- 標準ライブラリ -----#
import numbers
import random

#----- 専用ライブラリ -----#
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

#----- 自作モジュール -----#
#None


# to Tensor
class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, img, seg):
        # mutable https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        img = torch.from_numpy(img) / 255.
        seg = torch.from_numpy(seg)
        return img, seg


# compose
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg):
        for t in self.transforms:
            img, seg = t(img, seg)
        return img, seg

# random horizontal flip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(seg)
        return img, seg

#random vertical flip
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(seg)
        return img, seg

# pad
class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, seg):
        h, w = img.size
        ph = (h // 32 + 1) * 32 - h if h % 32 != 0 else 0
        pw = (w // 32 + 1) * 32 - w if w % 32 != 0 else 0
        im = F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        seg = F.pad(seg, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return im, seg


# Normalize
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        return F.normalize(img, self.mean, self.std), seg

# random crop
class RandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False):
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

    def __call__(self, img, seg):
        assert img.shape[-2:] == seg.shape[-2:], f'size of img and seg should be the same. {img.shape[-2:]}, {seg.shape[-2:]}'
        if self.padding > 0:
            img = F.pad(img, self.padding)
            seg = F.pad(seg, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.shape[-1] < self.shape[-2]:
            img = F.pad(img, padding=int((1 + self.shape[-2] - img.shape[-1]) / 2))
            seg = F.pad(seg, padding=int((1 + self.shape[-2] - seg.shape[-1]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.shape[-2] < self.shape[-1]:
            img = F.pad(img, padding=int((1 + self.shape[-1] - img.shape[-2]) / 2))
            seg = F.pad(seg, padding=int((1 + self.shape[-1] - seg.shape[-2]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(seg, i, j, h, w)

# resize
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, seg):
        return F.resize(img, self.size), F.resize(seg, self.size, Image.NEAREST)

