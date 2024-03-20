##### ライブラリ読み込み #####
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image
############################

# to Tensor
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic1, pic2, label):
        tensor1 = torch.from_numpy(pic1)
        tensor2 = torch.from_numpy(pic2)
        label = torch.from_numpy(label)
        #float32モードに変換
        tensor1 = tensor1.float()
        tensor2 = tensor2.float()
        
        tensor = torch.cat([tensor1, tensor2], dim=0)
        return tensor, label

# compose
class Compose(object):
    def __init__(self, transforms):
        self.ToTensor = ToTensor()
        self.transforms = transforms

    def __call__(self, img1, img2, label):
        img, label = self.ToTensor(img1, img2, label)
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

# center crop
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        return F.center_crop(img, self.size), F.center_crop(label, self.size)

# random scale
class RandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, label):
        assert img.shape[1:] == label.shape[1:]
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.shape[1]*scale), int(img.shape[2]*scale))
        return F.resize(img, target_size, self.interpolation), F.resize(label, target_size, Image.NEAREST)

# scale
class Scale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, label):
        assert img.shape[1:] == label.shape[1:]
        target_size = (int(img.shape[1]*self.scale),
                        int(img.shape[2]*self.scale))  # (H, W)
        return F.resize(img, target_size, self.interpolation), F.resize(label, target_size, Image.NEAREST)

# random rotation
class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, label):
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(label, angle, self.resample, self.expand, self.center)

# random horizontal flip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            #x軸の移動ベクトルを逆転
            label[0] = -label[0]
            return F.hflip(img), F.hflip(label)
        return img, label

#random vertical flip
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            #y軸の移動ベクトルを逆転
            label[1] = -label[1]
            return F.vflip(img), F.vflip(label)
        return img, label

# pad
class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser

    def __call__(self, img, label):
        h, w = img.shape[1:]
        ph = (h//32+1)*32 - h if h % 32 != 0 else 0
        pw = (w//32+1)*32 - w if w % 32 != 0 else 0
        im = F.pad(img, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        label = F.pad(label, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        return im, label


# Normalize
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, label):
        return F.normalize(tensor, self.mean, self.std), label

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

    def __call__(self, img, label):

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

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)

# resize


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, label):
        return F.resize(img, self.size, self.interpolation), F.resize(label, self.size, Image.NEAREST)

# color jitter


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                    clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(
                    "{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, label):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), label
'''
class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
'''

