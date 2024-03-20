import os
import numpy as np
import torch
from torch.utils.data import Dataset
import opensimplex as sx
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np


def through(mask):
    # 0% False
    return mask

def mesh(mask):
    # 25% Flase
    mask[::2,::2] = False
    return mask

def noise(mask):
    # 50% Flase
    noise = np.random.rand(mask.shape[0], mask.shape[1]) > 0.5
    mask[noise] = False
    return mask

def horizon_border(mask):
    # 50% Flase
    mask[::2] = False
    return mask

def vertical_border(mask):
    # 50% Flase
    mask[:,::2] = False
    return mask

def cross_check(mask):
    # 50% False
    y = np.arange(mask.shape[0])
    x = np.arange(mask.shape[1])
    mask = mask & ((y[:,None] + x[None]) % 2 == 1)
    return mask

class Perlin_Noise_Generater(Dataset):
    def __init__(self, noise_scale_min=0, noise_scale_max=6, th = 0.3,
                 texture=None):
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        self.th = th
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.texture = texture

    
    def __call__(self, size):
        # size[H,W]
        perlin_scalex = 2 ** np.random.randint(self.noise_scale_min, self.noise_scale_max, [1])[0]
        perlin_scaley = 2 ** np.random.randint(self.noise_scale_min, self.noise_scale_max, [1])[0]

        perlin_noise = rand_perlin_2d_np((size[1], size[0]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        perlin_thr = perlin_noise > self.th

        if self.texture is not None:
            func = np.random.choice(self.texture, 1)[0]
            perlin_thr = func(perlin_thr)
        
        perlin_thr = torch.from_numpy(perlin_thr)
        return perlin_thr[None]


class Simplex_Noise_Generater(Dataset):
    def __init__(self, noise_scale=[1, 7], noise_aspect=[0.3, 1/0.3], th = 0.7,
                 texture=None):
        self.noise_scale = noise_scale
        self.noise_aspect = noise_aspect
        self.th = th
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.texture = texture

    
    def __call__(self, size):
        # size[H,W]
        sx.random_seed()
        scale = (self.noise_scale[1] - self.noise_scale[0]) * np.random.rand(1)[0] + self.noise_scale[0]
        aspect = (self.noise_aspect[1] - self.noise_aspect[0]) * np.random.rand(1)[0] + self.noise_aspect[0]

        x = np.arange(size[0]) / (scale * aspect)
        y = np.arange(size[1]) / (scale / aspect)

        noize = sx.noise2array(x, y)
        noize = (noize + 1) / 2.
        noize = self.rot(image=noize)
        noize = noize > self.th


        if self.texture is not None:
            func = np.random.choice(self.texture, 1)[0]
            noize = func(noize)
        
        noize = torch.from_numpy(noize)
        return noize[None]



