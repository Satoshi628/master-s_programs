import os
import glob
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class TrainLoader:
    def __init__(self, classname, resize, crop, transform=None):
        self.classname = classname
        self.transform = transform
        self.img_paths = sorted(glob.glob(os.path.join("mvtec", classname, "train/good", "*.png")))
        self.totensor = transforms.Compose([
                                            transforms.ToTensor()
                                            ])
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.resize = resize
        self.crop = crop

    def crop_center(self, img, crop_width, crop_height):
        w, h = img.size
        return img.crop(((
                        (w-crop_width)//2,
                        (h-crop_height)//2,
                        (w+crop_width)//2,
                        (h+crop_height)//2,
                        )))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")

        img = img.resize((self.resize, self.resize))
        img = self.crop_center(img, self.crop, self.crop)
    
        if self.transform is not None:
            img = self.transform(img)

        img = self.totensor(img)
        img = self.norm(img)
        return img, img_path

    def __len__(self):
        return len(self.img_paths)

class TestLoader:
    def __init__(self, classname, resize, crop, transform=None):
        self.classname = classname
        self.transform = transform
        self.img_paths = sorted(glob.glob(os.path.join("mvtec", classname, "test/*", "*.png")))
        self.totensor = transforms.Compose([
                                            transforms.ToTensor()
                                            ])
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.resize = resize
        self.crop = crop

    def crop_center(self, img, crop_width, crop_height):
        w, h = img.size
        return img.crop(((
                        (w-crop_width)//2,
                        (h-crop_height)//2,
                        (w+crop_width)//2,
                        (h+crop_height)//2,
                        )))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")

        img = img.resize((self.resize, self.resize))
        img = self.crop_center(img, self.crop, self.crop)
    
        if self.transform is not None:
            img = self.transform(img)

        img = self.totensor(img)
        img = self.norm(img)
        return img, img_path

    def __len__(self):
        return len(self.img_paths)