import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from CutpasteAugment import cut_patch, paste_patch

from dalle_pytorch import VQGanVAE


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, mvtec_root, dtd_root, category, input_size,
                Faster=False,
                noise_generator=None,
                is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        
        self.input_size = input_size
        self.resize = transforms.Resize(input_size, antialias=True)
        self.recon_resize = [512,512] #VQGANVAEの再構成は高解像度ほど精度高い

        if is_train:
            self.image_files = glob.glob(os.path.join(mvtec_root, category, "train", "good", "*.png"))
        else:
            self.image_files = glob.glob(os.path.join(mvtec_root, category, "test", "*", "*.png"))

        if Faster:
            print("image loding...")
            self.images = [self.encoding(path) for path in tqdm(self.image_files)]

        self.Faster = Faster
        self.noise_generator = noise_generator
        self.is_train = is_train

        
        self.anomaly_source_paths = sorted(glob.glob(dtd_root+"/*/*.jpg"))

        self.anomaly_augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]


    def encoding(self, image_path, use_aug=False):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.recon_resize[1], self.recon_resize[0]))
        if use_aug:
            augs = np.random.choice(self.anomaly_augmenters, 3, replace=False)
            augs = iaa.Sequential(augs)
            image = augs(image=image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32)/255.).permute(2,0,1)
        return image
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        if self.Faster:
            image = self.images[index].detach()
        else:
            image = self.encoding(image_file)

        # make pseudo anomaly
        if self.is_train:
            anomaly_source_file = np.random.choice(self.anomaly_source_paths, 1)[0]
            anomaly_source_img = self.encoding(anomaly_source_file, use_aug=True)
            mask = self.noise_generator((image.shape[-2]//8, image.shape[-1]//8))

            return {"image":image, "anomaly":anomaly_source_img, "mask":mask}
        else:
            if os.path.dirname(image_file).endswith("good"):
                mask = torch.zeros([1, self.input_size[-2], self.input_size[-1]])
            else:
                mask = cv2.imread(image_file.replace("/test/", "/ground_truth/").replace(".png", "_mask.png"), cv2.IMREAD_GRAYSCALE)
                mask = self.resize(torch.from_numpy(mask[None].astype(np.float32))) / 255.
                mask = torch.clamp(mask, 0., 1.)
            return {"image":image, "mask":mask}

    def __len__(self):
        return len(self.image_files)


class MVTecDataset_Classification(torch.utils.data.Dataset):
    def __init__(self, mvtec_root, dtd_root, category, input_size,
                noise_generator=None,
                use_mask=False,
                bg_threshold=100,
                bg_reverse=False,
                is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        
        self.input_size = input_size
        self.resize = transforms.Resize(input_size, antialias=True)
        self.recon_resize = [512,512] #VQGANVAEの再構成は高解像度ほど精度高い
        self.use_mask = use_mask
        self.bg_threshold = bg_threshold
        self.bg_reverse = bg_reverse

        if is_train:
            self.image_files = glob.glob(os.path.join(mvtec_root, category, "train", "good", "*.png"))
        else:
            self.image_files = glob.glob(os.path.join(mvtec_root, category, "test", "*", "*.png"))

        self.noise_generator = noise_generator
        self.is_train = is_train

        
        self.anomaly_source_paths = sorted(glob.glob(dtd_root+"/*/*.jpg"))

        self.anomaly_augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]


    def encoding(self, image_path, use_aug=False, use_mask=False):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.recon_resize[1], self.recon_resize[0]))
        if use_mask:
            background_mask = cv2.resize(image, dsize=(self.recon_resize[1]//8, self.recon_resize[0]//8))
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2GRAY)
            _, background_mask = cv2.threshold(background_mask, self.bg_threshold, 255, cv2.THRESH_BINARY)
            background_mask = background_mask.astype(np.bool)
            #reverse= Trueならそのままらしい
            if self.bg_reverse:
                background_mask = background_mask
            else:
                background_mask = ~background_mask
            background_mask = torch.from_numpy(background_mask)[None]
        else:
            background_mask = torch.zeros((1, self.recon_resize[1]//8, self.recon_resize[0]//8), dtype=bool)
        
        if use_aug:
            augs = np.random.choice(self.anomaly_augmenters, 3, replace=False)
            augs = iaa.Sequential(augs)
            image = augs(image=image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32)/255.).permute(2,0,1)
        return image, background_mask
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image, background_mask = self.encoding(image_file, use_mask=self.use_mask)

        # make pseudo anomaly
        if self.is_train:
            anomaly_source_file = np.random.choice(self.anomaly_source_paths, 1)[0]
            anomaly_source_img, _ = self.encoding(anomaly_source_file, use_aug=True)
            mask = self.noise_generator((image.shape[-2]//8, image.shape[-1]//8))
            mask[~background_mask] = False

            return {"image":image, "anomaly":anomaly_source_img, "mask":mask}
        else:
            if os.path.dirname(image_file).endswith("good"):
                mask = torch.zeros([1, self.input_size[-2], self.input_size[-1]])
            else:
                mask = cv2.imread(image_file.replace("/test/", "/ground_truth/").replace(".png", "_mask.png"), cv2.IMREAD_GRAYSCALE)
                mask = self.resize(torch.from_numpy(mask[None].astype(np.float32))) / 255.
                mask = torch.clamp(mask, 0., 1.)
            return {"image":image, "mask":mask}

    def __len__(self):
        return len(self.image_files)

