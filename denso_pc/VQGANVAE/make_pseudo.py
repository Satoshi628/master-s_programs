import os
import math

import glob
import numpy as np
import PIL

import torch
import torchvision.transforms as T

from dalle_pytorch import VQGanVAE


def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

class Maker():
    def __init__(self, device="cuda:0"):
        self.vae = VQGanVAE(vqgan_model_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/last.ckpt", vqgan_config_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/model.yaml").to(device)
        
        self.resize_to_896 = T.Resize([896,896])
        self.resize_to_900 = T.Resize([900,900])
        self.resize_to_900_mask = torch.nn.Upsample(size=[900,900])

        self.toImage = T.ToPILImage(mode='RGB')
        self.tomaskImage = T.ToPILImage(mode='L')
        
        self.toTensor = T.ToTensor()
        
        self.device = device

    def pre_process(self, path):
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        img = self.resize_to_896(img)
        return self.toTensor(img)
    
    def post_process(self, img):
        img = self.toImage(img)
        img = self.resize_to_900(img)
        return img

    def mask_post_process(self, mask):
        mask = self.resize_to_900_mask(mask)[0,0]
        mask = (mask != 0)*1.0
        
        mask = self.tomaskImage(mask)
        return mask

    def get_mask(self):
        size = 112
        perlin_scalex = 4
        perlin_scaley = 4
        noize = rand_perlin_2d_np([size, size], [perlin_scalex, perlin_scaley])
        noize = noize > 0.5
        return noize.reshape(-1)

    def __call__(self, img_path, anomaly_path):
        img = self.pre_process(img_path).to(self.device)
        anomaly_img = self.pre_process(anomaly_path).to(self.device)
        
        image_idx = self.vae.get_codebook_indices(img[None])
        anomaly_idx = self.vae.get_codebook_indices(anomaly_img[None])
        mask = self.get_mask()
        
        image_idx[:, mask] = anomaly_idx[:, mask]

        img = self.vae.decode(image_idx)
        img = self.post_process(img[0])

        mask = 1.*(mask).reshape(1, 1, 112, 112)
        mask = torch.from_numpy(mask)
        mask = self.mask_post_process(mask)
        return img, mask


pseudo_maker = Maker()

# img_path = "/mnt/kamiya/dataset/MVtec_AD/leather/train/good/005.png"
# anomaly_img_path = '/mnt/kamiya/dataset/DTD/images/crosshatched/crosshatched_0081.jpg'

# img, mask = pseudo_maker(img_path, anomaly_img_path)
# img.save("a.png")
# mask.save("b.png")

dataset_path = "/mnt/kamiya/dataset/MVtec_AD-N2A"
anomaly_paths = [
                '/mnt/kamiya/dataset/DTD/images/crosshatched/crosshatched_0081.jpg',
                '/mnt/kamiya/dataset/DTD/images/crosshatched/crosshatched_0081.jpg',
                '/mnt/kamiya/dataset/DTD/images/crosshatched/crosshatched_0081.jpg',
                '/mnt/kamiya/dataset/DTD/images/crosshatched/crosshatched_0081.jpg',
                "/mnt/kamiya/dataset/DTD/images/bumpy/bumpy_0094.jpg",
                "/mnt/kamiya/dataset/DTD/images/bumpy/bumpy_0094.jpg",
                "/mnt/kamiya/dataset/DTD/images/bumpy/bumpy_0094.jpg",
                "/mnt/kamiya/dataset/DTD/images/bumpy/bumpy_0094.jpg",
                "/mnt/kamiya/dataset/DTD/images/cracked/cracked_0047.jpg",
                "/mnt/kamiya/dataset/DTD/images/cracked/cracked_0047.jpg",
                "/mnt/kamiya/dataset/DTD/images/cracked/cracked_0047.jpg",
                "/mnt/kamiya/dataset/DTD/images/cracked/cracked_0047.jpg",
                "/mnt/kamiya/dataset/DTD/images/knitted/knitted_0101.jpg",
                "/mnt/kamiya/dataset/DTD/images/knitted/knitted_0101.jpg",
                "/mnt/kamiya/dataset/DTD/images/knitted/knitted_0101.jpg",
                "/mnt/kamiya/dataset/DTD/images/knitted/knitted_0101.jpg",
                "/mnt/kamiya/dataset/DTD/images/scaly/scaly_0127.jpg",
                "/mnt/kamiya/dataset/DTD/images/scaly/scaly_0127.jpg",
                "/mnt/kamiya/dataset/DTD/images/scaly/scaly_0127.jpg",
                "/mnt/kamiya/dataset/DTD/images/scaly/scaly_0127.jpg",
                ]

category_paths = sorted(glob.glob(os.path.join(dataset_path, "*")))

for cate_path in category_paths:
    print(cate_path)
    image_path = sorted(glob.glob(os.path.join(cate_path, "train", "good", "*")))
    train_pseudo = image_path[:20]
    test_pseudo = image_path[20:40]
    
    makedirs(os.path.join(cate_path, "ground_truth_train"))
    makedirs(os.path.join(cate_path, "ground_truth", "pseudo_anomaly"))
    makedirs(os.path.join(cate_path, "test", "pseudo_anomaly"))
    
    # trainの疑似異常生成
    for idx, (img_path, anomaly_path) in enumerate(zip(train_pseudo, anomaly_paths)):
        img, mask = pseudo_maker(img_path, anomaly_path)

        img.save(os.path.join(cate_path, "train", "good", f"pseudo_{idx:04}.png"))
        mask.save(os.path.join(cate_path, "ground_truth_train", f"pseudo_{idx:04}.png"))

    # testの疑似異常生成
    for idx, (img_path, anomaly_path) in enumerate(zip(test_pseudo, anomaly_paths)):
        img, mask = pseudo_maker(img_path, anomaly_path)

        img.save(os.path.join(cate_path, "test", "pseudo_anomaly", f"pseudo_{idx:04}.png"))
        mask.save(os.path.join(cate_path, "ground_truth", "pseudo_anomaly", f"{idx:04}_mask.png"))