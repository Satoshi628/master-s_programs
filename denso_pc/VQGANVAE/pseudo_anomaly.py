import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from perlin import rand_perlin_2d_np
from dalle_pytorch import VQGanVAE



class Pseudo_Maker(nn.Module):
    def __init__(self, input_size, model_path=None, config_path=None):
        super().__init__()
        self.vae = VQGanVAE(vqgan_model_path=model_path, vqgan_config_path=config_path)
        for param in self.vae.parameters():
            param.requires_grad = False

        #evalモードにしなくてもおｋ
        # self.vae.eval()
        self.resize = nn.Upsample(input_size, mode='bilinear')

    def forward(self, image, anomaly=None, masks=None):
        if anomaly is None or masks is None:
            image_idx = self.vae.get_codebook_indices(image)
            recon_image = self.vae.decode(image_idx)
            
            image = self.resize(image)
            recon_image = self.resize(recon_image)
            return image, recon_image

        image_idx = self.vae.get_codebook_indices(image)
        anomaly_idx = self.vae.get_codebook_indices(anomaly)

        mask = masks.reshape([image.shape[0], -1])
        image_idx[mask] = anomaly_idx[mask]
        anomaly_img = self.vae.decode(image_idx)
        
        image = self.resize(image)
        anomaly_img = self.resize(anomaly_img)
        #UpSamplingによって0or1になっていないことに注意
        masks = self.resize(masks*1.)
        return image, anomaly_img, masks
