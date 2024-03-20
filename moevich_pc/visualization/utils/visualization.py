#coding: utf-8
#----- 標準ライブラリ -----#
import os

from PIL.Image import blend

#----- 専用ライブラリ -----#
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

#----- 自作モジュール -----#
#None

class Grad_CAM_3D():
    PVT = ["non PVT", "PVT"]
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model.to(device)
        self.target_layer = target_layer[0] # same pytorch_grad_cam list[torch.module]
        self.device = device
        self.hidden_out_list = []
        self.hidden_grad_list = []

        self.register_forword_hook(self.target_layer)

    def register_forword_hook(self,target_layer):
        self.handle = target_layer.register_forward_hook(self.hook)

    def hook(self, module, inputs, outputs):
        self.hidden_out_list.append(outputs.cpu().detach().clone())
        def save_grad(grad):
            self.hidden_grad_list.append(grad.cpu().detach().clone())
        outputs.register_hook(save_grad)

    def _reset(self):
        self.hidden_out_list = []
        self.hidden_grad_list = []
    
    def _reset_image_number(self):
        self.image_num = 0

    def delete_handle(self):
        self.handle.remove()

    def __call__(self,  inputs, targets):
        self._reset()
        self.model.eval()
        inputs = inputs.to(self.device)
        output = self.model(inputs).flatten()
        loss = output.sum() #binary classification
        
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        feature = self.hidden_out_list[0]
        alpha = self.hidden_grad_list[0].mean(dim=(-3, -2, -1))
        #重要度マップ作成
        alpha = F.relu(alpha)
        Imp_map = feature * alpha[:, :, None, None, None] #[batch, channel, T, H, W]
        Imp_map = Imp_map.sum(dim=1)  #[batch, T, H, W]
        
        #[0,1]になるようにmax-min正規化
        min_Imp = Imp_map.flatten(-3).min(dim=-1)[0]
        max_Imp = Imp_map.flatten(-3).max(dim=-1)[0]
        Imp_map = (Imp_map - min_Imp[:, None, None, None]) / (max_Imp[:, None, None, None] - min_Imp[:, None, None, None] + 1e-7)
        
        #Numpy化、inputsのサイズにリサイズ
        upsamp = nn.Upsample(inputs.shape[-3:], mode="trilinear", align_corners=True)
        #bilinearモードは4次元出なくてはならないため
        Imp_map = upsamp(Imp_map[:, None]).squeeze(1)
        Imp_map = Imp_map.cpu().detach().numpy()
        
        #Edit input image
        images = (inputs * 255).cpu().detach().numpy().transpose(0, 2, 3, 4, 1)
        images = np.tile(images, [1, 1, 1, 1, 3])

        #Edit targets
        targets = {k: v.cpu().detach().tolist() for k,v in targets.items()}

        #Edit predict
        predict = torch.round(torch.sigmoid(output)).to("cpu").detach().numpy().astype(np.int16)

        if not os.path.exists("CAM_image"):
            os.mkdir("CAM_image")

        alpha = 0.5
        colormap = plt.get_cmap('jet')
        for image, heatmap, pre_idx, label, frame, patient in zip(images, Imp_map, predict, targets['onset'], targets['frame_id'], targets['patient_id']):
            #heatmap.size => [b, H, W, 3]
            heatmap = (colormap(heatmap) * 255).astype(np.uint16)[:, :, :, :3]
            
            blended = image * alpha + heatmap * (1 - alpha)
            blended = blended.astype(np.uint8)

            self.save_gif(blended, pre_idx, label, frame, patient)
    
    def save_gif(self, image, predict, onset, frame, patient):
        plt.clf()
        fig = plt.figure()
        plt.title(f"label:{self.PVT[onset]}  predict:{self.PVT[predict]}")
        #メモリなし
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #video must be 2d list
        video =[[plt.imshow(img)] for img in image]
        
        video = ani.ArtistAnimation(fig, video,interval=125)
        
        #保存
        video.save(f"CAM_image/Grad_CAM_3D_P{patient:03}_F{frame:04}.gif",
                    writer="pillow")
        plt.close()

    def __del__(self):
        self.delete_handle()
