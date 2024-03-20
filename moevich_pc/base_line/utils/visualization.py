#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#----- 自作モジュール -----#
#None


class IoU_image():
    def __init__(self,
                color,
                root_dir="images",
                input_norm_mean=[0.485, 0.456, 0.406],
                input_norm_std=[0.229, 0.224, 0.225]):
        
        self.__color = torch.tensor(color)
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        self.input_norm_mean = torch.tensor(input_norm_mean)
        self.input_norm_std = torch.tensor(input_norm_std)

        self.__number_image = {"input": 0, "label": 0, "output": 0}
    
    def _reset(self):
        self.__number_image = {"input": 0, "label": 0, "output": 0}

    def save_fig(self, tensor, mode="input"):
        """入力されたtensorを可視化する関数

        Args:
            tensor (tensor[batch,channel,H,W]): 可視化したい画像,入力画像、ラベル、予測画像を入力できる

            mode (str, optional): "input","label","output"かを指定する。label,outputはsegmentation結果を出力する. Defaults to "input".
        """        

        if not mode in ["input", "label", "output"]:
            raise ValueError("modeにinput,label,output以外の値が入力されました")
        
        if mode == "input":
            if tensor.dim() != 4:
                raise ValueError("inputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            #channelが3(RGB)なら正規化を元に戻す
            if tensor.size(1) == 3:
                tensor = self.input_norm_std[None, :, None, None] * tensor + self.input_norm_mean[None, :, None, None]
            
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            tensor = (tensor * 255).permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)
            
            for image in tensor:
                self.plot_image(image, mode)
        
        if mode == "label":
            if tensor.dim() != 3:
                raise ValueError("labelモードですが入力が3次元ではありません")
            #long,cpuモードに変更
            tensor = tensor.long().to("cpu")
            
            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[tensor]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
        
        if mode == "output":
            if tensor.dim() != 4:
                raise ValueError("outputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            class_image = tensor.max(dim=1)[1]
            
            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[class_image]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
    
    def plot_image(self, image, mode):
        cv2.imwrite("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]), image)

        self.__number_image[mode] += 1


"""供養,cv2のほうが書き込みが早く、余白もできないためmatplotを使わなくなった

    def plot_image(self, image, mode):
        plt.clf()
        if image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

        #メモリなし
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #保存
        plt.savefig("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]),
                    bbox_inches='tight',
                    pad_inches=0)
        
        self.__number_image[mode] += 1
"""

class CAM():
    PVT = ["non PVT", "PVT"]
    CAM_dict = {"GradCAM": GradCAM,
                "ScoreCAM": ScoreCAM,
                "GradCAMPlusPlus": GradCAMPlusPlus,
                "AblationCAM": AblationCAM,
                "XGradCAM": XGradCAM,
                "EigenCAM": EigenCAM}
    
    def __init__(self, model, target_layer, device='cpu', CAM="Grad_CAM"):
        cam = self.CAM_dict.get(CAM)
        if cam is None:
            raise KeyError(f"CAM must be included in {list(CAM.keys())}")
        
        self.model = model
        self.cam = cam(model, target_layer, use_cuda=device)

        self.image_num = 0
        if not os.path.exists("CAM_image"):
            os.mkdir("CAM_image")
    
    def _reset(self):
        self.cam = None
        self.image_num = 0

    def __call__(self, inputs: torch.Tensor,label: torch.Tensor):
        """save Cam image

        Args:
            inputs (torch.Tensor[batch,c,H,W]): inputs image
            targets (torch.Tensor[batch]): label
            name (str): save image name
        """        
        if self.cam is None:
            raise ValueError("cam is None")
        
        label = label

        heatmaps = self.cam(input_tensor=inputs, target_category=0)
        predict = self.model(inputs).flatten()
        predict = torch.round(torch.sigmoid(predict)).to("cpu").detach().numpy().astype(np.int16)

        imgs = inputs.to("cpu").detach().permute(0, 2, 3, 1).numpy()
        imgs = np.tile(imgs, [1, 1, 1, 3])
        
        for heatmap, img, pre, onset, frame, patient in zip(heatmaps, imgs, predict, label['onset'].tolist(), label['frame_id'].tolist(), label['patient_id'].tolist()):
            vis = show_cam_on_image(img, heatmap, use_rgb=True)
            self.save_fig(vis, pre, onset, frame, patient)

    def save_fig(self, image, predict, onset, frame, patient):
        plt.clf()
        plt.imshow(image)
        plt.title(f"label:{self.PVT[onset]}  predict:{self.PVT[predict]}")
        #メモリなし
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #保存
        plt.savefig(f"CAM_image/{self.cam.__class__.__name__}_P{patient:03}_F{frame:04}.png",
                    bbox_inches='tight',
                    pad_inches=0.1)
        
        self.image_num += 1
        plt.close()
