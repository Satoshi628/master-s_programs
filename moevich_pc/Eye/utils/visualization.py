#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from tsne_torch import TorchTSNE as TSNE

#----- 自作モジュール -----#
#None

class CAM():
    PVT = ["non PVT", "PVT"]
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model.to(device)
        self.target_layer = target_layer[0] # same pytorch_grad_cam list[torch.module]
        self.device = device
        self.hidden_out_list = []
        self.class_weight = self.model.linear.weight
        self.class_conv3d = nn.Conv3d(11, 13, kernel_size=1, padding=0, bias=False)
        self.class_conv3d.weight = nn.Parameter(self.class_weight[:, :, None, None, None])
        self.class_conv3d = self.class_conv3d.to(device)

        self.handle_f = self.target_layer.register_forward_hook(self.foword_hook)
        
        for param in self.class_conv3d.parameters():
            param.requires_grad = False


    def foword_hook(self, module, inputs, outputs):
        self.hidden_out_list.append(outputs.detach().clone())
    
    def _reset(self):
        self.hidden_out_list = []
    
    def _reset_image_number(self):
        self.image_num = 0

    def delete_handle(self):
        self.handle_f.remove()

    def __call__(self,  inputs, targets):
        self._reset()
        self.model.eval()
        inputs = inputs.to(self.device)
        output = self.model(inputs)
        
        feature = self.hidden_out_list[0]
        class_weight = self.class_conv3d(feature)
        t,h,w = class_weight.shape[-3:]

        self._reset()

        #Edit predict
        predict = output.max(dim=-1)[1]
        class_weight = torch.gather(class_weight, 1, predict[:, None, None, None, None].expand(-1, -1, t, h, w))

        # Mr. haruki and sota's grad cam. This does not consider time dim.
        #重要度マップ作成
        class_weight = class_weight.sum(dim=1)  #[batch, T, H, W]
        
        #[0,1]になるようにmax-min正規化
        min_weight = class_weight.flatten(-3).min(dim=-1)[0]
        max_weight = class_weight.flatten(-3).max(dim=-1)[0]
        class_weight = (class_weight - min_weight[:, None, None, None]) / (max_weight[:, None, None, None] - min_weight[:, None, None, None] + 1e-7)
        
        #Numpy化、inputsのサイズにリサイズ
        upsamp = nn.Upsample(inputs.shape[-3:], mode="trilinear", align_corners=True)
        #bilinearモードは4次元出なくてはならないため
        class_weight = upsamp(class_weight[:, None]).squeeze(1)
        class_weight = class_weight.cpu().detach().numpy()
        
        #Edit input image
        images = (inputs * 255).cpu().detach().numpy().transpose(0, 2, 3, 4, 1)
        images = np.tile(images, [1, 1, 1, 1, 3])

        #Edit targets
        targets = {k: v.cpu().detach().tolist() for k,v in targets.items()}

        #Edit predict
        predict = predict.to("cpu").detach().numpy().astype(np.int16)

        if not os.path.exists("CAM_image"):
            os.mkdir("CAM_image")

        alpha = 0.5
        colormap = plt.get_cmap('jet')
        for image, heatmap, pre_idx, label, frame, patient in zip(images, class_weight, predict, targets['onset'], targets['frame_id'], targets['patient_id']):
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
        video.save(f"CAM_image/CAM_3D_P{patient:03}_F{frame:04}.gif",
                    writer="pillow")
        plt.close()

    def __del__(self):
        self.delete_handle()


class CAM_2D():
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model.to(device)
        self.target_layer = target_layer[0] # same pytorch_grad_cam list[torch.module]
        self.device = device
        self.hidden_out_list = []
        self.hidden_grad_list = []
        self.image_num = 0

        self.handle_f = self.target_layer.register_forward_hook(self.foword_hook)
        self.handle_b = self.target_layer.register_full_backward_hook(self.backword_hook)

    def foword_hook(self, module, inputs, outputs):
        self.hidden_out_list.append(outputs.cpu().detach().clone())
    
    def backword_hook(self, module, grad_in, grad_out):
        self.hidden_grad_list.append(grad_out[0].cpu().detach().clone())

    def _reset(self):
        self.hidden_out_list = []
        self.hidden_grad_list = []
    
    def _reset_image_number(self):
        self.image_num = 0

    def delete_handle(self):
        self.handle_f.remove()
        self.handle_b.remove()

    def __call__(self,  inputs, targets):
        self._reset()
        self.model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        targets = targets.long()
        output = self.model(inputs)

        if isinstance(output, tuple):
            predict = torch.stack([output[0].max(dim=-1)[1].to("cpu").detach(), 
                        output[1].max(dim=-1)[1].to("cpu").detach()],dim=-1)
            loss = torch.gather(output[0],-1,targets[:,:1]).sum() + torch.gather(output[1],-1,targets[:,1:]).sum()
        elif isinstance(output, torch.Tensor):
            predict = output.max(dim=-1)[1].to("cpu").detach().numpy()
            loss = torch.gather(output,-1,targets[:,None]).sum() #binary classification
        
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        feature = self.hidden_out_list[0]
        
        alpha = self.hidden_grad_list[0].mean(dim=(-2, -1))
        #重要度マップ作成
        alpha = F.relu(alpha)
        Imp_map = feature * alpha[:, :, None, None] #[batch, channel, H, W]
        Imp_map = Imp_map.sum(dim=1)  #[batch, H, W]
        
        #[0,1]になるようにmax-min正規化
        min_Imp = Imp_map.flatten(-2).min(dim=-1)[0]
        max_Imp = Imp_map.flatten(-2).max(dim=-1)[0]
        Imp_map = (Imp_map - min_Imp[:, None, None]) / (max_Imp[:, None, None] - min_Imp[:, None, None] + 1e-7)
        
        #Numpy化、inputsのサイズにリサイズ
        upsamp = nn.Upsample(inputs.shape[-2:], mode="bilinear", align_corners=True)
        #bilinearモードは4次元出なくてはならないため
        Imp_map = upsamp(Imp_map[:, None]).squeeze(1)
        Imp_map = Imp_map.cpu().detach().numpy()
        
        #Edit input image
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        images = inputs[:, :3] * std[None,:,None,None] + mean[None,:,None,None]
        images = (images * 255).cpu().detach().numpy().transpose(0, 2, 3, 1)

        if not os.path.exists("CAM_image"):
            os.mkdir("CAM_image")

        alpha = 0.7
        colormap = plt.get_cmap('jet')
        for image, heatmap, pre_idx, label in zip(images, Imp_map, predict, targets.to("cpu")):
            #heatmap.size => [b, H, W, 3]
            heatmap = (colormap(heatmap) * 255).astype(np.uint16)[:, :, :3]
            
            blended = image * alpha + heatmap * (1 - alpha)
            blended = blended.astype(np.uint8)

            self.save_fig(blended, pre_idx, label)
    
    def save_fig(self, image, predict, label):
        plt.clf()
        plt.imshow(image)
        plt.title(self._result(predict, label))
        #メモリなし
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #保存
        plt.savefig(f"CAM_image/{self.image_num:05}.png",
                    bbox_inches='tight',
                    pad_inches=0.1)
        
        self.image_num += 1
        plt.close()

    def _result(self,predict, label):
        if label.shape:
            return f"ARMS2:{predict[0]==label[0]}   CFH:{predict[1]==label[1]}"
        else:
            ARMS2 = f"{predict:02b}"[1] == f"{label:02b}"[1]
            CFH = f"{predict:02b}"[0] == f"{label:02b}"[0]
            return f"ARMS2:{ARMS2}   CFH:{CFH}"
    
    def __del__(self):
        self.delete_handle()



class CAMs():
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


class Edge_tSNE():
    def __init__(self,
                color,
                n_class,
                root_dir="images"):
        
        self.__color = torch.tensor(color)
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        self.n_class = n_class

        
        self.avgPool = nn.AvgPool2d(2 * 7 + 1, stride=1, padding=7)
        self.downsamp = nn.AvgPool2d(2)


    def __call__(self, feature, targets):
        onehot = one_hot_changer(targets, self.n_class, dim=1)

        smooth_map = self.avgPool(onehot)
        # 物体の曲線付近内側までを1.とするフラグを作成
        weight = onehot * smooth_map

        seg_image = self.__color[targets].cuda(targets.device).permute(0, 3, 1, 2)
        
        weight = self.downsamp(weight)
        weight = weight.sum(dim=1)
        seg_image = self.downsamp(seg_image.float()).long()
        feature = self.downsamp(feature)
        feature = feature.permute(0, 2, 3, 1).flatten(0, 2)
        weighted_color = (seg_image * weight.unsqueeze(1)).flatten().cpu().detach().numpy()
        weighted_color = weighted_color.astype(np.uint8).tolist()

        emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(feature.cpu(),device="cpu")

        plt.clf()
        for color, coord in zip(weighted_color, emb):
            plt.plot(coord[0], coord[1], marker=".", color=color)

        plt.savefig(f"edge_feature_Normal.png")
    
    def plot_image(self, image, mode):
        cv2.imwrite("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]), image)

        self.__number_image[mode] += 1

