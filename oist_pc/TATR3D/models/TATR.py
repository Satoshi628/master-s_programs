#coding: utf-8
#----- 標準ライブラリ -----#
#None
#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#----- 自作ライブラリ -----#
from .Unet import UNet
from .Unet_3D import UNet_3D, UNet_3D_FA
from .Transformer import Transformer_3D
from .Positional_Encoding import Sine_Positional_Encoding, Learned_Positional_Encoding, Linear_Positional_Encoding

PAD_ID = -1
None_ID = -2

#3DUNet Featrue addtional and distance atteniton
#None token exist from first time
class TATR_3D(nn.Module):
    def __init__(self, back_bone_path=None, channel=32,
                    pos_mode="Sin",
                    assignment=True,
                    **kwargs):
        super().__init__()
        if assignment:
            self.backbone = UNet_3D_FA(in_channels=1, n_classes=1, channel=channel, **kwargs)
        else:
            self.backbone = UNet_3D(in_channels=1, n_classes=1, channel=channel, **kwargs)
        
        self.d_model = channel * 8  # 32*8=256
        self.transformer = Transformer_3D(d_model=self.d_model, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=F.relu, **kwargs)

        self.vector_embedding = nn.Sequential(
            nn.Conv2d(channel, self.d_model, kernel_size=1),
            nn.ReLU())
        if pos_mode == "Sin":
            self.Positional_Encoding = Sine_Positional_Encoding(self.d_model)
        elif pos_mode == "Learned":
            self.Positional_Encoding = Learned_Positional_Encoding(self.d_model)
        elif pos_mode == "MLP":
            self.Positional_Encoding = Linear_Positional_Encoding(self.d_model)
        else:
            raise ValueError("encode_modeがSin、Learned以外の値を持っています")

        self.None_token = nn.Parameter(torch.randn(1, self.d_model))

        #学習済みモデルロード
        if back_bone_path is not None:
            model_path = back_bone_path
            self.backbone.load_state_dict(torch.load(model_path))

        #backboneの重みは固定しておく
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_backbone(self, inputs):
        #inputsは5次元でなければならない

        # 設定：パラメータの更新なし
        with torch.no_grad():
            _, feature, coordature = self.backbone(inputs)

        return feature, coordature

    def feature_embedding(self, feature, coordinate):
        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        #coordinateにNone追加
        None_token = None_ID * coordinate.new_ones(coordinate.shape[0], coordinate.shape[1], 1, 2)
        coordinate = torch.cat([None_token, coordinate], dim=2)

        #featrueからvector抽出
        coord_take = coordinate[:, :, :, [1, 0]]  #(x,y)=>(h,w)
        coord_take[:, :, :, 0] = 2 * coord_take[:, :, :, 0] / feature.shape[-2] - 1
        coord_take[:, :, :, 1] = 2 * coord_take[:, :, :, 1] / feature.shape[-1] - 1

        B, _, N, _ = coord_take.shape

        length_idx = torch.arange(coord_take.shape[1])[None, :, None, None].expand(B, -1, N, -1)
        length_idx = length_idx.to(coord_take)
        length_idx = 2 * length_idx / coord_take.shape[1] - 1

        coord_take = torch.cat([length_idx, coord_take], dim=-1)

        vector = F.grid_sample(feature, coord_take[:, :, :, None], align_corners=True)
        
        # embedding
        vector = vector.squeeze(-1)
        vector = self.vector_embedding(vector).permute(0, 2, 3, 1)  #[batch,T,num+1,d_model]

        #Noneとmaskのかかった部分は0にする
        vector[coord_take[:, :, :, -1] < 0, :] = 0.
        #0番目はNone token
        vector[:, :, 0] = self.None_token

        #coordinateがマイナスならmaskをTrueに
        mask = coordinate[:, :, :, 0] < 0
        #None tokenは学習対象
        mask[:, 0] = False

        #pos作成
        pos = self.Positional_Encoding(coordinate, feature.shape[-2:])

        return vector, pos, mask, coordinate

    def forward(self, inputs, point=None):
        """model実行関数

        Args:
            inputs (tensor[batch,1,T,H,W]): inputs images
            point (tensor[batch,T,num,2(x,y)], optional): specify position. Defaults to None.

        Returns:
            tensor[batch,T,num,dim]: object vector
            tensor[batch,T,num+1,2(x,y)]: object coordinate
        """
        batch, channel, T, H, W = inputs.size()

        #細胞位置特定
        if point is None:  # 検出位置が指定されていない場合
            #特徴量抽出
            feature, coordinate = self.forward_backbone(inputs)

        else:
            coordinate = point
            feature = self.backbone.get_feature(inputs, coordinate)

        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        #特徴マップを細胞ベクトルに変換
        vector, pos, mask, coordinate = self.feature_embedding(feature, coordinate)

        #vector.size() => [batch size, num cell, dim]
        inputs = {
            "inputs": vector,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        vector = self.transformer(inputs)
        return vector, coordinate

    def tracking_process(self, inputs, coord=None, add_F_dict=None):
        #バッチサイズは1である必要がある
        batch, channel, T, H, W = inputs.size()

        #細胞位置特定、特徴量抽出
        feature, coordinate, add_F_dict = self.backbone.tracking_process(inputs, coord, add_F_dict)

        #特徴マップを細胞ベクトルに変換
        vector, pos, mask, coordinate = self.feature_embedding(feature, coordinate)
        
        inputs = {
            "inputs": vector,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        #vector.size() => [batch size, num cell, dim]
        vector = self.transformer(inputs)
        vector = vector[:, :-1]
        coordinate = coordinate[:, :-1]

        return vector, coordinate, add_F_dict

    def feature_hist(self, inputs, point=None):
        batch, channel, T, H, W = inputs.size()

        #細胞位置特定
        if point is None:  # 検出位置が指定されていない場合
            #特徴量抽出
            feature, coordinate = self.forward_backbone(inputs)

        else:
            coordinate = point
            feature = self.backbone.get_feature(inputs, coordinate)
        
        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        #featrueからvector抽出
        coord_take = coordinate[:, :, :, [1, 0]]  #(x,y)=>(h,w)
        coord_take[:, :, :, 0] = 2 * coord_take[:, :, :, 0] / feature.shape[-2] - 1
        coord_take[:, :, :, 1] = 2 * coord_take[:, :, :, 1] / feature.shape[-1] - 1

        B, _, N, _ = coord_take.shape

        length_idx = torch.arange(coord_take.shape[1])[None, :, None, None].expand(B, -1, N, -1)
        length_idx = length_idx.to(coord_take)
        length_idx = 2 * length_idx / coord_take.shape[1] - 1

        coord_take = torch.cat([length_idx, coord_take], dim=-1)

        vector = F.grid_sample(feature, coord_take[:, :, :, None], align_corners=True)
        
        # embedding
        vector = vector.squeeze(-1)
        vector = self.vector_embedding(vector).permute(0, 2, 3, 1)  #[batch,T,num+1,d_model]
        vector = vector[coord_take[:, :, :, -1] < 0, :]
        vector = (vector ** 2).sum(dim=-1)
        vector = torch.sqrt(vector)
        return vector
