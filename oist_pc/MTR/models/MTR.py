#coding: utf-8
#----- Standard Library -----#
import os

#----- Public Package -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#----- Module -----#
from .Unet_3D import UNet_3D, UNet_3D_FA
from .Transformer import Transformer_3D, Transformer_3D_encoder
from .Positional_Encoding import Sine_Positional_Encoding, Learned_Positional_Encoding, Linear_Positional_Encoding

PAD_ID = -1
None_ID = -2


#3DUNet Featrue addtional and distance atteniton
#None token exist from first time
class MTR(nn.Module):
    def __init__(self, back_bone_path=None, channel=32,
                    pos_mode="Sin",
                    assignment=True,
                    move_limit=[25., 30., 35.],
                    **kwargs):
        super().__init__()
        if assignment:
            self.backbone = UNet_3D_FA(in_channels=1, n_classes=1, channel=channel, **kwargs)
        else:
            self.backbone = UNet_3D(in_channels=1, n_classes=1, channel=channel, **kwargs)
        
        self.d_model = channel * 8  # 32*8=256

        self.move_limit = torch.tensor(move_limit)

        self.transformer = Transformer_3D(d_model=self.d_model, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=F.relu, move_limit=move_limit[0], **kwargs)
        
        self.head = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 2*3, 1),
            nn.Sigmoid()
        )

        self.vector_embedding = nn.Sequential(
            nn.Conv2d(channel, self.d_model, kernel_size=1),
            nn.ReLU())

        if pos_mode == "Sin":
            self.Positional_Encoding = Sine_Positional_Encoding(self.d_model)
            self.Positional_Encoding2 = Sine_Positional_Encoding(self.d_model)
        elif pos_mode == "Learned":
            self.Positional_Encoding = Learned_Positional_Encoding(self.d_model)
            self.Positional_Encoding2 = Learned_Positional_Encoding(self.d_model)
        elif pos_mode == "MLP":
            self.Positional_Encoding = Linear_Positional_Encoding(self.d_model)
            self.Positional_Encoding2 = Linear_Positional_Encoding(self.d_model)
        else:
            raise ValueError("encode_modeがSin、Learned以外の値を持っています")

        # load pretrained model
        if back_bone_path is not None:
            root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
            model_path = "/" + os.path.join(*root_dir, back_bone_path, "result", "model.pth")
            self.backbone.load_state_dict(torch.load(model_path))

        # backbone freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_backbone(self, inputs):
        with torch.no_grad():
            _, feature, coordature = self.backbone(inputs)

        return feature, coordature

    def feature_embedding(self, feature, coordinate):
        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        # 特徴マップから検出位置の特徴を取得
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
        vector = self.vector_embedding(vector).permute(0, 2, 3, 1)  #[batch,T,num,d_model]

        vector[coord_take[:, :, :, -1] < 0, :] = 0.

        mask = coordinate[:, :, :, 0] < 0

        pos = self.Positional_Encoding(coordinate, feature.shape[-2:])
        decoder_pos = self.Positional_Encoding2(coordinate, feature.shape[-2:])


        return vector, pos, decoder_pos, mask, coordinate

    def forward(self, inputs, point=None):
        """

        Args:
            inputs (tensor[batch,1,T,H,W]): inputs images
            point (tensor[batch,T,num,2(x,y)], optional): specify position. Defaults to None.

        Returns:
            tensor[batch,T,num,dim]: object vector
            tensor[batch,T,num+1,2(x,y)]: object coordinate
        """
        batch, channel, T, H, W = inputs.size()

        if point is None:
            feature, coordinate = self.forward_backbone(inputs)

        else:
            coordinate = point
            feature = self.backbone.get_feature(inputs, coordinate)

        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]
        vector, pos, decoder_pos, mask, coordinate = self.feature_embedding(feature, coordinate)

        #vector.size() => [batch size, num cell, dim]
        inputs = {
            "encoder": vector,
            "decoder": decoder_pos,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        #vector.size() => [batch, T, num, d_model]
        vector = self.transformer(inputs)
        #move.size() => [batch, 6, T, num]
        move = self.head(vector.permute(0, 3, 1, 2))

        #move.size() => [batch, T, num, 3, 2(x, y)]
        move = move.view(move.size(0), 3, 2, move.size(2), move.size(3)).permute(0, 3, 4, 1, 2)


        # value's range convert to [-1, 1] from [0, 1]
        move = 2 * move - 1
        move_limit = self.move_limit.to(move.device)
        move = move * move_limit[None, None, None :, None]
        
        return move, coordinate

    def tracking_process(self, inputs, coord=None, add_F_dict=None):
        # Batch size must be 1
        batch, channel, T, H, W = inputs.size()

        feature, coordinate, add_F_dict = self.backbone.tracking_process(inputs, coord, add_F_dict)

        vector, pos, mask, coordinate = self.feature_embedding(feature, coordinate)
        
        inputs = {
            "inputs": vector,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        #vector.size() => [batch size, num cell, dim]
        vector = self.transformer(inputs)
        move = self.head(vector.permute(0, 3, 1, 2))
        move = move.view(move.size(0), 3, 2, move.size(2), move.size(3))
        move = move[:, :, :, :-1]
        coordinate = coordinate[:, :-1]

        return move, coordinate, add_F_dict



#3DUNet Featrue addtional and distance atteniton
#None token exist from first time
class MTR2(nn.Module):
    def __init__(self, back_bone_path=None, channel=32,
                    pos_mode="Sin",
                    assignment=True,
                    move_limit=[25., 30., 35.],
                    **kwargs):
        super().__init__()
        self.backbone = UNet_3D(in_channels=3, n_classes=1, channel=channel, **kwargs)
        
        self.d_model = channel * 8  # 32*8=256

        self.move_limit = torch.tensor(move_limit)

        self.transformer = Transformer_3D_encoder(d_model=self.d_model, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=F.relu, move_limit=move_limit[0], **kwargs)
        
        self.head = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 2, 1),
            nn.Sigmoid()
        )

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

        # load pretrained model

    def forward_backbone(self, inputs):
        x, feature, _ = self.backbone(inputs)

        return x, feature

    def feature_embedding(self, feature, coordinate):
        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        # 特徴マップから検出位置の特徴を取得
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
        vector = self.vector_embedding(vector).permute(0, 2, 3, 1)  #[batch,T,num,d_model]

        vector[coord_take[:, :, :, -1] < 0, :] = 0.

        mask = coordinate[:, :, :, 0] < 0

        pos = self.Positional_Encoding(coordinate, feature.shape[-2:])


        return vector, pos, mask, coordinate

    def forward(self, inputs, point=None):
        """

        Args:
            inputs (tensor[batch,1,T,H,W]): inputs images
            point (tensor[batch,T,num,2(x,y)], optional): specify position. Defaults to None.

        Returns:
            tensor[batch,T,num,dim]: object vector
            tensor[batch,T,num+1,2(x,y)]: object coordinate
        """
        batch, channel, T, H, W = inputs.size()

        x, feature = self.forward_backbone(inputs)

        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]
        vector, pos, mask, point = self.feature_embedding(feature, point)

        #vector.size() => [batch size, num cell, dim]
        inputs = {
            "encoder": vector,
            "pos": pos,
            "mask": mask,
            "coord": point
        }

        #vector.size() => [batch, T, num, d_model]
        vector = self.transformer(inputs)
        #move.size() => [batch, 6, T, num]
        move = self.head(vector.permute(0, 3, 1, 2))

        #move.size() => [batch, T, num, 3, 2(x, y)]
        move = move.view(move.size(0), 1, 2, move.size(2), move.size(3)).permute(0, 3, 4, 1, 2)


        # value's range convert to [-1, 1] from [0, 1]
        move = 2 * move - 1
        move_limit = self.move_limit.to(move.device)
        move = move * move_limit[None, None, None :, None]
        
        return x, feature, move, point

    def tracking_process(self, inputs, coord=None, add_F_dict=None):
        # Batch size must be 1
        batch, channel, T, H, W = inputs.size()

        feature, coordinate, add_F_dict = self.backbone.tracking_process(inputs, coord, add_F_dict)

        vector, pos, mask, coordinate = self.feature_embedding(feature, coordinate)
        
        inputs = {
            "inputs": vector,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        #vector.size() => [batch size, num cell, dim]
        vector = self.transformer(inputs)
        move = self.head(vector.permute(0, 3, 1, 2))
        move = move.view(move.size(0), 3, 2, move.size(2), move.size(3))
        move = move[:, :, :, :-1]
        coordinate = coordinate[:, :-1]

        return move, coordinate, add_F_dict



#3DUNet Featrue addtional and distance atteniton
#None token exist from first time
class MTR3(nn.Module):
    def __init__(self, back_bone_path=None, channel=32,
                    pos_mode="Sin",
                    assignment=True,
                    move_limit=[25., 30., 35.],
                    **kwargs):
        super().__init__()
        self.backbone = UNet_3D(in_channels=1, n_classes=1, channel=channel, **kwargs)
        
        self.d_model = channel * 8  # 32*8=256

        self.move_limit = torch.tensor(move_limit)

        self.head = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, self.d_model, 1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 2, 1),
            nn.Sigmoid()
        )

        self.vector_embedding = nn.Sequential(
            nn.Conv2d(channel, self.d_model, kernel_size=1),
            nn.ReLU())

        # load pretrained model

    def forward_backbone(self, inputs):
        x, feature, _ = self.backbone(inputs)

        return x, feature

    def feature_embedding(self, feature, coordinate):
        #feature.size => [batch,channel,T,H,W]
        #coordinate.size => [batch,T,num,2(x,y)]

        # 特徴マップから検出位置の特徴を取得
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
        vector = self.vector_embedding(vector).permute(0, 2, 3, 1)  #[batch,T,num,d_model]

        vector[coord_take[:, :, :, -1] < 0, :] = 0.

        return vector, coordinate

    def forward(self, inputs, point=None):
        """

        Args:
            inputs (tensor[batch,1,T,H,W]): inputs images
            point (tensor[batch,T,num,2(x,y)], optional): specify position. Defaults to None.

        Returns:
            tensor[batch,T,num,dim]: object vector
            tensor[batch,T,num+1,2(x,y)]: object coordinate
        """
        batch, channel, T, H, W = inputs.size()

        x, feature = self.forward_backbone(inputs)

        #feature.size => [batch,channel,T,H,W]
        vector, point = self.feature_embedding(feature, point)

        #move.size() => [batch, 6, T, num]
        move = self.head(vector.permute(0, 3, 1, 2))

        #move.size() => [batch, T, num, 3, 2(x, y)]
        move = move.view(move.size(0), 1, 2, move.size(2), move.size(3)).permute(0, 3, 4, 1, 2)


        # value's range convert to [-1, 1] from [0, 1]
        move = 2 * move - 1
        move_limit = self.move_limit.to(move.device)
        move = move * move_limit[None, None, None :, None]
        
        return x, feature, move, point

    def tracking_process(self, inputs, coord=None, add_F_dict=None):
        # Batch size must be 1
        batch, channel, T, H, W = inputs.size()

        feature, coordinate, add_F_dict = self.backbone.tracking_process(inputs, coord, add_F_dict)

        vector, pos, mask, coordinate = self.feature_embedding(feature, coordinate)
        
        inputs = {
            "inputs": vector,
            "pos": pos,
            "mask": mask,
            "coord": coordinate
        }

        #vector.size() => [batch size, num cell, dim]
        vector = self.transformer(inputs)
        move = self.head(vector.permute(0, 3, 1, 2))
        move = move.view(move.size(0), 3, 2, move.size(2), move.size(3))
        move = move[:, :, :, :-1]
        coordinate = coordinate[:, :-1]

        return move, coordinate, add_F_dict
