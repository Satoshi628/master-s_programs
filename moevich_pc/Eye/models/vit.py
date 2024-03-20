#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import VisionTransformer
#----- 自作モジュール -----#
#None





class VisionTransformer(VisionTransformer):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.gender_token = MLP([1, *dim_feedforward_layers, embed_dim])
        self.age_token = MLP([1, *dim_feedforward_layers, embed_dim])
        self.smoke_token = MLP([1, *dim_feedforward_layers, embed_dim])


    def forward_features(self, x, info={"gender":None, "age":None, "smoke":None}):
        x = self.patch_embed(x)
        print(x.shape)
        if info.get("gender") is not None:
            g_token = self.gender_token(info["gender"])
            x = torch.cat((x, g_token), dim=1)
        
        if info.get("age") is not None:
            a_token = self.age_token(info["age"])
            x = torch.cat((x, a_token), dim=1)
        
        if info.get("smoke") is not None:
            s_token = self.smoke_token(info["smoke"])
            print(s_token.shape)
            x = torch.cat((x, s_token), dim=1)
        

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x_dist = {"x":x, "atten":[]}
        x_dist = self.blocks(x_dist)
        x = x_dist["x"]
        attens = x_dist["atten"]
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attens
        else:
            return [x[:, 0], x[:, 1]], attens


    def forward(self, x, info={"gender":None, "age":None, "smoke":None}):
        x, attens = self.forward_features(x, info)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, attens



def MLP(channels):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        if i < (n - 1):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=False))
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
    return nn.Sequential(*layers)